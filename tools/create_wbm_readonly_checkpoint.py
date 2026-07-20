"""Create a read-only provenance checkpoint before a WBM physical run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CODE_PREFIXES = ("src/", "tools/", "tests/", "configs/")
CODE_FILES = {"pyproject.toml", "README.md", "AGENTS.md"}


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    ).stdout


def _current_code_tree(repo: Path) -> tuple[str, list[dict[str, Any]]]:
    tracked = _git(repo, "ls-files", "-z").decode("utf-8").split("\0")
    untracked = _git(
        repo,
        "ls-files",
        "--others",
        "--exclude-standard",
        "-z",
    ).decode("utf-8").split("\0")
    names = sorted(
        {
            name
            for name in (*tracked, *untracked)
            if name
            and (name in CODE_FILES or any(name.startswith(prefix) for prefix in CODE_PREFIXES))
        }
    )
    records = []
    encoded = bytearray()
    for name in names:
        path = repo / name
        if not path.is_file():
            continue
        digest = _sha256_file(path)
        records.append({"path": name, "sha256": digest, "bytes": path.stat().st_size})
        encoded.extend(name.encode("utf-8"))
        encoded.extend(b"\0")
        encoded.extend(digest.encode())
        encoded.extend(b"\n")
    return _sha256_bytes(bytes(encoded)), records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--conda-environment", default="llm")
    parser.add_argument("--artifact", type=Path, action="append", default=[])
    parser.add_argument("--execution-source", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.output.resolve().is_relative_to(repo):
        parser.error("checkpoint must remain outside the repository")
    if args.output.exists():
        raise FileExistsError(f"immutable checkpoint exists: {args.output}")
    missing = [path for path in (*args.artifact, *args.execution_source) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint inputs do not exist: {missing}")
    tree_sha, tree_files = _current_code_tree(repo)
    diff = _git(repo, "diff", "--binary", "HEAD", "--", ".")
    status = _git(repo, "status", "--porcelain=v1", "-z").decode("utf-8")
    environment_lock = subprocess.run(
        ["conda", "list", "--explicit", "-n", args.conda_environment],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    payload = {
        "schema_version": "wbm-readonly-checkpoint-v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "scope": "pre_physical_run_provenance_checkpoint_not_a_git_commit",
        "repo": str(repo),
        "git_head": _git(repo, "rev-parse", "HEAD").decode().strip(),
        "git_branch": _git(repo, "branch", "--show-current").decode().strip(),
        "git_status_porcelain_v1_z_sha256": _sha256_bytes(status.encode("utf-8")),
        "tracked_diff_binary_sha256": _sha256_bytes(diff),
        "current_code_tree_sha256": tree_sha,
        "current_code_tree_files": tree_files,
        "execution_sources": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in args.execution_source
        ],
        "external_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in args.artifact
        ],
        "environment": {
            "conda_environment": args.conda_environment,
            "explicit_lock_sha256": _sha256_bytes(environment_lock.encode("utf-8")),
            "explicit_lock": environment_lock.splitlines(),
        },
        "guardrails": [
            "checkpoint records a dirty worktree without committing or pushing it",
            "external datasets and outputs remain outside Git",
            "a post-checkpoint source or artifact change requires a new checkpoint",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.chmod(args.output, stat.S_IREAD)
    print(f"code_tree={tree_sha}")
    print(f"diff={payload['tracked_diff_binary_sha256']}")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
