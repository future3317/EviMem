"""Freeze a small development-only roster for the E52 two-step audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_sha256(values: list[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(values)).encode()
    ).hexdigest()


def build(
    *,
    task_path: Path,
    development_crossfit_path: Path,
    output: Path,
    system_count: int = 10,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("equivalence manifests must remain outside Git")
    crossfit = json.loads(development_crossfit_path.read_text(encoding="utf-8"))
    if crossfit.get("task_sha256") != _sha256(task_path):
        raise ValueError("development cross-fit manifest does not match task")
    eligible = sorted(str(value) for value in crossfit["eligible_systems"])
    if len(eligible) != len(set(eligible)):
        raise ValueError("development systems must be unique")
    if system_count < 1 or system_count >= len(eligible):
        raise ValueError("system_count must leave at least one fit system")
    release_id = str(crossfit["release_id"])
    ranked = sorted(
        eligible,
        key=lambda system: hashlib.sha256(
            f"{release_id}||e52-two-step-equivalence-v1||{system}".encode()
        ).hexdigest(),
    )
    selected = ranked[:system_count]
    result = {
        "schema_version": 1,
        "status": "e52_two_step_equivalence_roster_frozen",
        "release_id": release_id,
        "task_sha256": _sha256(task_path),
        "eligible_system_count": len(eligible),
        "eligible_systems": eligible,
        "fold_count": 1,
        "folds": [
            {
                "fold_index": 0,
                "query_systems": selected,
                "system_count": len(selected),
                "fit_system_count": len(eligible) - len(selected),
            }
        ],
        "selected_system_set_sha256": _set_sha256(selected),
        "assignment_rule": (
            "lowest SHA256(release_id || e52-two-step-equivalence-v1 || system)"
        ),
        "assignment_uses_target_outcomes": False,
        "parent_development_crossfit_sha256": _sha256(development_crossfit_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--development-crossfit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-count", type=int, default=10)
    args = parser.parse_args()
    result = build(
        task_path=args.task,
        development_crossfit_path=args.development_crossfit,
        output=args.output,
        system_count=args.system_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
