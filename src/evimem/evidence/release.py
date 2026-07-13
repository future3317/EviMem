"""Versioned immutable evidence release manager.

A release is an immutable, checksumed snapshot of evidence blocks:

    dataset/evidence/releases/<release_id>/
        blocks.parquet          # all evidence blocks
        manifest.json           # provenance + counts + checksums
        source_index.parquet    # one row per (doi, source, block_count)
        checksums.sha256        # checksums of the three files above

``dataset/evidence/CURRENT`` is a pointer file naming the active release.

Design goals:
* Immutability: once created, a release is never modified.
* Reproducibility: checksums + manifest make every release auditable.
* Source of truth: the release directory is the canonical evidence store;
  ``CURRENT`` is only a convenience pointer.
* Isolation: a release is self-contained and never depends on mutable legacy
  evidence files.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from structlog import get_logger

from evimem.evidence.doi import normalize_doi

logger = get_logger()

RELEASE_SCHEMA_VERSION = "v2.0.0"
CURRENT_POINTER_NAME = "CURRENT"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _get_git_commit(fallback: str = "unknown") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[3],
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return fallback


def _compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _stable_json_bytes(obj: Any) -> bytes:
    """Canonical JSON bytes for deterministic hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def block_checksum(block: dict[str, Any]) -> str:
    """Return the checksum of a release block without runtime-only annotations."""

    payload = {
        key: value
        for key, value in block.items()
        if key not in {"evidence_release_id", "evidence_block_checksum"}
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class EvidenceRelease:
    """A resolved evidence release."""

    release_id: str
    path: Path
    blocks_path: Path
    manifest_path: Path
    source_index_path: Path
    checksums_path: Path
    manifest: dict[str, Any] = field(repr=False)

    @property
    def blocks_sha256(self) -> str:
        return str(self.manifest.get("blocks_sha256", ""))

    @property
    def doi_count(self) -> int:
        return int(self.manifest.get("doi_count", 0))

    @property
    def block_count(self) -> int:
        return int(self.manifest.get("block_count", 0))


class EvidenceReleaseManager:
    """Create, list, verify, and load versioned evidence releases."""

    def __init__(self, root: str | Path | None = None):
        if root is None:
            root = Path(__file__).resolve().parents[3] / "dataset" / "evidence"
        self.root = Path(root).resolve()
        self.releases_dir = self.root / "releases"
        self.current_path = self.root / CURRENT_POINTER_NAME

    def _release_dir(self, release_id: str) -> Path:
        return (self.releases_dir / release_id).resolve()

    def _validate_release_id(self, release_id: str) -> None:
        if not release_id:
            raise ValueError("release_id must be non-empty")
        if "/" in release_id or "\\" in release_id or ".." in release_id:
            raise ValueError(f"release_id contains path separators: {release_id}")

    def _build_source_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate a per-(doi, source) index from block-level data."""
        required = {"doi", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"blocks DataFrame missing required columns: {missing}")

        grouped = (
            df.groupby(["doi", "source"], dropna=False)
            .size()
            .reset_index(name="block_count")
        )
        grouped["doi_normalized"] = grouped["doi"].apply(normalize_doi)
        return grouped[["doi", "doi_normalized", "source", "block_count"]]

    def _build_manifest(
        self,
        *,
        release_id: str,
        blocks_path: Path,
        source_index_path: Path,
        blocks_df: pd.DataFrame,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        blocks_sha256 = _compute_sha256(blocks_path)
        source_index_sha256 = _compute_sha256(source_index_path)

        doi_count = int(blocks_df["doi"].nunique()) if "doi" in blocks_df.columns else 0
        block_count = len(blocks_df)

        source_distribution: dict[str, int] = {}
        if "source" in blocks_df.columns:
            source_distribution = blocks_df["source"].value_counts().to_dict()

        domain_distribution: dict[str, int] = {}
        if "domain_name" in blocks_df.columns:
            domain_distribution = blocks_df["domain_name"].value_counts().to_dict()

        manifest: dict[str, Any] = {
            "schema_version": RELEASE_SCHEMA_VERSION,
            "release_id": release_id,
            "created_at": _now_iso(),
            "blocks_file": blocks_path.name,
            "blocks_sha256": blocks_sha256,
            "source_index_file": source_index_path.name,
            "source_index_sha256": source_index_sha256,
            "doi_count": doi_count,
            "block_count": block_count,
            "source_distribution": source_distribution,
            "domain_distribution": domain_distribution,
            "git_commit": _get_git_commit(),
        }
        if metadata:
            manifest["metadata"] = metadata
        return manifest

    def _write_checksums(self, release_dir: Path, manifest: dict[str, Any]) -> Path:
        checksums = {
            manifest["blocks_file"]: manifest["blocks_sha256"],
            manifest["source_index_file"]: manifest["source_index_sha256"],
            "manifest.json": _compute_sha256(release_dir / "manifest.json"),
        }
        path = release_dir / "checksums.sha256"
        lines = [f"{digest}  {name}\n" for name, digest in sorted(checksums.items())]
        path.write_text("".join(lines), encoding="utf-8")
        return path

    def create_release(
        self,
        blocks: pd.DataFrame | list[dict[str, Any]],
        *,
        release_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRelease:
        """Create a new immutable evidence release.

        Args:
            blocks: Evidence block records. Must contain at least ``doi`` and
                ``source`` columns.
            release_id: Optional release identifier. If omitted, a timestamp
                identifier is generated.
            metadata: Optional provenance metadata (e.g. build command, input
                manifest path, git branch).

        Returns:
            ``EvidenceRelease`` describing the created release.
        """
        if isinstance(blocks, pd.DataFrame):
            df = blocks.copy()
        else:
            df = pd.DataFrame(blocks)

        if df.empty:
            raise ValueError("Cannot create an evidence release from empty blocks")

        required = {"doi", "source"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"blocks DataFrame missing required columns: {missing}")
        df["doi"] = df["doi"].map(normalize_doi)
        if (df["doi"] == "").any():
            raise ValueError("every evidence block requires a valid DOI")
        if "block_id" not in df.columns:
            df["block_id"] = ""
        for position, index in enumerate(df.index):
            block_id = str(df.at[index, "block_id"] or "").strip()
            if not block_id:
                digest = hashlib.sha256(
                    _stable_json_bytes(
                        {
                            "doi": df.at[index, "doi"],
                            "source": str(df.at[index, "source"]),
                            "text": str(df.at[index, "text"] if "text" in df.columns else ""),
                            "index": position,
                        }
                    )
                ).hexdigest()[:24]
                df.at[index, "block_id"] = f"block-{digest}"
        if df["block_id"].astype(str).duplicated().any():
            raise ValueError("evidence block_id values must be unique within a release")

        if release_id is None:
            release_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self._validate_release_id(release_id)

        release_dir = self._release_dir(release_id)
        if release_dir.exists():
            raise FileExistsError(f"Release already exists: {release_dir}")
        release_dir.mkdir(parents=True, exist_ok=False)

        try:
            blocks_path = release_dir / "blocks.parquet"
            source_index_path = release_dir / "source_index.parquet"
            manifest_path = release_dir / "manifest.json"

            df.to_parquet(blocks_path, index=False)
            source_index = self._build_source_index(df)
            source_index.to_parquet(source_index_path, index=False)

            manifest = self._build_manifest(
                release_id=release_id,
                blocks_path=blocks_path,
                source_index_path=source_index_path,
                blocks_df=df,
                metadata=metadata,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8",
            )

            checksums_path = self._write_checksums(release_dir, manifest)

            logger.info(
                "evidence_release_created",
                release_id=release_id,
                release_dir=str(release_dir),
                doi_count=manifest["doi_count"],
                block_count=manifest["block_count"],
            )

            return EvidenceRelease(
                release_id=release_id,
                path=release_dir,
                blocks_path=blocks_path,
                manifest_path=manifest_path,
                source_index_path=source_index_path,
                checksums_path=checksums_path,
                manifest=manifest,
            )
        except Exception:
            # Fail clean: do not leave a partially-written release behind.
            shutil.rmtree(release_dir, ignore_errors=True)
            raise

    def list_releases(self) -> list[str]:
        """Return sorted release IDs."""
        if not self.releases_dir.exists():
            return []
        return sorted(
            d.name for d in self.releases_dir.iterdir() if d.is_dir()
        )

    def get_release(self, release_id: str) -> EvidenceRelease:
        """Resolve a release by ID."""
        self._validate_release_id(release_id)
        release_dir = self._release_dir(release_id)
        if not release_dir.exists():
            raise FileNotFoundError(f"Release not found: {release_id}")

        manifest_path = release_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Release manifest missing: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return EvidenceRelease(
            release_id=release_id,
            path=release_dir,
            blocks_path=release_dir / manifest.get("blocks_file", "blocks.parquet"),
            manifest_path=manifest_path,
            source_index_path=release_dir / manifest.get(
                "source_index_file", "source_index.parquet"
            ),
            checksums_path=release_dir / "checksums.sha256",
            manifest=manifest,
        )

    def get_current_release(self) -> EvidenceRelease | None:
        """Return the release named by ``CURRENT``, or None."""
        if not self.current_path.exists():
            return None
        try:
            data = json.loads(self.current_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        release_id = data.get("release_id")
        if not release_id:
            return None
        return self.get_release(release_id)

    def set_current_release(self, release_id: str) -> None:
        """Atomically update the ``CURRENT`` pointer."""
        release = self.get_release(release_id)
        pointer = {
            "release_id": release.release_id,
            "release_path": str(release.path.relative_to(self.root)),
            "updated_at": _now_iso(),
        }
        tmp = self.current_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(pointer, indent=2, sort_keys=True, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.current_path)
        logger.info("evidence_current_release_set", release_id=release_id)

    def verify_release(self, release_id: str) -> dict[str, Any]:
        """Verify a release's checksums and manifest consistency.

        Returns a dict with ``ok`` (bool) and per-file results.
        """
        release = self.get_release(release_id)
        results: dict[str, Any] = {"release_id": release_id, "ok": True, "files": {}}

        files_to_check = {
            release.manifest["blocks_file"]: release.manifest["blocks_sha256"],
            release.manifest["source_index_file"]: release.manifest["source_index_sha256"],
            "manifest.json": _compute_sha256(release.manifest_path),
        }

        for name, expected in files_to_check.items():
            path = release.path / name
            if not path.exists():
                results["files"][name] = {"exists": False, "ok": False}
                results["ok"] = False
                continue
            actual = _compute_sha256(path)
            ok = actual == expected
            results["files"][name] = {
                "exists": True,
                "expected": expected,
                "actual": actual,
                "ok": ok,
            }
            if not ok:
                results["ok"] = False

        # Cross-check checksums.sha256 if present.
        checksums_path = release.path / "checksums.sha256"
        if checksums_path.exists():
            checksums_ok = True
            try:
                for line in checksums_path.read_text(encoding="utf-8").strip().splitlines():
                    digest, _, fname = line.strip().partition("  ")
                    file_path = release.path / fname
                    if file_path.exists():
                        actual = _compute_sha256(file_path)
                        if actual != digest:
                            checksums_ok = False
                            break
            except Exception:
                checksums_ok = False
            results["files"]["checksums.sha256"] = {"ok": checksums_ok}
            if not checksums_ok:
                results["ok"] = False

        return results

    def load_blocks(self, release_id: str) -> pd.DataFrame:
        """Load the blocks parquet for a release."""
        release = self.get_release(release_id)
        return pd.read_parquet(release.blocks_path)

    def load_source_index(self, release_id: str) -> pd.DataFrame:
        """Load the source index parquet for a release."""
        release = self.get_release(release_id)
        return pd.read_parquet(release.source_index_path)

    def load_by_doi(
        self,
        release_id: str,
        doi: str,
        *,
        domain_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return evidence blocks for a DOI from a release.

        This is the release-aware replacement for ``load_evidence_by_doi``.
        """
        doi_clean = normalize_doi(doi)
        df = self.load_blocks(release_id)
        if df.empty or "doi" not in df.columns or not doi_clean:
            return []

        normalized = df["doi"].astype(str).map(normalize_doi)
        rows = df[normalized == doi_clean]
        if domain_name and "domain_name" in df.columns:
            rows = rows[rows["domain_name"] == domain_name]

        records = rows.to_dict("records")
        for block in records:
            block["evidence_release_id"] = release_id
            block["evidence_block_checksum"] = block_checksum(block)
        return records


__all__ = [
    "CURRENT_POINTER_NAME",
    "EvidenceRelease",
    "EvidenceReleaseManager",
    "RELEASE_SCHEMA_VERSION",
    "block_checksum",
]
