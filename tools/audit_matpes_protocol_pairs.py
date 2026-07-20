"""Audit official MatPES PBE/r2SCAN files before building a protocol task.

The audit is deliberately independent of the policy runner.  It proves that a
``matpes_id`` denotes the same configuration in both protocols and records the
places where a future train/evaluation split can leak through an original MP
parent.  It never writes paired target outcomes into the Git repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from matmem.matpes_data import (
    MATPES_PBE_STEM,
    MATPES_R2SCAN_STEM,
    MATPES_SPLITS,
    MatPESCompactConfiguration,
    compact_matpes_configuration,
)


@dataclass(frozen=True, slots=True)
class FileProfile:
    path: str
    size_bytes: int
    row_count: int
    sha256: str
    hub_revision: str | None
    hub_object_sha256: str | None


def _metadata(path: Path) -> tuple[str | None, str | None]:
    metadata = path.parent / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
    if not metadata.exists():
        return None, None
    lines = metadata.read_text(encoding="utf-8").splitlines()
    revision = lines[0].strip() if lines else None
    object_hash = lines[1].strip() if len(lines) > 1 else None
    return revision or None, object_hash or None


def _rows(
    path: Path,
    *,
    split: str,
    digest: Any,
):
    """Yield rows while hashing bytes; no full JSONL payload is retained."""

    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {split}:{line_number}") from exc
            identifier = str(row.get("matpes_id", "")).strip()
            yield identifier, row


def _file_profile(path: Path, *, row_count: int, digest: Any) -> FileProfile:
    revision, object_hash = _metadata(path)
    profile = FileProfile(
        path=str(path.resolve()),
        size_bytes=path.stat().st_size,
        row_count=row_count,
        sha256=digest.hexdigest(),
        hub_revision=revision,
        hub_object_sha256=object_hash,
    )
    if object_hash is not None and object_hash != profile.sha256:
        raise ValueError(f"Hugging Face object hash mismatch for {path}")
    return profile


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _count_parent_split_leakage(
    parent_splits: dict[str, set[str]],
) -> dict[str, Any]:
    leaking = {parent: splits for parent, splits in parent_splits.items() if len(splits) > 1}
    return {
        "unique_original_mp_ids": len(parent_splits),
        "cross_split_original_mp_id_count": len(leaking),
        "cross_split_original_mp_id_examples": [
            {"original_mp_id": parent, "splits": sorted(splits)}
            for parent, splits in sorted(leaking.items())[:20]
        ],
    }


def run(*, pbe_root: Path, r2scan_root: Path, output: Path) -> dict[str, Any]:
    """Run a complete strict-pair audit and write one immutable JSON summary."""

    if output.exists():
        raise FileExistsError("MatPES protocol-pair audit cannot overwrite output")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("MatPES audit output must remain outside Git")

    pbe_index: dict[str, MatPESCompactConfiguration] = {}
    pbe_duplicates: Counter[str] = Counter()
    pbe_profiles: list[FileProfile] = []
    parent_splits: dict[str, set[str]] = defaultdict(set)
    functional_counts: dict[str, Counter[str]] = {
        "pbe": Counter(),
        "r2scan": Counter(),
    }
    for split in MATPES_SPLITS:
        path = pbe_root / f"{MATPES_PBE_STEM}-{split}.jsonl"
        digest = hashlib.sha256()
        row_count = 0
        for identifier, row in _rows(path, split=split, digest=digest):
            row_count += 1
            functional_counts["pbe"][str(row.get("functional"))] += 1
            record = compact_matpes_configuration(row, split=split)
            if identifier in pbe_index:
                pbe_duplicates[identifier] += 1
                continue
            pbe_index[identifier] = record
            if record.original_mp_id is not None:
                parent_splits[record.original_mp_id].add(split)
        pbe_profiles.append(_file_profile(path, row_count=row_count, digest=digest))

    r2_seen: set[str] = set()
    r2_duplicates: Counter[str] = Counter()
    r2_profiles: list[FileProfile] = []
    mismatch_counts: Counter[str] = Counter()
    mismatch_examples: dict[str, list[str]] = defaultdict(list)
    deltas: list[float] = []
    formation_deltas: list[float] = []
    valid_pair_ids: list[str] = []
    same_upstream_split_count = 0
    chemsys_counts: Counter[str] = Counter()
    formation_chemsys_counts: Counter[str] = Counter()
    pbe_formation_count = sum(
        record.formation_energy_ev_per_atom is not None for record in pbe_index.values()
    )
    r2_formation_count = 0
    for split in MATPES_SPLITS:
        path = r2scan_root / f"{MATPES_R2SCAN_STEM}-{split}.jsonl"
        digest = hashlib.sha256()
        row_count = 0
        for identifier, row in _rows(path, split=split, digest=digest):
            row_count += 1
            functional_counts["r2scan"][str(row.get("functional"))] += 1
            if identifier in r2_seen:
                r2_duplicates[identifier] += 1
                continue
            r2_seen.add(identifier)
            target = compact_matpes_configuration(row, split=split)
            r2_formation_count += int(target.formation_energy_ev_per_atom is not None)
            source = pbe_index.get(identifier)
            if source is None:
                mismatch_counts["missing_in_pbe"] += 1
                if len(mismatch_examples["missing_in_pbe"]) < 20:
                    mismatch_examples["missing_in_pbe"].append(identifier)
                continue
            comparisons = {
                "split": source.split == target.split,
                "nsites": source.nsites == target.nsites,
                "chemsys": source.chemsys == target.chemsys,
                "composition": source.composition_key == target.composition_key,
                "original_mp_id": source.original_mp_id == target.original_mp_id,
                "exact_geometry": (
                    source.exact_geometry_sha256 == target.exact_geometry_sha256
                ),
                "rounded_geometry_1e-10": (
                    source.rounded_geometry_sha256 == target.rounded_geometry_sha256
                ),
                "raw_structure": source.raw_structure_sha256 == target.raw_structure_sha256,
            }
            for name, agrees in comparisons.items():
                if not agrees:
                    mismatch_counts[name] += 1
                    if len(mismatch_examples[name]) < 20:
                        mismatch_examples[name].append(identifier)
            task_pair_agrees = all(
                comparisons[name]
                for name in (
                    "nsites",
                    "chemsys",
                    "composition",
                    "original_mp_id",
                    "rounded_geometry_1e-10",
                )
            )
            if task_pair_agrees:
                deltas.append(target.energy_ev_per_atom - source.energy_ev_per_atom)
                chemsys_counts[target.chemsys] += 1
                valid_pair_ids.append(identifier)
                same_upstream_split_count += int(comparisons["split"])
                if (
                    source.formation_energy_ev_per_atom is not None
                    and target.formation_energy_ev_per_atom is not None
                ):
                    formation_deltas.append(
                        target.formation_energy_ev_per_atom
                        - source.formation_energy_ev_per_atom
                    )
                    formation_chemsys_counts[target.chemsys] += 1
        r2_profiles.append(_file_profile(path, row_count=row_count, digest=digest))

    missing_in_r2scan = sorted(set(pbe_index) - r2_seen)
    mismatch_counts["missing_in_r2scan"] = len(missing_in_r2scan)
    mismatch_examples["missing_in_r2scan"] = missing_in_r2scan[:20]
    for field in (
        "missing_in_pbe",
        "split",
        "nsites",
        "chemsys",
        "composition",
        "original_mp_id",
        "exact_geometry",
        "rounded_geometry_1e-10",
        "raw_structure",
    ):
        mismatch_counts[field] += 0
    task_pair_fields = (
        "nsites",
        "chemsys",
        "composition",
        "original_mp_id",
        "rounded_geometry_1e-10",
    )
    release_parity_gate = (
        not pbe_duplicates
        and not r2_duplicates
        and mismatch_counts["missing_in_pbe"] == 0
        and mismatch_counts["missing_in_r2scan"] == 0
        and mismatch_counts["split"] == 0
        and all(mismatch_counts[field] == 0 for field in task_pair_fields)
        and len(deltas) == len(pbe_index) == len(r2_seen)
    )
    same_configuration_pair_gate = (
        not pbe_duplicates
        and not r2_duplicates
        and all(mismatch_counts[field] == 0 for field in task_pair_fields)
        and len(deltas) == len(set(pbe_index) & r2_seen)
        and len(deltas) > 0
    )
    pair_id_set_sha256 = hashlib.sha256(
        "".join(f"{identifier}\n" for identifier in sorted(valid_pair_ids)).encode()
    ).hexdigest()
    counts = np.asarray(list(chemsys_counts.values()), dtype=np.int64)
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "strict_same_configuration_protocol_pair_audit",
        "pbe_root": str(pbe_root.resolve()),
        "r2scan_root": str(r2scan_root.resolve()),
        "files": {
            "pbe": [asdict(profile) for profile in pbe_profiles],
            "r2scan": [asdict(profile) for profile in r2_profiles],
        },
        "profiles": {
            "pbe_rows": sum(profile.row_count for profile in pbe_profiles),
            "r2scan_rows": sum(profile.row_count for profile in r2_profiles),
            "pbe_unique_ids": len(pbe_index),
            "r2scan_unique_ids": len(r2_seen),
            "pbe_duplicate_id_rows": sum(pbe_duplicates.values()),
            "r2scan_duplicate_id_rows": sum(r2_duplicates.values()),
            "functional_counts": {
                name: dict(sorted(counts_.items()))
                for name, counts_ in functional_counts.items()
            },
            "pbe_formation_energy_available": pbe_formation_count,
            "r2scan_formation_energy_available": r2_formation_count,
        },
        "pairing": {
            "common_id_count": len(set(pbe_index) & r2_seen),
            "same_configuration_pair_count": len(deltas),
            "same_upstream_split_pair_count": same_upstream_split_count,
            "same_configuration_pair_id_set_sha256": pair_id_set_sha256,
            "mismatch_counts": dict(sorted(mismatch_counts.items())),
            "mismatch_examples": {
                name: values for name, values in sorted(mismatch_examples.items()) if values
            },
            "r2scan_minus_pbe_energy_ev_per_atom": _quantiles(deltas),
            "both_formation_energy_pair_count": len(formation_deltas),
            "r2scan_minus_pbe_formation_energy_ev_per_atom": _quantiles(
                formation_deltas
            ),
        },
        "chemical_systems": {
            "exact_system_count": len(chemsys_counts),
            "candidate_count_quantiles": (
                {
                    "minimum": int(np.min(counts)),
                    "p25": float(np.quantile(counts, 0.25)),
                    "p50": float(np.quantile(counts, 0.50)),
                    "p75": float(np.quantile(counts, 0.75)),
                    "maximum": int(np.max(counts)),
                }
                if len(counts)
                else None
            ),
            "systems_with_at_least_8": sum(count >= 8 for count in chemsys_counts.values()),
            "systems_with_at_least_16": sum(count >= 16 for count in chemsys_counts.values()),
            "formation_energy_exact_system_count": len(formation_chemsys_counts),
            "formation_energy_systems_with_at_least_8": sum(
                count >= 8 for count in formation_chemsys_counts.values()
            ),
            "formation_energy_systems_with_at_least_16": sum(
                count >= 16 for count in formation_chemsys_counts.values()
            ),
            "largest_systems": [
                {"chemical_system": system, "candidate_count": count}
                for system, count in sorted(
                    chemsys_counts.items(), key=lambda item: (-item[1], item[0])
                )[:30]
            ],
        },
        "parent_split_audit": _count_parent_split_leakage(parent_splits),
        "decision": {
            "release_one_to_one_split_parity_gate_pass": release_parity_gate,
            "same_configuration_pair_gate_pass": same_configuration_pair_gate,
            "same_configuration_protocol_task_supported": (
                same_configuration_pair_gate and len(deltas) >= 1000
            ),
            "formation_energy_labels_available": (
                pbe_formation_count == len(pbe_index)
                and r2_formation_count == len(r2_seen)
            ),
            "hull_requirement": (
                "Use only pairs where both protocol formation_energy_per_atom fields "
                "are finite, or separately derive protocol-consistent references; "
                "never impute a null formation-energy label."
            ),
            "split_requirement": (
                "Future calibration/evaluation partitions must group exact chemical "
                "systems and original_mp_id parents; upstream row splits alone are unsafe."
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output={output.resolve()}")
    print(f"same_configuration_pair_count={len(deltas)}")
    print(f"same_configuration_pair_gate_pass={same_configuration_pair_gate}")
    print(f"release_one_to_one_split_parity_gate_pass={release_parity_gate}")
    print(
        "rounded_geometry_mismatches=",
        mismatch_counts["rounded_geometry_1e-10"],
        sep="",
    )
    print(
        "cross_split_original_mp_ids=",
        result["parent_split_audit"]["cross_split_original_mp_id_count"],
        sep="",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbe-root", type=Path, required=True)
    parser.add_argument("--r2scan-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(pbe_root=args.pbe_root, r2scan_root=args.r2scan_root, output=args.output)


if __name__ == "__main__":
    main()
