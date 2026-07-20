"""Stream-audit LeMat PBE/SCAN keys before building a protocol-pair task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

COLUMNS = (
    "immutable_id",
    "entalpic_fingerprint",
    "chemical_formula_reduced",
    "nsites",
    "energy",
    "functional",
    "cross_compatibility",
)


@dataclass(frozen=True)
class CompactRow:
    immutable_id: str
    fingerprint: str | None
    formula: str
    nsites: int
    energy_ev: float
    functional: str
    cross_compatible: bool

    @property
    def energy_ev_per_atom(self) -> float:
        return self.energy_ev / self.nsites


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(paths: list[Path]):
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=65536, columns=COLUMNS):
            values = batch.to_pydict()
            for index in range(batch.num_rows):
                fingerprint = values["entalpic_fingerprint"][index]
                row = CompactRow(
                    immutable_id=str(values["immutable_id"][index]).strip(),
                    fingerprint=(
                        str(fingerprint).strip() if fingerprint is not None else None
                    ),
                    formula=str(values["chemical_formula_reduced"][index]).strip(),
                    nsites=int(values["nsites"][index]),
                    energy_ev=float(values["energy"][index]),
                    functional=str(values["functional"][index]).strip().lower(),
                    cross_compatible=bool(values["cross_compatibility"][index]),
                )
                yield row


def _prefix(identifier: str) -> str:
    return identifier.split("-", maxsplit=1)[0].lower()


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p10": float(np.quantile(array, 0.10)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p99": float(np.quantile(array, 0.99)),
    }


def _pair_summary(
    *,
    scan_index: dict[str, tuple[int, CompactRow]],
    pbe_index: dict[str, tuple[int, CompactRow]],
) -> dict[str, Any]:
    shared = set(scan_index) & set(pbe_index)
    unambiguous = [
        key
        for key in shared
        if scan_index[key][0] == 1 and pbe_index[key][0] == 1
    ]
    deltas: list[float] = []
    formula_agreement = 0
    nsites_agreement = 0
    compatible_pairs = 0
    invalid_energy_rows = 0
    prefix_pairs: Counter[str] = Counter()
    for key in unambiguous:
        scan = scan_index[key][1]
        pbe = pbe_index[key][1]
        formula_agreement += int(scan.formula == pbe.formula)
        nsites_agreement += int(scan.nsites == pbe.nsites)
        compatible_pairs += int(scan.cross_compatible and pbe.cross_compatible)
        prefix_pairs[_prefix(scan.immutable_id)] += 1
        if (
            scan.nsites <= 0
            or pbe.nsites <= 0
            or not math.isfinite(scan.energy_ev)
            or not math.isfinite(pbe.energy_ev)
        ):
            invalid_energy_rows += 1
        else:
            deltas.append(scan.energy_ev_per_atom - pbe.energy_ev_per_atom)
    count = len(unambiguous)
    return {
        "shared_key_count": len(shared),
        "unambiguous_pair_count": count,
        "ambiguous_shared_key_count": len(shared) - count,
        "formula_agreement_count": formula_agreement,
        "formula_agreement_rate": formula_agreement / count if count else None,
        "nsites_agreement_count": nsites_agreement,
        "nsites_agreement_rate": nsites_agreement / count if count else None,
        "both_cross_compatible_count": compatible_pairs,
        "both_cross_compatible_rate": compatible_pairs / count if count else None,
        "invalid_energy_pair_count": invalid_energy_rows,
        "scan_minus_pbe_energy_ev_per_atom": _quantiles(deltas),
        "source_prefix_counts": dict(sorted(prefix_pairs.items())),
        "sample_pair_keys": sorted(
            unambiguous,
            key=lambda value: hashlib.sha256(value.encode()).hexdigest(),
        )[:20],
    }


def run(*, root: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError("LeMat protocol-pair audit cannot overwrite output")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("LeMat audit output must remain outside Git")
    pbe_paths = sorted((root / "unique_pbe").glob("*.parquet"))
    scan_paths = sorted((root / "unique_scan").glob("*.parquet"))
    if len(pbe_paths) != 16 or len(scan_paths) != 2:
        raise ValueError("LeMat PBE/SCAN shard set is incomplete")

    scan_id: dict[str, tuple[int, CompactRow]] = {}
    scan_fingerprint: dict[str, tuple[int, CompactRow]] = {}
    scan_rows = 0
    scan_missing_id = 0
    scan_missing_fingerprint = 0
    scan_functionals: Counter[str] = Counter()
    for row in _rows(scan_paths):
        scan_rows += 1
        scan_functionals[row.functional] += 1
        if not row.immutable_id:
            scan_missing_id += 1
        else:
            count, first = scan_id.get(row.immutable_id, (0, row))
            scan_id[row.immutable_id] = (count + 1, first)
        if not row.fingerprint:
            scan_missing_fingerprint += 1
        else:
            count, first = scan_fingerprint.get(row.fingerprint, (0, row))
            scan_fingerprint[row.fingerprint] = (count + 1, first)

    pbe_id_matches: dict[str, tuple[int, CompactRow]] = {}
    pbe_fingerprint_matches: dict[str, tuple[int, CompactRow]] = {}
    pbe_rows = 0
    pbe_missing_id = 0
    pbe_missing_fingerprint = 0
    pbe_functionals: Counter[str] = Counter()
    for row in _rows(pbe_paths):
        pbe_rows += 1
        pbe_functionals[row.functional] += 1
        if not row.immutable_id:
            pbe_missing_id += 1
        if not row.fingerprint:
            pbe_missing_fingerprint += 1
        if row.immutable_id in scan_id:
            count, first = pbe_id_matches.get(row.immutable_id, (0, row))
            pbe_id_matches[row.immutable_id] = (count + 1, first)
        if row.fingerprint and row.fingerprint in scan_fingerprint:
            count, first = pbe_fingerprint_matches.get(row.fingerprint, (0, row))
            pbe_fingerprint_matches[row.fingerprint] = (count + 1, first)

    direct = _pair_summary(scan_index=scan_id, pbe_index=pbe_id_matches)
    fingerprint = _pair_summary(
        scan_index=scan_fingerprint,
        pbe_index=pbe_fingerprint_matches,
    )
    content = {
        "schema_version": 1,
        "status": "read_only_pair_key_audit_not_yet_a_structure_match",
        "intended_grain": "one PBE and one SCAN calculation for the same material/structure",
        "root": str(root.resolve()),
        "input": {
            "pbe_shards": len(pbe_paths),
            "scan_shards": len(scan_paths),
            "pbe_bytes": sum(path.stat().st_size for path in pbe_paths),
            "scan_bytes": sum(path.stat().st_size for path in scan_paths),
            "download_metadata_sha256": _sha256(root / "DOWNLOAD_METADATA.json"),
        },
        "profiles": {
            "pbe": {
                "row_count": pbe_rows,
                "missing_id_count": pbe_missing_id,
                "missing_fingerprint_count": pbe_missing_fingerprint,
                "functional_counts": dict(sorted(pbe_functionals.items())),
            },
            "scan": {
                "row_count": scan_rows,
                "unique_id_count": len(scan_id),
                "duplicate_id_row_count": sum(
                    count - 1 for count, _ in scan_id.values() if count > 1
                ),
                "unique_fingerprint_count": len(scan_fingerprint),
                "duplicate_fingerprint_row_count": sum(
                    count - 1
                    for count, _ in scan_fingerprint.values()
                    if count > 1
                ),
                "missing_id_count": scan_missing_id,
                "missing_fingerprint_count": scan_missing_fingerprint,
                "functional_counts": dict(sorted(scan_functionals.items())),
            },
        },
        "direct_immutable_id_pairs": direct,
        "entalpic_fingerprint_pairs": fingerprint,
        "decision": {
            "safe_to_build_task": (
                direct["unambiguous_pair_count"] >= 1000
                and (direct["formula_agreement_rate"] or 0) >= 0.999
                and (direct["nsites_agreement_rate"] or 0) >= 0.99
            ),
            "next_gate": "deterministic structure matching on unambiguous direct-ID pairs",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"output={output.resolve()}")
    print(
        "direct_pairs",
        direct["unambiguous_pair_count"],
        "formula_agreement",
        direct["formula_agreement_rate"],
        "nsites_agreement",
        direct["nsites_agreement_rate"],
    )
    print(
        "fingerprint_pairs",
        fingerprint["unambiguous_pair_count"],
        "formula_agreement",
        fingerprint["formula_agreement_rate"],
        "nsites_agreement",
        fingerprint["nsites_agreement_rate"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(root=args.root, output=args.output)


if __name__ == "__main__":
    main()
