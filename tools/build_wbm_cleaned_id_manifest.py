"""Build an auditable WBM raw-to-cleaned-ID manifest outside the repository.

This reproduces only the identity alignment and documented raw formation-energy
outlier filter of Matbench Discovery's pinned WBM compiler.  It deliberately
does not compute MP2020 corrections or hull distances; those require the
separate frozen MP parity inputs.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

STEP_COUNTS = (61_848, 52_800, 79_205, 40_328, 23_308)
STEP3_ANOMALOUS_STRUCTURE_IDS = (
    "step_3_70802",
    "step_3_70803",
    "step_3_70825",
    "step_3_70826",
    "step_3_70828",
    "step_3_70829",
)
STEP5_MISSING_INITIAL_IDS = ("step_5_23165", "step_5_23293")
OUTLIER_CUTOFF_EV_PER_ATOM = 5.0


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _id_checksum(ids: Iterable[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(ids)) + "\n").encode("utf-8"))


def _source_to_wbm_id(source_id: str) -> str:
    prefix, step, index = source_id.split("_")
    if prefix != "step" or not step.isdigit() or not index.isdigit():
        raise ValueError(f"invalid WBM source ID: {source_id}")
    return f"wbm-{int(step)}-{int(index) + 1}"


def _fix_step3_alignment(wbm_id: str) -> str:
    prefix, step, number = wbm_id.split("-")
    if prefix != "wbm" or not step.isdigit() or not number.isdigit():
        raise ValueError(f"invalid WBM benchmark ID: {wbm_id}")
    if int(step) != 3:
        return wbm_id
    adjusted = int(number) - sum(
        int(number) > int(source_id.rsplit("_", maxsplit=1)[1]) + 1
        for source_id in STEP3_ANOMALOUS_STRUCTURE_IDS
    )
    return f"wbm-3-{adjusted}"


def _load_bz2_json(path: Path) -> Any:
    try:
        return json.loads(bz2.decompress(path.read_bytes()))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid compressed JSON: {path}") from exc


def _duplicate_count(values: Iterable[str]) -> int:
    items = tuple(values)
    return len(items) - len(set(items))


def _parse_summary(path: Path) -> dict[str, float]:
    """Return raw source ID -> upstream WBM formation energy in eV/atom."""

    energies: dict[str, float] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 8:
            raise ValueError(f"invalid WBM summary row {line_number}: expected eight columns")
        source_id = fields[7]
        if source_id in {"", "None", "nan", "NaN"}:
            continue
        if source_id in energies:
            raise ValueError(f"duplicate WBM summary source ID: {source_id}")
        try:
            energies[source_id] = float(fields[4])
        except ValueError as exc:
            raise ValueError(f"invalid WBM formation energy on row {line_number}") from exc
    return energies


def _file_record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def build_manifest(
    *,
    cse_root: Path,
    structures_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create the frozen cleaned-ID set and its raw-alignment audit."""

    all_structure_ids: list[str] = []
    initial_ids: list[str] = []
    relaxed_ids: list[str] = []
    cse_aligned_source_ids: list[str] = []
    source_files: dict[str, dict[str, object]] = {}

    for step, expected_count in enumerate(STEP_COUNTS, start=1):
        structure_path = structures_root / f"wbm-structures-step-{step}.json.bz2"
        cse_path = cse_root / f"step_{step}.json.bz2"
        if not structure_path.is_file() or not cse_path.is_file():
            raise FileNotFoundError(f"missing WBM step {step} source file")
        structures = _load_bz2_json(structure_path)
        cse_payload = _load_bz2_json(cse_path)
        if not isinstance(structures, dict):
            raise ValueError(f"WBM structures step {step} must be an object keyed by source ID")
        if not isinstance(cse_payload, dict) or not isinstance(cse_payload.get("entries"), list):
            raise ValueError(f"WBM CSE step {step} must contain an entries list")
        structure_ids = list(structures)
        if step == 3 and len(structure_ids) != expected_count + len(STEP3_ANOMALOUS_STRUCTURE_IDS):
            raise ValueError("WBM step 3 does not contain the six documented anomalous structures")
        if step != 3 and len(structure_ids) != expected_count:
            raise ValueError(f"WBM structures step {step} has unexpected count")
        retained_ids = [
            source_id for source_id in structure_ids if source_id not in STEP3_ANOMALOUS_STRUCTURE_IDS
        ]
        if len(cse_payload["entries"]) != expected_count or len(retained_ids) != expected_count:
            raise ValueError(f"WBM CSE/structure alignment count mismatch for step {step}")
        all_structure_ids.extend(structure_ids)
        for source_id, record in structures.items():
            if not isinstance(record, Mapping) or "org" not in record or "opt" not in record:
                raise ValueError(f"WBM structure record lacks org/opt fields: {source_id}")
            if record["org"] is not None:
                initial_ids.append(source_id)
            if record["opt"] is not None:
                relaxed_ids.append(source_id)
        cse_aligned_source_ids.extend(retained_ids)
        source_files[f"structures_step_{step}"] = _file_record(structure_path)
        source_files[f"cse_step_{step}"] = _file_record(cse_path)

    summary_path = structures_root / "wbm-summary.txt"
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing WBM Google Drive summary: {summary_path}")
    summary_e_form_by_source = _parse_summary(summary_path)
    source_files["summary_google_drive"] = _file_record(summary_path)

    cse_source_set = set(cse_aligned_source_ids)
    structure_set = set(all_structure_ids)
    initial_set = set(initial_ids)
    relaxed_set = set(relaxed_ids)
    summary_source_set = set(summary_e_form_by_source)
    if len(cse_source_set) != sum(STEP_COUNTS):
        raise ValueError("aligned CSE source IDs are not unique")
    if tuple(sorted(structure_set - cse_source_set)) != tuple(sorted(STEP3_ANOMALOUS_STRUCTURE_IDS)):
        raise ValueError("structure/CSE extra IDs differ from upstream step-3 anomalies")
    if tuple(sorted(cse_source_set - initial_set)) != STEP5_MISSING_INITIAL_IDS:
        raise ValueError("CSE/initial missing IDs differ from upstream step-5 anomalies")
    if relaxed_set - cse_source_set != set(STEP3_ANOMALOUS_STRUCTURE_IDS):
        raise ValueError("CSE/relaxed extra IDs differ from upstream step-3 anomalies")
    cse_benchmark_ids = {_fix_step3_alignment(_source_to_wbm_id(item)) for item in cse_source_set}
    summary_e_form_by_benchmark = {
        _source_to_wbm_id(source_id): value for source_id, value in summary_e_form_by_source.items()
    }
    if set(summary_e_form_by_benchmark) != cse_benchmark_ids:
        raise ValueError("summary and CSE benchmark IDs diverge after documented step-3 alignment")
    missing_initial_benchmark_ids = {_source_to_wbm_id(item) for item in STEP5_MISSING_INITIAL_IDS}
    pre_outlier_ids = cse_benchmark_ids - missing_initial_benchmark_ids
    below = {
        item
        for item in pre_outlier_ids
        if summary_e_form_by_benchmark[item] < -OUTLIER_CUTOFF_EV_PER_ATOM
    }
    above = {
        item
        for item in pre_outlier_ids
        if summary_e_form_by_benchmark[item] > OUTLIER_CUTOFF_EV_PER_ATOM
    }
    cleaned_ids = pre_outlier_ids - below - above
    if (len(pre_outlier_ids), len(below), len(above), len(cleaned_ids)) != (257_487, 502, 22, 256_963):
        raise ValueError("WBM raw-to-cleaned filter counts differ from the pinned compiler")

    output_dir.mkdir(parents=True, exist_ok=True)
    cleaned_ids_path = output_dir / "wbm-256963-cleaned-benchmark-ids.txt"
    cleaned_ids_path.write_text("\n".join(sorted(cleaned_ids)) + "\n", encoding="utf-8")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "scope": "raw_to_cleaned_wbm_identity_only_no_mp2020_or_hull_recalculation",
        "upstream_inconsistency": {
            "readme_text_count": 257_487,
            "raw_cse_count": 257_489,
            "compiler_step_counts": list(STEP_COUNTS),
            "compiler_step_count_sum": sum(STEP_COUNTS),
            "resolution": "257489 is accepted as the source CSE count; the README text count is not used as a download-integrity criterion.",
        },
        "source_files": source_files,
        "id_counts": {
            "structure_records": len(all_structure_ids),
            "initial_structure_records": len(initial_ids),
            "relaxed_structure_records": len(relaxed_ids),
            "aligned_cse_records": len(cse_aligned_source_ids),
            "summary_rows": sum(1 for _ in summary_path.open(encoding="utf-8")),
            "summary_defined_ids": len(summary_source_set),
            "aligned_cse_unique_ids": len(cse_source_set),
            "aligned_cse_duplicate_ids": _duplicate_count(cse_aligned_source_ids),
        },
        "source_id_set_differences": {
            "structure_minus_cse": sorted(structure_set - cse_source_set),
            "cse_minus_structure": sorted(cse_source_set - structure_set),
            "initial_minus_cse": sorted(initial_set - cse_source_set),
            "cse_minus_initial": sorted(cse_source_set - initial_set),
            "relaxed_minus_cse": sorted(relaxed_set - cse_source_set),
            "cse_minus_relaxed": sorted(cse_source_set - relaxed_set),
            "summary_minus_cse": sorted(summary_source_set - cse_source_set),
            "cse_minus_summary": sorted(cse_source_set - summary_source_set),
        },
        "benchmark_id_normalization_gate": {
            "cse_benchmark_id_count": len(cse_benchmark_ids),
            "summary_benchmark_id_count": len(summary_e_form_by_benchmark),
            "cse_benchmark_id_checksum": _id_checksum(cse_benchmark_ids),
            "summary_benchmark_id_checksum": _id_checksum(summary_e_form_by_benchmark),
            "exact_match_after_documented_step3_reindexing": True,
        },
        "documented_anomalies": {
            "step3_extra_structure_ids_without_cse_or_summary_id": list(STEP3_ANOMALOUS_STRUCTURE_IDS),
            "step5_missing_initial_structure_source_ids": list(STEP5_MISSING_INITIAL_IDS),
            "step5_missing_initial_structure_benchmark_ids": sorted(missing_initial_benchmark_ids),
        },
        "filter_chain": [
            {"name": "aligned_cse_benchmark_ids", "count": len(cse_benchmark_ids), "id_checksum": _id_checksum(cse_benchmark_ids)},
            {"name": "drop_missing_initial_structures", "removed_count": len(missing_initial_benchmark_ids), "remaining_count": len(pre_outlier_ids), "removed_id_checksum": _id_checksum(missing_initial_benchmark_ids), "remaining_id_checksum": _id_checksum(pre_outlier_ids)},
            {"name": "drop_formation_energy_below_minus_5_ev_per_atom", "removed_count": len(below), "removed_id_checksum": _id_checksum(below)},
            {"name": "drop_formation_energy_above_plus_5_ev_per_atom", "removed_count": len(above), "removed_id_checksum": _id_checksum(above)},
            {"name": "cleaned_matbench_discovery_benchmark_ids", "count": len(cleaned_ids), "id_checksum": _id_checksum(cleaned_ids), "id_file": str(cleaned_ids_path.resolve()), "id_file_sha256": _sha256_file(cleaned_ids_path)},
        ],
    }
    manifest_path = output_dir / "wbm-raw-to-cleaned-id-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cse-root", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = build_manifest(
        cse_root=args.cse_root,
        structures_root=args.structures_root,
        output_dir=args.output_dir,
    )
    cleaned = manifest["filter_chain"][-1]
    print(f"cleaned_benchmark_ids={cleaned['count']}")
    print(f"cleaned_benchmark_id_checksum={cleaned['id_checksum']}")


if __name__ == "__main__":
    main()
