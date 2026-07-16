"""Audit frozen official WBM, MP, and CHGNet inputs without policy execution."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

OFFICIAL_ARTIFACTS = {
    "mp_cse": {
        "filename": "2023-02-07-mp-computed-structure-entries.json.gz",
        "figshare_file_id": 40344436,
        "expected_md5": "76fc748db6b175bb80de4c276d27c235",
        "expected_sha256": "553d6272f049a8f4ec26e503b89751e2616dd3af53d086545f6ea00f317a361f",
    },
    "mp_ppd": {
        "filename": "2023-02-07-ppd-mp.pkl.gz",
        "figshare_file_id": 48241624,
        "expected_md5": "60d19d691fa1d338aa496a40a9641bef",
        "expected_sha256": "f7fc992230e88dcf26bab2e85fa1ed4fdf6b047c06479bfd8b7a1e003f242d1e",
    },
    "chgnet_predictions": {
        "filename": "2023-12-21-chgnet-0.3.0-discovery.csv.gz",
        "figshare_file_id": 66646268,
        "expected_md5": "fd7cd3781a24be465aaeadf97663ce58",
        "expected_sha256": "b04ce3b031827fcd532e22058d06dc77fb4705ce8b12aea54652a14504e41f16",
    },
}
EMPTY_ID_SET_SHA256 = "sha256:01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b"


def _digest_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_checksum(ids: Iterable[str]) -> str:
    return "sha256:" + hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()


def _difference_report(left: Iterable[str], right: Iterable[str]) -> dict[str, object]:
    left_set, right_set = set(left), set(right)
    return {
        "left_count": len(left_set),
        "right_count": len(right_set),
        "left_minus_right_count": len(left_set - right_set),
        "right_minus_left_count": len(right_set - left_set),
        "left_minus_right_checksum": _id_checksum(left_set - right_set),
        "right_minus_left_checksum": _id_checksum(right_set - left_set),
        "exact_match": left_set == right_set,
    }


def verify_official_files(artifact_dir: Path) -> dict[str, dict[str, object]]:
    """Validate registry MD5 and frozen SHA-256 for all three source files."""

    result: dict[str, dict[str, object]] = {}
    for role, expected in OFFICIAL_ARTIFACTS.items():
        path = artifact_dir / str(expected["filename"])
        if not path.is_file():
            raise FileNotFoundError(f"missing official {role} artifact: {path}")
        md5, sha256 = _digest_file(path, "md5"), _digest_file(path, "sha256")
        if md5 != expected["expected_md5"] or sha256 != expected["expected_sha256"]:
            raise ValueError(f"official {role} checksum mismatch")
        result[role] = {
            "path": str(path.resolve()), "bytes": path.stat().st_size,
            "figshare_file_id": expected["figshare_file_id"], "md5": md5, "sha256": sha256,
        }
    return result


def read_cleaned_ids(path: Path, *, expected_count: int = 256_963) -> set[str]:
    ids = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if len(ids) != expected_count:
        raise ValueError(f"cleaned WBM ID set has {len(ids)} IDs, expected {expected_count}")
    return ids


def inspect_prediction_join(prediction_path: Path, cleaned_ids: set[str]) -> dict[str, object]:
    """Require a unique, exact ID join for the official prediction artifact."""

    with gzip.open(prediction_path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["material_id", "e_form_per_atom"]:
            raise ValueError(f"unexpected official CHGNet prediction columns: {reader.fieldnames}")
        rows = list(reader)
    prediction_ids = [row["material_id"] for row in rows]
    duplicate_count = len(prediction_ids) - len(set(prediction_ids))
    if not all(prediction_ids) or duplicate_count:
        raise ValueError("official CHGNet predictions require unique, non-empty IDs")
    try:
        for row in rows:
            float(row["e_form_per_atom"])
    except (TypeError, ValueError) as exc:
        raise ValueError("official CHGNet prediction contains a non-numeric energy") from exc
    parity = _difference_report(prediction_ids, cleaned_ids)
    if not parity["exact_match"]:
        raise ValueError("official CHGNet prediction IDs do not exactly match cleaned WBM IDs")
    return {
        "row_count": len(rows), "unique_id_count": len(set(prediction_ids)),
        "duplicate_id_count": duplicate_count, "columns": ["material_id", "e_form_per_atom"],
        "declared_unit": "eV/atom (registry field name e_form_per_atom)",
        "target_task": "IS2RE-SR (pinned CHGNet registry configuration)",
        "mp2020_correction": "already included by registry; no additional correction is applied",
        "cleaned_id_parity": parity,
    }


def inspect_mp_cse(cse_path: Path) -> tuple[dict[str, object], set[str]]:
    """Check MP CSE mappings and return its unique serialized entry IDs."""

    with gzip.open(cse_path, "rt", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("MP CSE artifact must decode to a mapping")
    entries, material_ids = payload.get("entry"), payload.get("material_id")
    if not isinstance(entries, Mapping) or not isinstance(material_ids, Mapping):
        raise ValueError("MP CSE artifact requires entry and material_id mappings")
    if set(entries) != set(material_ids):
        raise ValueError("MP CSE entry and material_id mappings have different keys")
    entry_ids = [str(value.get("entry_id")) for value in entries.values() if isinstance(value, Mapping)]
    if len(entry_ids) != len(entries) or any(item == "None" for item in entry_ids):
        raise ValueError("MP CSE mapping has an invalid entry_id")
    duplicate_count = len(entry_ids) - len(set(entry_ids))
    if duplicate_count:
        raise ValueError(f"MP CSE contains {duplicate_count} duplicate entry IDs")
    return ({
        "top_level_keys": sorted(payload), "entry_mapping_count": len(entries),
        "material_id_mapping_count": len(material_ids), "entry_id_count": len(entry_ids),
        "duplicate_entry_id_count": duplicate_count, "entry_id_checksum": _id_checksum(entry_ids),
    }, set(entry_ids))


def _load_ppd_read_only(ppd_path: Path) -> Any:
    """Load historic PPD state without modifying it or recomputing corrections."""

    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    original_new, original_init = MaterialsProject2020Compatibility.__new__, MaterialsProject2020Compatibility.__init__
    MaterialsProject2020Compatibility.__new__ = staticmethod(lambda cls, *args, **kwargs: object.__new__(cls))
    MaterialsProject2020Compatibility.__init__ = lambda self, *args, **kwargs: None
    try:
        with gzip.open(ppd_path, "rb") as handle:
            return pickle.load(handle)
    finally:
        MaterialsProject2020Compatibility.__new__ = original_new
        MaterialsProject2020Compatibility.__init__ = original_init


def inspect_ppd_membership(ppd_path: Path, cse_entry_ids: set[str]) -> dict[str, object]:
    ppd = _load_ppd_read_only(ppd_path)
    entry_ids = [str(entry.entry_id) for entry in ppd.all_entries]
    duplicate_count = len(entry_ids) - len(set(entry_ids))
    if duplicate_count:
        raise ValueError(f"official PPD contains {duplicate_count} duplicate entry IDs")
    parity = _difference_report(entry_ids, cse_entry_ids)
    if not parity["exact_match"]:
        raise ValueError("official PPD phase membership differs from official MP CSE entry IDs")
    return {
        "ppd_type": f"{type(ppd).__module__}.{type(ppd).__qualname__}",
        "chemical_element_count": len(ppd.elements), "space_count": len(ppd.spaces),
        "all_entry_count": len(entry_ids), "duplicate_entry_id_count": duplicate_count,
        "entry_id_checksum": _id_checksum(entry_ids), "cse_phase_membership_parity": parity,
        "loading_note": "Read-only historic compatibility constructor bypass; no correction was recomputed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.artifact_dir.resolve().is_relative_to(repo_root) or args.output.resolve().is_relative_to(repo_root):
        raise ValueError("official artifacts and audit report must remain outside the repository")
    files = verify_official_files(args.artifact_dir)
    cleaned_ids = read_cleaned_ids(args.cleaned_ids)
    cse, cse_ids = inspect_mp_cse(args.artifact_dir / str(OFFICIAL_ARTIFACTS["mp_cse"]["filename"]))
    predictions = inspect_prediction_join(args.artifact_dir / str(OFFICIAL_ARTIFACTS["chgnet_predictions"]["filename"]), cleaned_ids)
    ppd = inspect_ppd_membership(args.artifact_dir / str(OFFICIAL_ARTIFACTS["mp_ppd"]["filename"]), cse_ids)
    report = {
        "schema_version": 1,
        "scope": "official_artifact_integrity_and_keyset_parity_only_no_policy_execution",
        "technical_gate_passed": True,
        "formal_gate_passed": False,
        "formal_gate_blocker": "human license and redistribution manifest review remains pending",
        "artifacts": files,
        "cleaned_wbm_ids": {"count": len(cleaned_ids), "checksum": _id_checksum(cleaned_ids)},
        "mp_cse": cse, "mp_ppd": ppd, "chgnet_official_prediction": predictions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("technical_gate_passed=True")
    print("formal_gate_passed=False")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
