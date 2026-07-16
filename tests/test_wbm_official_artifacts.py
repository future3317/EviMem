from __future__ import annotations

import csv
import gzip
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "audit_wbm_official_artifacts.py"))


def test_prediction_join_requires_exact_cleaned_ids_and_unique_keys(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["material_id", "e_form_per_atom"])
        writer.writeheader()
        writer.writerows([
            {"material_id": "wbm-1-1", "e_form_per_atom": "-1.2"},
            {"material_id": "wbm-1-2", "e_form_per_atom": "-0.4"},
        ])
    report = MODULE["inspect_prediction_join"](path, {"wbm-1-1", "wbm-1-2"})
    assert report["cleaned_id_parity"]["exact_match"] is True
    with pytest.raises(ValueError, match="exactly match"):
        MODULE["inspect_prediction_join"](path, {"wbm-1-1", "wbm-1-3"})


def test_cse_inspection_rejects_mapping_key_inconsistency(tmp_path: Path) -> None:
    path = tmp_path / "cse.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('{"entry":{"a":{"entry_id":"mp-1"}},"material_id":{"a":"mp-1"}}')
    report, ids = MODULE["inspect_mp_cse"](path)
    assert report["entry_id_count"] == 1
    assert ids == {"mp-1"}
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write('{"entry":{"a":{"entry_id":"mp-1"}},"material_id":{"b":"mp-1"}}')
    with pytest.raises(ValueError, match="different keys"):
        MODULE["inspect_mp_cse"](path)


def test_difference_report_uses_stable_empty_set_checksum() -> None:
    report = MODULE["_difference_report"]({"x", "y"}, {"y", "x"})
    assert report["exact_match"] is True
    assert report["left_minus_right_checksum"] == MODULE["EMPTY_ID_SET_SHA256"]
