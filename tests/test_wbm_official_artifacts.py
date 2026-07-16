from __future__ import annotations

import csv
import gzip
import runpy
import sys
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "audit_wbm_official_artifacts.py"))
TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
PARITY_MODULE = runpy.run_path(str(TOOLS / "build_wbm_candidate_parity_audit.py"))


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


def test_candidate_parity_requires_explicit_summary_ids(tmp_path: Path) -> None:
    explicit = tmp_path / "wbm-summary.txt"
    explicit.write_text(
        "Fe2\t2\t10.0\t-2.0\t-0.2\t0.0\t0.0\tstep_1_0\n"
        "Bad\t0\t0.0\t0.0\t0.0\t0.0\t0.0\tNone\n",
        encoding="utf-8",
    )
    rows = PARITY_MODULE["_official_summary"](explicit, {"wbm-1-1"})
    assert rows["wbm-1-1"]["official_raw_formation_energy_ev_per_atom"] == -0.2
    assert "official_corrected_formation_energy_ev_per_atom" not in rows["wbm-1-1"]

    positional = tmp_path / "positional-summary.txt"
    positional.write_text("Fe2 2 10.0 -2.0 -0.2 0.0 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="explicit-ID"):
        PARITY_MODULE["_official_summary"](positional, {"wbm-1-1"})
