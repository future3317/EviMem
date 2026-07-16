from __future__ import annotations

import runpy
from pathlib import Path

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "build_wbm_cleaned_id_manifest.py")
)


def test_step3_alignment_matches_the_pinned_compiler_rule() -> None:
    source_to_wbm_id = MODULE["_source_to_wbm_id"]
    fix_step3_alignment = MODULE["_fix_step3_alignment"]
    assert source_to_wbm_id("step_1_0") == "wbm-1-1"
    assert fix_step3_alignment("wbm-3-70802") == "wbm-3-70802"
    assert fix_step3_alignment("wbm-3-70805") == "wbm-3-70803"
    assert fix_step3_alignment("wbm-5-23166") == "wbm-5-23166"


def test_id_checksum_is_order_invariant_and_newline_delimited() -> None:
    id_checksum = MODULE["_id_checksum"]
    assert id_checksum(("wbm-2-2", "wbm-1-1")) == id_checksum(("wbm-1-1", "wbm-2-2"))
    assert id_checksum(("wbm-1-1",)) != id_checksum(("wbm-1-1", "wbm-1-2"))
