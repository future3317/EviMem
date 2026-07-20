from __future__ import annotations

from runpy import run_path

import pytest

from tools.build_wbm_small_pool_manifest import ObservableCandidate

MODULE = run_path("tools/build_wbm_compute_gate_manifest.py")


def _candidate(system: tuple[str, ...], index: int) -> ObservableCandidate:
    return ObservableCandidate(
        query_id=f"wbm-1-{index}",
        chemical_system=system,
        composition=tuple((element, 1.0) for element in system),
        exact_structure_sha256=f"sha256:{index:064x}",
    )


def test_compute_gate_selects_longest_exact_system_without_mixing() -> None:
    candidates = [
        *(_candidate(("A", "B"), index) for index in range(1, 42)),
        *(_candidate(("C", "D"), index) for index in range(100, 143)),
        *(_candidate(("E", "F"), index) for index in range(200, 240)),
    ]
    selection = MODULE["select_compute_gate_systems"](candidates, system_count=2)
    assert list(selection["pools"]) == ["C-D", "A-B"]
    assert selection["selected_candidate_count"] == 84
    assert all(
        set(candidate["chemical_system"]) == set(pool["chemical_system"])
        for pool in selection["pools"].values()
        for candidate in pool["candidates"]
    )
    assert selection["amdahl_gate"]["minimum_real_trace_gp_numerical_fraction"] > 0.09


def test_compute_gate_rejects_panels_without_a_b40_trajectory() -> None:
    candidates = [_candidate(("A", "B"), index) for index in range(1, 41)]
    with pytest.raises(ValueError, match="at least 41"):
        MODULE["select_compute_gate_systems"](candidates)
