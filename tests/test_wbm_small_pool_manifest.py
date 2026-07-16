from __future__ import annotations

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "build_wbm_small_pool_manifest.py"))
Candidate = MODULE["ObservableCandidate"]


def _candidate(system: tuple[str, ...], index: int) -> object:
    return Candidate(
        query_id=f"id-{''.join(system)}-{index}", chemical_system=system,
        composition=tuple((element, 1.0) for element in system),
        exact_structure_sha256=f"sha256:{''.join(system)}-{index}",
    )


def test_exact_duplicate_filter_is_deterministic() -> None:
    first = _candidate(("A", "B"), 1)
    second = Candidate(
        query_id="id-later", chemical_system=("A", "B"), composition=(("A", 1.0), ("B", 1.0)),
        exact_structure_sha256=first.exact_structure_sha256,
    )
    retained, duplicates = MODULE["deduplicate_exact_structures"]([second, first])
    assert duplicates == 1
    assert retained == [first]


def test_selector_fails_instead_of_silently_replacing_an_empty_stratum() -> None:
    candidates = [_candidate(("A", "B"), index) for index in range(20)]
    with pytest.raises(ValueError, match="no eligible systems"):
        MODULE["select_small_pools"](candidates, pool_size=16, calibration_fraction=0.01)
