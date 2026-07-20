from __future__ import annotations

import runpy
from collections import Counter
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "build_jarvis_mp_multifidelity_task.py")
)


def test_system_split_is_exact_disjoint_and_deterministic() -> None:
    systems = Counter(
        {
            "A-B": 20,
            "A-C": 20,
            "A-D": 20,
            "A-E": 20,
            "A-F": 20,
            "A-B-C": 20,
            "A-B-D": 20,
            "A-B-E": 20,
            "A-B-F": 20,
            "A-B-G": 20,
            "A-B-C-D": 20,
            "A-B-C-E": 20,
            "A-B-C-F": 20,
        }
    )
    kwargs = {
        "calibration_per_stratum": {
            "binary": 2,
            "ternary": 2,
            "quaternary_plus": 1,
        },
        "evaluation_per_stratum": {
            "binary": 2,
            "ternary": 2,
            "quaternary_plus": 1,
        },
    }
    first = MODULE["_choose_systems"](systems, **kwargs)
    second = MODULE["_choose_systems"](
        Counter(dict(reversed(list(systems.items())))), **kwargs
    )
    assert first == second
    calibration = {system for values in first[0].values() for system in values}
    evaluation = {system for values in first[1].values() for system in values}
    assert calibration.isdisjoint(evaluation)


def test_system_split_fails_instead_of_reusing_a_calibration_system() -> None:
    with pytest.raises(ValueError, match="insufficient binary"):
        MODULE["_choose_systems"](
            Counter({"Li-O": 20}),
            calibration_per_stratum={
                "binary": 1,
                "ternary": 0,
                "quaternary_plus": 0,
            },
            evaluation_per_stratum={
                "binary": 1,
                "ternary": 0,
                "quaternary_plus": 0,
            },
        )


def test_descriptor_has_frozen_dimension_and_no_outcome_input() -> None:
    from pymatgen.core import Lattice, Structure

    structure = Structure(Lattice.cubic(4.2), ["Li", "F"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    descriptor = MODULE["_descriptor"](structure)
    assert len(descriptor) == 124
    assert sum(descriptor[:118]) == pytest.approx(1.0)
