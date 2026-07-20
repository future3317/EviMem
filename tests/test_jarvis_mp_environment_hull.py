from __future__ import annotations

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "run_jarvis_mp_environment_hull.py")
)


def test_calibration_partition_is_exact_deterministic_and_disjoint() -> None:
    systems = {f"A-{letter}" for letter in "BCDEFGHIJKLM"}
    first = MODULE["_partition_calibration_systems"](systems, "release")
    second = MODULE["_partition_calibration_systems"](
        set(reversed(sorted(systems))), "release"
    )
    assert first == second
    assert set.union(*first) == systems
    assert not first[0] & first[1]
    assert not first[0] & first[2]
    assert not first[1] & first[2]


def test_calibration_only_vault_rejects_evaluation_rows() -> None:
    with pytest.raises(ValueError, match="evaluation"):
        MODULE["CalibrationOnlyVault"](
            {"target_outcomes": [{"pair_id": "x", "split": "evaluation"}]}
        )


def test_bootstrap_lower_bound_is_deterministic() -> None:
    values = [0.1, 0.2, 0.3, 0.4]
    first = MODULE["_bootstrap_lower_95"](values, "a" * 64)
    second = MODULE["_bootstrap_lower_95"](values, "a" * 64)
    assert first == second
    assert 0 <= first <= sum(values) / len(values)


def test_evaluation_only_vault_rejects_calibration_and_duplicate_reveal() -> None:
    with pytest.raises(ValueError, match="calibration"):
        MODULE["EvaluationOnlyVault"](
            {"target_outcomes": [{"pair_id": "x", "split": "calibration"}]}
        )
    vault = MODULE["EvaluationOnlyVault"](
        {"target_outcomes": [{"pair_id": "x", "split": "evaluation"}]}
    )
    assert vault.reveal("x")["pair_id"] == "x"
    with pytest.raises(ValueError, match="twice"):
        vault.reveal("x")
