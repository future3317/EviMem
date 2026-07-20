from __future__ import annotations

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(
        Path(__file__).resolve().parents[1]
        / "tools"
        / "run_jarvis_mp_protocol_activation_pilot.py"
    )
)


def test_expected_positive_part_is_monotone_in_better_hull_mean() -> None:
    expected = MODULE["_expected_positive_part"]
    assert expected(-0.1, 0.05, 0.05) > expected(0.1, 0.05, 0.05)


def test_gaussian_crps_is_smallest_at_the_target() -> None:
    crps = MODULE["_gaussian_crps"]
    assert crps(0.0, 0.1, 0.0) < crps(0.3, 0.1, 0.0)


def test_oracle_vault_separates_calibration_reveal_and_evaluator() -> None:
    vault = MODULE["EvaluationOracleVault"](
        {
            "target_outcomes": [
                {
                    "pair_id": "cal",
                    "split": "calibration",
                    "chemical_system": "Li-O",
                },
                {
                    "pair_id": "eval",
                    "split": "evaluation",
                    "chemical_system": "Na-Cl",
                },
            ]
        }
    )
    assert set(vault.calibration_rows({"Li-O"})) == {"cal"}
    with pytest.raises(ValueError, match="evaluation outcomes only"):
        vault.reveal("cal")
    assert vault.reveal("eval")["pair_id"] == "eval"
    with pytest.raises(ValueError, match="twice"):
        vault.reveal("eval")
    with pytest.raises(ValueError, match="evaluation outcomes only"):
        vault.evaluate(("cal",))
