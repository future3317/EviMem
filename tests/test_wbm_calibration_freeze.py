from __future__ import annotations

import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "freeze_wbm_calibration_gate.py")
)


def _manifest() -> dict:
    pools = {}
    for index in range(4):
        pools[f"B{index}-X"] = {"chemical_complexity_stratum": "binary"}
        pools[f"T{index}-X-Y"] = {"chemical_complexity_stratum": "ternary"}
    return {"selection": {"pools": pools}}


def _summary(*, include_dacc: bool = False) -> dict:
    runs = []
    for system in _manifest()["selection"]["pools"]:
        for strategy, brier, log_loss in (
            ("fifo", 0.20, 0.40),
            ("gp_variance_one_swap", 0.10, 0.20),
            ("full_history", 0.11, 0.21),
        ):
            runs.append(
                {
                    "pool": system,
                    "strategy": strategy,
                    "budget": 12,
                    "prequential_rounds": [{"round_index": item} for item in range(1, 13)],
                    "prequential": {
                        "boundary_weighted_causal_crps": 0.1,
                        "boundary_weighted_causal_brier": brier,
                        "boundary_weighted_causal_log_loss": log_loss,
                    },
                }
            )
    if include_dacc:
        runs.append({**runs[0], "strategy": "decision_coreset"})
    return {"runs": runs}


def test_calibration_freeze_uses_only_fifo_and_gp_variance_for_margins() -> None:
    config = {"posterior": {"kernel": "matern52", "length_scale": 0.35}}
    result = MODULE["freeze_gate"](
        config=config, manifest=_manifest(), summary=_summary()
    )
    assert result["gp_parameter_status"] == "frozen_on_disjoint_calibration_systems_v1"
    assert result["evaluation_results_accessed"] is False
    assert result["brier_margin"] == pytest.approx(0.01)
    assert result["log_loss_margin"] == pytest.approx(0.02)


def test_calibration_freeze_rejects_unregistered_dacc_runs() -> None:
    config = {"posterior": {"kernel": "matern52", "length_scale": 0.35}}
    with pytest.raises(ValueError, match="other selectors"):
        MODULE["freeze_gate"](
            config=config, manifest=_manifest(), summary=_summary(include_dacc=True)
        )
