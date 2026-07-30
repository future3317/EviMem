from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "tools" / "summarize_ic_sarr_rollout_calibration.py"
    spec = importlib.util.spec_from_file_location("summarize_ic_sarr_rollout_calibration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_summary_reports_registered_strata_and_false_positive_rate() -> None:
    summary = _module()._summary(
        [
            {
                "predicted_advantage": 0.1,
                "actual_complete_pool_t_advantage": 1.0,
                "actual_selected_history_f_advantage": 1.0,
                "accepted_deviation": True,
            },
            {
                "predicted_advantage": 0.2,
                "actual_complete_pool_t_advantage": -1.0,
                "actual_selected_history_f_advantage": 0.0,
                "accepted_deviation": True,
            },
            {
                "predicted_advantage": 0.3,
                "actual_complete_pool_t_advantage": 0.0,
                "actual_selected_history_f_advantage": 0.0,
                "accepted_deviation": False,
            },
        ]
    )

    assert summary["decision_state_count"] == 3
    assert summary["accepted_deviation_count"] == 2
    assert summary["false_positive_accepted_deviation_rate"] == 0.5
    assert summary["strata"]["accepted"]["count"] == 2
    assert summary["strata"]["rejected_or_screened"]["count"] == 1
