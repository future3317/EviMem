from __future__ import annotations

import json
from pathlib import Path

from tools.render_e52_validation_figure import render


def test_render_e52_validation_figure(tmp_path: Path) -> None:
    objective = {
        "curves": {
            pool: {
                "budgets": {
                    str(budget): {
                        "paired_delta_T": 0.01 * budget,
                        "bootstrap_95": {
                            "lower": 0.01 * budget - 0.01,
                            "upper": 0.01 * budget + 0.01,
                        },
                    }
                    for budget in range(1, 7)
                }
            }
            for pool in ("070", "085", "100")
        }
    }
    calibration = {
        "groups": {
            "task:delta": {
                "policy": "delta_hull_active_search",
                "all_candidates": {
                    "metrics": {
                        "brier_score": 0.05,
                        "bernoulli_nll": 0.2,
                        "roc_auc": 0.94,
                    },
                    "reliability_bins": [
                        {
                            "record_count": 10,
                            "mean_predicted_probability": 0.1,
                            "empirical_frequency": 0.12,
                        },
                        {
                            "record_count": 5,
                            "mean_predicted_probability": 0.8,
                            "empirical_frequency": 0.75,
                        },
                    ],
                },
            }
        }
    }
    objective_path = tmp_path / "objective.json"
    calibration_path = tmp_path / "calibration.json"
    objective_path.write_text(json.dumps(objective), encoding="utf-8")
    calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    output = tmp_path / "figure.pdf"

    render(
        objective_path=objective_path,
        calibration_path=calibration_path,
        output_path=output,
    )

    assert output.stat().st_size > 1_000
    assert output.with_suffix(".png").stat().st_size > 1_000
