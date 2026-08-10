from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.render_e53_objective_isolation_figure import render


def _summary() -> dict[str, object]:
    panels = {}
    for panel, offset in (("development", 0.0), ("secondary", -0.02)):
        panels[panel] = {
            "budgets": {
                str(budget): {
                    "absolute_mean_T": {
                        "posterior_mean_target_margin": 0.30 * budget,
                        "matched_local_hull_probability": 0.34 * budget,
                        "delta_hull_active_search": 0.38 * budget + offset,
                    },
                    "contrasts": {
                        "delta_minus_local": {
                            "mean_effect": 0.04 * budget + offset,
                            "ci_low": 0.04 * budget - 0.02 + offset,
                            "ci_high": 0.04 * budget + 0.02 + offset,
                        }
                    },
                }
                for budget in range(1, 7)
            }
        }
    return {
        "status": "e53_objective_isolation_complete",
        "policies": [
            "posterior_mean_target_margin",
            "matched_local_hull_probability",
            "delta_hull_active_search",
        ],
        "panels": panels,
    }


def test_render_e53_objective_isolation_figure(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
    output = tmp_path / "figure.pdf"

    render(summary_path=summary_path, output_path=output)

    assert output.stat().st_size > 1_000
    assert output.with_suffix(".png").stat().st_size > 1_000


def test_render_rejects_missing_secondary_panel(tmp_path: Path) -> None:
    payload = _summary()
    del payload["panels"]["secondary"]
    summary_path = tmp_path / "bad.json"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="development and secondary"):
        render(summary_path=summary_path, output_path=tmp_path / "bad.pdf")
