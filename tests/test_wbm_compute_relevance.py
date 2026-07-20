from __future__ import annotations

import pytest

from tools.analyze_wbm_compute_relevance import _real_trace_prefix, _timing_summary


def test_real_trace_prefix_counts_lazy_prediction_as_gp_numerical_work() -> None:
    rounds = [
        {
            "posterior_fit_seconds": 0.01,
            "prediction_seconds": 0.09,
            "round_pipeline_seconds": 1.0,
        }
        for _ in range(40)
    ]
    run = {
        "pool": "A-B",
        "prequential_rounds": rounds,
        "phase_timings": [
            {"round_index": index, "hull_update_seconds": 0.02}
            for index in range(1, 41)
        ],
        "peak_parent_rss_bytes": 1_000,
    }
    prefix = _real_trace_prefix(run, 40)
    assert prefix["gp_numerical_seconds"] == pytest.approx(4.0)
    assert prefix["gp_fraction_of_round_pipeline"] == pytest.approx(0.1)
    assert prefix["passes_10pct_ideal_speedup_gate"] is True


def test_timing_summary_reports_robust_dispersion() -> None:
    result = _timing_summary([1.0, 1.0, 1.0, 10.0])
    assert result["median_seconds"] == pytest.approx(1.0)
    assert result["mad_seconds"] == pytest.approx(0.0)
    assert result["p95_seconds"] > 1.0
