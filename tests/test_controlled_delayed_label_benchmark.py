from __future__ import annotations

import pytest

from matmem.controlled_delayed_label_benchmark import controlled_benchmark_grid


def test_exact_controlled_grid_respects_policy_information_order() -> None:
    rows = controlled_benchmark_grid()

    assert len(rows) == 27
    for row in rows:
        assert row["gated_source_rollout"] >= row["source_margin"]
        assert row["optimal_dp"] >= row["gated_source_rollout"]
        if row["budget"] == 1:
            assert row["gated_source_rollout"] == pytest.approx(row["greedy_final"])


def test_exact_dp_has_strict_adaptive_headroom_in_the_registered_grid() -> None:
    rows = controlled_benchmark_grid()

    assert any(row["optimal_dp"] > row["gated_source_rollout"] for row in rows)
