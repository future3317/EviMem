from __future__ import annotations

import numpy as np

from matmem.ridge_acquisition import (
    linear_ridge_hull_influence_acquisition,
    linear_ridge_predicted_final_hull_acquisition,
)


def test_hull_influence_acquisition_is_finite_and_nonconstant() -> None:
    result = linear_ridge_hull_influence_acquisition(
        query_features=np.asarray([[0.0, 1.0], [1.0, 0.0], [0.8, 0.8]]),
        query_source_energies=np.asarray([-0.15, -0.02, -0.12]),
        current_competing_hull_energies=np.zeros(3),
        history_features=np.empty((0, 2)),
        history_source_energies=np.empty(0),
        history_target_energies=np.empty(0),
    )
    assert np.isfinite(result.scores).all()
    assert np.ptp(result.scores) > 1e-8
    assert result.feasible_margin_count == 3
    assert result.decision_gradient_norm > 0


def test_predicted_final_hull_acquisition_self_removes_candidate() -> None:
    result = linear_ridge_predicted_final_hull_acquisition(
        query_features=np.asarray([[1.0, 0.0]]),
        query_source_energies=np.asarray([-10.0]),
        query_compositions=np.asarray([[0.5, 0.5]]),
        reference_compositions=np.eye(2),
        reference_energies=np.zeros(2),
        history_features=np.empty((0, 2)),
        history_source_energies=np.empty(0),
        history_target_energies=np.empty(0),
    )
    assert result.feasible_margin_count == 1
    assert result.predicted_final_hull_margins == (-10.0,)
    assert result.scores == (10.0,)


def test_predicted_final_hull_acquisition_prefers_forecast_support_phase() -> None:
    result = linear_ridge_predicted_final_hull_acquisition(
        query_features=np.eye(2),
        query_source_energies=np.asarray([-0.4, -0.1]),
        query_compositions=np.asarray([[0.5, 0.5], [0.25, 0.75]]),
        reference_compositions=np.eye(2),
        reference_energies=np.zeros(2),
        history_features=np.empty((0, 2)),
        history_source_energies=np.empty(0),
        history_target_energies=np.empty(0),
    )
    assert np.argmax(result.scores) == 0
    assert result.predicted_final_hull_margins[0] < 0
