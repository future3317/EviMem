from __future__ import annotations

import numpy as np
import pytest

from matmem.chic import (
    hull_margin_subgradient,
    joint_nonnegative_gradient_match,
    linear_ridge_hull_influence_acquisition,
    linear_ridge_predicted_final_hull_acquisition,
    smooth_decision_update_deviation_bound,
)


def test_hull_lp_subgradient_matches_finite_difference() -> None:
    predicted_x = np.asarray(
        [
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
        ]
    )
    predicted_e = np.asarray([-0.20, -0.10])
    reference_x = np.eye(3)
    reference_e = np.asarray([0.0, 0.0, 0.0])
    result = hull_margin_subgradient(
        candidate_index=0,
        predicted_compositions=predicted_x,
        predicted_energies=predicted_e,
        reference_compositions=reference_x,
        reference_energies=reference_e,
    )
    assert result is not None
    assert result.margin == pytest.approx(-0.20)
    analytic = np.asarray(result.predicted_energy_subgradient)
    epsilon = 1e-6
    numerical = []
    for index in range(len(predicted_e)):
        changed = predicted_e.copy()
        changed[index] += epsilon
        perturbed = hull_margin_subgradient(
            candidate_index=0,
            predicted_compositions=predicted_x,
            predicted_energies=changed,
            reference_compositions=reference_x,
            reference_energies=reference_e,
        )
        assert perturbed is not None
        numerical.append((perturbed.margin - result.margin) / epsilon)
    assert np.asarray(numerical) == pytest.approx(analytic, abs=1e-7)


def test_candidate_cannot_support_its_own_competing_hull() -> None:
    result = hull_margin_subgradient(
        candidate_index=0,
        predicted_compositions=np.asarray([[0.5, 0.5], [0.5, 0.5]]),
        predicted_energies=np.asarray([-10.0, 1.0]),
        reference_compositions=np.eye(2),
        reference_energies=np.asarray([0.0, 0.0]),
    )
    assert result is not None
    assert result.competing_hull_energy == pytest.approx(0.0)
    assert result.predicted_energy_subgradient[0] == 1.0


def test_joint_gradient_match_captures_complementarity() -> None:
    result = joint_nonnegative_gradient_match(
        np.asarray([1.0, 1.0]),
        np.asarray(
            [
                [0.9, 0.2],
                [0.0, 1.0],
                [-1.0, -1.0],
            ]
        ),
        max_items=2,
    )
    assert result.selected_indices == (0, 1)
    assert result.residual_norm < 1e-10
    assert all(weight >= 0 for weight in result.weights)


def test_gradient_match_respects_measured_budget() -> None:
    result = joint_nonnegative_gradient_match(
        np.asarray([1.0, 1.0]),
        np.eye(2),
        max_items=2,
        costs=np.asarray([2.0, 1.0]),
        budget=1.0,
    )
    assert result.selected_indices == (1,)
    assert result.budget_used == 1.0


def test_smooth_decision_bound_is_zero_only_for_exact_update() -> None:
    assert (
        smooth_decision_update_deviation_bound(
            update_direction_error=0.0,
            step_size=0.1,
            downstream_gradient_norm=2.0,
            smoothness=3.0,
        )
        == 0.0
    )
    assert smooth_decision_update_deviation_bound(
        update_direction_error=0.2,
        step_size=0.1,
        downstream_gradient_norm=2.0,
        smoothness=3.0,
    ) == pytest.approx(0.0406)


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
    assert result.predicted_final_hull_margins == pytest.approx((-10.0,))
    assert result.scores == pytest.approx((10.0,))


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
