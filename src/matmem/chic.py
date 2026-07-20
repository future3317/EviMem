"""Convex-hull influence gradients and joint gradient matching.

CHIC does not delete outcomes from the scientific archive.  It selects which
revealed examples an expensive optimizer reads in one update.  Its target is a
downstream hull-decision gradient obtained from the subgradient of the
composition-constrained competing-hull LP.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import linprog, nnls


class HullMarginSubgradient(BaseModel):
    """Candidate margin and its subgradient over predicted phase energies."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    candidate_index: int = Field(ge=0)
    competing_hull_energy: float
    margin: float
    predicted_energy_subgradient: tuple[float, ...]
    competing_weights: tuple[float, ...]


class GradientMatchResult(BaseModel):
    """Deterministic non-negative joint gradient approximation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_indices: tuple[int, ...]
    weights: tuple[float, ...]
    initial_error_norm: float
    residual_norm: float
    budget_used: float


class HullInfluenceAcquisitionResult(BaseModel):
    """Observable ridge predictions and hull-influence acquisition scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    predicted_target_energies: tuple[float, ...]
    predictive_standard_deviations: tuple[float, ...]
    decision_gradient_norm: float
    feasible_margin_count: int = Field(ge=0)
    fallback: str | None = None


class PredictedFinalHullAcquisitionResult(BaseModel):
    """Self-removed margins on the model-predicted final candidate hull."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    predicted_final_hull_margins: tuple[float, ...]
    predicted_target_energies: tuple[float, ...]
    predictive_standard_deviations: tuple[float, ...]
    feasible_margin_count: int = Field(ge=0)


def _validate_phase_arrays(
    compositions: np.ndarray,
    energies: np.ndarray,
    *,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    fractions = np.asarray(compositions, dtype=np.float64)
    values = np.asarray(energies, dtype=np.float64)
    if fractions.ndim != 2 or values.ndim != 1 or len(fractions) != len(values):
        raise ValueError(f"{name} phase arrays have inconsistent shapes")
    if not len(values) or not np.isfinite(fractions).all() or not np.isfinite(values).all():
        raise ValueError(f"{name} phase arrays must be nonempty and finite")
    if np.any(fractions < 0) or not np.allclose(fractions.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError(f"{name} compositions must be normalized non-negative fractions")
    return fractions, values


def hull_margin_subgradient(
    *,
    candidate_index: int,
    predicted_compositions: np.ndarray,
    predicted_energies: np.ndarray,
    reference_compositions: np.ndarray,
    reference_energies: np.ndarray,
    feasibility_tolerance: float = 1e-9,
) -> HullMarginSubgradient | None:
    """Solve the competing hull and return a valid LP subgradient.

    The candidate is explicitly removed from its competing phase set.  LP
    weights on other predicted phases become negative coefficients in the
    candidate-margin subgradient; reference phases are constants and therefore
    contribute no model-energy derivative.
    """

    predicted_x, predicted_e = _validate_phase_arrays(
        predicted_compositions, predicted_energies, name="predicted"
    )
    reference_x, reference_e = _validate_phase_arrays(
        reference_compositions, reference_energies, name="reference"
    )
    if predicted_x.shape[1] != reference_x.shape[1]:
        raise ValueError("predicted and reference phases use different element spaces")
    if not 0 <= candidate_index < len(predicted_e):
        raise IndexError("candidate index is outside predicted phase arrays")
    if not math.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
        raise ValueError("feasibility tolerance must be finite and positive")

    other_indices = [index for index in range(len(predicted_e)) if index != candidate_index]
    competitor_x = np.vstack((reference_x, predicted_x[other_indices]))
    competitor_e = np.concatenate((reference_e, predicted_e[other_indices]))
    target = predicted_x[candidate_index]
    a_eq = np.vstack((np.ones(len(competitor_e)), competitor_x[:, :-1].T))
    b_eq = np.concatenate(([1.0], target[:-1]))
    solved = linprog(
        competitor_e,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if solved.status == 2:
        return None
    if not solved.success or solved.fun is None or solved.x is None:
        raise RuntimeError(f"CHIC competing-hull LP failed: {solved.message}")
    if np.max(np.abs(a_eq @ solved.x - b_eq)) > feasibility_tolerance:
        raise RuntimeError("CHIC competing-hull LP returned an infeasible mixture")

    reference_count = len(reference_e)
    energy_gradient = np.zeros(len(predicted_e), dtype=np.float64)
    energy_gradient[candidate_index] = 1.0
    for local_index, predicted_index in enumerate(other_indices):
        energy_gradient[predicted_index] -= solved.x[reference_count + local_index]
    hull_energy = float(solved.fun)
    return HullMarginSubgradient(
        candidate_index=candidate_index,
        competing_hull_energy=hull_energy,
        margin=float(predicted_e[candidate_index] - hull_energy),
        predicted_energy_subgradient=tuple(float(value) for value in energy_gradient),
        competing_weights=tuple(float(value) for value in solved.x),
    )


def joint_nonnegative_gradient_match(
    target_gradient: np.ndarray,
    sample_gradients: np.ndarray,
    *,
    max_items: int,
    costs: np.ndarray | None = None,
    budget: float | None = None,
    minimum_improvement: float = 1e-12,
) -> GradientMatchResult:
    """Greedily choose a set, refitting all non-negative weights jointly.

    Each step tries every legal addition and solves the exact NNLS problem for
    that candidate set.  This avoids singleton utility: redundancy and
    complementarity are evaluated after jointly refitting the selected set.
    """

    target = np.asarray(target_gradient, dtype=np.float64).reshape(-1)
    gradients = np.asarray(sample_gradients, dtype=np.float64)
    if gradients.ndim != 2 or gradients.shape[1] != len(target):
        raise ValueError("sample gradients and target gradient have inconsistent shapes")
    if not np.isfinite(target).all() or not np.isfinite(gradients).all():
        raise ValueError("gradient matching inputs must be finite")
    if max_items < 0:
        raise ValueError("max_items cannot be negative")
    item_costs = (
        np.ones(len(gradients), dtype=np.float64)
        if costs is None
        else np.asarray(costs, dtype=np.float64).reshape(-1)
    )
    if len(item_costs) != len(gradients) or np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("gradient costs must be finite and positive")
    allowed_budget = float(item_costs.sum()) if budget is None else float(budget)
    if not math.isfinite(allowed_budget) or allowed_budget < 0:
        raise ValueError("gradient budget must be finite and non-negative")
    if not math.isfinite(minimum_improvement) or minimum_improvement < 0:
        raise ValueError("minimum improvement must be finite and non-negative")

    initial_norm = float(np.linalg.norm(target))
    selected: list[int] = []
    weights = np.empty(0, dtype=np.float64)
    residual_norm = initial_norm
    budget_used = 0.0
    while len(selected) < min(max_items, len(gradients)):
        best: tuple[float, float, int, np.ndarray] | None = None
        for index in range(len(gradients)):
            if index in selected or budget_used + item_costs[index] > allowed_budget + 1e-12:
                continue
            trial = [*selected, index]
            trial_weights, trial_residual = nnls(gradients[trial].T, target)
            improvement = residual_norm - float(trial_residual)
            score = improvement / item_costs[index]
            candidate = (score, improvement, -index, trial_weights)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        if best is None or best[1] <= minimum_improvement:
            break
        index = -best[2]
        selected.append(index)
        budget_used += float(item_costs[index])
        weights, residual_norm = nnls(gradients[selected].T, target)
        residual_norm = float(residual_norm)

    return GradientMatchResult(
        selected_indices=tuple(selected),
        weights=tuple(float(value) for value in weights),
        initial_error_norm=initial_norm,
        residual_norm=residual_norm,
        budget_used=budget_used,
    )


def _ridge_working_state(
    *,
    query_x: np.ndarray,
    source: np.ndarray,
    history_x: np.ndarray,
    history_source: np.ndarray,
    history_target: np.ndarray,
    ridge_penalty: float,
    prior_standard_deviation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    augmented_query = np.column_stack((query_x, source))
    augmented_history = np.column_stack((history_x, history_source))
    pooled = np.vstack((augmented_query, augmented_history))
    feature_mean = pooled.mean(axis=0)
    feature_scale = pooled.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized_query = (augmented_query - feature_mean) / feature_scale
    standardized_history = (augmented_history - feature_mean) / feature_scale
    query_design = np.column_stack((np.ones(len(query_x)), standardized_query))
    history_design = np.column_stack((np.ones(len(history_x)), standardized_history))
    precision = history_design.T @ history_design
    precision += ridge_penalty * np.eye(precision.shape[0])
    discrepancy = history_target - history_source
    natural_parameter = history_design.T @ discrepancy
    coefficients = np.linalg.solve(precision, natural_parameter)
    predicted = source + query_design @ coefficients
    if len(history_x) >= 2:
        residual = discrepancy - history_design @ coefficients
        noise_scale = max(
            prior_standard_deviation / 2.0,
            float(np.sqrt(np.mean(residual**2))),
        )
    else:
        noise_scale = prior_standard_deviation
    covariance_shape = np.linalg.inv(precision)
    leverage = np.einsum("ij,jk,ik->i", query_design, covariance_shape, query_design)
    predictive_std = noise_scale * np.sqrt(1.0 + np.maximum(leverage, 0.0))
    return (
        query_design,
        precision,
        natural_parameter,
        coefficients,
        predicted,
        predictive_std,
    )


def linear_ridge_hull_influence_acquisition(
    *,
    query_features: np.ndarray,
    query_source_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    history_features: np.ndarray,
    history_source_energies: np.ndarray,
    history_target_energies: np.ndarray,
    costs: np.ndarray | None = None,
    ridge_penalty: float = 1.0,
    prior_standard_deviation: float = 0.1,
    boundary_temperature: float = 0.05,
) -> HullInfluenceAcquisitionResult:
    """Score legal queries by expected alignment with the hull-decision gradient.

    A ridge model predicts the target-minus-source protocol discrepancy from
    policy-visible source features.  For a possible query ``i``, a Gaussian
    working model gives an expected absolute sample-gradient magnitude
    proportional to its predictive standard deviation.  CHIC multiplies that
    quantity by the alignment between the sample direction and the current
    composition-dependent hull-risk gradient.  No unrevealed target outcome is
    an input, and the uncertainty is a working acquisition scale rather than a
    calibration certificate.
    """

    query_x = np.asarray(query_features, dtype=np.float64)
    source = np.asarray(query_source_energies, dtype=np.float64).reshape(-1)
    competing_hull = np.asarray(
        current_competing_hull_energies, dtype=np.float64
    ).reshape(-1)
    history_x = np.asarray(history_features, dtype=np.float64)
    history_source = np.asarray(history_source_energies, dtype=np.float64).reshape(-1)
    history_target = np.asarray(history_target_energies, dtype=np.float64).reshape(-1)
    if query_x.ndim != 2 or not len(query_x) or len(source) != len(query_x):
        raise ValueError("CHIC query features and source energies have inconsistent shapes")
    if len(competing_hull) != len(query_x):
        raise ValueError("CHIC competing hull energies have inconsistent shapes")
    if history_x.ndim != 2 or history_x.shape[1] != query_x.shape[1]:
        raise ValueError("CHIC history and query features use different dimensions")
    if len(history_x) != len(history_source) or len(history_x) != len(history_target):
        raise ValueError("CHIC history arrays have inconsistent shapes")
    arrays = (
        query_x,
        source,
        competing_hull,
        history_x,
        history_source,
        history_target,
    )
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("CHIC acquisition inputs must be finite")
    if (
        not math.isfinite(ridge_penalty)
        or ridge_penalty <= 0
        or not math.isfinite(prior_standard_deviation)
        or prior_standard_deviation <= 0
        or not math.isfinite(boundary_temperature)
        or boundary_temperature <= 0
    ):
        raise ValueError("CHIC ridge, prior and boundary scales must be finite and positive")
    item_costs = (
        np.ones(len(query_x), dtype=np.float64)
        if costs is None
        else np.asarray(costs, dtype=np.float64).reshape(-1)
    )
    if len(item_costs) != len(query_x) or np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("CHIC query costs must be finite and positive")

    query_design, _, _, _, predicted, predictive_std = _ridge_working_state(
        query_x=query_x,
        source=source,
        history_x=history_x,
        history_source=history_source,
        history_target=history_target,
        ridge_penalty=ridge_penalty,
        prior_standard_deviation=prior_standard_deviation,
    )

    decision_gradient = np.zeros(query_design.shape[1], dtype=np.float64)
    for candidate_index in range(len(query_x)):
        margin = predicted[candidate_index] - competing_hull[candidate_index]
        smoothed_abs = math.sqrt(margin * margin + 1e-8)
        boundary_weight = math.exp(-smoothed_abs / boundary_temperature)
        derivative = -boundary_weight * margin / (boundary_temperature * smoothed_abs)
        decision_gradient += derivative * query_design[candidate_index]
    decision_gradient /= len(query_x)
    decision_norm = float(np.linalg.norm(decision_gradient))
    if decision_norm <= 1e-12:
        scores = predictive_std / item_costs
        fallback = "ridge_uncertainty_zero_hull_gradient"
    else:
        alignment = np.abs(query_design @ (decision_gradient / decision_norm))
        scores = predictive_std * alignment / item_costs
        fallback = None
    return HullInfluenceAcquisitionResult(
        scores=tuple(float(value) for value in scores),
        predicted_target_energies=tuple(float(value) for value in predicted),
        predictive_standard_deviations=tuple(float(value) for value in predictive_std),
        decision_gradient_norm=decision_norm,
        feasible_margin_count=len(query_x),
        fallback=fallback,
    )


def linear_ridge_predicted_final_hull_acquisition(
    *,
    query_features: np.ndarray,
    query_source_energies: np.ndarray,
    query_compositions: np.ndarray,
    reference_compositions: np.ndarray,
    reference_energies: np.ndarray,
    history_features: np.ndarray,
    history_source_energies: np.ndarray,
    history_target_energies: np.ndarray,
    costs: np.ndarray | None = None,
    ridge_penalty: float = 1.0,
    prior_standard_deviation: float = 0.1,
) -> PredictedFinalHullAcquisitionResult:
    """Rank candidates by self-removed support on a predicted future hull.

    All remaining candidate energies are predictions from policy-visible PBE
    features and revealed target outcomes.  They may compete in this forecast,
    but never enter the causal evaluator or revealed phase archive.  Removing
    the queried candidate from its own competing set prevents self-removal gain.
    """

    query_x = np.asarray(query_features, dtype=np.float64)
    source = np.asarray(query_source_energies, dtype=np.float64).reshape(-1)
    query_c = np.asarray(query_compositions, dtype=np.float64)
    history_x = np.asarray(history_features, dtype=np.float64)
    history_source = np.asarray(history_source_energies, dtype=np.float64).reshape(-1)
    history_target = np.asarray(history_target_energies, dtype=np.float64).reshape(-1)
    if query_x.ndim != 2 or not len(query_x) or len(source) != len(query_x):
        raise ValueError("predicted-final query features and energies disagree")
    if query_c.ndim != 2 or len(query_c) != len(query_x):
        raise ValueError("predicted-final query compositions disagree")
    if history_x.ndim != 2 or history_x.shape[1] != query_x.shape[1]:
        raise ValueError("predicted-final history feature dimension disagrees")
    if len(history_x) != len(history_source) or len(history_x) != len(history_target):
        raise ValueError("predicted-final history arrays disagree")
    arrays = (query_x, source, query_c, history_x, history_source, history_target)
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("predicted-final acquisition inputs must be finite")
    item_costs = (
        np.ones(len(query_x), dtype=np.float64)
        if costs is None
        else np.asarray(costs, dtype=np.float64).reshape(-1)
    )
    if len(item_costs) != len(query_x) or np.any(~np.isfinite(item_costs)) or np.any(
        item_costs <= 0
    ):
        raise ValueError("predicted-final query costs must be finite and positive")
    _, _, _, _, predicted, predictive_std = _ridge_working_state(
        query_x=query_x,
        source=source,
        history_x=history_x,
        history_source=history_source,
        history_target=history_target,
        ridge_penalty=ridge_penalty,
        prior_standard_deviation=prior_standard_deviation,
    )
    predicted_x, predicted_e = _validate_phase_arrays(
        query_c, predicted, name="predicted-final candidates"
    )
    reference_x, reference_e = _validate_phase_arrays(
        reference_compositions, reference_energies, name="predicted-final reference"
    )
    if predicted_x.shape[1] != reference_x.shape[1]:
        raise ValueError("predicted-final candidate and reference element spaces disagree")
    margins = np.full(len(query_x), np.inf, dtype=np.float64)
    feasible = 0
    for index in range(len(query_x)):
        result = hull_margin_subgradient(
            candidate_index=index,
            predicted_compositions=predicted_x,
            predicted_energies=predicted_e,
            reference_compositions=reference_x,
            reference_energies=reference_e,
        )
        if result is not None:
            margins[index] = result.margin
            feasible += 1
    if not feasible:
        raise ValueError("predicted-final hull cannot represent any query composition")
    scores = -margins / item_costs
    return PredictedFinalHullAcquisitionResult(
        scores=tuple(float(value) for value in scores),
        predicted_final_hull_margins=tuple(float(value) for value in margins),
        predicted_target_energies=tuple(float(value) for value in predicted),
        predictive_standard_deviations=tuple(float(value) for value in predictive_std),
        feasible_margin_count=feasible,
    )


def smooth_decision_update_deviation_bound(
    *,
    update_direction_error: float,
    step_size: float,
    downstream_gradient_norm: float,
    smoothness: float,
) -> float:
    """One-step decision-loss deviation bound for a selected update."""

    values = (
        update_direction_error,
        step_size,
        downstream_gradient_norm,
        smoothness,
    )
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("decision-update bound inputs must be finite and non-negative")
    return (
        step_size * downstream_gradient_norm * update_direction_error
        + 0.5 * smoothness * step_size**2 * update_direction_error**2
    )
