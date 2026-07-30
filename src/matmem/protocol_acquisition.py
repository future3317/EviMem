"""Acquisition and rollout policies for protocol-aware hull discovery.

This module contains the public policy functions that were originally in
``protocol_knowledge_gradient.py``: delta-hull active search, source-rollout,
dual-horizon rollout, independent confirmation, conformal one-deviation,
protocol hull knowledge gradient, and protocol hull risk reduction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import t as student_t

from .hull_geometry import (
    FixedCompositionHullTemplate,
    FixedHullRuntimePlan,
    _CausalHullEnvelope,
    _final_hull_membership,
    _final_hull_values,
    _fixed_evaluation_compositions,
    fixed_composition_hull_membership,
)
from .posterior import ProtocolTargetEnergyPosterior, _sample_gaussian, _sample_gaussian_blocks


class ProtocolHullKnowledgeGradientResult(BaseModel):
    """Two-step final-hull discovery values under a working posterior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    final_stability_probabilities: tuple[float, ...]
    expected_second_step_values: tuple[float, ...]
    posterior_risk: float = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)
    fantasy_count: int = Field(gt=0)
    horizon: int = Field(ge=1, le=2)


class DeltaHullActiveSearchResult(BaseModel):
    """Myopic Bayes action values for target-protocol hull discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    final_stability_probabilities: tuple[float, ...]
    posterior_sample_count: int = Field(gt=0)


class ProtocolHullRiskReductionResult(BaseModel):
    """Myopic Bayes-risk reduction for the target-protocol hull function."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    risk_reductions: tuple[float, ...]
    expected_posterior_risks: tuple[float, ...]
    current_hull_risk: float = Field(ge=0)
    evaluation_composition_count: int = Field(gt=0)
    posterior_sample_count: int = Field(gt=0)
    fantasy_count: int = Field(gt=0)


class ProtocolHullPosteriorSummary(BaseModel):
    """Posterior moments of the random target hull on the fixed pool grid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_compositions: tuple[dict[str, float], ...]
    mean_hull_energies: tuple[float, ...]
    hull_variances: tuple[float, ...]
    bayes_risk: float = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)


class SourceRolloutDeltaHullResult(BaseModel):
    """Full-budget rollout values using source margin as continuation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    block_scores: tuple[tuple[float, ...], ...]
    final_stability_probabilities: tuple[float, ...]
    paired_advantages_over_source: tuple[float, ...]
    paired_advantage_lower_bounds: tuple[float, ...]
    source_action_index: int = Field(ge=0)
    selected_action_index: int = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)
    sobol_scramble_count: int = Field(gt=1)
    simultaneous_comparison_count: int = Field(gt=0)
    horizon: int = Field(gt=0)
    fallback_reason: str | None = None


class DualHorizonSourceRolloutResult(BaseModel):
    """Source-rollout action values under terminal and causal horizons.

    ``terminal`` evaluates membership in the complete target-protocol hull,
    while ``causal`` evaluates the same simulated selected set against the
    hull that would be available after only those selected outcomes had been
    revealed.  A deviation is legal only when both simultaneous lower-bound
    gates pass; the source-margin action is always a legal fallback.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    terminal_scores: tuple[float, ...]
    causal_scores: tuple[float, ...]
    terminal_block_scores: tuple[tuple[float, ...], ...]
    causal_block_scores: tuple[tuple[float, ...], ...]
    terminal_paired_advantages: tuple[float, ...]
    causal_paired_advantages: tuple[float, ...]
    terminal_lower_bounds: tuple[float, ...]
    causal_lower_bounds: tuple[float, ...]
    terminal_final_stability_probabilities: tuple[float, ...]
    source_action_index: int = Field(ge=0)
    selected_action_index: int = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)
    sobol_scramble_count: int = Field(gt=1)
    simultaneous_comparison_count: int = Field(gt=0)
    horizon: int = Field(gt=0)
    feasible_mask: tuple[bool, ...]
    fallback_reason: str | None = None


class IndependentConfirmationSourceRolloutResult(BaseModel):
    """Two-stage, independently randomized numerical gate for SARR.

    The first stage is the frozen simultaneous SARR screen.  A second stage is
    permitted only for its preselected positive-but-unresolved candidate and
    uses a disjoint RQMC stream for one paired comparison against source.
    This controls integration noise conditional on the screen; it says
    nothing about posterior calibration or oracle-final utility.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_one: SourceRolloutDeltaHullResult
    source_action_index: int = Field(ge=0)
    screened_action_index: int | None = Field(default=None, ge=0)
    selected_action_index: int = Field(ge=0)
    stage_two_used: bool
    stage_two_paired_advantage: float | None = None
    stage_two_paired_lower_bound: float | None = None
    stage_two_block_advantages: tuple[float, ...] = ()
    stage_one_seed: int
    stage_two_seed: int | None = None
    stage_one_posterior_sample_count: int = Field(gt=0)
    stage_two_posterior_sample_count: int | None = Field(default=None, gt=0)
    sobol_scramble_count: int = Field(gt=1)
    fallback_reason: str | None = None


class ConformalSourceRolloutCalibration(BaseModel):
    """Exact-system calibration for a single source-relative deviation.

    ``radius`` is an upper quantile of the system-level maximum rollout
    over-estimation.  It is a deployment threshold, not a posterior standard
    deviation or a guarantee for arbitrary adaptive policies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(gt=0, lt=1)
    system_ids: tuple[str, ...]
    system_scores: tuple[float, ...]
    order_statistic_one_based: int = Field(gt=0)
    radius: float = Field(ge=0)

    @model_validator(mode="after")
    def _calibration_dimensions(self) -> ConformalSourceRolloutCalibration:
        if not self.system_ids or len(set(self.system_ids)) != len(self.system_ids):
            raise ValueError("conformal rollout systems must be unique and nonempty")
        if len(self.system_scores) != len(self.system_ids):
            raise ValueError("conformal rollout scores and systems disagree")
        if any(not math.isfinite(value) or value < 0 for value in self.system_scores):
            raise ValueError("conformal rollout scores must be finite and non-negative")
        if not 1 <= self.order_statistic_one_based <= len(self.system_ids):
            raise ValueError("conformal rollout order statistic is out of range")
        if not math.isfinite(self.radius):
            raise ValueError("conformal rollout radius must be finite")
        return self

    @property
    def identity_checksum(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class ConformalSourceRolloutResult(BaseModel):
    """One-deviation source-rollout decision and its numerical diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    paired_advantages_over_source: tuple[float, ...]
    rqmc_radii: tuple[float, ...]
    conformal_adjusted_advantages: tuple[float, ...]
    source_action_index: int = Field(ge=0)
    selected_action_index: int = Field(ge=0)
    deviation_used_before: bool
    deviation_selected: bool
    fallback_reason: str | None = None
    conformal_radius: float = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)
    sobol_scramble_count: int = Field(gt=1)
    horizon: int = Field(gt=0)


def source_margin_action_indices(
    *,
    source_energies: np.ndarray,
    competing_hull_energies: np.ndarray,
    query_ids: Sequence[str],
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Select source-margin actions with immutable-ID tie breaking.

    This vectorized primitive is shared by the deployed source policy and by
    posterior rollout continuations. Rows of ``competing_hull_energies`` are
    independent simulated causal-hull states; columns follow ``query_ids``.
    """

    source = np.asarray(source_energies, dtype=np.float64).reshape(-1)
    hull = np.asarray(competing_hull_energies, dtype=np.float64)
    if hull.ndim == 1:
        hull = hull[None, :]
    if hull.ndim != 2 or hull.shape[1] != len(source) or len(query_ids) != len(source):
        raise ValueError("source-margin arrays disagree")
    if not len(source) or not np.isfinite(source).all() or not np.isfinite(hull).all():
        raise ValueError("source-margin inputs must be nonempty and finite")
    if len(set(query_ids)) != len(query_ids) or any(not str(value) for value in query_ids):
        raise ValueError("source-margin query IDs must be unique and nonempty")
    mask = np.ones(hull.shape, dtype=bool)
    if eligible is not None:
        provided = np.asarray(eligible, dtype=bool)
        if provided.ndim == 1:
            provided = np.broadcast_to(provided[None, :], hull.shape)
        if provided.shape != hull.shape:
            raise ValueError("source-margin eligibility mask disagrees")
        mask = provided
    if np.any(~np.any(mask, axis=1)):
        raise ValueError("source-margin state has no eligible action")
    margins = source[None, :] - hull
    margins = np.where(mask, margins, np.inf)
    identifier_order = np.argsort(np.asarray(query_ids, dtype=str), kind="stable")
    ordered_actions = np.argmin(margins[:, identifier_order], axis=1)
    return identifier_order[ordered_actions]


def _source_rollout_rewards(
    *,
    sampled_query_energies: np.ndarray,
    final_hull_membership: np.ndarray,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    horizon: int,
    first_action_indices: Sequence[int] | None = None,
    causal_rewards_output: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate requested first actions under a source-margin continuation.

    ``first_action_indices`` makes an independently confirmed one-comparison
    audit proportional to two rollouts rather than to every legal action.  It
    never changes the continuation or accesses unobserved oracle outcomes.
    """

    samples = np.asarray(sampled_query_energies, dtype=np.float64)
    labels = np.asarray(final_hull_membership, dtype=bool)
    source = np.asarray(query_source_energies, dtype=np.float64).reshape(-1)
    references = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or labels.shape != samples.shape:
        raise ValueError("source rollout samples and final-hull labels disagree")
    if (
        samples.shape[1] != len(query_compositions)
        or len(source) != samples.shape[1]
        or len(query_ids) != samples.shape[1]
        or len(references) != len(reference_compositions)
    ):
        raise ValueError("source rollout arrays disagree")
    if horizon < 1 or horizon > samples.shape[1]:
        raise ValueError("source rollout horizon is invalid")
    if (
        not np.isfinite(samples).all()
        or not np.isfinite(source).all()
        or not np.isfinite(references).all()
    ):
        raise ValueError("source rollout energies must be finite")

    sample_count, query_count = samples.shape
    action_indices = (
        tuple(range(query_count))
        if first_action_indices is None
        else tuple(int(index) for index in first_action_indices)
    )
    if not action_indices or len(set(action_indices)) != len(action_indices):
        raise ValueError("source rollout first actions must be unique and nonempty")
    if any(index < 0 or index >= query_count for index in action_indices):
        raise ValueError("source rollout first action is out of range")
    rewards = np.empty((sample_count, len(action_indices)), dtype=np.float64)
    if causal_rewards_output is not None:
        if np.asarray(causal_rewards_output).shape != rewards.shape:
            raise ValueError("causal rollout output has inconsistent shape")
        causal_values = np.asarray(causal_rewards_output)
        if not np.issubdtype(causal_values.dtype, np.floating):
            raise ValueError("causal rollout output must have a floating dtype")
    else:
        causal_values = None
    geometry_cache: dict[tuple[int, ...], _CausalHullEnvelope] = {}
    causal_geometry_cache: dict[tuple[int, ...], tuple[FixedCompositionHullTemplate, FixedHullRuntimePlan]] = {}

    def geometry(selected: tuple[int, ...]) -> _CausalHullEnvelope:
        cached = geometry_cache.get(selected)
        if cached is None:
            cached = _CausalHullEnvelope.build(
                query_compositions=query_compositions,
                reference_compositions=reference_compositions,
                selected_query_indices=selected,
            )
            geometry_cache[selected] = cached
        return cached

    for output_index, first_action in enumerate(action_indices):
        # A rollout can select at most ``horizon`` candidates.  Keeping that
        # compact canonical set avoids repeatedly scanning a sample-by-query
        # boolean matrix merely to reconstruct the same selected tuple.
        selected_indices = np.empty((sample_count, horizon), dtype=np.int64)
        selected_indices[:, 0] = first_action
        for selected_count in range(1, horizon):
            groups: dict[tuple[int, ...], list[int]] = {}
            for sample_index in range(sample_count):
                key = tuple(int(index) for index in selected_indices[sample_index, :selected_count])
                groups.setdefault(key, []).append(sample_index)
            for key, row_indices in groups.items():
                rows = np.asarray(row_indices, dtype=np.int64)
                envelope = geometry(key)
                active_energies = np.column_stack(
                    (
                        np.broadcast_to(references, (len(rows), len(references))),
                        samples[np.ix_(rows, np.asarray(key, dtype=np.int64))],
                    )
                )
                hull = envelope.competing_hull_energies(active_energies)
                eligible = np.ones(query_count, dtype=bool)
                eligible[np.asarray(key, dtype=np.int64)] = False
                next_actions = source_margin_action_indices(
                    source_energies=source,
                    competing_hull_energies=hull,
                    query_ids=query_ids,
                    eligible=eligible,
                )
                selected_indices[rows, selected_count] = next_actions
            # The historical bool-mask key was ascending query index.  Preserve
            # that exact canonical state representation after each transition.
            selected_indices[:, : selected_count + 1].sort(axis=1)
        rewards[:, output_index] = labels[
            np.arange(sample_count, dtype=np.int64)[:, None], selected_indices
        ].sum(axis=1)
        if causal_values is not None:
            # The causal horizon uses exactly the outcomes that the simulated
            # source continuation selected.  No unselected sampled outcome is
            # allowed to enter this hull.  Grouping by the selected-set key
            # lets us reuse the composition-only template across samples.
            causal_groups: dict[tuple[int, ...], list[int]] = {}
            for sample_index in range(sample_count):
                key = tuple(int(index) for index in selected_indices[sample_index])
                causal_groups.setdefault(key, []).append(sample_index)
            for key, row_indices in causal_groups.items():
                cached = causal_geometry_cache.get(key)
                if cached is None:
                    template = FixedCompositionHullTemplate.from_compositions(
                        query_compositions=tuple(query_compositions[index] for index in key),
                        reference_compositions=reference_compositions,
                    )
                    cached = (template, FixedHullRuntimePlan.from_template(template))
                    causal_geometry_cache[key] = cached
                template, runtime_plan = cached
                rows = np.asarray(row_indices, dtype=np.int64)
                selected_energies = samples[np.ix_(rows, np.asarray(key, dtype=np.int64))]
                stable = fixed_composition_hull_membership(
                    template,
                    query_energies=selected_energies,
                    reference_energies=references,
                    runtime_plan=runtime_plan,
                )
                causal_values[rows, output_index] = np.sum(stable, axis=1)
    return rewards


def _simultaneous_paired_lower_bounds(
    block_differences: np.ndarray,
    *,
    confidence: float,
    comparison_count: int,
) -> np.ndarray:
    """Return one-sided Bonferroni-t lower bounds for all paired advantages.

    Rows are independent randomized-QMC blocks and columns are candidate
    advantages against the same source action.  The correction is applied to
    the non-source candidate family, so a positive returned bound supports a
    simultaneous source-relative statement rather than a collection of
    marginal tests.
    """

    values = np.asarray(block_differences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("simultaneous bounds require a block-by-candidate matrix")
    if not np.isfinite(values).all():
        raise ValueError("simultaneous bound inputs must be finite")
    if not 0.5 < confidence < 1.0:
        raise ValueError("simultaneous bound confidence must lie in (0.5, 1)")
    if comparison_count < 1:
        raise ValueError("simultaneous bound comparison count must be positive")
    alpha = (1.0 - confidence) / float(comparison_count)
    critical_value = float(student_t.ppf(1.0 - alpha, values.shape[0] - 1))
    if not math.isfinite(critical_value):
        raise ValueError("simultaneous bound critical value is not finite")
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    return means - critical_value * standard_errors


def source_rollout_system_score(
    estimated_advantages: np.ndarray,
    counterfactual_advantages: np.ndarray,
) -> float:
    """Return one exact-system conformal score for rollout over-estimation.

    Both arrays contain source-relative advantages indexed by round and legal
    first action.  Calibration must use exact-system counterfactual oracle
    traces that are disjoint from deployment systems.  Clipping at zero makes
    this an upper-error nonconformity score rather than a signed effect.
    """

    estimated = np.asarray(estimated_advantages, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_advantages, dtype=np.float64)
    if estimated.shape != counterfactual.shape or estimated.ndim < 1:
        raise ValueError("rollout advantage arrays must have the same nonempty shape")
    if not np.isfinite(estimated).all() or not np.isfinite(counterfactual).all():
        raise ValueError("rollout advantage arrays must be finite")
    return float(max(0.0, np.max(estimated - counterfactual)))


def fit_conformal_source_rollout_calibration(
    system_scores: Sequence[float],
    *,
    system_ids: Sequence[str],
    alpha: float = 0.1,
) -> ConformalSourceRolloutCalibration:
    """Fit a finite-sample exact-system split-conformal rollout threshold."""

    scores = tuple(float(value) for value in system_scores)
    identifiers = tuple(str(value) for value in system_ids)
    if len(scores) != len(identifiers) or not identifiers:
        raise ValueError("conformal rollout systems and scores disagree")
    if any(not value for value in identifiers):
        raise ValueError("conformal rollout system IDs must be nonempty")
    if not 0 < alpha < 1:
        raise ValueError("conformal rollout alpha must be in (0, 1)")
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("conformal rollout scores must be finite and non-negative")
    order = math.ceil((len(scores) + 1) * (1.0 - alpha))
    if order > len(scores):
        raise ValueError("too few exact systems for a finite conformal rollout threshold")
    radius = sorted(scores)[order - 1]
    return ConformalSourceRolloutCalibration(
        alpha=alpha,
        system_ids=identifiers,
        system_scores=scores,
        order_statistic_one_based=order,
        radius=radius,
    )


def _mean_hull_squared_error_risk(hull_values: np.ndarray) -> float:
    values = np.asarray(hull_values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("hull risk requires sampled hull functions")
    if not np.isfinite(values).all():
        raise ValueError("sampled hull functions must be finite")
    return float(np.var(values, axis=0, ddof=0).mean())


def protocol_hull_posterior_summary(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    posterior_sample_count: int = 16,
    seed: int = 0,
) -> ProtocolHullPosteriorSummary:
    """Summarize the random final hull without exposing target outcomes."""

    if posterior_sample_count < 4:
        raise ValueError("protocol hull posterior summary needs at least four samples")
    evaluation_compositions = _fixed_evaluation_compositions(
        query_compositions,
        reference_compositions,
    )
    hull_values = _final_hull_values(
        query_compositions=query_compositions,
        sampled_query_energies=_sample_gaussian(
            np.asarray(posterior.mean, dtype=np.float64),
            np.asarray(posterior.covariance, dtype=np.float64),
            sample_count=posterior_sample_count,
            seed=seed,
        ),
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        evaluation_compositions=evaluation_compositions,
    )
    means = np.mean(hull_values, axis=0)
    variances = np.var(hull_values, axis=0, ddof=0)
    return ProtocolHullPosteriorSummary(
        evaluation_compositions=evaluation_compositions,
        mean_hull_energies=tuple(float(value) for value in means),
        hull_variances=tuple(float(value) for value in variances),
        bayes_risk=float(np.mean(variances)),
        posterior_sample_count=posterior_sample_count,
    )


def delta_hull_active_search(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    costs: np.ndarray,
    posterior_sample_count: int = 16,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
) -> DeltaHullActiveSearchResult:
    """Return the exact one-step active-search objective under the posterior.

    The reward is one iff the queried configuration belongs to the final
    target-protocol hull over the complete visible fixed pool.  With one legal
    query remaining and equal query costs, maximizing its posterior membership
    probability is Bayes optimal.  Unequal costs are rejected rather than
    silently turning this finite-budget objective into a ratio heuristic.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if len(query_compositions) != len(mean) or len(item_costs) != len(mean):
        raise ValueError("delta-hull active-search inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("delta-hull query costs must be finite and positive")
    if not np.allclose(item_costs, item_costs[0], atol=1e-12):
        raise ValueError("delta-hull active search requires equal query costs")
    if posterior_sample_count < 4:
        raise ValueError("delta-hull active search needs at least four posterior samples")

    runtime_plan = (
        None if fixed_template is None else FixedHullRuntimePlan.from_template(fixed_template)
    )
    labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=_sample_gaussian(
            mean,
            covariance,
            sample_count=posterior_sample_count,
            seed=seed,
        ),
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        fixed_template=fixed_template,
        fixed_runtime_plan=runtime_plan,
    )
    probabilities = labels.mean(axis=0)
    return DeltaHullActiveSearchResult(
        scores=tuple(float(value) for value in probabilities),
        final_stability_probabilities=tuple(float(value) for value in probabilities),
        posterior_sample_count=posterior_sample_count,
    )


def source_rollout_delta_hull(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    posterior_sample_count: int = 1024,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
    sobol_scramble_count: int = 16,
    integration_confidence: float = 0.95,
) -> SourceRolloutDeltaHullResult:
    """Improve source margin by a full-remaining-budget posterior rollout.

    Every candidate first action is followed by the deployed source-margin
    policy inside each complete target-energy sample. The sampled target
    energy of every simulated query is added to the composition-dependent
    causal hull before choosing the next action. A Bonferroni-simultaneous
    paired scrambled-Sobol lower bound prevents numerically unresolved gains
    from changing the strong source action; it is only an integration
    safeguard, not a calibration or real-world safety guarantee.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    source = np.asarray(query_source_energies, dtype=np.float64).reshape(-1)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    current_hull = np.asarray(current_competing_hull_energies, dtype=np.float64).reshape(-1)
    size = len(mean)
    if (
        len(query_compositions) != size
        or len(query_ids) != size
        or len(source) != size
        or len(item_costs) != size
        or len(current_hull) != size
    ):
        raise ValueError("source-rollout Delta-Hull inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("source-rollout query costs must be finite and positive")
    if not np.allclose(item_costs, item_costs[0], atol=1e-12):
        raise ValueError("source-rollout Delta-Hull requires equal query costs")
    if not math.isfinite(remaining_budget) or remaining_budget < item_costs[0]:
        raise ValueError("remaining protocol budget cannot pay for a rollout query")
    if sobol_scramble_count < 2 or posterior_sample_count % sobol_scramble_count:
        raise ValueError("posterior samples must divide into independent Sobol scrambles")
    block_size = posterior_sample_count // sobol_scramble_count
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("each source-rollout Sobol block must have power-of-two size")
    if not 0.5 < integration_confidence < 1.0:
        raise ValueError("source-rollout integration confidence must lie in (0.5, 1)")
    horizon = min(size, int(math.floor((remaining_budget + 1e-12) / item_costs[0])))

    sample_blocks = _sample_gaussian_blocks(
        mean,
        covariance,
        sample_count=block_size,
        seeds=tuple(seed + 104729 * block_index for block_index in range(sobol_scramble_count)),
    )
    samples = np.concatenate(sample_blocks, axis=0)
    runtime_plan = (
        None if fixed_template is None else FixedHullRuntimePlan.from_template(fixed_template)
    )
    labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        fixed_template=fixed_template,
        fixed_runtime_plan=runtime_plan,
    )
    rewards = _source_rollout_rewards(
        sampled_query_energies=samples,
        final_hull_membership=labels,
        query_compositions=query_compositions,
        query_source_energies=source,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        horizon=horizon,
    )
    block_scores = rewards.reshape(sobol_scramble_count, block_size, size).mean(axis=1)
    scores = block_scores.mean(axis=0)
    source_action = int(
        source_margin_action_indices(
            source_energies=source,
            competing_hull_energies=current_hull,
            query_ids=query_ids,
        )[0]
    )
    differences = block_scores - block_scores[:, [source_action]]
    mean_advantages = differences.mean(axis=0)
    lower_bounds = _simultaneous_paired_lower_bounds(
        differences,
        confidence=integration_confidence,
        comparison_count=max(size - 1, 1),
    )
    lower_bounds[source_action] = 0.0
    improving = np.flatnonzero(lower_bounds > 0.0)
    if len(improving):
        selected_action = min(
            (int(index) for index in improving),
            key=lambda index: (-scores[index], str(query_ids[index])),
        )
        fallback_reason = None
    else:
        selected_action = source_action
        fallback_reason = (
            "source_is_only_legal_action" if size == 1 else "no_positive_simultaneous_lower_bound"
        )
    probabilities = labels.mean(axis=0)
    return SourceRolloutDeltaHullResult(
        scores=tuple(float(value) for value in scores),
        block_scores=tuple(tuple(float(value) for value in row) for row in block_scores),
        final_stability_probabilities=tuple(float(value) for value in probabilities),
        paired_advantages_over_source=tuple(float(value) for value in mean_advantages),
        paired_advantage_lower_bounds=tuple(float(value) for value in lower_bounds),
        source_action_index=source_action,
        selected_action_index=selected_action,
        posterior_sample_count=posterior_sample_count,
        sobol_scramble_count=sobol_scramble_count,
        simultaneous_comparison_count=max(size - 1, 1),
        horizon=horizon,
        fallback_reason=fallback_reason,
    )


def ungated_source_rollout_delta_hull(
    posterior: ProtocolTargetEnergyPosterior,
    **kwargs: object,
) -> SourceRolloutDeltaHullResult:
    """Return the source-continuation rollout action without a gate.

    This is an ablation-only comparator.  It computes exactly the same joint
    posterior worlds and source continuation as SARR but always follows the
    largest mean rollout value.  The numerical bounds are retained solely for
    diagnostics and never affect its action.
    """

    gated = source_rollout_delta_hull(posterior, **kwargs)  # type: ignore[arg-type]
    selected = min(
        range(len(gated.scores)),
        key=lambda index: (-gated.scores[index], index),
    )
    return gated.model_copy(
        update={
            "selected_action_index": selected,
            "fallback_reason": None if selected != gated.source_action_index else "ungated_score_tie_to_source",
        }
    )


def diagonal_independent_confirmation_source_rollout(
    posterior: ProtocolTargetEnergyPosterior,
    **kwargs: object,
) -> IndependentConfirmationSourceRolloutResult:
    """Run the frozen IC gate after removing posterior cross-covariances.

    The target mean, marginal variances, query order, source continuation,
    seeds and numerical gates are unchanged.  This comparator is therefore a
    direct test of the joint-world covariance component, not a new IC-SARR
    setting.
    """

    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    diagonal = posterior.model_copy(
        update={"covariance": tuple(tuple(float(value) for value in row) for row in np.diag(np.diag(covariance)))}
    )
    return independent_confirmation_source_rollout(diagonal, **kwargs)  # type: ignore[arg-type]


def _independent_world_stage_one(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    posterior_sample_count: int,
    seed: int,
    fixed_template: FixedCompositionHullTemplate | None,
    sobol_scramble_count: int,
    integration_confidence: float,
) -> SourceRolloutDeltaHullResult:
    """Stage one with independently sampled worlds for each first action."""

    # Reuse SARR's complete validation and geometry/horizon conventions; only
    # the action-specific randomized world streams below differ.
    checked = source_rollout_delta_hull(
        posterior,
        query_compositions=query_compositions,
        query_source_energies=query_source_energies,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        current_competing_hull_energies=current_competing_hull_energies,
        costs=costs,
        remaining_budget=remaining_budget,
        posterior_sample_count=posterior_sample_count,
        seed=seed,
        fixed_template=fixed_template,
        sobol_scramble_count=sobol_scramble_count,
        integration_confidence=integration_confidence,
    )
    size = len(query_ids)
    block_size = posterior_sample_count // sobol_scramble_count
    blocks = np.empty((sobol_scramble_count, size), dtype=np.float64)
    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    for action in range(size):
        samples = np.concatenate(
            tuple(
                np.random.default_rng(seed + 104729 * block + 15485863 * (action + 1)).multivariate_normal(
                    mean, covariance, size=block_size, check_valid="raise"
                )
                for block in range(sobol_scramble_count)
            ),
            axis=0,
        )
        labels = _final_hull_membership(
            query_compositions=query_compositions,
            sampled_query_energies=samples,
            reference_compositions=reference_compositions,
            reference_energies=reference_energies,
            fixed_template=fixed_template,
        )
        rewards = _source_rollout_rewards(
            sampled_query_energies=samples,
            final_hull_membership=labels,
            query_compositions=query_compositions,
            query_source_energies=query_source_energies,
            query_ids=query_ids,
            reference_compositions=reference_compositions,
            reference_energies=reference_energies,
            horizon=checked.horizon,
            first_action_indices=(action,),
        )[:, 0]
        blocks[:, action] = rewards.reshape(sobol_scramble_count, block_size).mean(axis=1)
    differences = blocks - blocks[:, [checked.source_action_index]]
    advantages = differences.mean(axis=0)
    lower_bounds = _simultaneous_paired_lower_bounds(
        differences,
        confidence=integration_confidence,
        comparison_count=max(size - 1, 1),
    )
    lower_bounds[checked.source_action_index] = 0.0
    improving = np.flatnonzero(lower_bounds > 0.0)
    selected = (
        min(improving, key=lambda index: (-blocks[:, index].mean(), str(query_ids[index])))
        if len(improving)
        else checked.source_action_index
    )
    return checked.model_copy(
        update={
            "scores": tuple(float(value) for value in blocks.mean(axis=0)),
            "block_scores": tuple(tuple(float(value) for value in row) for row in blocks),
            "paired_advantages_over_source": tuple(float(value) for value in advantages),
            "paired_advantage_lower_bounds": tuple(float(value) for value in lower_bounds),
            "selected_action_index": int(selected),
            "fallback_reason": None if selected != checked.source_action_index else "no_positive_independent_world_lower_bound",
        }
    )


def independent_world_confirmation_source_rollout(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    stage_one_posterior_sample_count: int = 1024,
    stage_two_posterior_sample_count: int = 8192,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
    sobol_scramble_count: int = 16,
    integration_confidence: float = 0.95,
) -> IndependentConfirmationSourceRolloutResult:
    """IC-style two-stage gate with independent worlds for candidate actions."""

    stage_one = _independent_world_stage_one(
        posterior,
        query_compositions=query_compositions,
        query_source_energies=query_source_energies,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        current_competing_hull_energies=current_competing_hull_energies,
        costs=costs,
        remaining_budget=remaining_budget,
        posterior_sample_count=stage_one_posterior_sample_count,
        seed=seed,
        fixed_template=fixed_template,
        sobol_scramble_count=sobol_scramble_count,
        integration_confidence=integration_confidence,
    )
    source = stage_one.source_action_index
    if stage_one.selected_action_index != source:
        return IndependentConfirmationSourceRolloutResult(
            stage_one=stage_one,
            source_action_index=source,
            selected_action_index=stage_one.selected_action_index,
            stage_two_used=False,
            stage_one_seed=seed,
            stage_one_posterior_sample_count=stage_one_posterior_sample_count,
            sobol_scramble_count=sobol_scramble_count,
            fallback_reason="stage_one_accepted_independent_worlds",
        )
    advantages = np.asarray(stage_one.paired_advantages_over_source, dtype=np.float64)
    positive = np.flatnonzero(advantages > 0.0)
    positive = positive[positive != source]
    if not len(positive):
        return IndependentConfirmationSourceRolloutResult(
            stage_one=stage_one,
            source_action_index=source,
            selected_action_index=source,
            stage_two_used=False,
            stage_one_seed=seed,
            stage_one_posterior_sample_count=stage_one_posterior_sample_count,
            sobol_scramble_count=sobol_scramble_count,
            fallback_reason="no_positive_independent_world_advantage",
        )
    screened = min(positive, key=lambda index: (-advantages[index], str(query_ids[index])))
    if stage_two_posterior_sample_count % sobol_scramble_count:
        raise ValueError("stage-two independent-world samples must divide into Sobol blocks")
    block_size = stage_two_posterior_sample_count // sobol_scramble_count
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("each stage-two independent-world block must have power-of-two size")
    stage_two_seed = _independent_confirmation_seed(seed)
    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)

    def rewards_for(action: int, stream: int) -> np.ndarray:
        samples = np.concatenate(
            tuple(
                np.random.default_rng(stage_two_seed + 104729 * block + stream).multivariate_normal(
                    mean, covariance, size=block_size, check_valid="raise"
                )
                for block in range(sobol_scramble_count)
            ),
            axis=0,
        )
        labels = _final_hull_membership(
            query_compositions=query_compositions,
            sampled_query_energies=samples,
            reference_compositions=reference_compositions,
            reference_energies=reference_energies,
            fixed_template=fixed_template,
        )
        return _source_rollout_rewards(
            sampled_query_energies=samples,
            final_hull_membership=labels,
            query_compositions=query_compositions,
            query_source_energies=query_source_energies,
            query_ids=query_ids,
            reference_compositions=reference_compositions,
            reference_energies=reference_energies,
            horizon=stage_one.horizon,
            first_action_indices=(action,),
        )[:, 0]

    difference = rewards_for(int(screened), 15485863) - rewards_for(source, 32452843)
    block_advantages = difference.reshape(sobol_scramble_count, block_size).mean(axis=1)
    advantage = float(block_advantages.mean())
    lower = float(
        _simultaneous_paired_lower_bounds(
            block_advantages.reshape(-1, 1), confidence=integration_confidence, comparison_count=1
        )[0]
    )
    selected = int(screened) if lower > 0.0 else source
    return IndependentConfirmationSourceRolloutResult(
        stage_one=stage_one,
        source_action_index=source,
        screened_action_index=int(screened),
        selected_action_index=selected,
        stage_two_used=True,
        stage_two_paired_advantage=advantage,
        stage_two_paired_lower_bound=lower,
        stage_two_block_advantages=tuple(float(value) for value in block_advantages),
        stage_one_seed=seed,
        stage_two_seed=stage_two_seed,
        stage_one_posterior_sample_count=stage_one_posterior_sample_count,
        stage_two_posterior_sample_count=stage_two_posterior_sample_count,
        sobol_scramble_count=sobol_scramble_count,
        fallback_reason=None if selected != source else "stage_two_independent_world_lower_bound_not_positive",
    )


def constrained_dual_horizon_source_rollout(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    posterior_sample_count: int = 1024,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
    sobol_scramble_count: int = 16,
    integration_confidence: float = 0.95,
) -> DualHorizonSourceRolloutResult:
    """Select a source-rollout action with a dual terminal/causal gate.

    For each first action, one common-random-number rollout is generated.  The
    terminal reward counts selected candidates stable in the complete sampled
    target hull.  The causal reward counts the same selected candidates stable
    in the hull containing only references plus the outcomes selected along
    that simulated continuation.  A non-source action is admissible iff its
    simultaneous one-sided lower bound is strictly positive for terminal
    reward and non-negative for causal reward.  Among admissible actions the
    terminal mean is maximized; otherwise the source-margin action is returned.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    source = np.asarray(query_source_energies, dtype=np.float64).reshape(-1)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    current_hull = np.asarray(current_competing_hull_energies, dtype=np.float64).reshape(-1)
    size = len(mean)
    if (
        len(query_compositions) != size
        or len(query_ids) != size
        or len(source) != size
        or len(item_costs) != size
        or len(current_hull) != size
    ):
        raise ValueError("dual-horizon source-rollout inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("dual-horizon query costs must be finite and positive")
    if not np.allclose(item_costs, item_costs[0], atol=1e-12):
        raise ValueError("dual-horizon source-rollout requires equal query costs")
    if not math.isfinite(remaining_budget) or remaining_budget < item_costs[0]:
        raise ValueError("remaining protocol budget cannot pay for a rollout query")
    if sobol_scramble_count < 2 or posterior_sample_count % sobol_scramble_count:
        raise ValueError("posterior samples must divide into independent Sobol scrambles")
    block_size = posterior_sample_count // sobol_scramble_count
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("each dual-horizon Sobol block must have power-of-two size")
    if not 0.5 < integration_confidence < 1.0:
        raise ValueError("dual-horizon integration confidence must lie in (0.5, 1)")
    horizon = min(size, int(math.floor((remaining_budget + 1e-12) / item_costs[0])))

    samples = np.concatenate(
        tuple(
            _sample_gaussian(
                mean,
                covariance,
                sample_count=block_size,
                seed=seed + 104729 * block_index,
            )
            for block_index in range(sobol_scramble_count)
        ),
        axis=0,
    )
    terminal_labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        fixed_template=fixed_template,
    )
    causal_rewards = np.empty((posterior_sample_count, size), dtype=np.float64)
    terminal_rewards = _source_rollout_rewards(
        sampled_query_energies=samples,
        final_hull_membership=terminal_labels,
        query_compositions=query_compositions,
        query_source_energies=source,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        horizon=horizon,
        causal_rewards_output=causal_rewards,
    )
    terminal_blocks = terminal_rewards.reshape(sobol_scramble_count, block_size, size).mean(axis=1)
    causal_blocks = causal_rewards.reshape(sobol_scramble_count, block_size, size).mean(axis=1)
    terminal_scores = terminal_blocks.mean(axis=0)
    causal_scores = causal_blocks.mean(axis=0)
    source_action = int(
        source_margin_action_indices(
            source_energies=source,
            competing_hull_energies=current_hull,
            query_ids=query_ids,
        )[0]
    )
    comparison_count = 2 * max(size - 1, 1)
    terminal_differences = terminal_blocks - terminal_blocks[:, [source_action]]
    causal_differences = causal_blocks - causal_blocks[:, [source_action]]
    terminal_advantages = terminal_differences.mean(axis=0)
    causal_advantages = causal_differences.mean(axis=0)
    terminal_bounds = _simultaneous_paired_lower_bounds(
        terminal_differences,
        confidence=integration_confidence,
        comparison_count=comparison_count,
    )
    causal_bounds = _simultaneous_paired_lower_bounds(
        causal_differences,
        confidence=integration_confidence,
        comparison_count=comparison_count,
    )
    terminal_bounds[source_action] = 0.0
    causal_bounds[source_action] = 0.0
    feasible = (terminal_bounds > 0.0) & (causal_bounds >= 0.0)
    feasible[source_action] = False
    improving = np.flatnonzero(feasible)
    if len(improving):
        selected_action = min(
            (int(index) for index in improving),
            key=lambda index: (-terminal_scores[index], str(query_ids[index])),
        )
        fallback_reason = None
    else:
        selected_action = source_action
        fallback_reason = (
            "source_is_only_legal_action" if size == 1 else "no_dual_horizon_feasible_deviation"
        )
    return DualHorizonSourceRolloutResult(
        terminal_scores=tuple(float(value) for value in terminal_scores),
        causal_scores=tuple(float(value) for value in causal_scores),
        terminal_block_scores=tuple(
            tuple(float(value) for value in row) for row in terminal_blocks
        ),
        causal_block_scores=tuple(tuple(float(value) for value in row) for row in causal_blocks),
        terminal_paired_advantages=tuple(float(value) for value in terminal_advantages),
        causal_paired_advantages=tuple(float(value) for value in causal_advantages),
        terminal_lower_bounds=tuple(float(value) for value in terminal_bounds),
        causal_lower_bounds=tuple(float(value) for value in causal_bounds),
        terminal_final_stability_probabilities=tuple(
            float(value) for value in terminal_labels.mean(axis=0)
        ),
        source_action_index=source_action,
        selected_action_index=selected_action,
        posterior_sample_count=posterior_sample_count,
        sobol_scramble_count=sobol_scramble_count,
        simultaneous_comparison_count=comparison_count,
        horizon=horizon,
        feasible_mask=tuple(bool(value) for value in feasible),
        fallback_reason=fallback_reason,
    )


def _independent_confirmation_seed(stage_one_seed: int) -> int:
    """Derive a fixed stream disjoint from SARR's sixteen stage-one streams."""

    # Stage one uses ``seed + 104729 * block`` for block 0--15.  This domain
    # separated offset cannot equal any of those seeds and is part of IC-SARR
    # v1's frozen numerical protocol.
    return int(stage_one_seed) + 1_000_000_007


def independent_confirmation_source_rollout(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    stage_one_posterior_sample_count: int = 1024,
    stage_two_posterior_sample_count: int = 8192,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
    sobol_scramble_count: int = 16,
    integration_confidence: float = 0.95,
) -> IndependentConfirmationSourceRolloutResult:
    """Run frozen SARR followed by one independent numerical confirmation.

    The method deliberately does *not* search candidates in stage two.  It
    receives exactly one candidate selected by stage-one point advantage and
    makes exactly one paired source comparison using an independent Sobol
    stream.  In production, callers use the protocol-fixed 1024/8192 sample
    counts; smaller counts are accepted here only for deterministic unit tests.
    """

    stage_one = source_rollout_delta_hull(
        posterior,
        query_compositions=query_compositions,
        query_source_energies=query_source_energies,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        current_competing_hull_energies=current_competing_hull_energies,
        costs=costs,
        remaining_budget=remaining_budget,
        posterior_sample_count=stage_one_posterior_sample_count,
        seed=seed,
        fixed_template=fixed_template,
        sobol_scramble_count=sobol_scramble_count,
        integration_confidence=integration_confidence,
    )
    source_action = stage_one.source_action_index
    if stage_one.selected_action_index != source_action:
        return IndependentConfirmationSourceRolloutResult(
            stage_one=stage_one,
            source_action_index=source_action,
            selected_action_index=stage_one.selected_action_index,
            stage_two_used=False,
            stage_one_seed=seed,
            stage_one_posterior_sample_count=stage_one_posterior_sample_count,
            sobol_scramble_count=sobol_scramble_count,
            fallback_reason="stage_one_accepted",
        )

    advantages = np.asarray(stage_one.paired_advantages_over_source, dtype=np.float64)
    positive = np.flatnonzero(advantages > 0.0)
    positive = positive[positive != source_action]
    if not len(positive):
        return IndependentConfirmationSourceRolloutResult(
            stage_one=stage_one,
            source_action_index=source_action,
            selected_action_index=source_action,
            stage_two_used=False,
            stage_one_seed=seed,
            stage_one_posterior_sample_count=stage_one_posterior_sample_count,
            sobol_scramble_count=sobol_scramble_count,
            fallback_reason="no_positive_stage_one_advantage",
        )
    screened_action = min(
        (int(index) for index in positive),
        key=lambda index: (-advantages[index], str(query_ids[index])),
    )

    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    size = len(posterior.mean)
    if (
        len(query_compositions) != size
        or len(query_ids) != size
        or len(item_costs) != size
        or not np.allclose(item_costs, item_costs[0], atol=1e-12)
    ):
        raise ValueError("IC-SARR source-rollout inputs disagree or have unequal costs")
    if stage_two_posterior_sample_count % sobol_scramble_count:
        raise ValueError("IC-SARR stage-two samples must divide into Sobol blocks")
    block_size = stage_two_posterior_sample_count // sobol_scramble_count
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("each IC-SARR stage-two Sobol block must have power-of-two size")
    stage_two_seed = _independent_confirmation_seed(seed)
    samples = np.concatenate(
        _sample_gaussian_blocks(
            np.asarray(posterior.mean, dtype=np.float64),
            np.asarray(posterior.covariance, dtype=np.float64),
            sample_count=block_size,
            seeds=tuple(
                stage_two_seed + 104729 * block_index
                for block_index in range(sobol_scramble_count)
            ),
        ),
        axis=0,
    )
    runtime_plan = (
        None if fixed_template is None else FixedHullRuntimePlan.from_template(fixed_template)
    )
    labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        fixed_template=fixed_template,
        fixed_runtime_plan=runtime_plan,
    )
    horizon = stage_one.horizon
    rewards = _source_rollout_rewards(
        sampled_query_energies=samples,
        final_hull_membership=labels,
        query_compositions=query_compositions,
        query_source_energies=query_source_energies,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        horizon=horizon,
        first_action_indices=(source_action, screened_action),
    )
    block_rewards = rewards.reshape(sobol_scramble_count, block_size, 2).mean(axis=1)
    block_advantages = block_rewards[:, 1] - block_rewards[:, 0]
    advantage = float(block_advantages.mean())
    lower_bound = float(
        _simultaneous_paired_lower_bounds(
            block_advantages.reshape(-1, 1),
            confidence=integration_confidence,
            comparison_count=1,
        )[0]
    )
    selected_action = screened_action if lower_bound > 0.0 else source_action
    return IndependentConfirmationSourceRolloutResult(
        stage_one=stage_one,
        source_action_index=source_action,
        screened_action_index=screened_action,
        selected_action_index=selected_action,
        stage_two_used=True,
        stage_two_paired_advantage=advantage,
        stage_two_paired_lower_bound=lower_bound,
        stage_two_block_advantages=tuple(float(value) for value in block_advantages),
        stage_one_seed=seed,
        stage_two_seed=stage_two_seed,
        stage_one_posterior_sample_count=stage_one_posterior_sample_count,
        stage_two_posterior_sample_count=stage_two_posterior_sample_count,
        sobol_scramble_count=sobol_scramble_count,
        fallback_reason=(
            None if selected_action != source_action else "stage_two_lower_bound_not_positive"
        ),
    )


def conformal_one_deviation_source_rollout(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    conformal_radius: float,
    deviation_used: bool = False,
    posterior_sample_count: int = 1024,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
    sobol_scramble_count: int = 16,
    integration_confidence: float = 0.95,
) -> ConformalSourceRolloutResult:
    """Allow at most one calibrated deviation from source continuation.

    The rollout estimate is paired against the same source action and uses the
    existing simultaneous RQMC lower-bound radius as ``c_RQMC(x)``.  A
    non-source action is legal only when

    ``estimated_advantage - c_RQMC(x) > conformal_radius``.

    The conformal radius is calibrated on exact-system maxima of rollout
    over-estimation.  This function therefore supplies a safe, source-anchored
    policy rule; it does not turn posterior correctness into a distribution-free
    guarantee.  Once ``deviation_used`` is true, callers must execute the source
    policy directly for all remaining rounds.
    """

    radius = float(conformal_radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("conformal rollout radius must be finite and non-negative")
    rollout = source_rollout_delta_hull(
        posterior,
        query_compositions=query_compositions,
        query_source_energies=query_source_energies,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        current_competing_hull_energies=current_competing_hull_energies,
        costs=costs,
        remaining_budget=remaining_budget,
        posterior_sample_count=posterior_sample_count,
        seed=seed,
        fixed_template=fixed_template,
        sobol_scramble_count=sobol_scramble_count,
        integration_confidence=integration_confidence,
    )
    advantages = np.asarray(rollout.paired_advantages_over_source, dtype=np.float64)
    lower_bounds = np.asarray(rollout.paired_advantage_lower_bounds, dtype=np.float64)
    rqmc_radii = np.maximum(advantages - lower_bounds, 0.0)
    adjusted = advantages - rqmc_radii
    source_index = rollout.source_action_index
    adjusted[source_index] = 0.0
    eligible = np.flatnonzero(adjusted > radius)
    if deviation_used:
        selected = source_index
        selected_deviation = False
        reason = "deviation_already_used"
    elif len(eligible):
        selected = min(
            (int(index) for index in eligible),
            key=lambda index: (-adjusted[index], str(query_ids[index])),
        )
        selected_deviation = selected != source_index
        reason = None
    else:
        selected = source_index
        selected_deviation = False
        reason = "conformal_gate_not_positive"
    return ConformalSourceRolloutResult(
        scores=rollout.scores,
        paired_advantages_over_source=rollout.paired_advantages_over_source,
        rqmc_radii=tuple(float(value) for value in rqmc_radii),
        conformal_adjusted_advantages=tuple(float(value) for value in adjusted),
        source_action_index=source_index,
        selected_action_index=selected,
        deviation_used_before=deviation_used,
        deviation_selected=selected_deviation,
        fallback_reason=reason,
        conformal_radius=radius,
        posterior_sample_count=rollout.posterior_sample_count,
        sobol_scramble_count=rollout.sobol_scramble_count,
        horizon=rollout.horizon,
    )


def protocol_hull_risk_reduction(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    costs: np.ndarray,
    posterior_sample_count: int = 16,
    fantasy_count: int = 3,
    seed: int = 0,
) -> ProtocolHullRiskReductionResult:
    """Reduce Bayes risk of the complete target-protocol hull function.

    The terminal estimator is the posterior mean hull on the fixed union of
    initial, revealed and still-queryable compositions.  Under squared loss,
    its Bayes risk is the mean posterior variance of that random hull.  Each
    action is valued by the expected reduction in this risk per unit cost.
    This continuous objective remains informative when binary hull-membership
    probabilities saturate.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    size = len(mean)
    if len(query_compositions) != size or len(item_costs) != size:
        raise ValueError("protocol hull risk-reduction inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("protocol hull query costs must be finite and positive")
    if posterior_sample_count < 4 or fantasy_count < 1:
        raise ValueError("protocol hull Monte Carlo settings are too small")

    evaluation_compositions = _fixed_evaluation_compositions(
        query_compositions,
        reference_compositions,
    )
    current_samples = _sample_gaussian(
        mean,
        covariance,
        sample_count=posterior_sample_count,
        seed=seed,
    )
    current_hulls = _final_hull_values(
        query_compositions=query_compositions,
        sampled_query_energies=current_samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        evaluation_compositions=evaluation_compositions,
    )
    current_risk = _mean_hull_squared_error_risk(current_hulls)
    nodes, weights = np.polynomial.hermite.hermgauss(fantasy_count)
    weights = weights / math.sqrt(math.pi)
    expected_risks = np.empty(size, dtype=np.float64)
    for query_index in range(size):
        variance = float(covariance[query_index, query_index])
        if variance <= 1e-15:
            expected_risks[query_index] = current_risk
            continue
        cross = covariance[:, query_index]
        conditional_covariance = covariance - np.outer(cross, cross) / variance
        conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
        conditional_covariance[query_index, :] = 0.0
        conditional_covariance[:, query_index] = 0.0
        expected_risk = 0.0
        for fantasy_index, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
            outcome = mean[query_index] + math.sqrt(2.0 * variance) * node
            conditional_mean = mean + cross * ((outcome - mean[query_index]) / variance)
            conditional_samples = _sample_gaussian(
                conditional_mean,
                conditional_covariance,
                sample_count=posterior_sample_count,
                seed=seed + 104729 * (fantasy_index + 1),
            )
            conditional_hulls = _final_hull_values(
                query_compositions=query_compositions,
                sampled_query_energies=conditional_samples,
                reference_compositions=reference_compositions,
                reference_energies=reference_energies,
                evaluation_compositions=evaluation_compositions,
            )
            expected_risk += float(weight) * _mean_hull_squared_error_risk(conditional_hulls)
        expected_risks[query_index] = expected_risk
    reductions = current_risk - expected_risks
    scores = reductions / item_costs
    return ProtocolHullRiskReductionResult(
        scores=tuple(float(value) for value in scores),
        risk_reductions=tuple(float(value) for value in reductions),
        expected_posterior_risks=tuple(float(value) for value in expected_risks),
        current_hull_risk=current_risk,
        evaluation_composition_count=len(evaluation_compositions),
        posterior_sample_count=posterior_sample_count,
        fantasy_count=fantasy_count,
    )


def protocol_hull_knowledge_gradient(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    posterior_sample_count: int = 16,
    fantasy_count: int = 3,
    seed: int = 0,
) -> ProtocolHullKnowledgeGradientResult:
    """Evaluate the exact two-step Bayes objective under the working posterior.

    With unit query costs, the terminal utility is the number of queried phases
    that belong to the final target-protocol hull.  The first term is the
    current query's final-stability probability; the second is the expected
    optimal final-stability probability for one subsequent query.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    size = len(mean)
    if len(query_compositions) != size or len(item_costs) != size:
        raise ValueError("protocol hull knowledge-gradient inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("protocol hull query costs must be finite and positive")
    if not np.allclose(item_costs, item_costs[0], atol=1e-12):
        raise ValueError("two-step protocol hull knowledge gradient requires unit costs")
    if posterior_sample_count < 4 or fantasy_count < 1:
        raise ValueError("protocol hull Monte Carlo settings are too small")
    if not math.isfinite(remaining_budget) or remaining_budget < item_costs[0]:
        raise ValueError("remaining protocol budget cannot pay for a query")

    current_samples = _sample_gaussian(
        mean,
        covariance,
        sample_count=posterior_sample_count,
        seed=seed,
    )
    current_labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=current_samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
    )
    probabilities = current_labels.mean(axis=0)
    posterior_risk = float(np.minimum(probabilities, 1.0 - probabilities).sum())
    horizon = 2 if remaining_budget + 1e-12 >= 2.0 * item_costs[0] and size > 1 else 1
    expected_second = np.zeros(size, dtype=np.float64)
    if horizon == 2:
        nodes, weights = np.polynomial.hermite.hermgauss(fantasy_count)
        weights = weights / math.sqrt(math.pi)
        for query_index in range(size):
            variance = float(covariance[query_index, query_index])
            cross = covariance[:, query_index]
            conditional_covariance = covariance - np.outer(cross, cross) / variance
            conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
            conditional_covariance[query_index, :] = 0.0
            conditional_covariance[:, query_index] = 0.0
            value = 0.0
            for fantasy_index, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
                outcome = mean[query_index] + math.sqrt(2.0 * variance) * node
                conditional_mean = mean + cross * ((outcome - mean[query_index]) / variance)
                conditional_samples = _sample_gaussian(
                    conditional_mean,
                    conditional_covariance,
                    sample_count=posterior_sample_count,
                    seed=seed + 104729 * (fantasy_index + 1),
                )
                conditional_labels = _final_hull_membership(
                    query_compositions=query_compositions,
                    sampled_query_energies=conditional_samples,
                    reference_compositions=reference_compositions,
                    reference_energies=reference_energies,
                )
                next_probabilities = conditional_labels.mean(axis=0)
                next_probabilities[query_index] = -np.inf
                value += float(weight) * float(np.max(next_probabilities))
            expected_second[query_index] = value
    scores = (probabilities + expected_second) / item_costs
    return ProtocolHullKnowledgeGradientResult(
        scores=tuple(float(value) for value in scores),
        final_stability_probabilities=tuple(float(value) for value in probabilities),
        expected_second_step_values=tuple(float(value) for value in expected_second),
        posterior_risk=posterior_risk,
        posterior_sample_count=posterior_sample_count,
        fantasy_count=fantasy_count,
        horizon=horizon,
    )
