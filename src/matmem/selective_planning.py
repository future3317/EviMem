"""Small, auditable utilities for selective delayed-label planning.

The module contains only decision algebra and a deliberately simple gate.  It
does not fit a posterior, access an oracle, or select a calibration threshold.
Those responsibilities stay in the registered runner/evaluator so that a
future selective-policy experiment can keep policy-facing and evaluator-only
quantities separate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateTrace:
    """Policy-measurable trace for one delayed-label planning state.

    The fields contain only posterior predictions and revealed-observation
    branch information.  Ground-truth labels and realized returns belong in
    :class:`EvaluatorTrace` and must never be passed to a policy gate.
    """

    system_id: str
    fold_id: str
    round_id: int
    remaining_budget: int
    candidate_ids: tuple[str, ...]
    p_final: tuple[float, ...]
    greedy_order: tuple[str, ...]
    rank_gaps: tuple[float, ...]
    fantasy_candidate: str | None
    fantasy_energy: float | None
    conditional_p_final: tuple[tuple[float, ...], ...]
    signed_delta_p: tuple[tuple[float, ...], ...]
    information_value: tuple[float, ...]
    q_two_step: tuple[float, ...]
    q_rollout: tuple[float, ...]
    q_standard_error: tuple[float, ...]
    greedy_action: str
    planner_action: str
    posterior_stream_id: str
    inner_stream_id: str

    def __post_init__(self) -> None:
        size = len(self.candidate_ids)
        if size == 0:
            raise ValueError("state traces require at least one candidate")
        if len(set(self.candidate_ids)) != size:
            raise ValueError("state-trace candidate IDs must be unique")
        if len(self.p_final) != size or len(self.information_value) != size:
            raise ValueError("state-trace candidate vectors are misaligned")
        if len(self.q_two_step) != size or len(self.q_rollout) != size:
            raise ValueError("state-trace Q vectors are misaligned")
        if len(self.q_standard_error) != size:
            raise ValueError("state-trace standard errors are misaligned")
        if len(self.greedy_order) != size:
            raise ValueError("state-trace greedy order is incomplete")
        if len(self.rank_gaps) != max(0, size - 1):
            raise ValueError("state-trace rank gaps are misaligned")
        if len(self.conditional_p_final) != size or len(self.signed_delta_p) != size:
            raise ValueError("state-trace conditional matrices are misaligned")
        if any(len(row) != size for row in (*self.conditional_p_final, *self.signed_delta_p)):
            raise ValueError("state-trace conditional rows are misaligned")
        if self.remaining_budget < 0 or self.round_id < 0:
            raise ValueError("state-trace time fields must be nonnegative")


@dataclass(frozen=True)
class EvaluatorTrace:
    """Ground-truth companion trace kept outside policy-facing state."""

    realized_final_labels: tuple[bool, ...]
    realized_greedy_return: float
    realized_planner_return: float


@dataclass(frozen=True)
class PlanningValueDecomposition:
    """Model, solver, and distribution-shift terms for a policy contrast."""

    structural_headroom: float
    solver_regret: float
    differential_model_error: float
    planning_gain: float
    cost_adjusted_gain: float


@dataclass(frozen=True)
class SelectiveGateDecision:
    """A transparent Delta-Hull-relative planning-gate decision."""

    action_index: int
    greedy_action_index: int
    gate_used: bool
    robust_gain: float
    headroom_mean: float
    headroom_standard_error: float
    model_penalty: float
    cost_penalty: float
    fallback_reason: str | None


def stable_argmax(values: Sequence[float], ids: Sequence[str] | None = None) -> int:
    """Return a deterministic maximum using an immutable-ID tie break."""

    if len(values) == 0:
        raise ValueError("cannot select from an empty value vector")
    if ids is None:
        ids = tuple(str(index) for index in range(len(values)))
    if len(ids) != len(values):
        raise ValueError("value and ID vectors must have the same length")
    return min(range(len(values)), key=lambda index: (-float(values[index]), str(ids[index])))


def top_two_exchange_gap(
    probabilities: Sequence[float],
    information_values: Sequence[float],
    first_action: int,
    second_action: int,
) -> float:
    """Return the two-step ``Q(second)-Q(first)`` exchange gap.

    The helper is intentionally restricted to the current top-two actions.
    Their direct reward terms are the same set of two probabilities regardless
    of order, so the exchange gap is exactly ``I(second)-I(first)``.
    """

    p = np.asarray(probabilities, dtype=float)
    info = np.asarray(information_values, dtype=float)
    if p.ndim != 1 or info.shape != p.shape:
        raise ValueError("probabilities and information values must be 1-D and aligned")
    if not 0 <= first_action < len(p) or not 0 <= second_action < len(p):
        raise IndexError("action index is outside the candidate vector")
    if first_action == second_action:
        raise ValueError("top-two actions must be distinct")
    top_two = tuple(np.argsort(-p, kind="stable")[:2])
    if {first_action, second_action} != set(top_two):
        raise ValueError("actions must be the current top two candidates")
    return float(info[second_action] - info[first_action])


def planning_value_decomposition(
    *,
    true_planner_value: float,
    true_greedy_value: float,
    posterior_optimal_value: float,
    posterior_greedy_value: float,
    posterior_planner_value: float,
    planner_cost: float = 0.0,
    greedy_cost: float = 0.0,
    cost_weight: float = 0.0,
) -> PlanningValueDecomposition:
    """Evaluate the exact three-term planning-value identity.

    The first three returned quantities satisfy
    ``true_planner - true_greedy = headroom - regret + model_error``.
    ``cost_adjusted_gain`` subtracts only the declared incremental planning
    cost; it does not treat wall time as a target-query cost.
    """

    headroom = float(posterior_optimal_value - posterior_greedy_value)
    solver_regret = float(posterior_optimal_value - posterior_planner_value)
    model_error = float(
        (true_planner_value - posterior_planner_value)
        - (true_greedy_value - posterior_greedy_value)
    )
    gain = float(headroom - solver_regret + model_error)
    cost_adjusted = float(gain - cost_weight * (planner_cost - greedy_cost))
    return PlanningValueDecomposition(
        structural_headroom=headroom,
        solver_regret=solver_regret,
        differential_model_error=model_error,
        planning_gain=gain,
        cost_adjusted_gain=cost_adjusted,
    )


def selective_gate(
    *,
    greedy_action_index: int,
    rollout_action_index: int,
    headroom_mean: float,
    headroom_standard_error: float,
    model_penalty: float,
    cost_penalty: float,
    critical_value: float = 1.96,
) -> SelectiveGateDecision:
    """Choose rollout only when its conservative net headroom is positive.

    ``model_penalty`` and ``cost_penalty`` must be fixed or nested-calibrated
    before the evaluated system is opened.  The function intentionally does
    not estimate either quantity and never reads realized outcomes.
    """

    values = (
        headroom_mean,
        headroom_standard_error,
        model_penalty,
        cost_penalty,
        critical_value,
    )
    if any(not np.isfinite(value) for value in values) or any(
        value < 0.0 for value in (headroom_standard_error, model_penalty, cost_penalty)
    ):
        raise ValueError("selective-gate inputs must be finite and nonnegative where required")
    robust_gain = float(
        headroom_mean - critical_value * headroom_standard_error - model_penalty - cost_penalty
    )
    if robust_gain > 0.0:
        return SelectiveGateDecision(
            action_index=int(rollout_action_index),
            greedy_action_index=int(greedy_action_index),
            gate_used=True,
            robust_gain=robust_gain,
            headroom_mean=float(headroom_mean),
            headroom_standard_error=float(headroom_standard_error),
            model_penalty=float(model_penalty),
            cost_penalty=float(cost_penalty),
            fallback_reason=None,
        )
    return SelectiveGateDecision(
        action_index=int(greedy_action_index),
        greedy_action_index=int(greedy_action_index),
        gate_used=False,
        robust_gain=robust_gain,
        headroom_mean=float(headroom_mean),
        headroom_standard_error=float(headroom_standard_error),
        model_penalty=float(model_penalty),
        cost_penalty=float(cost_penalty),
        fallback_reason="nonpositive_robust_headroom",
    )
