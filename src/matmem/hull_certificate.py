"""Robust convex-hull decisions from simultaneous energy intervals.

The certificate in this module is deliberately distribution agnostic.  It
accepts simultaneous per-phase energy intervals and solves the lower and upper
competing hulls directly.  A conformal radius may be used to construct those
intervals, but it is never interpreted as a Gaussian standard deviation.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from scipy.optimize import linprog


class PhaseEnergyInterval(BaseModel):
    """One phase composition with a simultaneous per-atom energy interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase_id: str
    element_fractions: dict[str, float]
    lower_energy_ev_per_atom: float
    upper_energy_ev_per_atom: float

    @field_validator("phase_id")
    @classmethod
    def _identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("phase interval requires a non-empty identity")
        return value

    @field_validator("element_fractions")
    @classmethod
    def _composition(cls, values: dict[str, float]) -> dict[str, float]:
        normalized = {key.strip(): float(value) for key, value in values.items()}
        if (
            not normalized
            or any(
                not key or not math.isfinite(value) or value < 0
                for key, value in normalized.items()
            )
        ):
            raise ValueError("phase fractions must be finite and non-negative")
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("phase fractions must have positive mass")
        return dict(sorted((key, value / total) for key, value in normalized.items()))

    @field_validator("lower_energy_ev_per_atom", "upper_energy_ev_per_atom")
    @classmethod
    def _finite_energy(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("phase interval energies must be finite")
        return value

    @model_validator(mode="after")
    def _ordered_interval(self) -> PhaseEnergyInterval:
        if self.lower_energy_ev_per_atom > self.upper_energy_ev_per_atom:
            raise ValueError("phase interval lower endpoint exceeds upper endpoint")
        return self


class RobustHullDecisionKind(StrEnum):
    STABLE = "stable"
    UNSTABLE = "unstable"
    ABSTAIN = "abstain"


class RobustHullDecision(BaseModel):
    """Auditable stable/unstable/abstain result for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str
    kind: RobustHullDecisionKind
    lower_competing_hull_ev_per_atom: float | None
    upper_competing_hull_ev_per_atom: float | None
    stable_worst_case_margin_ev_per_atom: float | None
    unstable_worst_case_margin_ev_per_atom: float | None
    tolerance_ev_per_atom: float
    reason: str


class ActionValueInterval(BaseModel):
    """Certified interval for a value-maximizing legal action."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str
    lower_value: float
    upper_value: float

    @field_validator("action_id")
    @classmethod
    def _action_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("action interval requires a non-empty identity")
        return value

    @field_validator("lower_value", "upper_value")
    @classmethod
    def _finite_value(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("action values must be finite")
        return value

    @model_validator(mode="after")
    def _ordered_values(self) -> ActionValueInterval:
        if self.lower_value > self.upper_value:
            raise ValueError("action interval lower endpoint exceeds upper endpoint")
        return self


class CertifiedActionSet(BaseModel):
    """Actions guaranteed or still possibly epsilon-optimal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epsilon: float
    guaranteed_epsilon_optimal_action_ids: tuple[str, ...]
    possible_epsilon_optimal_action_ids: tuple[str, ...]


class ClusteredConformalCalibration(BaseModel):
    """Finite-sample quantile over exchangeable cluster-level scores."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float
    cluster_count: int
    order_statistic_one_based: int
    radius: float


class RobustHullDecisionCertifier:
    """Solve candidate-specific lower and upper competing convex hulls.

    The candidate is never inserted into its own competing hull.  This avoids
    the same self-removal artifact that invalidated an earlier information-gain
    objective.
    """

    def __init__(
        self,
        *,
        stability_tolerance_ev_per_atom: float = 0.0,
        feasibility_tolerance: float = 1e-9,
    ) -> None:
        if (
            not math.isfinite(stability_tolerance_ev_per_atom)
            or stability_tolerance_ev_per_atom < 0
        ):
            raise ValueError("stability tolerance must be finite and non-negative")
        if not math.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0:
            raise ValueError("feasibility tolerance must be finite and positive")
        self.stability_tolerance_ev_per_atom = stability_tolerance_ev_per_atom
        self.feasibility_tolerance = feasibility_tolerance

    @staticmethod
    def _elements(
        target: PhaseEnergyInterval, phases: tuple[PhaseEnergyInterval, ...]
    ) -> tuple[str, ...]:
        return tuple(
            sorted(
                set(target.element_fractions).union(
                    *(set(phase.element_fractions) for phase in phases)
                )
            )
        )

    def competing_hull_energy(
        self,
        target: PhaseEnergyInterval,
        competing_phases: tuple[PhaseEnergyInterval, ...],
        *,
        endpoint: Literal["lower", "upper"],
    ) -> float | None:
        """Return the feasible competing hull energy or ``None`` if infeasible."""

        phases = tuple(
            phase for phase in competing_phases if phase.phase_id != target.phase_id
        )
        if not phases:
            return None
        ids = tuple(phase.phase_id for phase in phases)
        if len(set(ids)) != len(ids):
            raise ValueError("competing phase IDs must be unique")
        elements = self._elements(target, phases)
        target_vector = np.asarray(
            [target.element_fractions.get(element, 0.0) for element in elements],
            dtype=np.float64,
        )
        phase_matrix = np.asarray(
            [
                [phase.element_fractions.get(element, 0.0) for phase in phases]
                for element in elements[:-1]
            ],
            dtype=np.float64,
        )
        a_eq = np.vstack((np.ones(len(phases), dtype=np.float64), phase_matrix))
        b_eq = np.concatenate(([1.0], target_vector[:-1]))
        objective = np.asarray(
            [
                (
                    phase.lower_energy_ev_per_atom
                    if endpoint == "lower"
                    else phase.upper_energy_ev_per_atom
                )
                for phase in phases
            ],
            dtype=np.float64,
        )
        result = linprog(
            objective,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=(0.0, None),
            method="highs",
        )
        if result.status == 2:
            return None
        if not result.success or result.fun is None:
            raise RuntimeError(f"robust hull LP failed: {result.message}")
        if np.max(np.abs(a_eq @ result.x - b_eq)) > self.feasibility_tolerance:
            raise RuntimeError("robust hull LP returned an infeasible mixture")
        return float(result.fun)

    def certify(
        self,
        candidate: PhaseEnergyInterval,
        competing_phases: tuple[PhaseEnergyInterval, ...],
    ) -> RobustHullDecision:
        lower_hull = self.competing_hull_energy(
            candidate, competing_phases, endpoint="lower"
        )
        upper_hull = self.competing_hull_energy(
            candidate, competing_phases, endpoint="upper"
        )
        if lower_hull is None or upper_hull is None:
            return RobustHullDecision(
                candidate_id=candidate.phase_id,
                kind=RobustHullDecisionKind.ABSTAIN,
                lower_competing_hull_ev_per_atom=lower_hull,
                upper_competing_hull_ev_per_atom=upper_hull,
                stable_worst_case_margin_ev_per_atom=None,
                unstable_worst_case_margin_ev_per_atom=None,
                tolerance_ev_per_atom=self.stability_tolerance_ev_per_atom,
                reason="competing_hull_infeasible",
            )
        stable_margin = candidate.upper_energy_ev_per_atom - lower_hull
        unstable_margin = candidate.lower_energy_ev_per_atom - upper_hull
        tolerance = self.stability_tolerance_ev_per_atom
        if stable_margin <= tolerance:
            kind = RobustHullDecisionKind.STABLE
            reason = "upper_candidate_below_lower_competing_hull"
        elif unstable_margin > tolerance:
            kind = RobustHullDecisionKind.UNSTABLE
            reason = "lower_candidate_above_upper_competing_hull"
        else:
            kind = RobustHullDecisionKind.ABSTAIN
            reason = "intervals_do_not_preserve_hull_decision"
        return RobustHullDecision(
            candidate_id=candidate.phase_id,
            kind=kind,
            lower_competing_hull_ev_per_atom=lower_hull,
            upper_competing_hull_ev_per_atom=upper_hull,
            stable_worst_case_margin_ev_per_atom=stable_margin,
            unstable_worst_case_margin_ev_per_atom=unstable_margin,
            tolerance_ev_per_atom=tolerance,
            reason=reason,
        )


def certify_epsilon_optimal_actions(
    intervals: tuple[ActionValueInterval, ...], *, epsilon: float
) -> CertifiedActionSet:
    """Certify epsilon-optimal actions under simultaneous value intervals."""

    if not intervals:
        raise ValueError("action certification requires at least one legal action")
    if not math.isfinite(epsilon) or epsilon < 0:
        raise ValueError("epsilon must be finite and non-negative")
    ids = tuple(item.action_id for item in intervals)
    if len(set(ids)) != len(ids):
        raise ValueError("action IDs must be unique")
    best_upper = max(item.upper_value for item in intervals)
    best_lower = max(item.lower_value for item in intervals)
    guaranteed = tuple(
        sorted(
            item.action_id
            for item in intervals
            if item.lower_value + 1e-12 >= best_upper - epsilon
        )
    )
    possible = tuple(
        sorted(
            item.action_id
            for item in intervals
            if item.upper_value + 1e-12 >= best_lower - epsilon
        )
    )
    return CertifiedActionSet(
        epsilon=epsilon,
        guaranteed_epsilon_optimal_action_ids=guaranteed,
        possible_epsilon_optimal_action_ids=possible,
    )


def clustered_conformal_quantile(
    cluster_scores: tuple[float, ...], *, alpha: float
) -> ClusteredConformalCalibration:
    """Return the split-conformal cluster quantile or fail if it is infinite."""

    if not 0 < alpha < 1:
        raise ValueError("clustered conformal alpha must be in (0, 1)")
    if not cluster_scores or any(
        not math.isfinite(score) or score < 0 for score in cluster_scores
    ):
        raise ValueError("clustered conformal scores must be finite and non-negative")
    ordered = sorted(cluster_scores)
    order = math.ceil((len(ordered) + 1) * (1 - alpha))
    if order > len(ordered):
        raise ValueError("too few clusters for a finite conformal quantile")
    return ClusteredConformalCalibration(
        alpha=alpha,
        cluster_count=len(ordered),
        order_statistic_one_based=order,
        radius=ordered[order - 1],
    )
