"""Exact small-state diagnostic for query/active-witness coupling.

This is a deliberately abstract binary environment, not a materials result.
Each candidate belongs to a residual cluster with a beta-Bernoulli predictive
model.  Only observations in the certified active working set contribute to
the next predictive state.  An immutable audit archive may exist outside this
decision state, but inactive observations are not free online evidence.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from functools import cache
from typing import Literal

BinaryObservation = tuple[int, int]
PolicyKind = Literal["one_step", "decoupled"]


@dataclass(frozen=True)
class BinaryWitnessState:
    remaining_by_cluster: tuple[int, ...]
    active_observations: tuple[BinaryObservation, ...] = ()


@dataclass(frozen=True)
class BinaryWitnessDP:
    """Finite-horizon beta-Bernoulli environment with exact retention search."""

    active_witness_budget: int
    prior_alpha: float = 1.0
    prior_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.active_witness_budget < 0:
            raise ValueError("active witness budget cannot be negative")
        if min(self.prior_alpha, self.prior_beta) <= 0:
            raise ValueError("beta prior parameters must be positive")

    def predictive_stable(self, state: BinaryWitnessState, cluster: int) -> float:
        successes = sum(
            outcome
            for observed_cluster, outcome in state.active_observations
            if observed_cluster == cluster
        )
        failures = sum(
            1 - outcome
            for observed_cluster, outcome in state.active_observations
            if observed_cluster == cluster
        )
        return (self.prior_alpha + successes) / (
            self.prior_alpha + self.prior_beta + successes + failures
        )

    def retention_options(
        self,
        observations: tuple[BinaryObservation, ...],
    ) -> tuple[tuple[BinaryObservation, ...], ...]:
        capacity = min(self.active_witness_budget, len(observations))
        if capacity == len(observations):
            return (tuple(sorted(observations)),)
        options = {
            tuple(sorted(retained))
            for retained in itertools.combinations(observations, capacity)
        }
        return tuple(sorted(options))

    @staticmethod
    def _entropy(probability: float) -> float:
        if probability in (0.0, 1.0):
            return 0.0
        return -probability * math.log(probability) - (1 - probability) * math.log(
            1 - probability
        )

    def entropy_retention(
        self,
        remaining: tuple[int, ...],
        observations: tuple[BinaryObservation, ...],
    ) -> tuple[BinaryObservation, ...]:
        choices = []
        for retained in self.retention_options(observations):
            retained_state = BinaryWitnessState(remaining, retained)
            uncertainty = sum(
                count * self._entropy(self.predictive_stable(retained_state, cluster))
                for cluster, count in enumerate(remaining)
            )
            choices.append((uncertainty, retained))
        return min(choices, key=lambda item: (item[0], item[1]))[1]

    @staticmethod
    def _after_query(remaining: tuple[int, ...], cluster: int) -> tuple[int, ...]:
        updated = list(remaining)
        updated[cluster] -= 1
        return tuple(updated)

    @cache
    def optimal_value(self, state: BinaryWitnessState, horizon: int) -> float:
        """Bayes-optimal value with outcome-dependent exact retention."""

        if horizon <= 0 or not any(state.remaining_by_cluster):
            return 0.0
        action_values = []
        for cluster, count in enumerate(state.remaining_by_cluster):
            if count == 0:
                continue
            probability = self.predictive_stable(state, cluster)
            remaining = self._after_query(state.remaining_by_cluster, cluster)
            value = 0.0
            for outcome, outcome_probability in (
                (1, probability),
                (0, 1.0 - probability),
            ):
                observations = (*state.active_observations, (cluster, outcome))
                future = max(
                    self.optimal_value(BinaryWitnessState(remaining, retained), horizon - 1)
                    for retained in self.retention_options(observations)
                )
                value += outcome_probability * (outcome + future)
            action_values.append((value, cluster))
        return max(action_values, key=lambda item: (item[0], -item[1]))[0]

    def _decoupled_action(self, state: BinaryWitnessState) -> int:
        candidates = (
            (self.predictive_stable(state, cluster), cluster)
            for cluster, count in enumerate(state.remaining_by_cluster)
            if count > 0
        )
        return max(candidates, key=lambda item: (item[0], -item[1]))[1]

    def _one_step_action(self, state: BinaryWitnessState, horizon: int) -> int:
        values = []
        for cluster, count in enumerate(state.remaining_by_cluster):
            if count == 0:
                continue
            probability = self.predictive_stable(state, cluster)
            remaining = self._after_query(state.remaining_by_cluster, cluster)
            next_reward = 0.0
            if horizon > 1 and any(remaining):
                for outcome, outcome_probability in (
                    (1, probability),
                    (0, 1.0 - probability),
                ):
                    observations = (*state.active_observations, (cluster, outcome))
                    retained = self.entropy_retention(remaining, observations)
                    next_state = BinaryWitnessState(remaining, retained)
                    best_next = max(
                        self.predictive_stable(next_state, next_cluster)
                        for next_cluster, next_count in enumerate(remaining)
                        if next_count > 0
                    )
                    next_reward += outcome_probability * best_next
            values.append((probability + next_reward, cluster))
        return max(values, key=lambda item: (item[0], -item[1]))[1]

    @cache
    def policy_value(
        self,
        state: BinaryWitnessState,
        horizon: int,
        policy: PolicyKind,
    ) -> float:
        if horizon <= 0 or not any(state.remaining_by_cluster):
            return 0.0
        cluster = (
            self._one_step_action(state, horizon)
            if policy == "one_step"
            else self._decoupled_action(state)
        )
        probability = self.predictive_stable(state, cluster)
        remaining = self._after_query(state.remaining_by_cluster, cluster)
        value = 0.0
        for outcome, outcome_probability in (
            (1, probability),
            (0, 1.0 - probability),
        ):
            observations = (*state.active_observations, (cluster, outcome))
            retained = self.entropy_retention(remaining, observations)
            future = self.policy_value(
                BinaryWitnessState(remaining, retained),
                horizon - 1,
                policy,
            )
            value += outcome_probability * (outcome + future)
        return value


def exact_policy_comparison(
    remaining_by_cluster: tuple[int, ...],
    *,
    oracle_budget: int,
    active_witness_budget: int,
) -> dict[str, float]:
    model = BinaryWitnessDP(active_witness_budget=active_witness_budget)
    state = BinaryWitnessState(remaining_by_cluster)
    return {
        "exact_joint": model.optimal_value(state, oracle_budget),
        "one_step_joint": model.policy_value(state, oracle_budget, "one_step"),
        "decoupled": model.policy_value(state, oracle_budget, "decoupled"),
    }
