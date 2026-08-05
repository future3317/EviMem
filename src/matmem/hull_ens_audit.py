"""Exact and sampled Hull-ENS audits on the finite-world benchmark.

The audit is deliberately separate from the materials runner.  It makes the
measurement boundary explicit: a continuation may depend on the observed
energy and the posterior branch, but never on unrevealed coordinates of the
world.  The exact evaluator is useful for distinguishing batch approximation
regret from Monte-Carlo regret without opening a materials vault.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .random_delayed_label_benchmark import (
    RandomDelayedLabelInstance,
    _belief_probability,
    _normalize_belief,
    _stable_indices,
    _update_belief,
)


@dataclass(frozen=True)
class HullENSAuditResult:
    """First-action values and the selected action for one finite world."""

    scores: tuple[float, ...]
    selected_action: int
    posterior_sample_count: int | None
    inner_sample_count: int | None
    independent_inner_stream: bool


def _branch_for_observation(
    instance: RandomDelayedLabelInstance,
    belief: tuple[int, ...],
    action: int,
    observed_value: float,
) -> tuple[int, ...]:
    """Return the legal posterior branch for one observable energy."""

    return tuple(
        world
        for world in belief
        if abs(instance.world_energies[world][action] - observed_value) <= 1e-12
    )


def _membership_probabilities(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
    belief: tuple[int, ...],
) -> np.ndarray:
    weights = np.asarray(_normalize_belief(instance, belief), dtype=float)
    return np.asarray(
        [
            sum(weight * (candidate in stable[world]) for world, weight in zip(belief, weights))
            for candidate in range(instance.pool_size)
        ],
        dtype=float,
    )


def _stable_argmax(values: Iterable[float]) -> int:
    values = tuple(float(value) for value in values)
    return min(range(len(values)), key=lambda index: (-values[index], index))


def exact_hull_ens_values(
    instance: RandomDelayedLabelInstance,
    *,
    selected: tuple[int, ...] = (),
    belief: tuple[int, ...] | None = None,
    remaining_budget: int | None = None,
) -> tuple[float, ...]:
    """Compute the exact one-fantasy batch Hull-ENS values.

    The future term is the sum of the largest conditional membership
    probabilities among the remaining candidates.  This is exact for the
    registered two-step objective and is the declared batch approximation for
    larger horizons.
    """

    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    current_belief = (
        tuple(range(len(instance.world_energies))) if belief is None else tuple(belief)
    )
    if not current_belief:
        raise ValueError("the posterior belief cannot be empty")
    available = tuple(index for index in range(instance.pool_size) if index not in selected)
    horizon = instance.budget - len(selected) if remaining_budget is None else int(remaining_budget)
    if horizon < 1 or horizon > len(available):
        raise ValueError("remaining budget is outside the legal action set")
    probabilities = _membership_probabilities(instance, stable, current_belief)
    scores = np.full(instance.pool_size, -np.inf, dtype=float)
    for action in available:
        branch_values: list[tuple[float, float]] = []
        observations = sorted({instance.world_energies[world][action] for world in current_belief})
        for observed in observations:
            branch = _branch_for_observation(instance, current_belief, action, observed)
            branch_probability = _belief_probability(instance, branch) / _belief_probability(
                instance, current_belief
            )
            conditional = _membership_probabilities(instance, stable, branch)
            remaining = tuple(index for index in available if index != action)
            future_count = min(horizon - 1, len(remaining))
            future = 0.0 if future_count == 0 else float(
                np.sort(conditional[np.asarray(remaining)])[-future_count:].sum()
            )
            branch_values.append((branch_probability, future))
        scores[action] = probabilities[action] + sum(
            branch_probability * future for branch_probability, future in branch_values
        )
    return tuple(float(value) for value in scores)


def exact_hull_ens(
    instance: RandomDelayedLabelInstance,
    *,
    selected: tuple[int, ...] = (),
    belief: tuple[int, ...] | None = None,
    remaining_budget: int | None = None,
) -> HullENSAuditResult:
    """Return exact Hull-ENS values and its deterministic first action."""

    scores = exact_hull_ens_values(
        instance,
        selected=selected,
        belief=belief,
        remaining_budget=remaining_budget,
    )
    available = tuple(index for index in range(instance.pool_size) if index not in selected)
    action = min(available, key=lambda index: (-scores[index], index))
    return HullENSAuditResult(
        scores=scores,
        selected_action=action,
        posterior_sample_count=None,
        inner_sample_count=None,
        independent_inner_stream=False,
    )


def sampled_hull_ens(
    instance: RandomDelayedLabelInstance,
    *,
    posterior_sample_count: int = 128,
    inner_sample_count: int = 8,
    seed: int = 0,
    selected: tuple[int, ...] = (),
    belief: tuple[int, ...] | None = None,
    remaining_budget: int | None = None,
    independent_inner_stream: bool = True,
) -> HullENSAuditResult:
    """Estimate Hull-ENS with explicit outer/inner sampling.

    The independent version uses a separate RNG for conditional branches.
    The non-independent version reuses the outer draws within each observed
    branch, making the selection/evaluation dependence visible to the audit.
    """

    if posterior_sample_count < 2 or inner_sample_count < 1:
        raise ValueError("sample counts are too small")
    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    current_belief = (
        tuple(range(len(instance.world_energies))) if belief is None else tuple(belief)
    )
    available = tuple(index for index in range(instance.pool_size) if index not in selected)
    horizon = instance.budget - len(selected) if remaining_budget is None else int(remaining_budget)
    if horizon < 1 or horizon > len(available):
        raise ValueError("remaining budget is outside the legal action set")
    weights = np.asarray(_normalize_belief(instance, current_belief), dtype=float)
    outer_rng = np.random.default_rng(seed)
    outer_worlds = outer_rng.choice(
        np.asarray(current_belief, dtype=int), size=posterior_sample_count, p=weights
    )
    inner_rng = np.random.default_rng(seed + 32452843)
    scores = np.full(instance.pool_size, -np.inf, dtype=float)
    for action in available:
        immediate = np.asarray(
            [action in stable[int(world)] for world in outer_worlds], dtype=float
        ).mean()
        future_values: list[float] = []
        for outer_world in outer_worlds:
            observed = instance.world_energies[int(outer_world)][action]
            branch = _branch_for_observation(instance, current_belief, action, observed)
            branch_weights = np.asarray(_normalize_belief(instance, branch), dtype=float)
            if independent_inner_stream:
                inner_worlds = inner_rng.choice(
                    np.asarray(branch, dtype=int), size=inner_sample_count, p=branch_weights
                )
            else:
                branch_mask = np.asarray(
                    [
                        abs(instance.world_energies[int(world)][action] - observed) <= 1e-12
                        for world in outer_worlds
                    ],
                    dtype=bool,
                )
                branch_outer = outer_worlds[branch_mask]
                inner_worlds = (
                    branch_outer
                    if len(branch_outer)
                    else np.asarray((int(outer_world),), dtype=int)
                )
            conditional = np.asarray(
                [
                    np.mean([candidate in stable[int(world)] for world in inner_worlds])
                    for candidate in available
                    if candidate != action
                ],
                dtype=float,
            )
            future_count = min(horizon - 1, len(conditional))
            future_values.append(
                0.0
                if future_count == 0
                else float(np.sort(conditional)[-future_count:].sum())
            )
        scores[action] = float(immediate + np.mean(future_values))
    action = min(available, key=lambda index: (-scores[index], index))
    return HullENSAuditResult(
        scores=tuple(float(value) for value in scores),
        selected_action=action,
        posterior_sample_count=posterior_sample_count,
        inner_sample_count=inner_sample_count,
        independent_inner_stream=independent_inner_stream,
    )


def exact_branch_value(
    instance: RandomDelayedLabelInstance,
    *,
    action: int,
    observed_world: int,
    selected: tuple[int, ...] = (),
    remaining_budget: int | None = None,
) -> tuple[int, ...]:
    """Return the exact continuation ranking for one observable branch."""

    belief = tuple(range(len(instance.world_energies)))
    observed = instance.world_energies[observed_world][action]
    branch = _branch_for_observation(instance, belief, action, observed)
    continuation_budget = (
        instance.budget - len(selected) - 1
        if remaining_budget is None
        else int(remaining_budget)
    )
    if continuation_budget <= 0:
        return ()
    return exact_hull_ens_values(
        instance,
        selected=selected + (action,),
        belief=branch,
        remaining_budget=continuation_budget,
    )


def evaluate_hull_ens_policy(
    instance: RandomDelayedLabelInstance,
    *,
    mode: str = "exact",
    posterior_sample_count: int = 128,
    inner_sample_count: int = 8,
    seed: int = 0,
    independent_inner_stream: bool = True,
) -> float:
    """Evaluate repeated Hull-ENS on the finite-world benchmark.

    The latent world is used only by this evaluator to average realized
    terminal labels. Every action is selected from the current belief and its
    observable energy branch. ``mode='exact'`` has no Monte Carlo error;
    ``mode='sampled'`` exposes numerical and selected-action regret.
    """

    if mode not in {"exact", "sampled"}:
        raise ValueError("Hull-ENS evaluation mode must be exact or sampled")
    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    initial_belief = tuple(range(len(instance.world_energies)))
    expected_value = 0.0
    for world, probability in enumerate(instance.world_probabilities):
        belief = initial_belief
        selected: tuple[int, ...] = ()
        for round_index in range(instance.budget):
            state_seed = seed + 104729 * (instance.instance_id + 17 * round_index + 1)
            if mode == "exact":
                result = exact_hull_ens(
                    instance,
                    selected=selected,
                    belief=belief,
                    remaining_budget=instance.budget - len(selected),
                )
            else:
                result = sampled_hull_ens(
                    instance,
                    posterior_sample_count=posterior_sample_count,
                    inner_sample_count=inner_sample_count,
                    seed=state_seed,
                    selected=selected,
                    belief=belief,
                    remaining_budget=instance.budget - len(selected),
                    independent_inner_stream=independent_inner_stream,
                )
            action = int(result.selected_action)
            selected += (action,)
            belief = _update_belief(instance, belief, action, world)
        expected_value += float(probability) * sum(action in stable[world] for action in selected)
    return float(expected_value)
