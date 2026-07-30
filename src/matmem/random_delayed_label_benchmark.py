"""Registered random exact-DP suite for delayed full-pool labels.

This module is deliberately independent of materials tasks and oracle vaults.
It exposes a small finite-world problem: a query reveals one energy, while the
reward is the membership of selected candidates in the lower hull of the
*complete* world.  The pool is kept small enough that the belief-state optimum
can be enumerated exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

import numpy as np

PolicyName = Literal["source_margin", "greedy_final", "source_rollout", "optimal_dp", "ic_sarr"]


@dataclass(frozen=True)
class RandomDelayedLabelInstance:
    """One generated finite-world delayed-adjudication problem."""

    instance_id: int
    budget: int
    source_signal: float
    energy_correlation: float
    delayed_label_coupling: float
    posterior_noise: float
    competing_facet_count: int
    source_energies: tuple[float, ...]
    world_energies: tuple[tuple[float, ...], ...]
    world_probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        size = len(self.source_energies)
        if not 5 <= size <= 10 or not 1 <= self.budget <= 4 or self.budget > size:
            raise ValueError("registered pool and budget bounds are violated")
        if len(self.world_energies) != 4 or len(self.world_probabilities) != 4:
            raise ValueError("the registered suite has four finite worlds")
        if any(len(world) != size for world in self.world_energies):
            raise ValueError("world energies have inconsistent pool size")
        if not np.isclose(sum(self.world_probabilities), 1.0):
            raise ValueError("world probabilities must sum to one")
        if any(value <= 0.0 for value in self.world_probabilities):
            raise ValueError("world probabilities must be positive")

    @property
    def pool_size(self) -> int:
        return len(self.source_energies)


@dataclass(frozen=True)
class RandomBenchmarkRow:
    """Exact expected value and first action for one policy/instance."""

    instance_id: int
    policy: PolicyName
    value: float
    first_action: int


def _stable_indices(energies: tuple[float, ...]) -> frozenset[int]:
    """Return vertices of a one-dimensional lower hull with zero endpoints."""

    size = len(energies)
    points = [(0.0, 0.0, -1)] + [
        ((index + 1) / (size + 1), energy, index) for index, energy in enumerate(energies)
    ] + [(1.0, 0.0, -2)]
    lower: list[tuple[float, float, int]] = []
    for point in points:
        while len(lower) >= 2:
            ax, ay, _ = lower[-2]
            bx, by, _ = lower[-1]
            cx, cy, _ = point
            if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 1e-12:
                break
            lower.pop()
        lower.append(point)
    return frozenset(index for _, _, index in lower if index >= 0)


def generate_random_instances(*, count: int = 1000, seed: int = 20260730) -> tuple[RandomDelayedLabelInstance, ...]:
    """Generate the predeclared independent suite without material inputs."""

    if count < 1:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    instances: list[RandomDelayedLabelInstance] = []
    for instance_id in range(count):
        size = int(rng.integers(5, 11))
        budget = int(rng.integers(1, min(4, size) + 1))
        source_signal = float(rng.choice((0.0, 0.5, 1.0)))
        correlation = float(rng.choice((0.0, 0.5, 1.0)))
        coupling = float(rng.choice((0.0, 0.5, 1.0)))
        posterior_noise = float(rng.choice((0.02, 0.05, 0.10)))
        facets = int(rng.integers(1, 4))

        # A shared latent component gives the requested energy correlation;
        # world/facet terms make omitted competitors alter the terminal hull.
        base = -0.025 - 0.070 * rng.random(size)
        shared = rng.normal(0.0, 0.035, size)
        world_terms = rng.normal(0.0, 0.050, (4, size))
        facet_positions = rng.choice(size, size=facets, replace=False)
        facet_effects = np.zeros((4, size), dtype=float)
        for world in range(4):
            facet_effects[world, facet_positions] = rng.normal(-0.080, 0.025, facets)
            facet_effects[world, facet_positions[world % facets]] -= 0.045 * coupling
        world_energies = base + correlation * shared[None, :] + (1.0 - correlation) * world_terms
        world_energies = world_energies + coupling * facet_effects
        mean = world_energies.mean(axis=0)
        nuisance = rng.normal(-0.025, 0.060, size)
        source = source_signal * mean + (1.0 - source_signal) * nuisance
        source = source + rng.normal(0.0, posterior_noise, size)
        probabilities = rng.dirichlet(np.ones(4, dtype=float))
        instances.append(
            RandomDelayedLabelInstance(
                instance_id=instance_id,
                budget=budget,
                source_signal=source_signal,
                energy_correlation=correlation,
                delayed_label_coupling=coupling,
                posterior_noise=posterior_noise,
                competing_facet_count=facets,
                source_energies=tuple(float(value) for value in source),
                world_energies=tuple(
                    tuple(float(value) for value in row) for row in world_energies
                ),
                world_probabilities=tuple(float(value) for value in probabilities),
            )
        )
    return tuple(instances)


def _belief_probability(instance: RandomDelayedLabelInstance, belief: tuple[int, ...]) -> float:
    return float(sum(instance.world_probabilities[world] for world in belief))


def _normalize_belief(instance: RandomDelayedLabelInstance, belief: tuple[int, ...]) -> tuple[float, ...]:
    total = _belief_probability(instance, belief)
    return tuple(instance.world_probabilities[world] / total for world in belief)


def _update_belief(
    instance: RandomDelayedLabelInstance, belief: tuple[int, ...], action: int, world: int
) -> tuple[int, ...]:
    observed = instance.world_energies[world][action]
    return tuple(
        candidate
        for candidate in belief
        if abs(instance.world_energies[candidate][action] - observed) <= 1e-12
    )


def _terminal_value(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
    selected: tuple[int, ...],
    belief: tuple[int, ...],
) -> float:
    weights = _normalize_belief(instance, belief)
    return float(
        sum(
            weight * sum(action in stable[world] for action in selected)
            for world, weight in zip(belief, weights, strict=True)
        )
    )


def _source_action(instance: RandomDelayedLabelInstance, selected: tuple[int, ...]) -> int:
    return min(
        (index for index in range(instance.pool_size) if index not in selected),
        key=lambda index: (instance.source_energies[index], index),
    )


def _source_completion(
    instance: RandomDelayedLabelInstance, selected: tuple[int, ...], action: int
) -> tuple[int, ...]:
    completion = selected + (action,)
    for candidate in sorted(range(instance.pool_size), key=lambda i: (instance.source_energies[i], i)):
        if len(completion) == instance.budget:
            break
        if candidate not in completion:
            completion += (candidate,)
    return completion


def _rollout_action_values(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
    belief: tuple[int, ...],
    selected: tuple[int, ...],
) -> dict[int, float]:
    return {
        action: _terminal_value(instance, stable, _source_completion(instance, selected, action), belief)
        for action in range(instance.pool_size)
        if action not in selected
    }


def _policy_action(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
    policy: PolicyName,
    belief: tuple[int, ...],
    selected: tuple[int, ...],
    *,
    random_seed: int,
) -> int:
    remaining = tuple(index for index in range(instance.pool_size) if index not in selected)
    source = _source_action(instance, selected)
    if policy == "source_margin":
        return source
    if policy == "greedy_final":
        weights = _normalize_belief(instance, belief)
        return min(
            remaining,
            key=lambda action: (
                -sum(weight * (action in stable[world]) for world, weight in zip(belief, weights, strict=True)),
                action,
            ),
        )
    values = _rollout_action_values(instance, stable, belief, selected)
    if policy == "source_rollout":
        return min(remaining, key=lambda action: (-values[action], action))
    if policy == "ic_sarr":
        return _sampled_ic_sarr_action(instance, stable, belief, selected, random_seed=random_seed)
    raise ValueError("optimal DP is handled by its Bellman recursion")


def _sampled_ic_sarr_action(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
    belief: tuple[int, ...],
    selected: tuple[int, ...],
    *,
    random_seed: int,
) -> int:
    """Frozen-logic two-stage sampled gate for the independent synthetic suite.

    The suite uses 128/512 finite-world samples (16 blocks) rather than the
    material runner's 1024/8192 Gaussian samples; the action, screen, fallback
    and independent confirmation logic are otherwise the same.  This is a
    mechanism comparator, not a material IC-SARR run.
    """

    remaining = tuple(index for index in range(instance.pool_size) if index not in selected)
    source = _source_action(instance, selected)
    weights = np.asarray(_normalize_belief(instance, belief), dtype=float)
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(np.asarray(belief, dtype=int), size=128, p=weights)
    scores = np.empty((16, len(remaining)), dtype=float)
    for block in range(16):
        worlds = samples[block * 8 : (block + 1) * 8]
        for position, action in enumerate(remaining):
            completed = _source_completion(instance, selected, action)
            scores[block, position] = np.mean(
                [sum(candidate in stable[int(world)] for candidate in completed) for world in worlds]
            )
    source_position = remaining.index(source)
    advantages = scores - scores[:, [source_position]]
    means = advantages.mean(axis=0)
    # Conservative normal lower screen mirrors a simultaneous numerical gate.
    standard_error = advantages.std(axis=0, ddof=1) / 4.0
    lower = means - 2.64 * standard_error
    lower[source_position] = 0.0
    improving = [position for position, bound in enumerate(lower) if bound > 0.0]
    if improving:
        selected_position = min(
            improving, key=lambda position: (-scores[:, position].mean(), remaining[position])
        )
        return remaining[selected_position]
    positive = [position for position, value in enumerate(means) if position != source_position and value > 0.0]
    if not positive:
        return source
    screened = min(positive, key=lambda position: (-means[position], remaining[position]))
    stage_two_worlds = rng.choice(np.asarray(belief, dtype=int), size=512, p=weights)
    difference = np.asarray(
        [
            sum(candidate in stable[int(world)] for candidate in _source_completion(instance, selected, remaining[screened]))
            - sum(candidate in stable[int(world)] for candidate in _source_completion(instance, selected, source))
            for world in stage_two_worlds
        ],
        dtype=float,
    )
    # Independent stage-two block lower bound; no second candidate search.
    blocks = difference.reshape(16, 32).mean(axis=1)
    return remaining[screened] if blocks.mean() - 1.96 * blocks.std(ddof=1) / 4.0 > 0.0 else source


def evaluate_random_instance(
    instance: RandomDelayedLabelInstance, policy: PolicyName
) -> RandomBenchmarkRow:
    """Return an exact world-average policy value (except sampled IC-SARR)."""

    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    initial_belief = tuple(range(4))

    @cache
    def optimal_value(belief: tuple[int, ...], selected: tuple[int, ...]) -> float:
        if len(selected) == instance.budget:
            return _terminal_value(instance, stable, selected, belief)
        total = _belief_probability(instance, belief)
        values: dict[int, float] = {}
        for action in range(instance.pool_size):
            if action in selected:
                continue
            branches = {_update_belief(instance, belief, action, world) for world in belief}
            values[action] = sum(
                _belief_probability(instance, branch) / total
                * optimal_value(branch, selected + (action,))
                for branch in branches
            )
        return max(values.values())

    def action(policy_name: PolicyName, belief: tuple[int, ...], selected: tuple[int, ...], round_index: int) -> int:
        if policy_name != "optimal_dp":
            return _policy_action(
                instance,
                stable,
                policy_name,
                belief,
                selected,
                random_seed=20270720 + 104729 * (instance.instance_id + 17 * round_index),
            )
        candidates: dict[int, float] = {}
        total = _belief_probability(instance, belief)
        for candidate in range(instance.pool_size):
            if candidate in selected:
                continue
            branches = {_update_belief(instance, belief, candidate, world) for world in belief}
            candidates[candidate] = sum(
                _belief_probability(instance, branch) / total
                * optimal_value(branch, selected + (candidate,))
                for branch in branches
            )
        return min(candidates, key=lambda candidate: (-candidates[candidate], candidate))

    initial_action = action(policy, initial_belief, (), 0)
    total_value = 0.0
    for world, probability in enumerate(instance.world_probabilities):
        belief = initial_belief
        selected: tuple[int, ...] = ()
        for round_index in range(instance.budget):
            selected_action = action(policy, belief, selected, round_index)
            selected += (selected_action,)
            belief = _update_belief(instance, belief, selected_action, world)
        total_value += probability * sum(candidate in stable[world] for candidate in selected)
    return RandomBenchmarkRow(instance.instance_id, policy, float(total_value), initial_action)


def evaluate_random_suite(*, count: int = 1000, seed: int = 20260730) -> tuple[RandomBenchmarkRow, ...]:
    """Evaluate every registered policy for every generated instance."""

    rows = []
    for instance in generate_random_instances(count=count, seed=seed):
        for policy in ("source_margin", "greedy_final", "source_rollout", "optimal_dp", "ic_sarr"):
            rows.append(evaluate_random_instance(instance, policy))
    return tuple(rows)
