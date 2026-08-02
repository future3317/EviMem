"""Exact controlled benchmark for delayed, globally coupled discovery labels.

This is deliberately separate from the frozen materials runners.  A query
reveals one discrete target energy, whereas reward is assessed only after the
full finite-pool lower hull is known.  The benchmark enumerates posterior
worlds and therefore distinguishes a finite-horizon source-policy rollout from
both greedy final-membership ranking and the Bayes-optimal DP.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from typing import Literal

Policy = Literal["source_margin", "greedy_final", "gated_source_rollout", "optimal_dp"]


@dataclass(frozen=True)
class ControlledDelayedLabelInstance:
    """A fixed finite pool with four equally likely, observable energy worlds."""

    budget: int = 3
    source_signal: float = 0.7
    coupling: float = 1.0
    pool_size: int = 5

    def __post_init__(self) -> None:
        if self.pool_size != 5:
            raise ValueError("The registered controlled benchmark has five candidates.")
        if not 1 <= self.budget <= self.pool_size:
            raise ValueError("budget must lie in 1..pool_size.")
        if not 0.0 <= self.source_signal <= 1.0:
            raise ValueError("source_signal must lie in [0, 1].")
        if not 0.0 <= self.coupling <= 1.0:
            raise ValueError("coupling must lie in [0, 1].")

    @property
    def worlds(self) -> tuple[tuple[float, ...], ...]:
        # Each world contains a different low-energy competitor pair.  At
        # coupling=0 all worlds coincide (the information null); at coupling=1
        # their lower facets differ, so a queried energy can change later value.
        base = (-0.045, -0.038, -0.034, -0.038, -0.045)
        patterns = (
            (-0.055, -0.035, 0.020, 0.050, 0.055),
            (0.020, -0.060, -0.040, 0.030, 0.055),
            (0.055, 0.030, -0.040, -0.060, 0.020),
            (0.055, 0.050, 0.020, -0.035, -0.055),
        )
        return tuple(
            tuple(round(base[i] + self.coupling * pattern[i], 6) for i in range(self.pool_size))
            for pattern in patterns
        )

    @property
    def source_energies(self) -> tuple[float, ...]:
        mean = tuple(sum(world[i] for world in self.worlds) / len(self.worlds) for i in range(self.pool_size))
        nuisance = (0.030, -0.020, 0.025, -0.020, 0.030)
        return tuple(
            self.source_signal * mean[i] + (1.0 - self.source_signal) * nuisance[i]
            for i in range(self.pool_size)
        )


def _stable_indices(energies: tuple[float, ...]) -> frozenset[int]:
    """Return candidate vertices of the exact one-dimensional lower hull."""

    points = [(0.0, 0.0, -1)] + [
        ((index + 1) / 6.0, energy, index) for index, energy in enumerate(energies)
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


def evaluate_controlled_policy(
    instance: ControlledDelayedLabelInstance, policy: Policy
) -> float:
    """Return exact expected durable discoveries for one declared policy."""

    worlds = instance.worlds
    stable = tuple(_stable_indices(world) for world in worlds)
    source_order = tuple(sorted(range(instance.pool_size), key=lambda i: (instance.source_energies[i], i)))

    def terminal(selected: tuple[int, ...], belief: tuple[int, ...]) -> float:
        return sum(sum(candidate in stable[world] for candidate in selected) for world in belief) / len(belief)

    def update(belief: tuple[int, ...], action: int, world: int) -> tuple[int, ...]:
        observed = worlds[world][action]
        return tuple(candidate for candidate in belief if worlds[candidate][action] == observed)

    def source_completion(selected: tuple[int, ...], action: int) -> tuple[int, ...]:
        picked = selected + (action,)
        for candidate in source_order:
            if len(picked) == instance.budget:
                break
            if candidate not in picked:
                picked += (candidate,)
        return picked

    @cache
    def optimal(belief: tuple[int, ...], selected: tuple[int, ...]) -> float:
        if len(selected) == instance.budget:
            return terminal(selected, belief)
        values = []
        for action in range(instance.pool_size):
            if action in selected:
                continue
            branches = {update(belief, action, world) for world in belief}
            values.append(sum(len(branch) / len(belief) * optimal(branch, selected + (action,)) for branch in branches))
        return max(values)

    def action_for(belief: tuple[int, ...], selected: tuple[int, ...]) -> int:
        remaining = tuple(i for i in range(instance.pool_size) if i not in selected)
        source = next(i for i in source_order if i in remaining)
        if policy == "source_margin":
            return source
        if policy == "greedy_final":
            return min(remaining, key=lambda i: (-sum(i in stable[w] for w in belief), i))
        if policy == "gated_source_rollout":
            values = {i: terminal(source_completion(selected, i), belief) for i in remaining}
            best = min(remaining, key=lambda i: (-values[i], i))
            return best if values[best] > values[source] + 1e-12 else source
        def optimal_action_value(action: int) -> float:
            branches = {update(belief, action, world) for world in belief}
            return sum(
                len(branch) / len(belief) * optimal(branch, selected + (action,))
                for branch in branches
            )

        return min(remaining, key=lambda i: (-optimal_action_value(i), i))

    rewards = []
    for world in range(len(worlds)):
        belief = tuple(range(len(worlds)))
        selected: tuple[int, ...] = ()
        for _ in range(instance.budget):
            action = action_for(belief, selected)
            selected += (action,)
            belief = update(belief, action, world)
        rewards.append(sum(candidate in stable[world] for candidate in selected))
    return sum(rewards) / len(rewards)


def controlled_benchmark_grid() -> tuple[dict[str, float], ...]:
    """Frozen grid for figures: horizon, source signal, and coupling only."""

    rows = []
    for budget in (1, 2, 3):
        for source_signal in (0.0, 0.5, 1.0):
            for coupling in (0.0, 0.5, 1.0):
                instance = ControlledDelayedLabelInstance(budget, source_signal, coupling)
                row: dict[str, float] = {
                    "budget": float(budget),
                    "source_signal": source_signal,
                    "coupling": coupling,
                }
                for policy in ("source_margin", "greedy_final", "gated_source_rollout", "optimal_dp"):
                    row[policy] = evaluate_controlled_policy(instance, policy)  # type: ignore[arg-type]
                rows.append(row)
    return tuple(rows)
