"""Audit the greedy weak-coupling certificate on the frozen exact-DP suite.

This is a finite-world mechanism audit.  It reconstructs the exact belief
state from the already-opened synthetic worlds, enumerates the exact dynamic
program and repeated-greedy values, and computes a conservative posterior
drift certificate over every reachable continuation from the initial state.
It does not use MatPES/MAD outcomes and does not change any policy result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"

if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from matmem.random_delayed_label_benchmark import (  # noqa: E402
    RandomDelayedLabelInstance,
    _belief_probability,
    _normalize_belief,
    _stable_indices,
    _update_belief,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _instance(payload: dict[str, Any]) -> RandomDelayedLabelInstance:
    return RandomDelayedLabelInstance(
        instance_id=int(payload["instance_id"]),
        budget=int(payload["budget"]),
        source_signal=float(payload["source_signal"]),
        energy_correlation=float(payload["energy_correlation"]),
        delayed_label_coupling=float(payload["delayed_label_coupling"]),
        posterior_noise=float(payload["posterior_noise"]),
        competing_facet_count=int(payload["competing_facet_count"]),
        source_energies=tuple(float(value) for value in payload["source_energies"]),
        world_energies=tuple(
            tuple(float(value) for value in row.split())
            if isinstance(row, str)
            else tuple(float(value) for value in row)
            for row in payload["world_energies"]
        ),
        world_probabilities=tuple(float(value) for value in payload["world_probabilities"]),
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


def _membership_probabilities(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
    belief: tuple[int, ...],
) -> np.ndarray:
    weights = _normalize_belief(instance, belief)
    return np.asarray(
        [
            sum(weight * (candidate in stable[world]) for world, weight in zip(belief, weights, strict=True))
            for candidate in range(instance.pool_size)
        ],
        dtype=float,
    )


def _branches(
    instance: RandomDelayedLabelInstance,
    belief: tuple[int, ...],
    action: int,
) -> tuple[tuple[int, ...], ...]:
    branches = {
        _update_belief(instance, belief, action, world)
        for world in belief
    }
    return tuple(sorted(branches))


def _hull_height_without_candidate(energies: tuple[float, ...], excluded: int) -> float:
    size = len(energies)
    points = [(0.0, 0.0)]
    points.extend(
        ((index + 1) / (size + 1), value)
        for index, value in enumerate(energies)
        if index != excluded
    )
    points.append((1.0, 0.0))
    ordered = [(float(x), float(y)) for x, y in points]
    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2:
            ax, ay = lower[-2]
            bx, by = lower[-1]
            cx, cy = point
            cross = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if cross > 1e-12:
                break
            lower.pop()
        lower.append(point)
    x_target = (excluded + 1) / (size + 1)
    for (x0, y0), (x1, y1) in zip(lower, lower[1:], strict=True):
        if x0 <= x_target <= x1:
            fraction = (x_target - x0) / (x1 - x0)
            return float(y0 + fraction * (y1 - y0))
    raise RuntimeError("candidate composition was not bracketed by the lower hull")


def _signed_margin(energies: tuple[float, ...], candidate: int) -> float:
    return _hull_height_without_candidate(energies, candidate) - energies[candidate]


def _two_step_headroom(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
) -> dict[str, Any] | None:
    if instance.budget < 2:
        return None
    belief = tuple(range(len(instance.world_energies)))
    probabilities = _membership_probabilities(instance, stable, belief)
    remaining = tuple(range(instance.pool_size))
    information: dict[int, float] = {}
    q_values: dict[int, float] = {}
    total = _belief_probability(instance, belief)
    for action in remaining:
        future = 0.0
        for branch in _branches(instance, belief, action):
            branch_probabilities = _membership_probabilities(instance, stable, branch)
            legal = [candidate for candidate in remaining if candidate != action]
            future += _belief_probability(instance, branch) / total * float(
                np.max(branch_probabilities[legal])
            )
        baseline = float(np.max(np.delete(probabilities, action)))
        information[action] = future - baseline
        q_values[action] = float(probabilities[action] + future)
    order = sorted(remaining, key=lambda candidate: (-probabilities[candidate], candidate))
    greedy_action = order[0]
    direct_headroom = max(0.0, max(q_values.values()) - q_values[greedy_action])
    formula_terms = [
        information[order[1]] - information[order[0]],
        *(
            information[candidate]
            - information[order[0]]
            - (probabilities[order[1]] - probabilities[candidate])
            for candidate in order[2:]
        ),
    ]
    formula_headroom = max(0.0, *formula_terms)
    candidate_terms = [
        {
            "rank": rank,
            "rank_penalty": float(probabilities[order[1]] - probabilities[candidate]),
            "information_gain": float(information[candidate] - information[order[0]]),
            "headroom_term": float(
                information[candidate]
                - information[order[0]]
                - (probabilities[order[1]] - probabilities[candidate])
            ),
        }
        for rank, candidate in enumerate(order[1:], start=2)
    ]
    result = {
        "two_step_greedy_action_rank": 1,
        "two_step_headroom": float(direct_headroom),
        "two_step_headroom_identity": float(formula_headroom),
        "two_step_headroom_identity_error": float(abs(direct_headroom - formula_headroom)),
        "two_step_max_information_value": float(max(information.values())),
        "two_step_top_two_information_difference": float(
            information[order[1]] - information[order[0]]
        ),
        "two_step_rank_penalty_top3": float(
            probabilities[order[1]] - probabilities[order[2]]
            if len(order) >= 3
            else 0.0
        ),
        "two_step_candidate_terms": candidate_terms,
    }
    return result


def _analyze_instance(instance: RandomDelayedLabelInstance) -> dict[str, Any]:
    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    root_belief = tuple(range(len(instance.world_energies)))
    headroom = _two_step_headroom(instance, stable)

    @cache
    def optimal_value(belief: tuple[int, ...], selected: tuple[int, ...]) -> float:
        if len(selected) == instance.budget:
            return _terminal_value(instance, stable, selected, belief)
        total = _belief_probability(instance, belief)
        values: list[float] = []
        for action in range(instance.pool_size):
            if action in selected:
                continue
            branches = _branches(instance, belief, action)
            values.append(
                max(
                    sum(
                        _belief_probability(instance, branch) / total
                        * optimal_value(branch, selected + (action,))
                        for branch in branches
                    ),
                    0.0,
                )
            )
        return float(max(values))

    @cache
    def greedy_value(belief: tuple[int, ...], selected: tuple[int, ...]) -> float:
        if len(selected) == instance.budget:
            return _terminal_value(instance, stable, selected, belief)
        probabilities = _membership_probabilities(instance, stable, belief)
        remaining = tuple(index for index in range(instance.pool_size) if index not in selected)
        action = min(remaining, key=lambda index: (-probabilities[index], index))
        total = _belief_probability(instance, belief)
        return float(
            sum(
                _belief_probability(instance, branch) / total
                * greedy_value(branch, selected + (action,))
                for branch in _branches(instance, belief, action)
            )
        )

    @cache
    def reachable_states(
        belief: tuple[int, ...], selected: tuple[int, ...]
    ) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
        states: set[tuple[tuple[int, ...], tuple[int, ...]]] = {(belief, selected)}
        if len(selected) == instance.budget:
            return tuple(sorted(states))
        for action in range(instance.pool_size):
            if action in selected:
                continue
            for branch in _branches(instance, belief, action):
                states.update(reachable_states(branch, selected + (action,)))
        return tuple(sorted(states))

    root_probabilities = _membership_probabilities(instance, stable, root_belief)
    root_states = reachable_states(root_belief, ())
    root_order = sorted(
        range(instance.pool_size), key=lambda index: (-root_probabilities[index], index)
    )
    rank_stable_all_order = True
    rank_stable_top_action = True
    epsilon = 0.0
    compared_coordinates = 0
    for belief, selected in root_states:
        probabilities = _membership_probabilities(instance, stable, belief)
        remaining = [index for index in range(instance.pool_size) if index not in selected]
        if not remaining:
            continue
        expected_order = [index for index in root_order if index in remaining]
        current_order = sorted(
            remaining, key=lambda index: (-probabilities[index], index)
        )
        rank_stable_all_order &= current_order == expected_order
        rank_stable_top_action &= current_order[0] == expected_order[0]
        epsilon = max(epsilon, max(abs(probabilities[index] - root_probabilities[index]) for index in remaining))
        compared_coordinates += len(remaining)

    margins = np.asarray(
        [
            _signed_margin(instance.world_energies[world], candidate)
            for world in range(len(instance.world_energies))
            for candidate in range(instance.pool_size)
        ],
        dtype=float,
    )
    world_weights = np.asarray(instance.world_probabilities, dtype=float)
    boundary_mass = float(
        max(
            sum(
                world_weights[world]
                for world in range(len(instance.world_energies))
                if abs(_signed_margin(instance.world_energies[world], candidate)) <= 1e-12
            )
            for candidate in range(instance.pool_size)
        )
    )
    boundary_mass_delta_0_02 = float(
        max(
            sum(
                world_weights[world]
                for world in range(len(instance.world_energies))
                if abs(_signed_margin(instance.world_energies[world], candidate)) <= 0.04
            )
            for candidate in range(instance.pool_size)
        )
    )
    gap = float(optimal_value(root_belief, ()) - greedy_value(root_belief, ()))
    bound = float(2 * instance.budget * epsilon)
    result = {
        "instance_id": instance.instance_id,
        "budget": instance.budget,
        "source_signal": instance.source_signal,
        "energy_correlation": instance.energy_correlation,
        "delayed_label_coupling": instance.delayed_label_coupling,
        "competing_facet_count": instance.competing_facet_count,
        "reachable_state_count": len(root_states),
        "compared_coordinates": compared_coordinates,
        "rank_stable_all_order": bool(rank_stable_all_order),
        "rank_stable_top_action": bool(rank_stable_top_action),
        "epsilon_weak_coupling": float(epsilon),
        "certificate_bound_2n_epsilon": bound,
        "exact_dp_minus_greedy": gap,
        "certificate_holds": bool(gap <= bound + 1e-12),
        "certificate_tightness": float(gap / bound) if bound > 1e-12 else 0.0,
        "zero_tolerance_boundary_mass": boundary_mass,
        "boundary_mass_delta_0_02": boundary_mass_delta_0_02,
        "minimum_absolute_hull_margin": float(np.min(np.abs(margins))),
        "median_absolute_hull_margin": float(np.median(np.abs(margins))),
    }
    if headroom is not None:
        result.update(headroom)
    return result


def analyze(input_path: Path, output: Path, limit: int | None) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    raw_instances = payload["instances"]
    if limit is not None:
        raw_instances = raw_instances[:limit]
    rows = [_analyze_instance(_instance(item)) for item in raw_instances]
    gaps = np.asarray([row["exact_dp_minus_greedy"] for row in rows], dtype=float)
    bounds = np.asarray([row["certificate_bound_2n_epsilon"] for row in rows], dtype=float)
    headroom_rows = [row for row in rows if "two_step_headroom" in row]
    headrooms = np.asarray([row["two_step_headroom"] for row in headroom_rows], dtype=float)
    identity_errors = np.asarray(
        [row["two_step_headroom_identity_error"] for row in headroom_rows], dtype=float
    )
    summary = {
        "instance_count": len(rows),
        "certificate_holds_fraction": float(np.mean([row["certificate_holds"] for row in rows])),
        "maximum_bound_violation": float(np.max(gaps - bounds)),
        "median_exact_dp_minus_greedy": float(np.median(gaps)),
        "p90_exact_dp_minus_greedy": float(np.quantile(gaps, 0.90)),
        "median_certificate_bound": float(np.median(bounds)),
        "p90_certificate_bound": float(np.quantile(bounds, 0.90)),
        "certificate_bound_le_0_01_fraction": float(np.mean(bounds <= 0.01)),
        "certificate_bound_le_0_05_fraction": float(np.mean(bounds <= 0.05)),
        "certificate_bound_le_0_10_fraction": float(np.mean(bounds <= 0.10)),
        "median_certificate_tightness_nonzero": float(
            np.median([row["certificate_tightness"] for row in rows if row["certificate_bound_2n_epsilon"] > 1e-12])
        ),
        "zero_tolerance_boundary_mass_nonzero_fraction": float(
            np.mean([row["zero_tolerance_boundary_mass"] > 0.0 for row in rows])
        ),
        "two_step_headroom_instance_count": len(headroom_rows),
        "two_step_headroom_zero_fraction": float(np.mean(headrooms <= 1e-12)),
        "two_step_headroom_median": float(np.median(headrooms)),
        "two_step_headroom_p90": float(np.quantile(headrooms, 0.90)),
        "two_step_headroom_identity_max_abs_error": float(np.max(identity_errors)),
        "positive_exact_dp_minus_greedy_count": int(np.count_nonzero(gaps > 1e-12)),
        "rank_stable_all_order_count": int(
            np.count_nonzero([row["rank_stable_all_order"] for row in rows])
        ),
        "rank_stable_all_order_fraction": float(
            np.mean([row["rank_stable_all_order"] for row in rows])
        ),
        "rank_stable_all_order_zero_gap_fraction": float(
            np.mean(
                [
                    row["exact_dp_minus_greedy"] <= 1e-12
                    for row in rows
                    if row["rank_stable_all_order"]
                ]
            )
            if any(row["rank_stable_all_order"] for row in rows)
            else float("nan")
        ),
        "rank_stable_top_action_fraction": float(
            np.mean([row["rank_stable_top_action"] for row in rows])
        ),
        "positive_h2_count": int(np.count_nonzero(headrooms > 1e-12)),
        "positive_gap_and_positive_h2_count": int(
            np.count_nonzero(
                [
                    row["exact_dp_minus_greedy"] > 1e-12
                    and row["two_step_headroom"] > 1e-12
                    for row in headroom_rows
                ]
            )
        ),
    }
    positive_gap_rows = [
        row for row in headroom_rows if row["exact_dp_minus_greedy"] > 1e-12
    ]
    summary["positive_gap_h2_recall"] = (
        float(np.mean([row["two_step_headroom"] > 1e-12 for row in positive_gap_rows]))
        if positive_gap_rows
        else float("nan")
    )
    result = {
        "schema_version": 1,
        "status": "complete_exact_dp_greedy_certificate_audit",
        "protocol": "frozen random_exact_dp_suite_v3; posterior-only certificate audit",
        "input": {"path": str(input_path), "sha256": _sha256(input_path)},
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    result = analyze(args.input, args.output, args.limit)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
