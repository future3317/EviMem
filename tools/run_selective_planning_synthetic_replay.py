"""Run the nested held-out synthetic selective-planning replay.

The gate sees only finite-world posterior quantities. The evaluator-only
realized labels are written in a separate trace field and are never used to
choose an action or penalty on the outer systems.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from matmem.hull_ens_audit import (  # noqa: E402
    _membership_probabilities,
    exact_hull_ens,
    sampled_hull_ens,
)
from matmem.random_delayed_label_benchmark import (  # noqa: E402
    RandomDelayedLabelInstance,
    _normalize_belief,
    _stable_indices,
    _update_belief,
    evaluate_random_instance,
    generate_random_instances,
)
from matmem.selective_planning import selective_gate  # noqa: E402

PENALTY_GRID = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
CRITICAL_VALUE = 1.96
HEADROOM_REPLICATES = 4
POSTERIOR_SAMPLES = 64
INNER_SAMPLES = 4


@dataclass(frozen=True)
class DecisionFeatures:
    """Posterior-only features cached for one legal state."""

    greedy_action: int
    planner_action: int
    headroom_mean: float
    headroom_standard_error: float
    headroom_replicates: tuple[float, ...]
    p_final: tuple[float, ...]
    greedy_order: tuple[int, ...]
    rank_gaps: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_seed(instance_id: int, selected: tuple[int, ...], belief: tuple[int, ...]) -> int:
    payload = json.dumps(
        {"instance": instance_id, "selected": selected, "belief": belief},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _greedy_action(
    instance: RandomDelayedLabelInstance,
    stable: tuple[frozenset[int], ...],
    belief: tuple[int, ...],
    selected: tuple[int, ...],
) -> int:
    weights = _normalize_belief(instance, belief)
    remaining = tuple(index for index in range(instance.pool_size) if index not in selected)
    return min(
        remaining,
        key=lambda action: (
            -sum(
                weight * (action in stable[world])
                for world, weight in zip(belief, weights, strict=True)
            ),
            action,
        ),
    )


def _decision_features(
    instance: RandomDelayedLabelInstance,
    *,
    stable: tuple[frozenset[int], ...],
    belief: tuple[int, ...],
    selected: tuple[int, ...],
) -> DecisionFeatures:
    remaining_budget = instance.budget - len(selected)
    greedy_action = _greedy_action(instance, stable, belief, selected)
    exact = exact_hull_ens(
        instance,
        selected=selected,
        belief=belief,
        remaining_budget=remaining_budget,
    )
    state_seed = _state_seed(instance.instance_id, selected, belief)
    headroom_replicates: list[float] = []
    p_final = tuple(float(value) for value in _membership_probabilities(instance, stable, belief))
    for replicate in range(HEADROOM_REPLICATES):
        sampled = sampled_hull_ens(
            instance,
            posterior_sample_count=POSTERIOR_SAMPLES,
            inner_sample_count=INNER_SAMPLES,
            seed=state_seed + 15485863 * (replicate + 1),
            selected=selected,
            belief=belief,
            remaining_budget=remaining_budget,
            independent_inner_stream=True,
        )
        available = tuple(index for index in range(instance.pool_size) if index not in selected)
        headroom_replicates.append(
            float(max(sampled.scores[index] for index in available) - sampled.scores[greedy_action])
        )
    replicate_array = np.asarray(headroom_replicates, dtype=float)
    return DecisionFeatures(
        greedy_action=greedy_action,
        planner_action=int(exact.selected_action),
        headroom_mean=float(replicate_array.mean()),
        headroom_standard_error=float(replicate_array.std(ddof=1) / np.sqrt(len(replicate_array))),
        headroom_replicates=tuple(float(value) for value in replicate_array),
        p_final=p_final,
        greedy_order=tuple(
            sorted(
                range(instance.pool_size),
                key=lambda index: (-float(p_final[index]), index),
            )
        ),
        rank_gaps=tuple(
            float(p_final[order] - p_final[next_order])
            for order, next_order in zip(
                sorted(
                    range(instance.pool_size),
                    key=lambda index: (-float(p_final[index]), index),
                )[:-1],
                sorted(
                    range(instance.pool_size),
                    key=lambda index: (-float(p_final[index]), index),
                )[1:],
                strict=True,
            )
        ),
    )


def _simulate(
    instance: RandomDelayedLabelInstance,
    *,
    model_penalty: float,
    save_trace: bool,
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], DecisionFeatures],
) -> dict[str, object]:
    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    initial_belief = tuple(range(len(instance.world_energies)))
    expected_value = 0.0
    gate_used_states = 0
    total_states = 0
    traces: list[dict[str, object]] = []
    for world, probability in enumerate(instance.world_probabilities):
        belief = initial_belief
        selected: tuple[int, ...] = ()
        posterior_trace: list[dict[str, object]] = []
        for round_index in range(instance.budget):
            key = (selected, belief)
            if key not in cache:
                cache[key] = _decision_features(
                    instance,
                    stable=stable,
                    belief=belief,
                    selected=selected,
                )
            features = cache[key]
            gate = selective_gate(
                greedy_action_index=features.greedy_action,
                rollout_action_index=features.planner_action,
                headroom_mean=features.headroom_mean,
                headroom_standard_error=features.headroom_standard_error,
                model_penalty=model_penalty,
                cost_penalty=0.0,
                critical_value=CRITICAL_VALUE,
            )
            gate_used_states += int(gate.gate_used)
            total_states += 1
            action = int(gate.action_index)
            selected += (action,)
            if save_trace:
                posterior_trace.append(
                    {
                        "round_id": round_index,
                        "selected_before": list(key[0]),
                        "belief_before": list(key[1]),
                        "greedy_action": features.greedy_action,
                        "planner_action": features.planner_action,
                        "selected_action": action,
                        "gate_used": gate.gate_used,
                        "robust_gain": gate.robust_gain,
                        "headroom_mean": features.headroom_mean,
                        "headroom_standard_error": features.headroom_standard_error,
                        "headroom_replicates": list(features.headroom_replicates),
                        "p_final": list(features.p_final),
                        "greedy_order": list(features.greedy_order),
                        "rank_gaps": list(features.rank_gaps),
                    }
                )
            belief = _update_belief(instance, belief, action, world)
        realized_labels = [bool(action in stable[world]) for action in selected]
        realized_return = float(sum(realized_labels))
        expected_value += float(probability) * realized_return
        if save_trace:
            traces.append(
                {
                    "world": world,
                    "posterior_trace": posterior_trace,
                    "evaluator_trace": {
                        "realized_final_labels": realized_labels,
                        "realized_selective_return": realized_return,
                    },
                }
            )
    result: dict[str, object] = {
        "value": float(expected_value),
        "gate_invocation_rate": float(gate_used_states / total_states),
        "gate_used_states": gate_used_states,
        "total_states": total_states,
    }
    if save_trace:
        result["traces"] = traces
    return result


def _system_metrics(
    instance: RandomDelayedLabelInstance,
    *,
    model_penalty: float,
    save_trace: bool,
) -> dict[str, object]:
    cache: dict[tuple[tuple[int, ...], tuple[int, ...]], DecisionFeatures] = {}
    selective = _simulate(
        instance,
        model_penalty=model_penalty,
        save_trace=save_trace,
        cache=cache,
    )
    greedy = evaluate_random_instance(instance, "greedy_final").value
    # The exact-HENS action is recovered directly from the cached state path;
    # no sentinel gate value is used for this reference policy.
    full_cache = cache
    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    initial_belief = tuple(range(len(instance.world_energies)))
    full_value = 0.0
    for world, probability in enumerate(instance.world_probabilities):
        belief = initial_belief
        selected: tuple[int, ...] = ()
        for _ in range(instance.budget):
            key = (selected, belief)
            if key not in full_cache:
                full_cache[key] = _decision_features(
                    instance,
                    stable=stable,
                    belief=belief,
                    selected=selected,
                )
            action = full_cache[key].planner_action
            selected += (action,)
            belief = _update_belief(instance, belief, action, world)
        full_value += float(probability) * sum(action in stable[world] for action in selected)
    return {
        "instance_id": instance.instance_id,
        "greedy_value": float(greedy),
        "full_planner_value": float(full_value),
        "selective_value": float(selective["value"]),
        "selective_gate_invocation_rate": float(selective["gate_invocation_rate"]),
        "selective_gate_used_states": int(selective["gate_used_states"]),
        "selective_total_states": int(selective["total_states"]),
        "traces": selective.get("traces", []),
    }


def _paired_summary(differences: np.ndarray) -> dict[str, object]:
    return {
        "n_systems": int(len(differences)),
        "mean": float(np.mean(differences)),
        "wins": int(np.count_nonzero(differences > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
        "losses": int(np.count_nonzero(differences < -1e-12)),
    }


def _calibrate(inner: tuple[RandomDelayedLabelInstance, ...]) -> tuple[float, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for penalty in PENALTY_GRID:
        metrics = [
            _system_metrics(instance, model_penalty=penalty, save_trace=False) for instance in inner
        ]
        greedy = np.asarray([row["greedy_value"] for row in metrics], dtype=float)
        full = np.asarray([row["full_planner_value"] for row in metrics], dtype=float)
        selective = np.asarray([row["selective_value"] for row in metrics], dtype=float)
        full_gain = float(np.mean(full - greedy))
        selective_gain = float(np.mean(selective - greedy))
        retained = float(selective_gain / full_gain) if full_gain > 1e-12 else 0.0
        invocation = float(np.mean([row["selective_gate_invocation_rate"] for row in metrics]))
        rows.append(
            {
                "model_penalty": penalty,
                "mean_full_gain": full_gain,
                "mean_selective_gain": selective_gain,
                "retained_full_gain": retained,
                "mean_invocation_rate": invocation,
                "meets_targets": bool(invocation <= 0.25 and retained >= 0.80),
            }
        )
    eligible = [row for row in rows if row["meets_targets"]]
    if eligible:
        selected = max(eligible, key=lambda row: (float(row["model_penalty"]),))
    else:
        selected = max(
            rows,
            key=lambda row: (float(row["mean_selective_gain"]), float(row["model_penalty"])),
        )
    return float(selected["model_penalty"]), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20270805)
    args = parser.parse_args()
    if args.count != 80:
        raise ValueError("the registered replay is frozen at 80 systems")
    instances = generate_random_instances(count=args.count, seed=args.seed)
    inner = instances[:40]
    outer = instances[40:]
    selected_penalty, calibration = _calibrate(inner)
    outer_metrics = [
        _system_metrics(instance, model_penalty=selected_penalty, save_trace=True)
        for instance in outer
    ]
    greedy = np.asarray([row["greedy_value"] for row in outer_metrics], dtype=float)
    full = np.asarray([row["full_planner_value"] for row in outer_metrics], dtype=float)
    selective = np.asarray([row["selective_value"] for row in outer_metrics], dtype=float)
    full_gain = float(np.mean(full - greedy))
    selective_gain = float(np.mean(selective - greedy))
    output = {
        "schema_version": 1,
        "status": "complete_nested_selective_synthetic_replay_not_material_evidence",
        "protocol": "docs/SELECTIVE_PLANNING_PROTOCOL_V1.md",
        "generator": {
            "count": args.count,
            "seed": args.seed,
            "inner_count": len(inner),
            "outer_count": len(outer),
            "penalty_grid": PENALTY_GRID,
            "critical_value": CRITICAL_VALUE,
            "headroom_replicates": HEADROOM_REPLICATES,
            "posterior_sample_count": POSTERIOR_SAMPLES,
            "inner_sample_count": INNER_SAMPLES,
        },
        "code_sha256": _sha256(Path(__file__).resolve()),
        "benchmark_sha256": _sha256(
            Path(__file__).resolve().parent.parent / "src" / "matmem" / "random_delayed_label_benchmark.py"
        ),
        "audit_sha256": _sha256(
            Path(__file__).resolve().parent.parent / "src" / "matmem" / "hull_ens_audit.py"
        ),
        "inner_calibration": {
            "selected_model_penalty": selected_penalty,
            "candidates": calibration,
        },
        "outer_summary": {
            "n_systems": len(outer_metrics),
            "mean_greedy_value": float(np.mean(greedy)),
            "mean_full_planner_value": float(np.mean(full)),
            "mean_selective_value": float(np.mean(selective)),
            "full_planner_minus_greedy": _paired_summary(full - greedy),
            "selective_minus_greedy": _paired_summary(selective - greedy),
            "selective_minus_full_planner": _paired_summary(selective - full),
            "retained_full_gain": float(selective_gain / full_gain) if full_gain > 1e-12 else 0.0,
            "mean_gate_invocation_rate": float(
                np.mean([row["selective_gate_invocation_rate"] for row in outer_metrics])
            ),
            "mean_gate_used_states": float(
                np.mean([row["selective_gate_used_states"] for row in outer_metrics])
            ),
            "mean_total_states": float(
                np.mean([row["selective_total_states"] for row in outer_metrics])
            ),
        },
        "outer_systems": outer_metrics,
        "inner_instances": [asdict(instance) for instance in inner],
        "outer_instances": [asdict(instance) for instance in outer],
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing replay output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["inner_calibration"], indent=2, sort_keys=True))
    print(json.dumps(output["outer_summary"], indent=2, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
