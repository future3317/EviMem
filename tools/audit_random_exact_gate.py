"""Audit the frozen synthetic IC-SARR gate against exact finite-world values.

This is post-processing only.  It reads the already registered 1,000-instance
finite-world suite, reproduces its 128/512-sample two-stage gate, and compares
every sampled decision with the exact posterior rollout values.  It does not
run a material policy, alter the registered protocol, or write inside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from matmem.random_delayed_label_benchmark import RandomDelayedLabelInstance, _stable_indices

STAGE_ONE_SAMPLES = 128
STAGE_TWO_SAMPLES = 512
BLOCKS = 16
STAGE_ONE_BLOCK_SIZE = STAGE_ONE_SAMPLES // BLOCKS
STAGE_TWO_BLOCK_SIZE = STAGE_TWO_SAMPLES // BLOCKS
STAGE_ONE_CRITICAL = 2.64
STAGE_TWO_CRITICAL = 1.96
EPSILON = 1e-12


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_instances(path: Path) -> tuple[RandomDelayedLabelInstance, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "independent_synthetic_mechanism_only_not_material_evidence":
        raise ValueError("unexpected exact-suite status")
    if payload.get("schema_version") != 1:
        raise ValueError("unexpected exact-suite schema")
    rows = payload.get("instances")
    if not isinstance(rows, list) or len(rows) != 1000:
        raise ValueError("the registered audit requires exactly 1,000 instances")
    instances = tuple(RandomDelayedLabelInstance(**row) for row in rows)
    if tuple(instance.instance_id for instance in instances) != tuple(range(1000)):
        raise ValueError("instance IDs are not the registered 0..999 roster")
    return instances


def _source_completion(instance: RandomDelayedLabelInstance, action: int) -> tuple[int, ...]:
    completion = (action,)
    order = sorted(range(instance.pool_size), key=lambda index: (instance.source_energies[index], index))
    for candidate in order:
        if len(completion) == instance.budget:
            break
        if candidate not in completion:
            completion += (candidate,)
    return completion


def _exact_action_values(instance: RandomDelayedLabelInstance) -> tuple[np.ndarray, int, tuple[frozenset[int], ...]]:
    stable = tuple(_stable_indices(world) for world in instance.world_energies)
    values = np.asarray(
        [
            sum(
                probability * sum(candidate in stable[world] for candidate in _source_completion(instance, action))
                for world, probability in enumerate(instance.world_probabilities)
            )
            for action in range(instance.pool_size)
        ],
        dtype=float,
    )
    source = min(range(instance.pool_size), key=lambda index: (instance.source_energies[index], index))
    return values, source, stable


def _audit_instance(instance: RandomDelayedLabelInstance) -> dict[str, Any]:
    values, source, stable = _exact_action_values(instance)
    rng = np.random.default_rng(20270720 + 104729 * instance.instance_id)
    belief = np.arange(4, dtype=int)
    weights = np.asarray(instance.world_probabilities, dtype=float)
    stage_one_worlds = rng.choice(belief, size=STAGE_ONE_SAMPLES, p=weights)
    action_count = instance.pool_size
    stage_one_rewards = np.empty((BLOCKS, action_count), dtype=float)
    for block in range(BLOCKS):
        worlds = stage_one_worlds[block * STAGE_ONE_BLOCK_SIZE : (block + 1) * STAGE_ONE_BLOCK_SIZE]
        for action in range(action_count):
            selected = _source_completion(instance, action)
            stage_one_rewards[block, action] = np.mean(
                [sum(candidate in stable[int(world)] for candidate in selected) for world in worlds]
            )
    stage_one_differences = stage_one_rewards - stage_one_rewards[:, [source]]
    stage_one_mean = stage_one_differences.mean(axis=0)
    stage_one_se = stage_one_differences.std(axis=0, ddof=1) / np.sqrt(BLOCKS)
    stage_one_lower = stage_one_mean - STAGE_ONE_CRITICAL * stage_one_se
    stage_one_lower[source] = 0.0
    improving = [
        action for action in range(action_count) if action != source and stage_one_lower[action] > 0.0
    ]
    stage_one_accepted = bool(improving)
    stage_one_action = (
        min(improving, key=lambda action: (-stage_one_rewards[:, action].mean(), action))
        if improving
        else source
    )
    positive = [
        action for action in range(action_count) if action != source and stage_one_mean[action] > 0.0
    ]
    stage_two_used = not stage_one_accepted and bool(positive)
    stage_two_screened = (
        min(positive, key=lambda action: (-stage_one_mean[action], action)) if stage_two_used else None
    )
    stage_two_accepted = False
    stage_two_lower = None
    if stage_two_used:
        assert stage_two_screened is not None
        stage_two_worlds = rng.choice(belief, size=STAGE_TWO_SAMPLES, p=weights)
        difference = np.asarray(
            [
                sum(candidate in stable[int(world)] for candidate in _source_completion(instance, stage_two_screened))
                - sum(candidate in stable[int(world)] for candidate in _source_completion(instance, source))
                for world in stage_two_worlds
            ],
            dtype=float,
        )
        stage_two_blocks = difference.reshape(BLOCKS, STAGE_TWO_BLOCK_SIZE).mean(axis=1)
        stage_two_mean = float(stage_two_blocks.mean())
        stage_two_se = float(stage_two_blocks.std(ddof=1) / np.sqrt(BLOCKS))
        stage_two_lower = stage_two_mean - STAGE_TWO_CRITICAL * stage_two_se
        stage_two_accepted = bool(stage_two_lower > 0.0)
    if stage_one_accepted:
        final_action = stage_one_action
    elif stage_two_accepted:
        assert stage_two_screened is not None
        final_action = stage_two_screened
    else:
        final_action = source
    exact_advantages = values - values[source]
    exact_best_action = min(range(action_count), key=lambda action: (-values[action], action))
    return {
        "instance_id": instance.instance_id,
        "exact_best_advantage": float(exact_advantages[exact_best_action]),
        "exact_final_advantage": float(exact_advantages[final_action]),
        "exact_best_action": int(exact_best_action),
        "final_action": int(final_action),
        "source_action": int(source),
        "stage_one_accepted": stage_one_accepted,
        "stage_one_action": int(stage_one_action),
        "stage_two_used": stage_two_used,
        "stage_two_screened": None if stage_two_screened is None else int(stage_two_screened),
        "exact_screened_advantage": (
            None if stage_two_screened is None else float(exact_advantages[stage_two_screened])
        ),
        "stage_two_accepted": stage_two_accepted,
        "stage_two_lower": stage_two_lower,
        "stage_one_false_accept": bool(stage_one_accepted and exact_advantages[stage_one_action] <= EPSILON),
        "stage_two_false_accept": bool(stage_two_accepted and stage_two_screened is not None and exact_advantages[stage_two_screened] <= EPSILON),
        "final_false_accept": bool(final_action != source and exact_advantages[final_action] <= EPSILON),
        "stage_one_missed_positive": bool(exact_advantages[exact_best_action] > EPSILON and not stage_one_accepted),
        "stage_two_missed_positive": bool(stage_two_used and stage_two_screened is not None and exact_advantages[stage_two_screened] > EPSILON and not stage_two_accepted),
        "stage_two_corrected_stage_one": bool(stage_two_used and not stage_two_accepted),
        "exact_regret": float(values[exact_best_action] - values[final_action]),
    }


def _rate(count: int, denominator: int) -> float:
    return None if denominator == 0 else float(count / denominator)


def audit(*, input_path: Path, output_path: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("exact gate audit must remain outside Git")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    instances = _load_instances(input_path)
    rows = [_audit_instance(instance) for instance in instances]
    registered_payload = json.loads(input_path.read_text(encoding="utf-8"))
    registered_actions = {
        int(row["instance_id"]): int(row["first_action"])
        for row in registered_payload["rows"]
        if row.get("policy") == "ic_sarr"
    }
    reproduced_actions = {int(row["instance_id"]): int(row["final_action"]) for row in rows}
    if set(registered_actions) != set(reproduced_actions):
        raise ValueError("registered ic_sarr roster does not match the exact audit roster")
    action_agreement = sum(
        registered_actions[instance_id] == reproduced_actions[instance_id]
        for instance_id in registered_actions
    )
    total = len(rows)
    stage_one_accepted = sum(row["stage_one_accepted"] for row in rows)
    stage_two_used = sum(row["stage_two_used"] for row in rows)
    stage_two_accepted = sum(row["stage_two_accepted"] for row in rows)
    stage_two_corrected = sum(row["stage_two_corrected_stage_one"] for row in rows)
    exact_positive = sum(row["exact_best_advantage"] > EPSILON for row in rows)
    final_non_source = sum(row["final_action"] != row["source_action"] for row in rows)
    summary = {
        "stage_one": {
            "accepted_count": stage_one_accepted,
            "acceptance_rate": _rate(stage_one_accepted, total),
            "false_accept_count": sum(row["stage_one_false_accept"] for row in rows),
            "false_accept_rate_all_instances": _rate(sum(row["stage_one_false_accept"] for row in rows), total),
            "false_accept_rate_among_accepts": _rate(sum(row["stage_one_false_accept"] for row in rows), stage_one_accepted),
            "missed_positive_count": sum(row["stage_one_missed_positive"] for row in rows),
            "missed_positive_rate_among_exact_positive": _rate(sum(row["stage_one_missed_positive"] for row in rows), exact_positive),
        },
        "stage_two": {
            "invocation_count": stage_two_used,
            "invocation_rate": _rate(stage_two_used, total),
            "accepted_count": stage_two_accepted,
            "acceptance_rate_among_invocations": _rate(stage_two_accepted, stage_two_used),
            "false_accept_count": sum(row["stage_two_false_accept"] for row in rows),
            "false_accept_rate_all_instances": _rate(sum(row["stage_two_false_accept"] for row in rows), total),
            "false_accept_rate_among_accepts": _rate(sum(row["stage_two_false_accept"] for row in rows), stage_two_accepted),
            "missed_positive_count": sum(row["stage_two_missed_positive"] for row in rows),
            "missed_positive_rate_among_positive_screened": _rate(
                sum(row["stage_two_missed_positive"] for row in rows),
                sum(row["stage_two_used"] and row["exact_screened_advantage"] > EPSILON for row in rows),
            ),
            "corrected_stage_one_count": stage_two_corrected,
            "corrected_stage_one_rate_among_invocations": _rate(stage_two_corrected, stage_two_used),
        },
        "final_gate": {
            "non_source_count": final_non_source,
            "false_accept_count": sum(row["final_false_accept"] for row in rows),
            "false_accept_rate_all_instances": _rate(sum(row["final_false_accept"] for row in rows), total),
            "missed_positive_count": sum(row["exact_best_advantage"] > EPSILON and row["final_action"] == row["source_action"] for row in rows),
            "missed_positive_rate_among_exact_positive": _rate(
                sum(row["exact_best_advantage"] > EPSILON and row["final_action"] == row["source_action"] for row in rows),
                exact_positive,
            ),
            "mean_exact_regret": float(np.mean([row["exact_regret"] for row in rows])),
            "median_exact_regret": float(np.median([row["exact_regret"] for row in rows])),
        },
    }
    payload = {
        "schema_version": 1,
        "status": "complete_exact_world_numerical_gate_audit",
        "scope": "initial decision state of each registered 1,000-instance finite-world suite instance",
        "input": {"path": str(input_path), "sha256": _sha256(input_path), "instance_count": total},
        "protocol": {
            "stage_one_samples": STAGE_ONE_SAMPLES,
            "stage_two_samples": STAGE_TWO_SAMPLES,
            "blocks": BLOCKS,
            "stage_one_samples_per_block": STAGE_ONE_BLOCK_SIZE,
            "stage_two_samples_per_block": STAGE_TWO_BLOCK_SIZE,
            "stage_one_critical_multiplier": STAGE_ONE_CRITICAL,
            "stage_two_critical_multiplier": STAGE_TWO_CRITICAL,
            "seed_rule": "20270720 + 104729 * instance_id",
            "exact_advantage": "exact posterior expected source-continuation terminal value minus source action value",
        },
        "summary": {
            "instance_count": total,
            "exact_positive_count": exact_positive,
            "registered_ic_sarr_action_agreement_count": action_agreement,
            "registered_ic_sarr_action_agreement_rate": _rate(action_agreement, total),
            **summary,
        },
        "rows": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(input_path=args.input, output_path=args.output)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
