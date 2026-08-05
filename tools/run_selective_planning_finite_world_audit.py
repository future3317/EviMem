"""Run the registered headroom-stratified finite-world planning audit.

The candidate pool is generated once with a frozen seed and accepted into
four exact-DP headroom bins. No materials data, vault, or posterior trace is
used. Raw/derived JSON should be written under an external DATA directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from matmem.hull_ens_audit import (  # noqa: E402
    evaluate_hull_ens_policy,
    exact_hull_ens,
    sampled_hull_ens,
)
from matmem.random_delayed_label_benchmark import (  # noqa: E402
    RandomDelayedLabelInstance,
    evaluate_random_instance,
    generate_random_instances,
)

BIN_NAMES = ("H*=0", "0<H*<=0.02", "0.02<H*<=0.10", "H*>0.10")
POLICIES = (
    "greedy_final",
    "exact_hull_ens",
    "sampled_hull_ens",
    "double_sampled_hull_ens",
    "optimal_dp",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _headroom_bin(headroom: float) -> str:
    if abs(headroom) <= 1e-12:
        return BIN_NAMES[0]
    if headroom <= 0.02 + 1e-12:
        return BIN_NAMES[1]
    if headroom <= 0.10 + 1e-12:
        return BIN_NAMES[2]
    return BIN_NAMES[3]


def _select_stratified(
    *,
    candidate_pool_count: int,
    seed: int,
    quota_per_bin: int,
) -> tuple[tuple[RandomDelayedLabelInstance, ...], dict[str, int]]:
    candidates = generate_random_instances(count=candidate_pool_count, seed=seed)
    buckets: dict[str, list[RandomDelayedLabelInstance]] = {name: [] for name in BIN_NAMES}
    for instance in candidates:
        dp = evaluate_random_instance(instance, "optimal_dp").value
        greedy = evaluate_random_instance(instance, "greedy_final").value
        buckets[_headroom_bin(dp - greedy)].append(instance)
    counts = {name: len(values) for name, values in buckets.items()}
    missing = {name: count for name, count in counts.items() if count < quota_per_bin}
    if missing:
        raise RuntimeError(
            "frozen candidate pool cannot populate headroom bins: "
            f"missing={missing}; counts={counts}"
        )
    selected = tuple(
        instance
        for name in BIN_NAMES
        for instance in buckets[name][:quota_per_bin]
    )
    return selected, counts


def _initial_action(instance: RandomDelayedLabelInstance, policy: str, *, seed: int) -> int:
    if policy == "exact_hull_ens":
        return exact_hull_ens(instance, remaining_budget=instance.budget).selected_action
    if policy == "sampled_hull_ens":
        return sampled_hull_ens(
            instance,
            posterior_sample_count=128,
            inner_sample_count=8,
            seed=seed,
            remaining_budget=instance.budget,
            independent_inner_stream=False,
        ).selected_action
    if policy == "double_sampled_hull_ens":
        return sampled_hull_ens(
            instance,
            posterior_sample_count=128,
            inner_sample_count=8,
            seed=seed,
            remaining_budget=instance.budget,
            independent_inner_stream=True,
        ).selected_action
    return evaluate_random_instance(instance, policy).first_action


def _evaluate_instance(instance: RandomDelayedLabelInstance, *, seed: int) -> dict[str, object]:
    dp = evaluate_random_instance(instance, "optimal_dp")
    greedy = evaluate_random_instance(instance, "greedy_final")
    values = {
        "optimal_dp": float(dp.value),
        "greedy_final": float(greedy.value),
        "exact_hull_ens": evaluate_hull_ens_policy(instance, mode="exact", seed=seed),
        "sampled_hull_ens": evaluate_hull_ens_policy(
            instance,
            mode="sampled",
            posterior_sample_count=128,
            inner_sample_count=8,
            seed=seed,
            independent_inner_stream=False,
        ),
        "double_sampled_hull_ens": evaluate_hull_ens_policy(
            instance,
            mode="sampled",
            posterior_sample_count=128,
            inner_sample_count=8,
            seed=seed,
            independent_inner_stream=True,
        ),
    }
    headroom = float(values["optimal_dp"] - values["greedy_final"])
    actions = {
        policy: _initial_action(instance, policy, seed=seed)
        for policy in ("greedy_final", "exact_hull_ens", "sampled_hull_ens", "double_sampled_hull_ens")
    }
    return {
        "instance_id": instance.instance_id,
        "headroom": headroom,
        "headroom_bin": _headroom_bin(headroom),
        "values": values,
        "actions": actions,
        "action_agreement_with_greedy": {
            policy: bool(actions[policy] == actions["greedy_final"])
            for policy in actions
            if policy != "greedy_final"
        },
    }


def _summarize(records: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for bin_name in BIN_NAMES:
        selected = [record for record in records if record["headroom_bin"] == bin_name]
        if not selected:
            summary[bin_name] = {"count": 0}
            continue
        values = {
            policy: np.asarray(
                [float(record["values"][policy]) for record in selected], dtype=float  # type: ignore[index]
            )
            for policy in POLICIES
        }
        summary[bin_name] = {
            "count": len(selected),
            "mean_headroom": float(np.mean([record["headroom"] for record in selected])),
            "mean_dp_minus_policy": {
                policy: float(np.mean(values["optimal_dp"] - values[policy]))
                for policy in POLICIES
                if policy != "optimal_dp"
            },
            "mean_policy_minus_greedy": {
                policy: float(np.mean(values[policy] - values["greedy_final"]))
                for policy in POLICIES
                if policy != "greedy_final"
            },
            "initial_action_agreement_with_greedy": {
                policy: float(
                    np.mean(
                        [
                            bool(record["action_agreement_with_greedy"][policy])  # type: ignore[index]
                            for record in selected
                        ]
                    )
                )
                for policy in ("exact_hull_ens", "sampled_hull_ens", "double_sampled_hull_ens")
            },
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-pool-count", type=int, default=16000)
    parser.add_argument("--quota-per-bin", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20270804)
    args = parser.parse_args()
    if args.candidate_pool_count < args.quota_per_bin * len(BIN_NAMES):
        raise ValueError("candidate pool is smaller than the requested stratified quota")
    if args.quota_per_bin < 1:
        raise ValueError("quota must be positive")

    selected, candidate_counts = _select_stratified(
        candidate_pool_count=args.candidate_pool_count,
        seed=args.seed,
        quota_per_bin=args.quota_per_bin,
    )
    records = [
        _evaluate_instance(instance, seed=args.seed + 15485863 * (position + 1))
        for position, instance in enumerate(selected)
    ]
    output = {
        "schema_version": 1,
        "status": "complete_finite_world_selective_planning_audit_not_material_evidence",
        "protocol": "docs/SELECTIVE_PLANNING_PROTOCOL_V1.md",
        "generator": {
            "candidate_pool_count": args.candidate_pool_count,
            "candidate_pool_seed": args.seed,
            "quota_per_bin": args.quota_per_bin,
            "world_count": 4,
            "sampled_hull_ens": {
                "posterior_sample_count": 128,
                "inner_sample_count": 8,
                "double_sampled_inner_stream": True,
            },
        },
        "code_sha256": _sha256(Path(__file__).resolve()),
        "benchmark_sha256": _sha256(
            Path(__file__).resolve().parent.parent / "src" / "matmem" / "random_delayed_label_benchmark.py"
        ),
        "audit_sha256": _sha256(
            Path(__file__).resolve().parent.parent / "src" / "matmem" / "hull_ens_audit.py"
        ),
        "candidate_pool_bin_counts": candidate_counts,
        "selected_instances": [asdict(instance) for instance in selected],
        "records": records,
        "summary": _summarize(records),
    }
    if args.output.exists():
        raise ValueError(f"refusing to overwrite existing audit output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
