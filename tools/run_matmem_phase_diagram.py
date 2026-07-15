"""Run a compact B x K synthetic phase diagram without writing artifacts."""

from __future__ import annotations

import argparse
import statistics
import time

from run_matmem_dual_budget_pilot import policy_factories, retention_competition_pool

from evimem.matmem import ActiveDiscoveryEvaluator

POLICIES = (
    "caw_joint",
    "decoupled_boundary",
    "uncertainty_fifo",
    "compatible_knn_archive_topk",
    "compatible_knn_full_history",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--budgets", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--capacities", nargs="+", default=["0", "1", "2", "4", "8", "inf"])
    args = parser.parse_args()
    if args.seeds < 1 or args.candidates < 1 or min(args.budgets) < 1:
        parser.error("seeds, candidates, and budgets must be positive")

    print(
        "B,K,joint,decoupled,uncertainty,archive_topk,full_history,"
        "joint_decoupled_disagreement,joint_runtime_seconds"
    )
    for budget in args.budgets:
        for raw_capacity in args.capacities:
            capacity = budget if raw_capacity == "inf" else int(raw_capacity)
            if capacity < 0:
                parser.error("capacities must be non-negative or 'inf'")
            discoveries: dict[str, list[float]] = {name: [] for name in POLICIES}
            disagreements: list[float] = []
            joint_runtimes: list[float] = []
            for seed in range(args.seeds):
                candidates = retention_competition_pool(seed, args.candidates)
                results = {}
                factories = policy_factories(seed, capacity, budget, candidates)
                for name in POLICIES:
                    acquisition, retention = factories[name]()
                    started = time.perf_counter()
                    result = ActiveDiscoveryEvaluator(
                        acquisition,
                        retention,
                        oracle_budget=budget,
                    ).evaluate(candidates)
                    elapsed = time.perf_counter() - started
                    results[name] = result
                    discoveries[name].append(result.cumulative_true_discoveries)
                    if name == "caw_joint":
                        joint_runtimes.append(elapsed)
                joint_actions = results["caw_joint"].selected_query_ids
                decoupled_actions = results["decoupled_boundary"].selected_query_ids
                disagreements.append(
                    sum(
                        joint != decoupled
                        for joint, decoupled in zip(
                            joint_actions,
                            decoupled_actions,
                            strict=True,
                        )
                    )
                    / len(joint_actions)
                )
            values = [statistics.fmean(discoveries[name]) for name in POLICIES]
            print(
                f"{budget},{raw_capacity},"
                + ",".join(f"{value:.3f}" for value in values)
                + f",{statistics.fmean(disagreements):.3f},"
                + f"{statistics.fmean(joint_runtimes):.6f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
