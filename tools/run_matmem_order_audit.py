"""Audit input-order invariance across synthetic pools without writing artifacts."""

from __future__ import annotations

import argparse
import random

from run_matmem_dual_budget_pilot import (
    hull_revision_pool,
    iid_pool,
    local_boundary_pool,
    nonrecurring_pool,
    policy_factories,
    protocol_shift_pool,
    recurring_pool,
    retention_competition_pool,
)

from evimem.matmem import ActiveDiscoveryEvaluator

SCENARIOS = {
    "local_boundary_correlation": local_boundary_pool,
    "recurring_residual": recurring_pool,
    "iid_residual": iid_pool,
    "nonrecurring_chemistry": nonrecurring_pool,
    "causal_hull_revision": hull_revision_pool,
    "unsupported_protocol_shift": protocol_shift_pool,
    "retention_competition": retention_competition_pool,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=2)
    args = parser.parse_args()
    if min(args.permutations, args.candidates, args.budget) < 1 or args.capacity < 0:
        parser.error("positive permutations/candidates/budget and non-negative capacity required")

    print("scenario,policy,permutations,action_sequence_mismatches")
    for scenario_name, builder in SCENARIOS.items():
        original = builder(0, args.candidates)
        factories = policy_factories(0, args.capacity, args.budget, original)
        for policy_name in ("caw_joint", "decoupled_boundary", "uncertainty_fifo"):
            acquisition, retention = factories[policy_name]()
            reference = ActiveDiscoveryEvaluator(
                acquisition,
                retention,
                oracle_budget=args.budget,
                causal_hull_updates=scenario_name == "causal_hull_revision",
            ).evaluate(original).selected_query_ids
            mismatches = 0
            for permutation in range(args.permutations):
                shuffled = list(original)
                random.Random(10_000 + permutation).shuffle(shuffled)
                acquisition, retention = policy_factories(
                    0,
                    args.capacity,
                    args.budget,
                    shuffled,
                )[policy_name]()
                selected = ActiveDiscoveryEvaluator(
                    acquisition,
                    retention,
                    oracle_budget=args.budget,
                    causal_hull_updates=scenario_name == "causal_hull_revision",
                ).evaluate(shuffled).selected_query_ids
                mismatches += int(selected != reference)
            print(
                f"{scenario_name},{policy_name},{args.permutations},{mismatches}",
                flush=True,
            )


if __name__ == "__main__":
    main()
