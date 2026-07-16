"""Run a small matched-access economics falsification pilot without artifacts.

This compares persistent FIFO with on-demand reconstruction of the *same* FIFO
set. It does not compare acquisition policies and is not materials evidence.
"""

from __future__ import annotations

import argparse
import statistics

from run_matmem_dual_budget_pilot import (
    SyntheticCausalHullReviser,
    evaluate_candidates,
    hull_revision_pool,
    recurring_pool,
)

from evimem.matmem import (
    ActiveDiscoveryEvaluator,
    BaseBoundaryAcquisition,
    FIFOBoundedMemory,
    MatchedAccessCostModel,
    MatchedAccessOperationLedger,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=30)
    parser.add_argument("--budget", type=int, default=10)
    parser.add_argument("--capacity", type=int, default=2)
    args = parser.parse_args()
    if min(args.seeds, args.candidates, args.budget) < 1 or args.capacity < 0:
        parser.error("seeds, candidates, and budget must be positive; capacity may be zero")

    scenarios = {
        "recurring_no_hull_update": (recurring_pool, False),
        "causal_hull_revision": (hull_revision_pool, True),
    }
    price_models = {
        "free_retrieval": MatchedAccessCostModel(
            archive_retrieval_cost=0.0,
            persistent_recertification_cost=1.0,
            on_demand_recertification_cost=1.0,
        ),
        "costed_retrieval": MatchedAccessCostModel(
            archive_retrieval_cost=1.0,
            persistent_recertification_cost=1.0,
            on_demand_recertification_cost=1.0,
        ),
    }
    print("scope=synthetic_matched_access_mechanism_not_materials_evidence")
    print(
        "scenario,mean_retrievals,mean_persistent_recertifications,"
        "free_mean_savings,costed_mean_savings"
    )
    for name, (builder, causal_hull_updates) in scenarios.items():
        ledgers = []
        for seed in range(args.seeds):
            candidates = builder(seed, args.candidates)
            metrics = evaluate_candidates(ActiveDiscoveryEvaluator(
                BaseBoundaryAcquisition(),
                FIFOBoundedMemory(capacity=args.capacity),
                oracle_budget=args.budget,
                causal_hull_updates=causal_hull_updates,
                causal_hull_reviser=(
                    SyntheticCausalHullReviser() if causal_hull_updates else None
                ),
            ), candidates)
            ledgers.append(MatchedAccessOperationLedger.from_metrics(metrics))
        savings = {
            label: [model.evaluate(ledger).persistent_net_savings for ledger in ledgers]
            for label, model in price_models.items()
        }
        print(
            ",".join(
                (
                    name,
                    f"{statistics.fmean(item.on_demand_archive_retrievals for item in ledgers):.3f}",
                    f"{statistics.fmean(item.persistent_hull_recertifications for item in ledgers):.3f}",
                    f"{statistics.fmean(savings['free_retrieval']):.3f}",
                    f"{statistics.fmean(savings['costed_retrieval']):.3f}",
                )
            )
        )


if __name__ == "__main__":
    main()
