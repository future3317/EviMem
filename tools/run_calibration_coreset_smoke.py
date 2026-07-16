"""Small deterministic calibration-compression smoke, not WBM evidence.

The evaluator owns future residuals only for metric computation.  Retention
receives observable future queries and previously revealed cards, never a
future oracle outcome.  No result artifact is written by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable

from run_matmem_dual_budget_pilot import iid_pool, recurring_pool

from evimem.matmem import (
    CalibrationUtilityBuilder,
    DiversityBoundedMemory,
    FacilityLocationCoresetPlanner,
    FIFOBoundedMemory,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    FullHistoryMemory,
    ProtocolCompatibilityResolver,
    StreamingCalibrationCoreset,
)


def _coreset(capacity: int, resolver: ProtocolCompatibilityResolver):
    template = FixedKernelResidualGP(
        resolver,
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    return StreamingCalibrationCoreset(
        FacilityLocationCoresetPlanner(
            capacity,
            CalibrationUtilityBuilder(template),
            min_admission_gain=1e-12,
        )
    )


def _evaluate_stream(cases, strategy, resolver):
    squared_error = 0.0
    brier = 0.0
    nll = 0.0
    prediction_count = 0
    prediction_seconds = 0.0
    update_seconds = 0.0
    for index, case in enumerate(cases):
        future = tuple(item.query for item in cases[index + 1 :])
        if future:
            prediction_started = time.perf_counter()
            posterior = FixedKernelResidualGP(
                resolver,
                config=FixedKernelGPConfig(length_scale=0.35),
            ).fit(strategy.cards())
            prediction = posterior.predict(future)
            prediction_seconds += time.perf_counter() - prediction_started
            for query, mean, probability, future_case in zip(
                future,
                prediction.mean_ev_per_atom,
                prediction.stable_probability,
                cases[index + 1 :],
                strict=True,
            ):
                actual_residual = future_case.oracle_card.oracle_residual_ev_per_atom
                actual_stable = float(
                    query.base_hull_distance_ev_per_atom + actual_residual
                    <= query.stability_threshold_ev_per_atom
                )
                clipped = min(1.0 - 1e-12, max(1e-12, probability))
                squared_error += (mean - actual_residual) ** 2
                brier += (probability - actual_stable) ** 2
                nll -= actual_stable * math.log(clipped) + (
                    1.0 - actual_stable
                ) * math.log(1.0 - clipped)
                prediction_count += 1
        started = time.perf_counter()
        strategy.admit(case.oracle_card, future or (case.query,))
        update_seconds += time.perf_counter() - started
    return {
        "rmse_ev_per_atom": math.sqrt(squared_error / prediction_count),
        "brier": brier / prediction_count,
        "nll": nll / prediction_count,
        "update_seconds": update_seconds,
        "prediction_seconds": prediction_seconds,
        "online_seconds": update_seconds + prediction_seconds,
        "final_active_size": len(strategy.cards()),
        "prediction_count": prediction_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=24)
    parser.add_argument("--capacity", type=int, default=4)
    args = parser.parse_args()
    if min(args.seeds, args.candidates) < 1 or args.capacity < 1:
        parser.error("seeds, candidates, and capacity must be positive")

    scenarios: dict[str, Callable] = {
        "recurrence_sanity": recurring_pool,
        "iid_negative_control": iid_pool,
    }
    rows = []
    for scenario, builder in scenarios.items():
        for seed in range(args.seeds):
            cases = builder(seed, args.candidates)
            resolver = ProtocolCompatibilityResolver()
            strategies = {
                "full_history": FullHistoryMemory(args.candidates),
                "fifo": FIFOBoundedMemory(args.capacity),
                "diversity": DiversityBoundedMemory(args.capacity),
                "decision_aware_coreset": _coreset(args.capacity, resolver),
            }
            for name, strategy in strategies.items():
                rows.append(
                    {
                        "scope": "synthetic_calibration_smoke_not_materials_evidence",
                        "scenario": scenario,
                        "seed": seed,
                        "strategy": name,
                        "capacity": (
                            args.candidates if name == "full_history" else args.capacity
                        ),
                        **_evaluate_stream(cases, strategy, resolver),
                    }
                )
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
