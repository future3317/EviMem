"""Run the locked independent random exact-DP delayed-label mechanism suite."""

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

from matmem.random_delayed_label_benchmark import (  # noqa: E402
    evaluate_random_suite,
    generate_random_instances,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(rows: tuple[object, ...], instances: tuple[object, ...]) -> dict[str, object]:
    values = {(row.instance_id, row.policy): row.value for row in rows}  # type: ignore[attr-defined]
    actions = {(row.instance_id, row.policy): row.first_action for row in rows}  # type: ignore[attr-defined]
    dp_gap: dict[str, list[float]] = {policy: [] for policy in ("source_margin", "greedy_final", "source_rollout", "ic_sarr")}
    rollout_source: list[float] = []
    rollout_greedy: list[float] = []
    agreement: list[float] = []
    factor_cells: dict[str, dict[str, float]] = {}
    for instance in instances:
        instance_id = instance.instance_id  # type: ignore[attr-defined]
        dp = values[(instance_id, "optimal_dp")]
        for policy in dp_gap:
            dp_gap[policy].append(dp - values[(instance_id, policy)])
        rollout_source.append(values[(instance_id, "source_rollout")] - values[(instance_id, "source_margin")])
        rollout_greedy.append(values[(instance_id, "source_rollout")] - values[(instance_id, "greedy_final")])
        agreement.append(float(actions[(instance_id, "ic_sarr")] == actions[(instance_id, "source_rollout")]))
        key = (
            f"B={instance.budget}|signal={instance.source_signal:.1f}|corr="  # type: ignore[attr-defined]
            f"{instance.energy_correlation:.1f}|coupling={instance.delayed_label_coupling:.1f}"  # type: ignore[attr-defined]
        )
        cell = factor_cells.setdefault(key, {"count": 0.0, "dp_minus_rollout_sum": 0.0})
        cell["count"] += 1.0
        cell["dp_minus_rollout_sum"] += dp - values[(instance_id, "source_rollout")]
    for cell in factor_cells.values():
        cell["mean_dp_minus_rollout"] = cell.pop("dp_minus_rollout_sum") / cell["count"]
    return {
        "policy_dp_gap_distribution": {
            policy: {
                "mean": float(np.mean(gaps)),
                "median": float(np.median(gaps)),
                "p90": float(np.quantile(gaps, 0.90)),
                "max": float(np.max(gaps)),
            }
            for policy, gaps in dp_gap.items()
        },
        "rollout_equals_dp_fraction": float(np.mean(np.isclose(dp_gap["source_rollout"], 0.0))),
        "rollout_beats_source_fraction": float(np.mean(np.asarray(rollout_source) > 0.0)),
        "rollout_beats_greedy_fraction": float(np.mean(np.asarray(rollout_greedy) > 0.0)),
        "sampled_ic_sarr_exact_rollout_first_action_agreement": float(np.mean(agreement)),
        "factor_cells": factor_cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    instances = generate_random_instances(count=args.count, seed=args.seed)
    rows = evaluate_random_suite(count=args.count, seed=args.seed)
    output = {
        "schema_version": 1,
        "status": "independent_synthetic_mechanism_only_not_material_evidence",
        "protocol": "docs/P0_MECHANISM_SUITE_PROTOCOL.md",
        "generator": {
            "instance_count": args.count,
            "seed": args.seed,
            "world_count": 4,
            "pool_size": [5, 10],
            "budget": [1, 4],
            "ic_sarr_synthetic_samples": {"stage_one": 128, "stage_two": 512, "blocks": 16},
        },
        "code_sha256": _sha256(Path(__file__).resolve()),
        "instance_generator_sha256": _sha256(
            Path(__file__).resolve().parent.parent / "src" / "matmem" / "random_delayed_label_benchmark.py"
        ),
        "instances": [asdict(instance) for instance in instances],
        "rows": [asdict(row) for row in rows],
        "summary": _summary(rows, instances),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output={args.output.resolve()}")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
