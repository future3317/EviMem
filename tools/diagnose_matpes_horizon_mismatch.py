"""Diagnose when myopic Delta-Hull gains and losses arise on opened traces.

This is attribution-only analysis. It reads an already opened result and never
changes a policy, posterior, task split, or oracle outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnose(
    *,
    result_path: Path,
    output_path: Path,
    delta_policy: str = "delta_hull_active_search",
    source_policy: str = "source_margin",
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("split") != "confirmatory" or not result.get(
        "evaluation_systems_accessed"
    ):
        raise ValueError("horizon diagnosis requires an opened confirmatory result")
    systems = sorted(result["systems"])
    if not systems:
        raise ValueError("horizon diagnosis requires at least one exact system")
    maximum_rounds = max(
        len(result["systems"][system]["strategies"][source_policy]["selected_pair_ids"])
        for system in systems
    )
    prefix_differences: dict[str, list[int]] = {}
    system_rows: dict[str, Any] = {}
    for system in systems:
        strategies = result["systems"][system]["strategies"]
        source = strategies[source_policy]
        delta = strategies[delta_policy]
        source_confirmed = set(source["oracle_pool_confirmed_ids"])
        delta_confirmed = set(delta["oracle_pool_confirmed_ids"])
        source_actions = list(source["selected_pair_ids"])
        delta_actions = list(delta["selected_pair_ids"])
        rounds = min(len(source_actions), len(delta_actions))
        differences = [
            sum(pair_id in delta_confirmed for pair_id in delta_actions[:round_index])
            - sum(pair_id in source_confirmed for pair_id in source_actions[:round_index])
            for round_index in range(1, rounds + 1)
        ]
        prefix_differences[system] = differences
        negative_rounds = [index + 1 for index, value in enumerate(differences) if value < 0]
        persistent_loss_onset = next(
            (
                index + 1
                for index in range(rounds)
                if all(value < 0 for value in differences[index:])
            ),
            None,
        )
        source_ceiling = int(source["oracle_pool_discovery_ceiling"])
        source_confirmations = int(source["oracle_pool_confirmed_discoveries"])
        system_rows[system] = {
            "round_prefix_confirmation_difference": differences,
            "final_difference": differences[-1],
            "first_negative_round": None if not negative_rounds else negative_rounds[0],
            "persistent_loss_onset_round": persistent_loss_onset,
            "first_actions_differ": source_actions[0] != delta_actions[0],
            "full_traces_differ": source_actions != delta_actions,
            "early_delta_advantage_not_retained": (
                any(value > 0 for value in differences[:2]) and differences[-1] <= 0
            ),
            "source_oracle_headroom": source_ceiling - source_confirmations,
        }
    round_rows: list[dict[str, Any]] = []
    for round_index in range(maximum_rounds):
        values = np.asarray(
            [
                differences[round_index]
                for differences in prefix_differences.values()
                if round_index < len(differences)
            ],
            dtype=float,
        )
        round_rows.append(
            {
                "round_index": round_index + 1,
                "system_count": len(values),
                "mean_prefix_confirmation_difference": float(values.mean()),
                "wins": int(np.sum(values > 0)),
                "ties": int(np.sum(values == 0)),
                "losses": int(np.sum(values < 0)),
            }
        )
    nonzero_headroom = [
        row for row in system_rows.values() if row["source_oracle_headroom"] > 0
    ]
    final_losses = [
        system for system, row in system_rows.items() if row["final_difference"] < 0
    ]
    payload = {
        "schema_version": 1,
        "status": "opened_attribution_only_horizon_diagnostic",
        "result_sha256": _sha256(result_path),
        "delta_policy": delta_policy,
        "source_policy": source_policy,
        "round_prefix_summary": round_rows,
        "system_count": len(systems),
        "nonzero_headroom_system_count": len(nonzero_headroom),
        "nonzero_headroom_mean_final_difference": (
            None
            if not nonzero_headroom
            else float(np.mean([row["final_difference"] for row in nonzero_headroom]))
        ),
        "first_action_differs_final_tie_count": sum(
            row["first_actions_differ"] and row["final_difference"] == 0
            for row in system_rows.values()
        ),
        "early_advantage_not_retained_count": sum(
            row["early_delta_advantage_not_retained"] for row in system_rows.values()
        ),
        "final_loss_systems": final_losses,
        "final_loss_onset_counts": {
            str(round_index): sum(
                row["persistent_loss_onset_round"] == round_index
                for row in system_rows.values()
            )
            for round_index in range(1, maximum_rounds + 1)
        },
        "posterior_conditioned_source_counterfactual_available": False,
        "systems": system_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta-policy", default="delta_hull_active_search")
    parser.add_argument("--source-policy", default="source_margin")
    args = parser.parse_args()
    payload = diagnose(
        result_path=args.result,
        output_path=args.output,
        delta_policy=args.delta_policy,
        source_policy=args.source_policy,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
