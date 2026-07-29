"""Audit whether the MAD-1.5 atomization-hull task is nontrivial.

This is an offline oracle-side diagnostic.  It never feeds target values to a
policy and its output must remain outside Git.  It checks the registered task
contract and computes the complete-pool hull ceiling for every development
system so that a saturated benchmark is not mistaken for selector evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.entries.computed_entries import ComputedEntry


def _composition_key(composition: dict[str, float]) -> str:
    return json.dumps(
        {key: round(float(value), 12) for key, value in sorted(composition.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(math.floor(probability * len(ordered))))
    return ordered[index]


def run(*, task_path: Path, vault_path: Path, output_path: Path, budget: int = 6) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("MAD hull audit output must remain outside Git")
    if output_path.exists():
        raise FileExistsError("MAD hull audit cannot overwrite an existing output")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    if (
        task.get("hull_semantics")
        != "atomization-energy convex-hull proxy, not solid-state formation hull"
    ):
        raise ValueError("unexpected MAD hull semantics")
    task_rows = task["development_pairs"]
    outcomes = {row["pair_id"]: row for row in vault["target_outcomes"]}
    if set(outcomes) != {row["pair_id"] for row in task_rows}:
        raise ValueError("MAD task and vault IDs do not join exactly")
    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in task_rows:
        by_system.setdefault(row["chemical_system"], []).append(row)

    rows: list[dict[str, Any]] = []
    for system in sorted(by_system):
        candidates = by_system[system]
        initial = [
            ComputedEntry(
                entry["composition"],
                entry["corrected_total_energy_ev"],
                entry_id=entry["entry_id"],
            )
            for entry in task["development_initial_phase_entries"][system]
        ]
        entries = [
            ComputedEntry(
                outcomes[row["pair_id"]]["composition"],
                outcomes[row["pair_id"]]["target_corrected_total_energy_ev"],
                entry_id=row["pair_id"],
            )
            for row in candidates
        ]
        diagram = PhaseDiagram([*initial, *entries])
        candidate_ids = {row["pair_id"] for row in candidates}
        stable_ids = {str(entry.entry_id) for entry in diagram.stable_entries}
        stable_candidate_ids = stable_ids & candidate_ids
        stable_compositions = {
            _composition_key(outcomes[pair_id]["composition"]) for pair_id in stable_candidate_ids
        }
        candidate_compositions = {_composition_key(row["composition"]) for row in candidates}
        stable_count = len(stable_candidate_ids)
        rows.append(
            {
                "chemical_system": system,
                "candidate_count": len(candidates),
                "composition_count": len(candidate_compositions),
                "stable_candidate_count": stable_count,
                "stable_composition_count": len(stable_compositions),
                "stable_fraction": stable_count / len(candidates),
                "oracle_confirmation_ceiling": min(budget, stable_count),
                "budget_saturated": stable_count >= budget,
                "stable_candidate_ids": sorted(stable_candidate_ids),
            }
        )

    stable_counts = [row["stable_candidate_count"] for row in rows]
    ceilings = [row["oracle_confirmation_ceiling"] for row in rows]
    fractions = [row["stable_fraction"] for row in rows]
    summary = {
        "status": "offline_oracle_nontriviality_audit",
        "target_values_used": True,
        "task_sha256": _sha256(task_path),
        "vault_sha256": _sha256(vault_path),
        "budget": budget,
        "system_count": len(rows),
        "candidate_count": sum(row["candidate_count"] for row in rows),
        "stable_candidate_count_mean": mean(stable_counts) if stable_counts else None,
        "stable_candidate_count_quantiles": {
            str(probability): _quantile(stable_counts, probability)
            for probability in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
        "stable_fraction_mean": mean(fractions) if fractions else None,
        "oracle_confirmation_ceiling_mean": mean(ceilings) if ceilings else None,
        "budget_saturated_system_count": sum(row["budget_saturated"] for row in rows),
        "zero_stable_system_count": sum(row["stable_candidate_count"] == 0 for row in rows),
        "stable_count_histogram": dict(sorted(Counter(stable_counts).items())),
        "systems": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "budget",
                    "system_count",
                    "candidate_count",
                    "stable_candidate_count_mean",
                    "stable_fraction_mean",
                    "oracle_confirmation_ceiling_mean",
                    "budget_saturated_system_count",
                    "zero_stable_system_count",
                    "stable_candidate_count_quantiles",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=6)
    args = parser.parse_args()
    run(task_path=args.task, vault_path=args.vault, output_path=args.output, budget=args.budget)


if __name__ == "__main__":
    main()
