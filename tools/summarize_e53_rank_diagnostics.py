"""Summarize observed-path Delta-Hull probability drift and rank stability."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

POLICY = "delta_hull_active_search"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probabilities(decision_round: dict[str, Any]) -> dict[str, float] | None:
    diagnostics = decision_round.get("selection_diagnostics")
    if diagnostics is None:
        return None
    if diagnostics.get("kind") != POLICY:
        raise ValueError("unexpected policy diagnostic in Delta-Hull trajectory")
    candidates = tuple(str(value) for value in diagnostics.get("candidate_pair_ids", ()))
    probabilities = {
        str(pair_id): float(value)
        for pair_id, value in diagnostics.get("final_stability_probabilities", {}).items()
    }
    if len(candidates) != len(set(candidates)) or set(candidates) != set(probabilities):
        raise ValueError("candidate/probability roster mismatch")
    if not candidates or not np.isfinite(tuple(probabilities.values())).all():
        raise ValueError("Delta-Hull probabilities must be non-empty and finite")
    selected = str(decision_round.get("selected_pair_id"))
    if selected not in probabilities:
        raise ValueError("selected action is absent from Delta-Hull probabilities")
    return probabilities


def _rank(probabilities: dict[str, float], candidates: set[str]) -> tuple[str, ...]:
    return tuple(sorted(candidates, key=lambda pair_id: (-probabilities[pair_id], pair_id)))


def _system_summary(rounds: list[dict[str, Any]]) -> dict[str, float | int] | None:
    probability_rounds = [_probabilities(row) for row in rounds]
    available = [row for row in probability_rounds if row is not None]
    if not available:
        return None
    if len(available) != len(probability_rounds):
        raise ValueError("Delta-Hull diagnostics are missing for only part of a trajectory")
    if len(available) < 2:
        raise ValueError("rank diagnostics need at least two decision states")

    drifts: list[float] = []
    top_preserved: list[float] = []
    full_preserved: list[float] = []
    candidate_transition_count = 0
    for before, after in zip(available, available[1:]):
        common = set(before) & set(after)
        if not common:
            raise ValueError("consecutive Delta-Hull states have no common legal candidate")
        before_rank = _rank(before, common)
        after_rank = _rank(after, common)
        drifts.append(float(np.mean([abs(after[x] - before[x]) for x in common])))
        top_preserved.append(float(before_rank[0] == after_rank[0]))
        full_preserved.append(float(before_rank == after_rank))
        candidate_transition_count += len(common)
    return {
        "transition_count": len(drifts),
        "candidate_transition_count": candidate_transition_count,
        "mean_absolute_membership_drift": float(np.mean(drifts)),
        "top_rank_preservation": float(np.mean(top_preserved)),
        "full_rank_preservation": float(np.mean(full_preserved)),
    }


def summarize(
    *,
    inputs: list[Path],
    output: Path,
    expected_system_count: int,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("E53 rank summaries must remain outside Git")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite rank summary: {output}")
    systems: dict[str, dict[str, float | int]] = {}
    input_hashes: dict[str, str] = {}
    unsupported_systems: list[str] = []
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(f"missing E53 trajectory: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if POLICY not in payload.get("active_policies", ()):
            raise ValueError(f"E53 trajectory lacks Delta-Hull: {path}")
        input_hashes[str(path.resolve())] = _sha256(path)
        for system, system_payload in payload.get("systems", {}).items():
            if system in systems or system in unsupported_systems:
                raise ValueError(f"chemical system occurs twice: {system}")
            strategy = system_payload.get("strategies", {}).get(POLICY)
            if strategy is None:
                raise ValueError(f"missing Delta-Hull strategy for {system}")
            system_result = _system_summary(list(strategy.get("policy_decision_rounds", ())))
            if system_result is None:
                unsupported_systems.append(str(system))
            else:
                systems[str(system)] = system_result
    if len(systems) != expected_system_count:
        raise ValueError(
            f"expected {expected_system_count} supported systems, found {len(systems)}"
        )
    metrics = (
        "mean_absolute_membership_drift",
        "top_rank_preservation",
        "full_rank_preservation",
    )
    result = {
        "schema_version": 1,
        "status": "e53_observed_path_rank_diagnostics_complete",
        "policy": POLICY,
        "weighting": "equal exact chemical system",
        "system_count": len(systems),
        "unsupported_system_count": len(unsupported_systems),
        "unsupported_systems": sorted(unsupported_systems),
        "transition_count": int(sum(row["transition_count"] for row in systems.values())),
        "pooled_candidate_transition_count": int(
            sum(row["candidate_transition_count"] for row in systems.values())
        ),
        "equal_system": {
            metric: float(np.mean([float(row[metric]) for row in systems.values()]))
            for metric in metrics
        },
        "system_summaries": systems,
        "input_sha256": input_hashes,
        "interpretation": "observed-path diagnostic, not a counterfactual certificate",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-systems", type=int, default=217)
    args = parser.parse_args()
    result = summarize(
        inputs=args.inputs,
        output=args.output,
        expected_system_count=args.expected_systems,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "system_count": result["system_count"],
                "equal_system": result["equal_system"],
            }
        )
    )


if __name__ == "__main__":
    main()
