"""Freeze the state set for the Source-Rollout numerical opportunity-cost audit.

This planner reads only completed development result records.  It does not
read a task oracle and therefore cannot reveal an unqueried target outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rollout(payload: dict[str, Any], system: str) -> dict[str, Any]:
    return payload["systems"][system]["strategies"]["source_rollout_delta_hull"]


def _source(payload: dict[str, Any], system: str) -> dict[str, Any]:
    return payload["systems"][system]["strategies"]["source_margin"]


def build_plan(
    *,
    sarr_result_path: Path,
    mc512_result_path: Path,
    mc1024_result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("numerical-audit plan must remain outside Git")
    if output_path.exists():
        raise FileExistsError(output_path)
    sarr = json.loads(sarr_result_path.read_text(encoding="utf-8"))
    mc512 = json.loads(mc512_result_path.read_text(encoding="utf-8"))
    mc1024 = json.loads(mc1024_result_path.read_text(encoding="utf-8"))
    systems = tuple(sorted(sarr["systems"]))
    if set(mc512["systems"]) != set(systems) or set(mc1024["systems"]) != set(systems):
        raise ValueError("SARR and pre-SARR system sets disagree")
    if sarr["evaluation_systems_accessed"]:
        raise ValueError("numerical audit accepts development-only SARR output")

    reasons: dict[tuple[str, int], set[str]] = defaultdict(set)
    for system in systems:
        rollout = _rollout(sarr, system)
        source = _source(sarr, system)
        decision_rounds = rollout.get("policy_decision_rounds")
        if not isinstance(decision_rounds, list) or len(decision_rounds) != len(
            rollout["selected_pair_ids"]
        ):
            raise ValueError("SARR decision diagnostics are incomplete")
        final_difference = (
            rollout["oracle_pool_confirmed_discoveries"]
            - source["oracle_pool_confirmed_discoveries"]
        )
        if final_difference > 0:
            final_reason = "final_win_system"
        elif final_difference < 0:
            final_reason = "final_loss_system"
        else:
            final_reason = None
        for index, event in enumerate(decision_rounds, start=1):
            diagnostic = event.get("selection_diagnostics")
            if not isinstance(diagnostic, dict) or diagnostic.get("kind") != "source_rollout_sarr":
                raise ValueError("SARR decision diagnostic schema is invalid")
            key = (system, index)
            if diagnostic["selected_pair_id"] != diagnostic["source_pair_id"]:
                reasons[key].add("sarr_deviation")
            if diagnostic.get("fallback_reason") == "no_positive_simultaneous_lower_bound":
                positive_but_unresolved = any(
                    pair_id != diagnostic["source_pair_id"]
                    and float(diagnostic["mean_advantages_over_source"][pair_id]) > 0.0
                    and float(diagnostic["simultaneous_lower_bounds"][pair_id]) <= 0.0
                    for pair_id in diagnostic["candidate_pair_ids"]
                )
                if positive_but_unresolved:
                    reasons[key].add("positive_but_simultaneously_unresolved")
            if final_reason is not None:
                reasons[key].add(final_reason)
        left = _rollout(mc512, system)["selected_pair_ids"]
        right = _rollout(mc1024, system)["selected_pair_ids"]
        if len(left) != len(right) or len(left) != len(decision_rounds):
            raise ValueError("pre-SARR trajectory lengths disagree")
        for index, (left_id, right_id) in enumerate(zip(left, right, strict=True), start=1):
            if left_id != right_id:
                reasons[(system, index)].add("pre_sarr_mc512_mc1024_disagreement")

    states = [
        {"chemical_system": system, "round_index": round_index, "reasons": sorted(values)}
        for (system, round_index), values in sorted(reasons.items())
    ]
    plan = {
        "schema_version": 1,
        "status": "frozen_development_numerical_audit_plan",
        "high_precision_posterior_sample_count": 8192,
        "sobol_scramble_count": 16,
        "sarr_result_sha256": _sha256(sarr_result_path),
        "pre_sarr_mc512_result_sha256": _sha256(mc512_result_path),
        "pre_sarr_mc1024_result_sha256": _sha256(mc1024_result_path),
        "task_sha256": sarr["task_sha256"],
        "evaluation_systems_accessed": False,
        "state_count": len(states),
        "states": states,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sarr-result", type=Path, required=True)
    parser.add_argument("--pre-sarr-mc512-result", type=Path, required=True)
    parser.add_argument("--pre-sarr-mc1024-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan = build_plan(
        sarr_result_path=args.sarr_result,
        mc512_result_path=args.pre_sarr_mc512_result,
        mc1024_result_path=args.pre_sarr_mc1024_result,
        output_path=args.output,
    )
    print(f"output={args.output} states={plan['state_count']}")


if __name__ == "__main__":
    main()
