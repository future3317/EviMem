"""Compare fast and high-precision E52 two-step numerical audits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ANCHORED = "delta_hull_anchored_rollout"
KG = "protocol_hull_knowledge_gradient"
POLICIES = (ANCHORED, KG)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(result: dict[str, Any], policy: str) -> dict[str, Any]:
    event = result["strategies"][policy]["policy_decision_rounds"][0]
    diagnostics = event["selection_diagnostics"]
    if diagnostics is None:
        raise ValueError("convergence roster contains an unsupported fallback")
    candidate_ids = tuple(str(value) for value in diagnostics["candidate_pair_ids"])
    if policy == ANCHORED:
        scores = np.asarray(diagnostics["rollout_scores"], dtype=float)
    else:
        scores = np.asarray(
            [diagnostics["two_step_scores"][pair_id] for pair_id in candidate_ids],
            dtype=float,
        )
    if len(scores) != len(candidate_ids):
        raise ValueError("candidate and score dimensions differ")
    delta_action = str(diagnostics["delta_hull_action_id"])
    delta_index = candidate_ids.index(delta_action)
    return {
        "candidate_ids": candidate_ids,
        "scores": scores,
        "selected_action": str(event["selected_pair_id"]),
        "delta_action": delta_action,
        "headroom": float(np.max(scores) - scores[delta_index]),
    }


def _score_difference(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left - right
    centered = (left - np.mean(left)) - (right - np.mean(right))
    return {
        "mean_absolute": float(np.mean(np.abs(difference))),
        "max_absolute": float(np.max(np.abs(difference))),
        "centered_mean_absolute": float(np.mean(np.abs(centered))),
        "centered_max_absolute": float(np.max(np.abs(centered))),
    }


def summarize(*, fast_path: Path, high_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("convergence summaries must remain outside Git")
    fast = json.loads(fast_path.read_text(encoding="utf-8"))
    high = json.loads(high_path.read_text(encoding="utf-8"))
    if tuple(fast["active_policies"]) != POLICIES:
        raise ValueError("fast audit has wrong policy roster")
    if tuple(high["active_policies"]) != POLICIES:
        raise ValueError("high-precision audit has wrong policy roster")
    if fast["task_sha256"] != high["task_sha256"]:
        raise ValueError("fast and high-precision audits use different tasks")
    systems = sorted(high["systems"])
    if not systems or not set(systems) <= set(fast["systems"]):
        raise ValueError("high-precision roster is not a nonempty fast-audit subset")
    rows: list[dict[str, Any]] = []
    for system in systems:
        fast_decisions = {
            policy: _decision(fast["systems"][system], policy) for policy in POLICIES
        }
        high_decisions = {
            policy: _decision(high["systems"][system], policy) for policy in POLICIES
        }
        roster = high_decisions[ANCHORED]["candidate_ids"]
        if any(
            decision["candidate_ids"] != roster
            for decision in (*fast_decisions.values(), *high_decisions.values())
        ):
            raise ValueError(f"candidate roster mismatch for {system}")
        within_precision = {
            policy: {
                "action_agreement": (
                    fast_decisions[policy]["selected_action"]
                    == high_decisions[policy]["selected_action"]
                ),
                "delta_action_agreement": (
                    fast_decisions[policy]["delta_action"]
                    == high_decisions[policy]["delta_action"]
                ),
                "absolute_headroom_difference": abs(
                    fast_decisions[policy]["headroom"]
                    - high_decisions[policy]["headroom"]
                ),
                **_score_difference(
                    fast_decisions[policy]["scores"],
                    high_decisions[policy]["scores"],
                ),
            }
            for policy in POLICIES
        }
        high_cross = _score_difference(
            high_decisions[ANCHORED]["scores"], high_decisions[KG]["scores"]
        )
        rows.append(
            {
                "system": system,
                "fast_cross_method_action_agreement": (
                    fast_decisions[ANCHORED]["selected_action"]
                    == fast_decisions[KG]["selected_action"]
                ),
                "high_cross_method_action_agreement": (
                    high_decisions[ANCHORED]["selected_action"]
                    == high_decisions[KG]["selected_action"]
                ),
                "high_cross_method_delta_action_agreement": (
                    high_decisions[ANCHORED]["delta_action"]
                    == high_decisions[KG]["delta_action"]
                ),
                "high_cross_method_absolute_headroom_difference": abs(
                    high_decisions[ANCHORED]["headroom"]
                    - high_decisions[KG]["headroom"]
                ),
                "high_cross_method_score_difference": high_cross,
                "fast_to_high": within_precision,
            }
        )
    result = {
        "schema_version": 1,
        "status": "e52_two_step_high_precision_convergence_complete",
        "fast_input_sha256": _sha256(fast_path),
        "high_input_sha256": _sha256(high_path),
        "task_sha256": fast["task_sha256"],
        "system_count": len(rows),
        "fast_cross_method_action_agreement_rate": float(
            np.mean([row["fast_cross_method_action_agreement"] for row in rows])
        ),
        "high_cross_method_action_agreement_rate": float(
            np.mean([row["high_cross_method_action_agreement"] for row in rows])
        ),
        "high_cross_method_delta_action_agreement_rate": float(
            np.mean([row["high_cross_method_delta_action_agreement"] for row in rows])
        ),
        "high_cross_method_mean_centered_absolute_q_difference": float(
            np.mean(
                [
                    row["high_cross_method_score_difference"]["centered_mean_absolute"]
                    for row in rows
                ]
            )
        ),
        "high_cross_method_max_centered_absolute_q_difference": float(
            np.max(
                [
                    row["high_cross_method_score_difference"]["centered_max_absolute"]
                    for row in rows
                ]
            )
        ),
        "high_cross_method_mean_absolute_headroom_difference": float(
            np.mean(
                [row["high_cross_method_absolute_headroom_difference"] for row in rows]
            )
        ),
        "fast_to_high_action_agreement_rate": {
            policy: float(
                np.mean([row["fast_to_high"][policy]["action_agreement"] for row in rows])
            )
            for policy in POLICIES
        },
        "rows": rows,
        "interpretation": (
            "Both policies target the same exact two-step Bellman value. This audit measures "
            "whether their finite numerical integrations converge sufficiently for matched "
            "actions and centered Q-values; it is not a policy-effect comparison."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", type=Path, required=True)
    parser.add_argument("--high", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(fast_path=args.fast, high_path=args.high, output=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "systems": result["system_count"],
                "fast_action_agreement": result[
                    "fast_cross_method_action_agreement_rate"
                ],
                "high_action_agreement": result[
                    "high_cross_method_action_agreement_rate"
                ],
                "high_max_centered_q_difference": result[
                    "high_cross_method_max_centered_absolute_q_difference"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
