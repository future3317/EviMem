"""Compare two numerical implementations of the same two-step Bellman policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ANCHORED = "delta_hull_anchored_rollout"
KG = "protocol_hull_knowledge_gradient"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(*, input_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("equivalence summaries must remain outside Git")
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not {ANCHORED, KG} <= set(payload["active_policies"]):
        raise ValueError("equivalence audit requires anchored rollout and Hull-KG")
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for system, system_result in payload["systems"].items():
        strategies = system_result["strategies"]
        anchored_event = strategies[ANCHORED]["policy_decision_rounds"][0]
        kg_event = strategies[KG]["policy_decision_rounds"][0]
        anchored = anchored_event["selection_diagnostics"]
        kg = kg_event["selection_diagnostics"]
        if anchored is None or kg is None:
            if (
                anchored is not None
                or kg is not None
                or bool(system_result.get("transport_element_support"))
            ):
                raise ValueError(f"unexpected missing two-step diagnostics for {system}")
            excluded.append(
                {
                    "system": system,
                    "reason": "outside_transport_element_support_common_fallback",
                    "anchored_action": anchored_event["selected_pair_id"],
                    "kg_action": kg_event["selected_pair_id"],
                    "action_agreement": (
                        anchored_event["selected_pair_id"] == kg_event["selected_pair_id"]
                    ),
                }
            )
            continue
        anchored_ids = tuple(anchored["candidate_pair_ids"])
        kg_ids = tuple(kg["candidate_pair_ids"])
        if anchored_ids != kg_ids:
            raise ValueError(f"candidate roster mismatch for {system}")
        anchored_scores = np.asarray(anchored["rollout_scores"], dtype=float)
        kg_scores = np.asarray([kg["two_step_scores"][pair_id] for pair_id in kg_ids])
        if len(anchored_scores) != len(kg_scores):
            raise ValueError(f"score dimension mismatch for {system}")
        delta_id = str(anchored["delta_hull_action_id"])
        if delta_id != str(kg["delta_hull_action_id"]):
            raise ValueError(f"one-step Delta-Hull action mismatch for {system}")
        delta_index = anchored_ids.index(delta_id)
        anchored_headroom = float(np.max(anchored_scores) - anchored_scores[delta_index])
        kg_headroom = float(np.max(kg_scores) - kg_scores[delta_index])
        difference = anchored_scores - kg_scores
        rows.append(
            {
                "system": system,
                "candidate_count": len(anchored_ids),
                "anchored_action": anchored_event["selected_pair_id"],
                "kg_action": kg_event["selected_pair_id"],
                "action_agreement": (
                    anchored_event["selected_pair_id"] == kg_event["selected_pair_id"]
                ),
                "delta_hull_action": delta_id,
                "anchored_headroom": anchored_headroom,
                "kg_headroom": kg_headroom,
                "absolute_headroom_difference": abs(anchored_headroom - kg_headroom),
                "mean_absolute_q_difference": float(np.mean(np.abs(difference))),
                "max_absolute_q_difference": float(np.max(np.abs(difference))),
                "centered_mean_absolute_q_difference": float(
                    np.mean(np.abs((anchored_scores - anchored_scores.mean()) - (kg_scores - kg_scores.mean())))
                ),
                "centered_max_absolute_q_difference": float(
                    np.max(np.abs((anchored_scores - anchored_scores.mean()) - (kg_scores - kg_scores.mean())))
                ),
            }
        )
    if not rows:
        raise ValueError("no transport-supported two-step states found")
    result = {
        "schema_version": 1,
        "status": "e52_two_step_equivalence_audit_complete",
        "input_path": str(input_path.resolve()),
        "input_sha256": _sha256(input_path),
        "task_sha256": payload["task_sha256"],
        "roster_system_count": len(payload["systems"]),
        "system_count": len(rows),
        "transport_supported_system_count": len(rows),
        "common_fallback_system_count": len(excluded),
        "action_agreement_count": sum(row["action_agreement"] for row in rows),
        "action_agreement_rate": float(np.mean([row["action_agreement"] for row in rows])),
        "mean_absolute_q_difference": float(
            np.mean([row["mean_absolute_q_difference"] for row in rows])
        ),
        "max_absolute_q_difference": float(
            np.max([row["max_absolute_q_difference"] for row in rows])
        ),
        "mean_centered_absolute_q_difference": float(
            np.mean([row["centered_mean_absolute_q_difference"] for row in rows])
        ),
        "max_centered_absolute_q_difference": float(
            np.max([row["centered_max_absolute_q_difference"] for row in rows])
        ),
        "mean_absolute_headroom_difference": float(
            np.mean([row["absolute_headroom_difference"] for row in rows])
        ),
        "rows": rows,
        "excluded_common_fallback_rows": excluded,
        "interpretation": (
            "Both methods target the same exact two-step Bellman value; discrepancies measure "
            "finite numerical integration and argmax error, not a different planning class. "
            "Systems outside transport element support use a common fallback and are excluded "
            "from Q-value discrepancies."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(input_path=args.input, output=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "systems": result["system_count"],
                "action_agreement_rate": result["action_agreement_rate"],
                "max_centered_absolute_q_difference": result[
                    "max_centered_absolute_q_difference"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
