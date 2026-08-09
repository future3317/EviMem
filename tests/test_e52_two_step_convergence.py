from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_e52_two_step_convergence import summarize


def _strategy(policy: str, scores: list[float], selected: str) -> dict[str, object]:
    diagnostics: dict[str, object] = {
        "candidate_pair_ids": ["a", "b"],
        "delta_hull_action_id": "a",
    }
    if policy == "delta_hull_anchored_rollout":
        diagnostics["rollout_scores"] = scores
    else:
        diagnostics["two_step_scores"] = dict(zip(("a", "b"), scores, strict=True))
    return {
        "policy_decision_rounds": [
            {"selected_pair_id": selected, "selection_diagnostics": diagnostics}
        ]
    }


def _payload(offset: float) -> dict[str, object]:
    return {
        "task_sha256": "task",
        "active_policies": [
            "delta_hull_anchored_rollout",
            "protocol_hull_knowledge_gradient",
        ],
        "systems": {
            "A-B": {
                "strategies": {
                    "delta_hull_anchored_rollout": _strategy(
                        "delta_hull_anchored_rollout", [0.4 + offset, 0.6 + offset], "b"
                    ),
                    "protocol_hull_knowledge_gradient": _strategy(
                        "protocol_hull_knowledge_gradient", [0.4, 0.6], "b"
                    ),
                }
            }
        },
    }


def test_convergence_summary_ignores_additive_q_offset(tmp_path: Path) -> None:
    fast = tmp_path / "fast.json"
    high = tmp_path / "high.json"
    fast.write_text(json.dumps(_payload(1.0)), encoding="utf-8")
    high.write_text(json.dumps(_payload(2.0)), encoding="utf-8")

    result = summarize(fast_path=fast, high_path=high, output=tmp_path / "summary.json")

    assert result["high_cross_method_action_agreement_rate"] == 1.0
    assert result["fast_to_high_action_agreement_rate"][
        "delta_hull_anchored_rollout"
    ] == 1.0
    row = result["rows"][0]
    assert row["fast_to_high"]["delta_hull_anchored_rollout"][
        "centered_max_absolute"
    ] == pytest.approx(0.0)
