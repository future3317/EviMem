from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_e52_two_step_equivalence import summarize


def _strategy(
    *,
    kind: str,
    selected: str,
    scores: dict[str, float],
) -> dict[str, object]:
    candidate_ids = list(scores)
    diagnostics: dict[str, object] = {
        "kind": kind,
        "candidate_pair_ids": candidate_ids,
        "delta_hull_action_id": "a",
    }
    if kind == "delta_hull_anchored_rollout":
        diagnostics["rollout_scores"] = list(scores.values())
    else:
        diagnostics["two_step_scores"] = scores
    return {
        "policy_decision_rounds": [
            {
                "selected_pair_id": selected,
                "selection_diagnostics": diagnostics,
            }
        ]
    }


def test_summarize_two_step_equivalence_uses_centered_scores(tmp_path: Path) -> None:
    payload = {
        "task_sha256": "task-sha",
        "active_policies": [
            "delta_hull_anchored_rollout",
            "protocol_hull_knowledge_gradient",
        ],
        "systems": {
            "A-B": {
                "strategies": {
                    "delta_hull_anchored_rollout": _strategy(
                        kind="delta_hull_anchored_rollout",
                        selected="b",
                        scores={"a": 0.4, "b": 0.7},
                    ),
                    "protocol_hull_knowledge_gradient": _strategy(
                        kind="protocol_hull_knowledge_gradient",
                        selected="b",
                        scores={"a": 1.4, "b": 1.7},
                    ),
                }
            }
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "summary.json"

    result = summarize(input_path=input_path, output=output)

    assert result["system_count"] == 1
    assert result["action_agreement_rate"] == 1.0
    assert result["mean_absolute_q_difference"] == pytest.approx(1.0)
    assert result["max_centered_absolute_q_difference"] == pytest.approx(0.0)
    assert result["mean_absolute_headroom_difference"] == pytest.approx(0.0)
    assert json.loads(output.read_text(encoding="utf-8"))["status"].endswith(
        "complete"
    )


def test_summarize_two_step_equivalence_rejects_roster_mismatch(tmp_path: Path) -> None:
    payload = {
        "task_sha256": "task-sha",
        "active_policies": [
            "delta_hull_anchored_rollout",
            "protocol_hull_knowledge_gradient",
        ],
        "systems": {
            "A-B": {
                "strategies": {
                    "delta_hull_anchored_rollout": _strategy(
                        kind="delta_hull_anchored_rollout",
                        selected="a",
                        scores={"a": 0.7, "b": 0.4},
                    ),
                    "protocol_hull_knowledge_gradient": _strategy(
                        kind="protocol_hull_knowledge_gradient",
                        selected="a",
                        scores={"a": 0.7, "c": 0.4},
                    ),
                }
            }
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="candidate roster mismatch"):
        summarize(input_path=input_path, output=tmp_path / "summary.json")
