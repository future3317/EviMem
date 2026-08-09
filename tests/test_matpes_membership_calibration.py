from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_matpes_membership_calibration import summarize


def _strategy(system: str, *, selected: str) -> dict:
    labels = {f"{system}-a": True, f"{system}-b": False}
    event = {
        "round_index": 1,
        "selected_pair_id": selected,
        "pre_reveal_state_checksum": f"state-{system}",
        "selection_diagnostics": {
            "candidate_pair_ids": list(labels),
            "selected_pair_id": selected,
            "final_stability_probabilities": {
                f"{system}-a": 0.8,
                f"{system}-b": 0.2,
            },
        },
    }
    return {
        "oracle_pool_final_labels_by_pair_id": labels,
        "policy_decision_rounds": [event],
    }


def test_membership_summary_deduplicates_repeated_states(tmp_path: Path) -> None:
    payload = {
        "task_sha256": "task",
        "systems": {
            system: {
                "strategies": {
                    "delta_hull_active_search": _strategy(system, selected=f"{system}-a")
                }
            }
            for system in ("A-B", "C-D")
        },
    }
    first = tmp_path / "b1.json"
    second = tmp_path / "b2.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    second.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "summary.json"

    result = summarize(
        input_paths=(first, second),
        output=output,
        bin_count=5,
        bootstrap_count=20,
    )
    group = result["groups"]["task:delta_hull_active_search"]
    assert group["unique_state_count"] == 2
    assert group["all_candidates"]["metrics"]["record_count"] == 4
    assert group["all_candidates"]["metrics"]["brier_score"] == pytest.approx(0.04)
    assert group["selected_actions"]["metrics"]["record_count"] == 2
    assert group["all_candidates"]["metrics"]["roc_auc"] == 1.0


def test_membership_summary_rejects_policy_evaluator_join_mismatch(tmp_path: Path) -> None:
    strategy = _strategy("A-B", selected="A-B-a")
    strategy["oracle_pool_final_labels_by_pair_id"].pop("A-B-b")
    payload = {
        "task_sha256": "task",
        "systems": {"A-B": {"strategies": {"delta_hull_active_search": strategy}}},
    }
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        summarize(input_paths=(input_path,), output=tmp_path / "summary.json")
    except ValueError as exc:
        assert "candidate/label mismatch" in str(exc)
    else:
        raise AssertionError("mismatched policy/evaluator join should fail")
