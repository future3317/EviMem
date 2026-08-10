from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_e53_rank_diagnostics import summarize


def _round(
    index: int,
    selected: str,
    probabilities: dict[str, float],
) -> dict[str, object]:
    return {
        "round_index": index,
        "selected_pair_id": selected,
        "selection_diagnostics": {
            "kind": "delta_hull_active_search",
            "candidate_pair_ids": list(probabilities),
            "final_stability_probabilities": probabilities,
            "selected_pair_id": selected,
        },
    }


def test_rank_diagnostics_use_equal_system_weighting(tmp_path: Path) -> None:
    payload = {
        "active_policies": ["delta_hull_active_search"],
        "systems": {
            "A-B": {
                "strategies": {
                    "delta_hull_active_search": {
                        "policy_decision_rounds": [
                            _round(1, "a", {"a": 0.9, "b": 0.8, "c": 0.1}),
                            _round(2, "b", {"b": 0.7, "c": 0.2}),
                        ]
                    }
                }
            },
            "C-D": {
                "strategies": {
                    "delta_hull_active_search": {
                        "policy_decision_rounds": [
                            _round(
                                1,
                                "x",
                                {"x": 0.9, "y": 0.5, "z": 0.4, "w": 0.3},
                            ),
                            _round(2, "z", {"y": 0.2, "z": 0.7, "w": 0.1}),
                        ]
                    }
                }
            },
        },
    }
    input_path = tmp_path / "fold.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    result = summarize(
        inputs=[input_path],
        output=tmp_path / "rank-summary.json",
        expected_system_count=2,
    )

    assert result["system_count"] == 2
    assert result["transition_count"] == 2
    assert result["equal_system"]["top_rank_preservation"] == 0.5
    assert result["equal_system"]["full_rank_preservation"] == 0.5
    assert result["equal_system"]["mean_absolute_membership_drift"] == pytest.approx(
        (0.1 + (0.3 + 0.3 + 0.2) / 3.0) / 2.0
    )
    assert result["pooled_candidate_transition_count"] == 5


def test_rank_diagnostics_reject_probability_roster_mismatch(tmp_path: Path) -> None:
    bad_round = _round(1, "a", {"a": 0.9, "b": 0.8})
    bad_round["selection_diagnostics"]["candidate_pair_ids"] = ["a"]
    payload = {
        "active_policies": ["delta_hull_active_search"],
        "systems": {
            "A-B": {
                "strategies": {
                    "delta_hull_active_search": {
                        "policy_decision_rounds": [bad_round, _round(2, "b", {"b": 0.7})]
                    }
                }
            }
        },
    }
    input_path = tmp_path / "bad.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate/probability roster"):
        summarize(inputs=[input_path], output=tmp_path / "summary.json", expected_system_count=1)


def test_rank_diagnostics_reject_duplicate_systems(tmp_path: Path) -> None:
    system = {
        "strategies": {
            "delta_hull_active_search": {
                "policy_decision_rounds": [
                    _round(1, "a", {"a": 0.9, "b": 0.8}),
                    _round(2, "b", {"b": 0.7}),
                ]
            }
        }
    }
    paths = []
    for index in range(2):
        path = tmp_path / f"fold{index}.json"
        path.write_text(
            json.dumps(
                {"active_policies": ["delta_hull_active_search"], "systems": {"A-B": system}}
            ),
            encoding="utf-8",
        )
        paths.append(path)
    with pytest.raises(ValueError, match="occurs twice"):
        summarize(inputs=paths, output=tmp_path / "summary.json", expected_system_count=1)
