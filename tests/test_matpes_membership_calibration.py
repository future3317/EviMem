from __future__ import annotations

import json
from pathlib import Path

import pytest

import tools.summarize_matpes_membership_calibration as membership_calibration
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


def _scored_strategy(
    system: str,
    *,
    labels: dict[str, int],
    probabilities: dict[str, float],
    selected: str,
) -> dict:
    event = {
        "round_index": 1,
        "selected_pair_id": selected,
        "pre_reveal_state_checksum": f"state-{system}",
        "selection_diagnostics": {
            "candidate_pair_ids": list(labels),
            "selected_pair_id": selected,
            "final_stability_probabilities": probabilities,
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


def test_membership_summary_reports_equal_system_and_top3_metrics(tmp_path: Path) -> None:
    payload = {
        "task_sha256": "task",
        "systems": {
            "A-B": {
                "strategies": {
                    "delta_hull_active_search": _scored_strategy(
                        "A-B",
                        labels={"a": 1, "b": 1, "c": 0, "d": 0},
                        probabilities={"a": 0.9, "b": 0.9, "c": 0.1, "d": 0.1},
                        selected="a",
                    )
                }
            },
            "C-D": {
                "strategies": {
                    "delta_hull_active_search": _scored_strategy(
                        "C-D",
                        labels={"z": 1},
                        probabilities={"z": 0.1},
                        selected="z",
                    )
                }
            },
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    result = summarize(
        input_paths=(input_path,),
        output=tmp_path / "summary.json",
        bin_count=5,
        bootstrap_count=20,
    )
    group = result["groups"]["task:delta_hull_active_search"]
    assert result["schema_version"] == 2
    assert group["all_candidates"]["metrics"]["brier_score"] == pytest.approx(0.17)
    assert group["all_candidates"]["equal_system_metrics"]["brier_score"] == pytest.approx(0.41)
    equal_system_bins = group["all_candidates"]["equal_system_reliability_bins"]
    assert equal_system_bins[0]["record_count"] == 3
    assert equal_system_bins[0]["mean_predicted_probability"] == pytest.approx(0.1)
    assert equal_system_bins[0]["empirical_frequency"] == pytest.approx(2.0 / 3.0)
    assert equal_system_bins[4]["record_count"] == 2
    assert equal_system_bins[4]["mean_predicted_probability"] == pytest.approx(0.9)
    assert equal_system_bins[4]["empirical_frequency"] == pytest.approx(1.0)
    assert group["top3_candidates"]["metrics"]["record_count"] == 4
    assert "equal_system_cluster_bootstrap_95" in group["top3_candidates"]


def test_equal_system_bootstrap_preserves_duplicate_system_multiplicity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _DuplicateDrawRng:
        def choice(self, _systems: list[str], **_kwargs: object) -> list[str]:
            return ["A", "A", "B"]

    monkeypatch.setattr(
        membership_calibration.np.random,
        "default_rng",
        lambda _seed: _DuplicateDrawRng(),
    )
    rows = [
        {"system": "A", "probability": 0.0, "label": 0},
        {"system": "B", "probability": 1.0, "label": 0},
        {"system": "C", "probability": 0.0, "label": 0},
    ]

    intervals = membership_calibration._cluster_bootstrap(
        rows,
        bin_count=2,
        bootstrap_count=1,
        seed=7,
        equal_system=True,
    )

    assert intervals["brier_score"]["lower_95"] == pytest.approx(1.0 / 3.0)
    assert intervals["brier_score"]["upper_95"] == pytest.approx(1.0 / 3.0)


def test_decision_top_k_keeps_selected_then_breaks_score_ties_by_id() -> None:
    rows = [
        {
            "system": "A-B",
            "state_checksum": "state",
            "pair_id": "z",
            "probability": 0.1,
            "label": 1,
            "selected": True,
        },
        {
            "system": "A-B",
            "state_checksum": "state",
            "pair_id": "c",
            "probability": 0.9,
            "label": 0,
            "selected": False,
        },
        {
            "system": "A-B",
            "state_checksum": "state",
            "pair_id": "b",
            "probability": 0.9,
            "label": 1,
            "selected": False,
        },
        {
            "system": "A-B",
            "state_checksum": "state",
            "pair_id": "a",
            "probability": 0.9,
            "label": 0,
            "selected": False,
        },
    ]
    assert hasattr(membership_calibration, "_decision_top_k")
    assert [
        row["pair_id"] for row in membership_calibration._decision_top_k(rows, k=3)
    ] == ["z", "a", "b"]
