from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.run_matpes_cal_style_campaign import POLICIES, build_units
from tools.summarize_matpes_cal_style import summarize


def _write_input(path: Path, payload: object | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload if payload is not None else {}) + "\n", encoding="utf-8")


def _strategy(selected: list[str], positives: set[str]) -> dict[str, object]:
    return {
        "selected_pair_ids": selected,
        "oracle_pool_final_labels_by_pair_id": {
            pair_id: pair_id in positives for pair_id in selected
        },
        "policy_decision_rounds": [
            {
                "selection_diagnostics": {
                    "kind": "cal_style_hull_entropy",
                    "wall_time_seconds": 0.25 + index,
                    "state_candidate_count": 4 + index,
                    "evaluation_composition_count": 3,
                    "posterior_sample_count": 200,
                    "fantasy_count": 10,
                    "relative_ridge": 1e-10,
                }
            }
            for index in range(6)
        ],
    }


def _unit(path: Path, *, fold: int, system: str) -> None:
    ids = [f"{system}-{index}" for index in range(6)]
    payload = {
        "task_sha256": "task",
        "oracle_vault_sha256": "vault",
        "active_policies": list(POLICIES),
        "query_systems": [system],
        "transport_fit_systems": [f"fit-{fold}"],
        "transport_fit_system_count": 1,
        "transport_fit_and_query_systems_disjoint": True,
        "config": {
            "query_budget": 6,
            "posterior_sample_count": 200,
            "fantasy_count": 10,
            "hull_backend": "fixed_composition",
            "crossfit_manifest_sha256": "crossfit",
            "crossfit_fold_index": fold,
        },
        "systems": {
            system: {
                "budget": 6,
                "strategies": {
                    "posterior_mean_target_margin": _strategy(ids, set()),
                    "delta_hull_active_search": _strategy(ids, {ids[0], ids[1]}),
                    "cal_style_hull_entropy": _strategy(
                        ids, {ids[0]} if fold < 3 else set()
                    ),
                },
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_cal_campaign_builds_five_b6_development_units(tmp_path: Path) -> None:
    full_pool = tmp_path / "inputs"
    split_manifest = tmp_path / "split"
    for name in (
        "matpes-e52-pool-100-task.json",
        "matpes-e52-pool-100-vault.json",
        "matpes-e52-pool-100-crossfit.json",
    ):
        _write_input(full_pool / name)
    _write_input(split_manifest / "matpes-e52-secondary-confirmation-crossfit.json")

    units = build_units(
        full_pool_root=full_pool,
        split_manifest_root=split_manifest,
        secondary_task=None,
        secondary_vault=None,
        output_root=tmp_path / "outputs",
        runner=Path("runner.py"),
        stage="development",
        posterior_sample_count=200,
        fantasy_count=10,
        selection_timeout_seconds=7200.0,
    )

    assert len(units) == 5
    assert all(unit.identity["budget"] == 6 for unit in units)
    assert all(tuple(unit.identity["policies"]) == POLICIES for unit in units)
    assert all("--fantasy-count" in unit.command for unit in units)
    assert all("10" in unit.command for unit in units)


def test_cal_summary_reports_prefixes_contrasts_and_runtime_metadata(tmp_path: Path) -> None:
    development = tmp_path / "development"
    for fold in range(5):
        _unit(development / f"fold{fold + 1}-b6.json", fold=fold, system=f"S{fold}")

    result = summarize(
        development_root=development,
        secondary_path=None,
        output=tmp_path / "summary.json",
        expected_development_system_count=5,
        expected_secondary_system_count=None,
        randomization_draws=1_000,
    )

    panel = result["panels"]["development"]
    assert panel["budgets"]["1"]["absolute_mean_T"]["delta_hull_active_search"] == 1.0
    assert panel["budgets"]["1"]["absolute_mean_T"]["cal_style_hull_entropy"] == 0.6
    assert panel["budgets"]["1"]["contrasts"]["delta_minus_cal"]["mean_effect"] == 0.4
    assert panel["integrated_budget_effects"]["delta_minus_cal"]["mean_effect"] > 0.0
    assert panel["cal_diagnostics"]["state_count"] == 30
    assert panel["cal_diagnostics"]["posterior_sample_counts"] == [200]
    assert panel["cal_diagnostics"]["fantasy_counts"] == [10]


def test_cal_summary_rejects_missing_diagnostic(tmp_path: Path) -> None:
    development = tmp_path / "development"
    for fold in range(5):
        _unit(development / f"fold{fold + 1}-b6.json", fold=fold, system=f"S{fold}")
    path = development / "fold1-b6.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["systems"]["S0"]["strategies"]["cal_style_hull_entropy"][
        "policy_decision_rounds"
    ] = []
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="CAL diagnostics"):
        summarize(
            development_root=development,
            secondary_path=None,
            output=tmp_path / "summary.json",
            expected_development_system_count=5,
            expected_secondary_system_count=None,
            randomization_draws=1_000,
        )
