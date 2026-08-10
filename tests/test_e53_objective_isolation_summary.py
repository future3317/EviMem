from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.summarize_e53_objective_isolation import POLICIES, summarize


def _strategy(selected: list[str], positives: set[str]) -> dict[str, object]:
    return {
        "selected_pair_ids": selected,
        "oracle_pool_final_labels_by_pair_id": {
            pair_id: pair_id in positives for pair_id in selected
        },
    }


def _unit(path: Path, *, fold: int, system: str, local_positive: bool) -> None:
    ids = [f"{system}-{index}" for index in range(6)]
    payload = {
        "task_sha256": "task",
        "oracle_vault_sha256": "vault",
        "active_policies": list(POLICIES),
        "config": {"query_budget": 6, "posterior_sample_count": 1024},
        "systems": {
            system: {
                "budget": 6,
                "strategies": {
                    "posterior_mean_target_margin": _strategy(ids, set()),
                    "matched_local_hull_probability": _strategy(
                        ids, {ids[0]} if local_positive else set()
                    ),
                    "delta_hull_active_search": _strategy(ids, {ids[0], ids[1]}),
                },
            }
        },
        "fold": fold,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_e53_summary_reports_absolute_prefixes_and_matched_contrasts(tmp_path: Path) -> None:
    development = tmp_path / "development"
    for fold in range(5):
        _unit(
            development / f"fold{fold + 1}-b6.json",
            fold=fold,
            system=f"S{fold}",
            local_positive=fold < 3,
        )
    secondary = tmp_path / "secondary" / "heldout-b6.json"
    _unit(secondary, fold=0, system="H0", local_positive=False)

    result = summarize(
        development_root=development,
        secondary_path=secondary,
        output=tmp_path / "summary.json",
        expected_development_system_count=5,
        expected_secondary_system_count=1,
        randomization_draws=1_000,
    )

    b1 = result["panels"]["development"]["budgets"]["1"]
    b2 = result["panels"]["development"]["budgets"]["2"]
    assert b1["absolute_mean_T"]["delta_hull_active_search"] == 1.0
    assert b2["absolute_mean_T"]["delta_hull_active_search"] == 2.0
    assert b1["absolute_mean_T"]["matched_local_hull_probability"] == 0.6
    assert b2["contrasts"]["delta_minus_local"]["mean_effect"] == 1.4
    assert result["panels"]["development"]["system_count"] == 5
    assert result["panels"]["secondary"]["system_count"] == 1
    assert result["panels"]["development"]["integrated_budget_effects"][
        "delta_minus_local"
    ]["mean_effect"] > 0.0


def test_e53_summary_rejects_wrong_policy_roster(tmp_path: Path) -> None:
    development = tmp_path / "development"
    for fold in range(5):
        path = development / f"fold{fold + 1}-b6.json"
        _unit(path, fold=fold, system=f"S{fold}", local_positive=False)
    payload = json.loads((development / "fold1-b6.json").read_text(encoding="utf-8"))
    payload["active_policies"] = list(POLICIES[:-1])
    (development / "fold1-b6.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="policy roster"):
        summarize(
            development_root=development,
            secondary_path=None,
            output=tmp_path / "summary.json",
            expected_development_system_count=5,
            expected_secondary_system_count=None,
            randomization_draws=1_000,
        )


def test_e53_summary_rejects_duplicate_systems_across_folds(tmp_path: Path) -> None:
    development = tmp_path / "development"
    for fold in range(5):
        _unit(
            development / f"fold{fold + 1}-b6.json",
            fold=fold,
            system="duplicate",
            local_positive=False,
        )
    with pytest.raises(ValueError, match="occurs twice"):
        summarize(
            development_root=development,
            secondary_path=None,
            output=tmp_path / "summary.json",
            expected_development_system_count=5,
            expected_secondary_system_count=None,
            randomization_draws=1_000,
        )
