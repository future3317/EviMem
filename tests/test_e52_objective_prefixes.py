from __future__ import annotations

import json
from pathlib import Path

from tools.summarize_e52_objective_prefixes import summarize


def _strategy(selected: list[str], positives: set[str]) -> dict[str, object]:
    return {
        "selected_pair_ids": selected,
        "oracle_pool_final_labels_by_pair_id": {
            pair_id: pair_id in positives for pair_id in selected
        },
    }


def test_objective_summary_derives_budget_prefixes(tmp_path: Path) -> None:
    root = tmp_path / "objective"
    systems: list[str] = []
    for pool in ("070", "085", "100"):
        for fold in range(5):
            system = f"S{fold}"
            systems.append(system)
            selected = [f"{system}-{index}" for index in range(6)]
            payload = {
                "active_policies": [
                    "posterior_mean_target_margin",
                    "delta_hull_active_search",
                ],
                "task_sha256": "full" if pool == "100" else f"{pool}-{fold}",
                "systems": {
                    system: {
                        "budget": 6,
                        "strategies": {
                            "posterior_mean_target_margin": _strategy(selected, set()),
                            "delta_hull_active_search": _strategy(
                                selected, {selected[0], selected[2]}
                            ),
                        },
                    }
                },
            }
            path = root / pool / f"fold{fold + 1}-b6.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

    result = summarize(
        input_root=root,
        output=tmp_path / "summary.json",
        expected_system_count=5,
        bootstrap_count=100,
    )

    for pool in ("070", "085", "100"):
        assert result["curves"][pool]["budgets"]["1"]["paired_delta_T"] == 1.0
        assert result["curves"][pool]["budgets"]["2"]["paired_delta_T"] == 1.0
        assert result["curves"][pool]["budgets"]["3"]["paired_delta_T"] == 2.0
