from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_planner():
    path = Path(__file__).parents[1] / "tools" / "plan_matpes_source_rollout_numerical_audit.py"
    spec = importlib.util.spec_from_file_location("numerical_audit_plan", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _result(*, selected: list[str], source_total: int, rollout_total: int) -> dict:
    diagnostic = {
        "kind": "source_rollout_sarr",
        "candidate_pair_ids": ["source", "other"],
        "source_pair_id": "source",
        "selected_pair_id": selected[0],
        "fallback_reason": "no_positive_simultaneous_lower_bound",
        "mean_advantages_over_source": {"source": 0.0, "other": 0.1},
        "simultaneous_lower_bounds": {"source": 0.0, "other": -0.1},
    }
    return {
        "task_sha256": "task",
        "evaluation_systems_accessed": False,
        "systems": {
            "A-B": {
                "strategies": {
                    "source_margin": {"oracle_pool_confirmed_discoveries": source_total},
                    "source_rollout_delta_hull": {
                        "selected_pair_ids": selected,
                        "oracle_pool_confirmed_discoveries": rollout_total,
                        "policy_decision_rounds": [{"selection_diagnostics": diagnostic}],
                    },
                }
            }
        },
    }


def test_numerical_audit_plan_unions_registered_state_reasons(tmp_path: Path) -> None:
    planner = _load_planner()
    sarr_path = tmp_path / "sarr.json"
    mc512_path = tmp_path / "mc512.json"
    mc1024_path = tmp_path / "mc1024.json"
    output_path = tmp_path / "plan.json"
    sarr_path.write_text(json.dumps(_result(selected=["other"], source_total=1, rollout_total=2)))
    mc512_path.write_text(json.dumps(_result(selected=["source"], source_total=1, rollout_total=2)))
    mc1024_path.write_text(json.dumps(_result(selected=["other"], source_total=1, rollout_total=2)))

    plan = planner.build_plan(
        sarr_result_path=sarr_path,
        mc512_result_path=mc512_path,
        mc1024_result_path=mc1024_path,
        output_path=output_path,
    )

    assert plan["state_count"] == 1
    assert plan["states"] == [
        {
            "chemical_system": "A-B",
            "round_index": 1,
            "reasons": [
                "final_win_system",
                "positive_but_simultaneously_unresolved",
                "pre_sarr_mc512_mc1024_disagreement",
                "sarr_deviation",
            ],
        }
    ]
    assert json.loads(output_path.read_text()) == plan
