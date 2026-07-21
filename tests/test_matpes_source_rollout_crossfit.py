from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_builder():
    path = Path(__file__).parents[1] / "tools" / "build_matpes_source_rollout_crossfit.py"
    spec = importlib.util.spec_from_file_location("build_matpes_source_rollout_crossfit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_crossfit_plan_is_outcome_independent_disjoint_and_complete(tmp_path: Path) -> None:
    builder = _load_builder()
    eligible = ("A-B", "C-D", "E-F-G", "H-I-J", "K-L-M-N", "O-P-Q-R")
    opened_systems = ("S-T", "U-V-W")
    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "release_id": "fixture-release",
                "development_pairs": [
                    {"chemical_system": system, "pair_id": f"{system}-1"}
                    for system in (*eligible, *opened_systems)
                ],
            }
        ),
        encoding="utf-8",
    )
    opened_path = tmp_path / "opened.json"
    opened_path.write_text(
        json.dumps(
            {
                "split": "confirmatory",
                "evaluation_systems_accessed": True,
                "transport_fit_systems": eligible,
                "transport_fit_system_count": len(eligible),
                "query_systems": opened_systems,
            }
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "crossfit.json"
    plan = builder.build(
        task_path=task_path,
        opened_result_path=opened_path,
        output_path=output_path,
        fold_count=3,
    )
    assigned = [
        system for fold in plan["folds"] for system in fold["query_systems"]
    ]
    assert sorted(assigned) == sorted(eligible)
    assert len(assigned) == len(set(assigned))
    assert not (set(assigned) & set(opened_systems))
    assert plan["assignment_uses_target_outcomes"] is False
    assert json.loads(output_path.read_text(encoding="utf-8")) == plan
