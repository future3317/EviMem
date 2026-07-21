from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_summary():
    path = Path(__file__).parents[1] / "tools" / "summarize_matpes_source_rollout_opportunity.py"
    spec = importlib.util.spec_from_file_location("opportunity_summary", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan() -> dict:
    return {
        "evaluation_systems_accessed": False,
        "state_count": 2,
        "states": [
            {"chemical_system": "A-B", "round_index": 1, "reasons": ["sarr_deviation"]},
            {"chemical_system": "A-B", "round_index": 2, "reasons": ["final_win_system", "positive_but_simultaneously_unresolved"]},
        ],
    }


def _audit(plan_sha: str) -> dict:
    return {
        "status": "development_high_precision_replay",
        "plan_sha256": plan_sha,
        "task_sha256": "task",
        "sarr_sha256": "sarr",
        "evaluation_systems_accessed": False,
        "state_count": 2,
        "states": [
            {"chemical_system": "A-B", "round_index": 1, "selected_pair_id": "x", "source_pair_id": "source", "selected_opportunity_cost": 0.0, "source_opportunity_cost": 0.2, "best_second_gap": 0.1, "selected_high_precision_advantage": 0.2, "selected_high_precision_lower_bound": 0.1},
            {"chemical_system": "A-B", "round_index": 2, "selected_pair_id": "source", "source_pair_id": "source", "selected_opportunity_cost": 0.1, "source_opportunity_cost": 0.1, "best_second_gap": 0.1, "selected_high_precision_advantage": 0.0, "selected_high_precision_lower_bound": -0.1},
        ],
    }


def test_opportunity_summary_requires_exact_plan_coverage(tmp_path: Path) -> None:
    module = _load_summary()
    plan_path = tmp_path / "plan.json"
    audit_path = tmp_path / "audit.json"
    output_path = tmp_path / "summary.json"
    plan_path.write_text(json.dumps(_plan()))
    audit_path.write_text(json.dumps(_audit(_sha(plan_path))))

    summary = module.summarize(audit_path=audit_path, plan_path=plan_path, output_path=output_path)

    assert summary["state_count"] == 2
    assert summary["counts"] == {
        "selected_non_source": 1,
        "selected_non_source_negative_high_precision_advantage": 0,
        "source_positive_opportunity_cost": 2,
    }
    assert summary["by_reason"]["sarr_deviation"]["source_opportunity_cost"]["mean"] == 0.2
    assert json.loads(output_path.read_text()) == summary

    audit = _audit(_sha(plan_path))
    audit["states"].pop()
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="exactly equal"):
        module.summarize(audit_path=audit_path, plan_path=plan_path, output_path=tmp_path / "bad.json")
