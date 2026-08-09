from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_e52_two_step_equivalence_manifest import build


def test_equivalence_roster_is_deterministic_and_development_only(tmp_path: Path) -> None:
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({"release_id": "fixture"}), encoding="utf-8")
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    development_path = tmp_path / "development.json"
    development_path.write_text(
        json.dumps(
            {
                "release_id": "fixture",
                "task_sha256": task_sha,
                "eligible_systems": ["A", "B", "C", "D"],
            }
        ),
        encoding="utf-8",
    )

    result = build(
        task_path=task_path,
        development_crossfit_path=development_path,
        output=tmp_path / "roster.json",
        system_count=2,
    )

    selected = result["folds"][0]["query_systems"]
    assert len(selected) == 2
    assert set(selected) <= {"A", "B", "C", "D"}
    assert result["folds"][0]["fit_system_count"] == 2
    assert result["assignment_uses_target_outcomes"] is False


def test_equivalence_roster_can_require_fit_element_support(tmp_path: Path) -> None:
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps({"release_id": "fixture"}), encoding="utf-8")
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    development_path = tmp_path / "development.json"
    development_path.write_text(
        json.dumps(
            {
                "release_id": "fixture",
                "task_sha256": task_sha,
                "eligible_systems": ["Ag-Cl", "A-B", "A-C", "B-C", "A-B-C"],
            }
        ),
        encoding="utf-8",
    )

    result = build(
        task_path=task_path,
        development_crossfit_path=development_path,
        output=tmp_path / "supported.json",
        system_count=2,
        require_fit_element_support=True,
    )

    assert "Ag-Cl" not in result["folds"][0]["query_systems"]
    assert result["requires_fit_element_support"] is True
