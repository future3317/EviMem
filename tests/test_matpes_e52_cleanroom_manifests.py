from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_matpes_e52_cleanroom_manifests import build


def test_cleanroom_manifest_freezes_development_and_complement(tmp_path: Path) -> None:
    task = {
        "release_id": "fixture",
        "development_systems": ["A-B", "C-D", "E-F", "G-H"],
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    prior = {
        "task_sha256": task_sha,
        "folds": [
            {"fold_index": 0, "query_systems": ["A-B"]},
            {"fold_index": 1, "query_systems": ["C-D"]},
            {"fold_index": 2, "query_systems": ["E-F"]},
        ],
    }
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps(prior), encoding="utf-8")

    result = build(
        task_path=task_path,
        prior_crossfit_path=prior_path,
        output_dir=tmp_path / "derived",
        development_fold_indices=(1, 2),
    )
    development = json.loads(Path(result["development_manifest_path"]).read_text())
    confirmation = json.loads(Path(result["secondary_confirmation_manifest_path"]).read_text())
    assert development["eligible_systems"] == ["C-D", "E-F"]
    assert confirmation["folds"][0]["query_systems"] == ["A-B", "G-H"]
    assert confirmation["eligible_systems"] == ["A-B", "C-D", "E-F", "G-H"]
    assert confirmation["fit_systems_are_exactly_development_roster"] is True
    assert result["secondary_confirmation_is_untouched"] is False
