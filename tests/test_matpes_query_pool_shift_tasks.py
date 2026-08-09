from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_matpes_query_pool_shift_tasks import build


def test_query_pool_shift_preserves_fit_rows_and_excludes_other_systems(tmp_path: Path) -> None:
    systems = ("A-B", "C-D", "E-F", "G-H", "I-J", "K-L")
    rows = [
        {"pair_id": f"{system}-{index}", "chemical_system": system}
        for system in systems
        for index in range(20)
    ]
    task = {
        "release_id": "fixture",
        "development_systems": list(systems),
        "development_pairs": rows,
        "development_initial_phase_entries": {system: [] for system in systems},
    }
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    vault = {
        "release_id": "fixture",
        "target_outcomes": [
            {"pair_id": row["pair_id"], "split": "development"} for row in rows
        ],
    }
    vault_path = tmp_path / "vault.json"
    vault_path.write_text(json.dumps(vault), encoding="utf-8")
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    development = {
        "task_sha256": task_sha,
        "eligible_systems": list(systems[:5]),
        "folds": [
            {"fold_index": index, "source_fold_index": index + 1, "query_systems": [system]}
            for index, system in enumerate(systems[:5])
        ],
    }
    development_path = tmp_path / "development.json"
    development_path.write_text(json.dumps(development), encoding="utf-8")
    output = tmp_path / "derived"

    result = build(
        task_path=task_path,
        vault_path=vault_path,
        development_crossfit_path=development_path,
        output_dir=output,
        fractions=(0.7,),
    )
    derived = json.loads((output / "pool-070-fold1-task.json").read_text())
    counts = {
        system: sum(row["chemical_system"] == system for row in derived["development_pairs"])
        for system in systems
    }
    assert counts["A-B"] == 14
    assert all(counts[system] == 20 for system in systems[1:5])
    assert counts["K-L"] == 0
    assert derived["pool_shift"]["fit_pool_fraction"] == 1.0
    assert result["fit_rows_preserved"] is True

    crossfit = json.loads((output / "pool-070-fold1-crossfit.json").read_text())
    assert crossfit["folds"][0]["query_systems"] == ["A-B"]
    assert crossfit["folds"][0]["fit_system_count"] == 4
