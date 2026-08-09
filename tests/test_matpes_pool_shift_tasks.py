from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.build_matpes_pool_shift_tasks import build


def _write_inputs(root: Path) -> tuple[Path, Path]:
    rows = [
        {
            "pair_id": f"pair-{index}",
            "chemical_system": "A-B",
            "composition": {"A": 1, "B": 1},
        }
        for index in range(10)
    ]
    task = {
        "schema_version": 2,
        "release_id": "fixture",
        "development_systems": ["A-B"],
        "development_pairs": rows,
        "development_initial_phase_entries": {"A-B": []},
        "status": "fixture",
    }
    vault = {
        "schema_version": 2,
        "release_id": "fixture",
        "status": "fixture",
        "target_outcomes": [
            {"pair_id": row["pair_id"], "split": "development", "secret": index}
            for index, row in enumerate(rows)
        ],
    }
    task_path = root / "task.json"
    vault_path = root / "vault.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    vault_path.write_text(json.dumps(vault), encoding="utf-8")
    return task_path, vault_path


def test_pool_shift_builder_is_nested_and_exact_join(tmp_path: Path) -> None:
    task_path, vault_path = _write_inputs(tmp_path)
    output = tmp_path / "derived"
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    crossfit_path = tmp_path / "crossfit.json"
    crossfit_path.write_text(
        json.dumps(
            {
                "task_sha256": task_sha,
                "eligible_systems": ["A-B"],
                "folds": [{"fold_index": 0, "query_systems": ["A-B"]}],
                "assignment_uses_target_outcomes": False,
            }
        ),
        encoding="utf-8",
    )
    result = build(
        task_path=task_path,
        vault_path=vault_path,
        output_dir=output,
        crossfit_manifest_path=crossfit_path,
    )

    ids_by_tag: dict[str, set[str]] = {}
    for tag, expected_count in (("070", 7), ("085", 9), ("100", 10)):
        task = json.loads((output / f"matpes-e52-pool-{tag}-task.json").read_text())
        vault = json.loads((output / f"matpes-e52-pool-{tag}-vault.json").read_text())
        task_ids = {row["pair_id"] for row in task["development_pairs"]}
        vault_ids = {row["pair_id"] for row in vault["target_outcomes"]}
        assert len(task_ids) == expected_count
        assert task_ids == vault_ids
        assert task["pool_shift"]["outcome_used_for_selection"] is False
        crossfit = json.loads((output / f"matpes-e52-pool-{tag}-crossfit.json").read_text())
        assert crossfit["task_sha256"] == result["variants"][tag]["task_sha256"]
        assert crossfit["folds"][0]["query_systems"] == ["A-B"]
        ids_by_tag[tag] = task_ids

    assert ids_by_tag["070"] < ids_by_tag["085"] < ids_by_tag["100"]
    assert result["selection_uses_target_outcomes"] is False


def test_pool_shift_builder_refuses_missing_reference_pool(tmp_path: Path) -> None:
    task_path, vault_path = _write_inputs(tmp_path)
    try:
        build(
            task_path=task_path,
            vault_path=vault_path,
            output_dir=tmp_path / "derived",
            fractions=(0.7, 0.85),
        )
    except ValueError as exc:
        assert "100%" in str(exc)
    else:
        raise AssertionError("missing 100% reference pool should fail")
