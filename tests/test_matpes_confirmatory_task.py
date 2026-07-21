from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_builder():
    path = Path(__file__).parents[1] / "tools" / "build_matpes_confirmatory_task.py"
    spec = importlib.util.spec_from_file_location("build_matpes_confirmatory_task", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(system: str, index: int) -> dict:
    return {
        "pair_id": f"{system}-{index}",
        "chemical_system": system,
        "composition": {system.split("-")[0]: 1.0},
        "source_formation_energy_ev_per_atom": -0.1,
        "source_structure_sha256": f"source-{system}-{index}",
        "source_environment_embedding": [float(index)],
        "original_mp_id": f"mp-{system}-{index}",
    }


def test_confirmatory_builder_excludes_development_and_is_outcome_independent(tmp_path: Path) -> None:
    builder = _load_builder()
    systems = ["A-B", "C-D", "E-F", "G-H", "I-J"]
    rows = [row for system in systems for row in (_row(system, i) for i in range(4))]
    task = {
        "schema_version": 1,
        "release_id": "fixture-release",
        "source_protocol": {},
        "target_protocol": {},
        "development_pairs": rows,
        "development_initial_phase_entries": {},
        "development_systems": systems[:1],
    }
    vault = {
        "release_id": "fixture-release",
        "target_outcomes": [
            {
                "pair_id": row["pair_id"],
                "composition": row["composition"],
                "target_corrected_total_energy_ev": -100.0 - index,
                "target_formation_energy_ev_per_atom": -0.2 - index,
                "split": "development",
            }
            for index, row in enumerate(rows)
        ],
    }
    task_path = tmp_path / "all-task.json"
    vault_path = tmp_path / "all-vault.json"
    dev_path = tmp_path / "dev-task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    vault_path.write_text(json.dumps(vault), encoding="utf-8")
    dev_path.write_text(json.dumps({"development_systems": systems[:1]}), encoding="utf-8")
    out_task = tmp_path / "fresh-task.json"
    out_vault = tmp_path / "fresh-vault.json"
    result = builder.build(
        all_task_path=task_path,
        all_vault_path=vault_path,
        development_task_path=dev_path,
        task_output=out_task,
        vault_output=out_vault,
        max_systems_per_stratum=8,
        minimum_candidates=4,
    )
    fresh_task = json.loads(out_task.read_text(encoding="utf-8"))
    fresh_vault = json.loads(out_vault.read_text(encoding="utf-8"))
    assert systems[0] not in fresh_task["confirmatory_systems"]
    assert set(fresh_task["confirmatory_systems"]) == set(systems[1:])
    assert all(row["split"] == "confirmatory" for row in fresh_vault["target_outcomes"])
    assert result["oracle_values_used_for_selection"] is False


def test_confirmatory_builder_can_reserve_before_transport_fit(tmp_path: Path) -> None:
    builder = _load_builder()
    systems = ["A-B", "C-D", "E-F"]
    rows = [row for system in systems for row in (_row(system, i) for i in range(4))]
    task = {
        "schema_version": 1,
        "release_id": "fixture-release",
        "source_protocol": {},
        "target_protocol": {},
        "development_pairs": rows,
        "development_initial_phase_entries": {},
        "development_systems": systems,
    }
    vault = {
        "schema_version": 1,
        "release_id": "fixture-release",
        "target_outcomes": [
            {
                "pair_id": row["pair_id"],
                "composition": row["composition"],
                "target_corrected_total_energy_ev": -1.0,
                "target_formation_energy_ev_per_atom": -0.1,
                "split": "development",
            }
            for row in rows
        ],
    }
    task_path = tmp_path / "all-task.json"
    vault_path = tmp_path / "all-vault.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    vault_path.write_text(json.dumps(vault), encoding="utf-8")
    out_task = tmp_path / "fresh-task.json"
    out_vault = tmp_path / "fresh-vault.json"
    result = builder.build(
        all_task_path=task_path,
        all_vault_path=vault_path,
        development_task_path=None,
        task_output=out_task,
        vault_output=out_vault,
        max_systems_per_stratum=2,
        minimum_candidates=4,
    )
    fresh_task = json.loads(out_task.read_text(encoding="utf-8"))
    assert result["split_mode"] == "new_outcome_independent_repartition"
    assert len(fresh_task["confirmatory_systems"]) == 2
    assert "development_pairs" not in fresh_task
    assert fresh_task["development_exclusion"]["systems"] == []
