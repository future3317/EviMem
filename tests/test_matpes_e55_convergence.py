from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.run_matpes_e55_convergence import (
    CAL_FANTASY_COUNTS,
    CAL_POSTERIOR_SAMPLE_COUNTS,
    CAL_SELECTION_TIMEOUT_SECONDS,
    CAL_WORKERS,
    DELTA_FANTASY_COUNT,
    DELTA_POSTERIOR_SAMPLE_COUNTS,
    DELTA_SELECTION_TIMEOUT_SECONDS,
    DELTA_WORKERS,
    SEED,
    _run_unit,
    _write_top_level_manifest,
    build_units,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    full_pool_root = tmp_path / "e52"
    task = full_pool_root / "matpes-e52-pool-100-task.json"
    vault = full_pool_root / "matpes-e52-pool-100-vault.json"
    task.parent.mkdir(parents=True)
    task.write_text('{"release_id": "fixture"}\n', encoding="utf-8")
    vault.write_text('{"oracle": "fixture"}\n', encoding="utf-8")

    systems = [f"S{index:03d}-X" for index in range(230)]
    folds = []
    for fold_index in range(5):
        query_systems = systems[46 * fold_index : 46 * (fold_index + 1)]
        folds.append(
            {
                "fold_index": fold_index,
                "query_systems": query_systems,
                "fit_systems": [system for system in systems if system not in query_systems],
            }
        )
    crossfit = full_pool_root / "matpes-e52-pool-100-crossfit.json"
    crossfit.write_text(
        json.dumps(
            {
                "task_sha256": _sha256(task),
                "eligible_systems": systems,
                "fold_count": 5,
                "folds": folds,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cal_folds = []
    for fold in folds:
        cal_folds.append(
            {
                "fold_index": fold["fold_index"],
                "query_systems": fold["query_systems"][:3],
                "fit_systems": fold["fit_systems"],
            }
        )
    cal_manifest = tmp_path / "e55-cal-manifest.json"
    cal_manifest.write_text(
        json.dumps(
            {
                "task_sha256": _sha256(task),
                "development_crossfit_sha256": _sha256(crossfit),
                "fold_count": 5,
                "folds": cal_folds,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return full_pool_root, cal_manifest, task, vault


def _units(tmp_path: Path, *, stages: tuple[str, ...] = ("delta", "cal")):
    full_pool_root, cal_manifest, _, _ = _write_inputs(tmp_path)
    return build_units(
        full_pool_root=full_pool_root,
        cal_manifest=cal_manifest,
        output_root=tmp_path / "outputs",
        runner=Path("runner.py"),
        stages=stages,
    )


def test_e55_builds_exact_delta_and_cal_grids_with_frozen_identity(tmp_path: Path) -> None:
    full_pool_root, cal_manifest, task, vault = _write_inputs(tmp_path)

    units = build_units(
        full_pool_root=full_pool_root,
        cal_manifest=cal_manifest,
        output_root=tmp_path / "outputs",
        runner=Path("runner.py"),
        stages=("delta", "cal"),
    )

    delta_units = [unit for unit in units if unit.identity["stage"] == "delta"]
    cal_units = [unit for unit in units if unit.identity["stage"] == "cal"]
    assert len(delta_units) == 25
    assert len(cal_units) == 45
    assert {
        (unit.identity["fold_index"], unit.identity["posterior_sample_count"])
        for unit in delta_units
    } == {(fold, samples) for fold in range(5) for samples in DELTA_POSTERIOR_SAMPLE_COUNTS}
    assert {
        (
            unit.identity["fold_index"],
            unit.identity["posterior_sample_count"],
            unit.identity["fantasy_count"],
        )
        for unit in cal_units
    } == {
        (fold, samples, fantasies)
        for fold in range(5)
        for samples in CAL_POSTERIOR_SAMPLE_COUNTS
        for fantasies in CAL_FANTASY_COUNTS
    }
    assert all(unit.identity["policy"] == "delta_hull_active_search" for unit in delta_units)
    assert all(unit.identity["policy"] == "cal_style_hull_entropy" for unit in cal_units)
    assert all(unit.identity["budget"] == 6 for unit in units)
    assert all(unit.identity["maximum_budget"] == 6 for unit in units)
    assert all(unit.identity["minimum_candidates"] == 12 for unit in units)
    assert all(unit.identity["seed"] == SEED for unit in units)
    assert all(unit.identity["hull_backend"] == "fixed_composition" for unit in units)
    assert all(unit.identity["transport_family"] == "hierarchical_matern52_frozen_structure" for unit in units)
    assert all(unit.identity["task_sha256"] == _sha256(task) for unit in units)
    assert all(unit.identity["vault_sha256"] == _sha256(vault) for unit in units)
    assert all(unit.identity["query_system_count"] == 46 for unit in delta_units)
    assert all(unit.identity["fit_system_count"] == 184 for unit in delta_units)
    assert all(unit.identity["query_system_count"] == 3 for unit in cal_units)
    assert all(unit.identity["fit_system_count"] == 184 for unit in cal_units)
    assert all(unit.identity["e55_manifest_sha256"] == _sha256(cal_manifest) for unit in cal_units)
    assert all(unit.identity["fantasy_count"] == DELTA_FANTASY_COUNT for unit in delta_units)
    assert all(unit.identity["hull_candidate_workers"] == DELTA_WORKERS for unit in delta_units)
    assert all(unit.identity["selection_timeout_seconds"] == DELTA_SELECTION_TIMEOUT_SECONDS for unit in delta_units)
    assert all(unit.identity["hull_candidate_workers"] == CAL_WORKERS for unit in cal_units)
    assert all(unit.identity["selection_timeout_seconds"] == CAL_SELECTION_TIMEOUT_SECONDS for unit in cal_units)
    assert all("delta-fold" in unit.output.name and "-m" in unit.output.name for unit in delta_units)
    assert all("cal-fold" in unit.output.name and "-m" in unit.output.name and "-k" in unit.output.name for unit in cal_units)


def test_e55_stage_filtering_deduplicates_and_rejects_unknown_stages(tmp_path: Path) -> None:
    assert len(_units(tmp_path / "delta", stages=("delta", "delta"))) == 25
    assert len(_units(tmp_path / "cal", stages=("cal", "cal"))) == 45
    with pytest.raises(ValueError, match="stage"):
        _units(tmp_path / "bad", stages=("delta", "unknown"))


def test_e55_rejects_output_under_a_git_worktree(tmp_path: Path) -> None:
    full_pool_root, cal_manifest, _, _ = _write_inputs(tmp_path)
    git_root = tmp_path / "other-worktree"
    (git_root / ".git").mkdir(parents=True)

    with pytest.raises(ValueError, match="outside Git"):
        build_units(
            full_pool_root=full_pool_root,
            cal_manifest=cal_manifest,
            output_root=git_root / "outputs",
            runner=Path("runner.py"),
            stages=("delta",),
        )


def _valid_output(unit) -> dict[str, object]:
    return {
        "task_sha256": unit.identity["task_sha256"],
        "oracle_vault_sha256": unit.identity["vault_sha256"],
        "active_policies": [unit.identity["policy"]],
        "query_systems": unit.identity["query_systems"],
        "transport_fit_systems": unit.identity["fit_systems"],
        "transport_fit_system_count": unit.identity["fit_system_count"],
        "transport_fit_and_query_systems_disjoint": True,
        "config": {
            "query_budget": 6,
            "maximum_budget": 6,
            "minimum_candidates": 12,
            "seed": SEED,
            "posterior_sample_count": unit.identity["posterior_sample_count"],
            "fantasy_count": unit.identity["fantasy_count"],
            "hull_candidate_workers": unit.identity["hull_candidate_workers"],
            "hull_backend": "fixed_composition",
            "transport_family": "hierarchical_matern52_frozen_structure",
            "rollout_selection_timeout_seconds": unit.identity["selection_timeout_seconds"],
            "crossfit_manifest_sha256": unit.identity["runner_crossfit_manifest_sha256"],
            "crossfit_fold_index": 0,
        },
    }


def test_e55_resume_requires_complete_existing_output_identity(tmp_path: Path) -> None:
    unit = _units(tmp_path, stages=("cal",))[0]
    unit.output.parent.mkdir(parents=True)
    unit.output.write_text(json.dumps(_valid_output(unit)), encoding="utf-8")

    assert _run_unit(unit) == f"resume-skip={unit.output}"

    invalid = _valid_output(unit)
    invalid["config"]["posterior_sample_count"] = 999
    unit.output.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="posterior_sample_count"):
        _run_unit(unit)


def test_e55_refuses_terminal_failures_and_overwriting_top_level_manifest(tmp_path: Path) -> None:
    units = _units(tmp_path, stages=("delta",))
    unit = units[0]
    unit.output.parent.mkdir(parents=True)
    unit.output.with_suffix(".failure.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="already failed"):
        _run_unit(unit)

    manifest = tmp_path / "outputs" / "e55-unit-manifest.json"
    _write_top_level_manifest(manifest, units)
    with pytest.raises(FileExistsError, match="overwrite"):
        _write_top_level_manifest(manifest, units)
