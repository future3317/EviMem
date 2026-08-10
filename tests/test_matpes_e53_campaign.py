from __future__ import annotations

from pathlib import Path

import pytest

from tools.run_matpes_e53_objective_isolation import POLICIES, build_units


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _fixture_roots(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    full = tmp_path / "full"
    split = tmp_path / "split"
    _write(full / "matpes-e52-pool-100-task.json", "task")
    _write(full / "matpes-e52-pool-100-vault.json", "vault")
    _write(full / "matpes-e52-pool-100-crossfit.json", "development")
    _write(split / "matpes-e52-secondary-confirmation-crossfit.json", "secondary")
    secondary_task = _write(tmp_path / "secondary-task.json", "secondary-task")
    secondary_vault = _write(tmp_path / "secondary-vault.json", "secondary-vault")
    return full, split, secondary_task, secondary_vault


def test_e53_development_stage_builds_five_frozen_units(tmp_path: Path) -> None:
    full, split, secondary_task, secondary_vault = _fixture_roots(tmp_path)
    units = build_units(
        full_pool_root=full,
        split_manifest_root=split,
        secondary_task=secondary_task,
        secondary_vault=secondary_vault,
        output_root=tmp_path / "external-output",
        runner=tmp_path / "runner.py",
        stage="development",
        posterior_sample_count=1024,
        selection_timeout_seconds=7200.0,
    )

    assert len(units) == 5
    assert tuple(unit.identity["fold_index"] for unit in units) == tuple(range(5))
    assert all(unit.identity["policies"] == POLICIES for unit in units)
    assert all(unit.identity["budget"] == 6 for unit in units)
    assert all(unit.identity["posterior_sample_count"] == 1024 for unit in units)
    assert all("development" in unit.output.parts for unit in units)
    assert all("secondary" not in unit.output.parts for unit in units)


def test_e53_secondary_stage_is_one_explicit_unit(tmp_path: Path) -> None:
    full, split, secondary_task, secondary_vault = _fixture_roots(tmp_path)
    units = build_units(
        full_pool_root=full,
        split_manifest_root=split,
        secondary_task=secondary_task,
        secondary_vault=secondary_vault,
        output_root=tmp_path / "external-output",
        runner=tmp_path / "runner.py",
        stage="secondary",
        posterior_sample_count=1024,
        selection_timeout_seconds=7200.0,
    )

    assert len(units) == 1
    unit = units[0]
    assert unit.identity["fold_index"] == 0
    assert unit.identity["stage"] == "secondary_heldout_matpes_rerun"
    assert unit.identity["secondary_is_untouched"] is False
    assert "secondary" in unit.output.parts
    assert any(
        Path(value).name == "matpes-e52-secondary-confirmation-crossfit.json"
        for value in unit.command
    )
    assert str(secondary_task) in unit.command
    assert str(secondary_vault) in unit.command
    assert unit.identity["task_sha256"] != units[0].identity.get("development_task_sha256")


def test_e53_campaign_rejects_output_inside_repository(tmp_path: Path) -> None:
    full, split, secondary_task, secondary_vault = _fixture_roots(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside Git"):
        build_units(
            full_pool_root=full,
            split_manifest_root=split,
            secondary_task=secondary_task,
            secondary_vault=secondary_vault,
            output_root=repo_root / "forbidden-e53-output",
            runner=tmp_path / "runner.py",
            stage="development",
            posterior_sample_count=1024,
            selection_timeout_seconds=7200.0,
        )


def test_e53_secondary_requires_explicit_task_and_vault(tmp_path: Path) -> None:
    full, split, _, _ = _fixture_roots(tmp_path)
    with pytest.raises(ValueError, match="explicit secondary task and vault"):
        build_units(
            full_pool_root=full,
            split_manifest_root=split,
            secondary_task=None,
            secondary_vault=None,
            output_root=tmp_path / "external-output",
            runner=tmp_path / "runner.py",
            stage="secondary",
            posterior_sample_count=1024,
            selection_timeout_seconds=7200.0,
        )
