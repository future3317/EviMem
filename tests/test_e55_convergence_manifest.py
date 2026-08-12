from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tools.build_e55_convergence_manifest import build


def _write_fixture(
    tmp_path: Path, *, unsupported_first_fold_query: bool = False
) -> tuple[Path, Path, dict[str, list[str]]]:
    query_by_fold: dict[str, list[str]] = {}
    task_rows: list[dict[str, str]] = []
    folds: list[dict[str, object]] = []
    for fold_index in range(5):
        prefix = str(fold_index)
        query_systems = [
            f"{chr(65 + offset)}{prefix}-{chr(66 + offset)}{prefix}"
            for offset in range(9)
        ]
        if unsupported_first_fold_query and fold_index == 0:
            query_systems[0] = "Q0-R0"
        query_by_fold[str(fold_index)] = query_systems
        for rank, system in enumerate(query_systems):
            task_rows.extend(
                {"chemical_system": system, "pair_id": f"{system}-{index}"}
                for index in range(rank + 3)
            )
        fit_systems = [f"{chr(65 + offset)}{prefix}-Z{prefix}" for offset in range(10)]
        task_rows.extend(
            {"chemical_system": system, "pair_id": f"{system}-fit"}
            for system in fit_systems
        )
        folds.append(
            {
                "fold_index": fold_index,
                "query_systems": query_systems,
                "fit_systems": fit_systems,
                "system_count": len(query_systems),
                "fit_system_count": len(fit_systems),
            }
        )

    task_path = tmp_path / "task.json"
    task_path.write_text(
        json.dumps(
            {
                "release_id": "fixture-release",
                "development_pairs": task_rows,
                "development_systems": sorted(
                    {row["chemical_system"] for row in task_rows}
                ),
            }
        ),
        encoding="utf-8",
    )
    crossfit_path = tmp_path / "crossfit.json"
    eligible = [
        system for systems in query_by_fold.values() for system in systems
    ] + [
        f"{chr(65 + offset)}{fold_index}-Z{fold_index}"
        for fold_index in range(5)
        for offset in range(10)
    ]
    for fold in folds:
        fold["fit_systems"] = sorted(set(eligible) - set(fold["query_systems"]))
        fold["fit_system_count"] = len(fold["fit_systems"])
    crossfit_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release_id": "fixture-release",
                "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                "eligible_systems": eligible,
                "fold_count": 5,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )
    return task_path, crossfit_path, query_by_fold


def test_manifest_selects_one_deterministic_supported_system_per_tercile(
    tmp_path: Path,
) -> None:
    task_path, crossfit_path, query_by_fold = _write_fixture(tmp_path)

    result = build(task_path, crossfit_path, tmp_path / "manifest.json")
    repeat = build(task_path, crossfit_path, tmp_path / "repeat.json")

    assert result["assignment_uses_target_outcomes"] is False
    assert result["fold_count"] == 5
    assert result["folds"] == repeat["folds"]
    for fold in result["folds"]:
        assert set(fold["query_systems"]) == {
            str(fold["candidate_count_strata"][bin_name]["selected_system"])
            for bin_name in ("low", "middle", "high")
        }
        assert set(fold["query_systems"]) <= set(query_by_fold[str(fold["fold_index"])])
        assert all(
            fold["candidate_count_strata"][bin_name]["selected_system"]
            in fold["candidate_count_strata"][bin_name]["supported_systems"]
            for bin_name in ("low", "middle", "high")
        )


def test_manifest_preserves_each_original_fit_roster_and_fit_element_support(
    tmp_path: Path,
) -> None:
    task_path, crossfit_path, _ = _write_fixture(
        tmp_path, unsupported_first_fold_query=True
    )
    crossfit = json.loads(crossfit_path.read_text(encoding="utf-8"))

    result = build(task_path, crossfit_path, tmp_path / "manifest.json")

    for source_fold, manifest_fold in zip(crossfit["folds"], result["folds"]):
        assert manifest_fold["fit_systems"] == source_fold["fit_systems"]
        fit_elements = {
            element
            for system in source_fold["fit_systems"]
            for element in system.split("-")
        }
        assert all(
            set(system.split("-")) <= fit_elements
            for system in manifest_fold["query_systems"]
        )
    assert "Q0-R0" not in result["folds"][0]["query_systems"]


def test_manifest_rejects_task_hash_mismatch(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    task_path.write_text(
        task_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match task"):
        build(task_path, crossfit_path, tmp_path / "manifest.json")


def test_manifest_rejects_output_inside_git(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    repo_output = Path(__file__).resolve().parents[1] / ".e55-convergence-test.json"

    with pytest.raises(ValueError, match="outside Git"):
        build(task_path, crossfit_path, repo_output)


def test_manifest_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    build(task_path, crossfit_path, output)

    with pytest.raises(FileExistsError, match="overwrite"):
        build(task_path, crossfit_path, output)
