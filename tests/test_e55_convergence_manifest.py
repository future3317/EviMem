from __future__ import annotations

import hashlib
import itertools
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
    all_systems = [
        f"{chr(65 + left)}-{chr(65 + right)}"
        for left in range(10)
        for right in range(left + 1, 10)
    ]
    for fold_index in range(5):
        query_systems = all_systems[fold_index::5]
        if unsupported_first_fold_query and fold_index == 0:
            query_systems[0] = "Q0-R0"
        query_by_fold[str(fold_index)] = query_systems
        for rank, system in enumerate(query_systems):
            task_rows.extend(
                {"chemical_system": system, "pair_id": f"{system}-{index}"}
                for index in range(rank + 3)
            )
        if unsupported_first_fold_query and fold_index == 0:
            task_rows.extend(
                {"chemical_system": "Q0-R0", "pair_id": f"Q0-R0-{index}"}
                for index in range(3)
            )
        folds.append(
            {
                "fold_index": fold_index,
                "query_systems": query_systems,
                "system_count": len(query_systems),
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
    eligible = [system for systems in query_by_fold.values() for system in systems]
    for fold in folds:
        fit_roster = sorted(set(eligible) - set(fold["query_systems"]))
        fold["fit_systems"] = (
            list(reversed(fit_roster)) if fold["fold_index"] == 0 else fit_roster
        )
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


def _rewrite_crossfit(path: Path, mutate: object) -> None:
    crossfit = json.loads(path.read_text(encoding="utf-8"))
    mutate(crossfit)
    path.write_text(json.dumps(crossfit), encoding="utf-8")


def _write_46_system_fixture(tmp_path: Path) -> tuple[Path, Path, list[str]]:
    all_elements = tuple("ABCDEFGHIJKL")
    all_systems = [
        "-".join(elements)
        for size in (2, 3)
        for elements in itertools.combinations(all_elements, size)
    ][:230]
    query_by_fold = [all_systems[index : index + 46] for index in range(0, 230, 46)]
    primary_systems = query_by_fold[0]
    task_rows = [
        {"chemical_system": system, "pair_id": f"{system}-{index}"}
        for system_index, system in enumerate(all_systems)
        for index in range(12 + system_index)
    ]
    eligible = [system for systems in query_by_fold for system in systems]
    folds = []
    for fold_index, query_systems in enumerate(query_by_fold):
        fit_systems = [system for system in eligible if system not in query_systems]
        folds.append(
            {
                "fold_index": fold_index,
                "query_systems": query_systems,
                "fit_systems": fit_systems,
            }
        )
    task_path = tmp_path / "task-46.json"
    task_path.write_text(
        json.dumps({"release_id": "fixture-46", "development_pairs": task_rows}),
        encoding="utf-8",
    )
    crossfit_path = tmp_path / "crossfit-46.json"
    crossfit_path.write_text(
        json.dumps(
            {
                "release_id": "fixture-46",
                "task_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
                "eligible_systems": eligible,
                "fold_count": 5,
                "folds": folds,
            }
        ),
        encoding="utf-8",
    )
    return task_path, crossfit_path, primary_systems


def test_manifest_selects_one_deterministic_supported_system_per_tercile(
    tmp_path: Path,
) -> None:
    task_path, crossfit_path, query_by_fold = _write_fixture(tmp_path)

    result = build(task_path, crossfit_path, tmp_path / "manifest.json")
    repeat = build(task_path, crossfit_path, tmp_path / "repeat.json")

    assert result["assignment_uses_target_outcomes"] is False
    assert result["fold_count"] == 5
    assert result["folds"] == repeat["folds"]
    assert len({system for fold in result["folds"] for system in fold["query_systems"]}) == 15
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

    first_fold = result["folds"][0]
    expected_order = sorted(
        query_by_fold["0"],
        key=lambda system: (3 + query_by_fold["0"].index(system), system),
    )
    expected_bins = {
        "low": expected_order[:3],
        "middle": expected_order[3:6],
        "high": expected_order[6:9],
    }
    for bin_name in ("low", "middle", "high"):
        stratum = first_fold["candidate_count_strata"][bin_name]
        assert stratum["ordered_systems"] == expected_bins[bin_name]
        expected_winner = min(
            stratum["supported_systems"],
            key=lambda system: hashlib.sha256(
                f"fixture-release||e55-cal-convergence-v1||0||{bin_name}||{system}".encode()
            ).hexdigest(),
        )
        assert stratum["selected_system"] == expected_winner
    assert [
        first_fold["candidate_count_strata"][name]["candidate_count_range"]
        for name in ("low", "middle", "high")
    ] == [[3, 5], [6, 8], [9, 11]]


def test_manifest_uses_exact_nondivisible_46_system_terciles_and_hash_winners(
    tmp_path: Path,
) -> None:
    task_path, crossfit_path, primary_systems = _write_46_system_fixture(tmp_path)

    result = build(task_path, crossfit_path, tmp_path / "manifest.json")
    primary_fold = result["folds"][0]
    ordered = sorted(
        primary_systems,
        key=lambda system: (12 + primary_systems.index(system), system),
    )
    expected_bins = {
        "low": ordered[:15],
        "middle": ordered[15:30],
        "high": ordered[30:46],
    }

    assert [len(expected_bins[name]) for name in ("low", "middle", "high")] == [
        15,
        15,
        16,
    ]
    for bin_name in ("low", "middle", "high"):
        stratum = primary_fold["candidate_count_strata"][bin_name]
        assert stratum["ordered_systems"] == expected_bins[bin_name]
        expected_winner = min(
            expected_bins[bin_name],
            key=lambda system: hashlib.sha256(
                f"fixture-46||e55-cal-convergence-v1||0||{bin_name}||{system}".encode()
            ).hexdigest(),
        )
        assert stratum["selected_system"] == expected_winner


def test_manifest_accepts_task_only_systems_but_ignores_them_for_selection(
    tmp_path: Path,
) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["development_pairs"].append(
        {"chemical_system": "TASK-ONLY", "pair_id": "TASK-ONLY-0"}
    )
    task_path.write_text(json.dumps(task), encoding="utf-8")
    crossfit = json.loads(crossfit_path.read_text(encoding="utf-8"))
    crossfit["task_sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
    crossfit_path.write_text(json.dumps(crossfit), encoding="utf-8")

    result = build(task_path, crossfit_path, tmp_path / "manifest.json")

    eligible = set(crossfit["eligible_systems"])
    assert result["eligible_system_count"] == len(eligible)
    assert all(
        "TASK-ONLY" not in stratum["candidate_counts"]
        for fold in result["folds"]
        for stratum in fold["candidate_count_strata"].values()
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


def test_manifest_rejects_task_missing_eligible_system(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    crossfit = json.loads(crossfit_path.read_text(encoding="utf-8"))
    missing_system = crossfit["eligible_systems"][0]
    task["development_pairs"] = [
        row
        for row in task["development_pairs"]
        if row["chemical_system"] != missing_system
    ]
    task_path.write_text(json.dumps(task), encoding="utf-8")
    crossfit["task_sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
    crossfit_path.write_text(json.dumps(crossfit), encoding="utf-8")

    with pytest.raises(ValueError, match="missing cross-fit eligible systems"):
        build(task_path, crossfit_path, tmp_path / "manifest.json")


def test_manifest_requires_explicit_eligible_systems(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)

    _rewrite_crossfit(
        crossfit_path,
        lambda crossfit: crossfit.pop("eligible_systems"),
    )

    with pytest.raises(ValueError, match="eligible_systems"):
        build(task_path, crossfit_path, tmp_path / "manifest.json")


def test_manifest_rejects_duplicate_fold_ids(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)

    def duplicate_fold(crossfit: dict[str, object]) -> None:
        crossfit["folds"][4]["fold_index"] = 3

    _rewrite_crossfit(crossfit_path, duplicate_fold)

    with pytest.raises(ValueError, match="fold indices"):
        build(task_path, crossfit_path, tmp_path / "manifest.json")


def test_manifest_rejects_overlapping_original_query_systems(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)

    def overlap_query(crossfit: dict[str, object]) -> None:
        crossfit["folds"][1]["query_systems"][0] = crossfit["folds"][0]["query_systems"][0]

    _rewrite_crossfit(crossfit_path, overlap_query)

    with pytest.raises(ValueError, match="query systems"):
        build(task_path, crossfit_path, tmp_path / "manifest.json")


def test_manifest_rejects_missing_original_fit_roster(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)

    _rewrite_crossfit(
        crossfit_path,
        lambda crossfit: crossfit["folds"][0].pop("fit_systems"),
    )

    with pytest.raises(ValueError, match="fit_systems"):
        build(task_path, crossfit_path, tmp_path / "manifest.json")


def test_manifest_rejects_output_inside_git(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    repo_output = Path(__file__).resolve().parents[1] / ".e55-convergence-test.json"

    with pytest.raises(ValueError, match="outside Git"):
        build(task_path, crossfit_path, repo_output)


def test_manifest_rejects_output_inside_second_git_repository(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    other_repo = tmp_path / "other-repo"
    (other_repo / ".git").mkdir(parents=True)
    output = other_repo / "nested" / "manifest.json"

    with pytest.raises(ValueError, match="outside Git"):
        build(task_path, crossfit_path, output)


def test_manifest_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    task_path, crossfit_path, _ = _write_fixture(tmp_path)
    output = tmp_path / "manifest.json"
    build(task_path, crossfit_path, output)

    with pytest.raises(FileExistsError, match="overwrite"):
        build(task_path, crossfit_path, output)
