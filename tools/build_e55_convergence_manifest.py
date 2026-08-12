"""Build the outcome-independent E55 CAL convergence roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

PROTOCOL = "e55-cal-convergence-v1"
BINS = ("low", "middle", "high")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rank(release_id: str, fold_index: int, bin_name: str, system: str) -> str:
    value = f"{release_id}||{PROTOCOL}||{fold_index}||{bin_name}||{system}"
    return hashlib.sha256(value.encode()).hexdigest()


def _elements(system: str) -> set[str]:
    return set(system.split("-"))


def _set_sha256(values: set[str]) -> str:
    return hashlib.sha256("".join(f"{value}\n" for value in sorted(values)).encode()).hexdigest()


def _assert_outside_git(path: Path) -> None:
    current = path.resolve()
    while True:
        if current.name == ".git" or (current / ".git").exists():
            raise ValueError("E55 convergence manifests must remain outside Git")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _fit_systems(fold: dict[str, Any], eligible: set[str]) -> list[str]:
    query_systems = {str(value) for value in fold["query_systems"]}
    expected = eligible - query_systems
    if "fit_systems" not in fold:
        if "fit_system_count" not in fold:
            raise ValueError("fold without fit_systems must provide fit_system_count")
        if int(fold["fit_system_count"]) != len(expected):
            raise ValueError("fold fit_system_count is not the query complement count")
        return sorted(expected)
    supplied = fold["fit_systems"]
    fit_systems = [str(value) for value in supplied]
    if len(fit_systems) != len(set(fit_systems)):
        raise ValueError("fold fit_systems must be unique")
    if set(fit_systems) != expected:
        raise ValueError("fold fit roster is not the original query complement")
    return fit_systems


def _tercile_boundaries(system_count: int) -> tuple[int, int]:
    return system_count // 3, (2 * system_count) // 3


def build(
    task_path: Path,
    development_crossfit_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Build a write-once, five-fold CAL convergence manifest."""
    _assert_outside_git(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    task = json.loads(task_path.read_text(encoding="utf-8"))
    crossfit = json.loads(development_crossfit_path.read_text(encoding="utf-8"))
    task_sha256 = _sha256(task_path)
    if crossfit.get("task_sha256") != task_sha256:
        raise ValueError("development cross-fit manifest does not match task")
    if int(crossfit.get("fold_count", 0)) != 5 or len(crossfit.get("folds", [])) != 5:
        raise ValueError("E55 convergence requires five development folds")
    source_folds = crossfit["folds"]
    fold_indices = [int(fold["fold_index"]) for fold in source_folds]
    if len(fold_indices) != 5 or set(fold_indices) != set(range(5)):
        raise ValueError("E55 convergence requires fold indices {0,1,2,3,4}")

    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task.get("development_pairs", []):
        system = str(row["chemical_system"])
        rows_by_system[system].append(row)
    if not rows_by_system:
        raise ValueError("task contains no development pairs")

    eligible_values = crossfit.get("eligible_systems")
    if eligible_values is None:
        raise ValueError("cross-fit must explicitly provide eligible_systems")
    eligible_list = [str(value) for value in eligible_values]
    if len(eligible_list) != len(set(eligible_list)):
        raise ValueError("development systems must be unique")
    eligible = set(eligible_list)
    if not eligible <= set(rows_by_system):
        raise ValueError("task is missing cross-fit eligible systems")
    release_id = str(crossfit.get("release_id", task.get("release_id", "")))
    if release_id != str(task.get("release_id", release_id)):
        raise ValueError("task and cross-fit release IDs differ")

    query_sets = [
        {str(value) for value in fold["query_systems"]} for fold in source_folds
    ]
    if any(
        len(set(str(value) for value in fold["query_systems"]))
        != len(fold["query_systems"])
        for fold in source_folds
    ):
        raise ValueError("fold query systems must be unique")
    if any(left & right for index, left in enumerate(query_sets) for right in query_sets[index + 1 :]):
        raise ValueError("original query systems must be pairwise disjoint")
    if set().union(*query_sets) != eligible:
        raise ValueError("original query systems must cover eligible_systems exactly")

    folds: list[dict[str, Any]] = []
    for source_fold in sorted(source_folds, key=lambda value: int(value["fold_index"])):
        fold_index = int(source_fold["fold_index"])
        query_systems = [str(value) for value in source_fold["query_systems"]]
        if not set(query_systems) <= eligible:
            raise ValueError(f"fold {fold_index} contains an unknown query system")
        fit_systems = _fit_systems(source_fold, eligible)
        fit_elements = set().union(*(_elements(system) for system in fit_systems))
        ordered = sorted(
            query_systems,
            key=lambda system: (len(rows_by_system[system]), system),
        )
        low_end, middle_end = _tercile_boundaries(len(ordered))
        slices = (
            ordered[:low_end],
            ordered[low_end:middle_end],
            ordered[middle_end:],
        )
        strata: dict[str, dict[str, Any]] = {}
        selected: list[str] = []
        for bin_name, systems in zip(BINS, slices):
            supported = [
                system for system in systems if _elements(system) <= fit_elements
            ]
            if not supported:
                raise ValueError(
                    f"fold {fold_index} has no fit-element-supported system in {bin_name} bin"
                )
            selected_system = min(
                supported,
                key=lambda system: _rank(release_id, fold_index, bin_name, system),
            )
            selected.append(selected_system)
            strata[bin_name] = {
                "candidate_count_range": [
                    len(rows_by_system[systems[0]]),
                    len(rows_by_system[systems[-1]]),
                ],
                "candidate_counts": {
                    system: len(rows_by_system[system]) for system in systems
                },
                "ordered_systems": systems,
                "supported_systems": supported,
                "selected_system": selected_system,
                "selection_rank": _rank(release_id, fold_index, bin_name, selected_system),
            }
        folds.append(
            {
                "fold_index": fold_index,
                "query_systems": selected,
                "query_system_count": len(selected),
                "fit_systems": fit_systems,
                "fit_system_count": len(fit_systems),
                "candidate_count_strata": strata,
            }
        )
    selected_systems = [system for fold in folds for system in fold["query_systems"]]
    if len(selected_systems) != 15 or len(set(selected_systems)) != 15:
        raise ValueError("E55 convergence must select 15 unique systems")

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "e55_cal_convergence_roster_frozen",
        "protocol": PROTOCOL,
        "release_id": release_id,
        "task_sha256": task_sha256,
        "development_crossfit_sha256": _sha256(development_crossfit_path),
        "eligible_system_count": len(eligible),
        "eligible_systems": eligible_list,
        "fold_count": 5,
        "folds": folds,
        "selected_system_set_sha256": _set_sha256(
            {system for fold in folds for system in fold["query_systems"]}
        ),
        "selection_rule": (
            "within each original fold, sort by task-public candidate count and system, "
            "partition into integer rank terciles, retain fit-element-supported systems, "
            "then select lowest SHA-256 rank of release_id || protocol || fold_index || "
            "bin || system"
        ),
        "assignment_uses_target_outcomes": False,
        "selection_uses_target_outcomes": False,
        "outcome_independence": (
            "Roster uses only task-public candidate counts, cross-fit assignment, "
            "chemical-system strings, and fit-element coverage; no vault is accepted."
        ),
        "original_fit_rosters_preserved": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--development-crossfit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.task, args.development_crossfit, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
