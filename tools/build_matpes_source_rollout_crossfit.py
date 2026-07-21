"""Build an outcome-independent six-fold Source-Rollout development plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def _stratum(system: str) -> str:
    element_count = len(system.split("-"))
    if element_count == 2:
        return "binary"
    if element_count == 3:
        return "ternary"
    return "quaternary_or_higher"


def build(
    *,
    task_path: Path,
    opened_result_path: Path,
    output_path: Path,
    fold_count: int = 6,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if fold_count < 2:
        raise ValueError("cross-fitting requires at least two folds")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    opened = json.loads(opened_result_path.read_text(encoding="utf-8"))
    if opened.get("split") != "confirmatory" or not opened.get(
        "evaluation_systems_accessed"
    ):
        raise ValueError("cross-fit exclusion source must be an opened confirmatory result")
    eligible = tuple(sorted(set(opened["transport_fit_systems"])))
    opened_systems = tuple(sorted(set(opened["query_systems"])))
    task_systems = {row["chemical_system"] for row in task["development_pairs"]}
    if (
        not eligible
        or set(eligible) - task_systems
        or set(opened_systems) & set(eligible)
        or len(eligible) != int(opened["transport_fit_system_count"])
    ):
        raise ValueError("opened split does not define a valid disjoint development set")
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    release_id = str(task["release_id"])
    for stratum in ("binary", "ternary", "quaternary_or_higher"):
        systems = sorted(
            (system for system in eligible if _stratum(system) == stratum),
            key=lambda system: _stable_hash(
                release_id,
                "source-rollout-delta-hull-crossfit-v1",
                system,
            ),
        )
        for index, system in enumerate(systems):
            folds[index % fold_count].append(system)
    fold_payload = [
        {
            "fold_index": index,
            "query_systems": sorted(systems),
            "system_count": len(systems),
            "stratum_counts": {
                stratum: sum(_stratum(system) == stratum for system in systems)
                for stratum in ("binary", "ternary", "quaternary_or_higher")
            },
        }
        for index, systems in enumerate(folds)
    ]
    if set().union(*(set(row["query_systems"]) for row in fold_payload)) != set(eligible):
        raise AssertionError("cross-fit folds do not cover the eligible systems exactly")
    payload = {
        "schema_version": 1,
        "status": "outcome_independent_development_crossfit_plan",
        "method": "source_rollout_delta_hull",
        "task_sha256": _sha256(task_path),
        "opened_result_sha256": _sha256(opened_result_path),
        "release_id": release_id,
        "fold_count": fold_count,
        "eligible_systems": list(eligible),
        "eligible_system_count": len(eligible),
        "opened_evaluation_systems_excluded": list(opened_systems),
        "opened_evaluation_system_count": len(opened_systems),
        "assignment_uses_target_outcomes": False,
        "assignment_rule": (
            "within complexity stratum, sort SHA256(release_id || "
            "source-rollout-delta-hull-crossfit-v1 || exact_system), then round-robin"
        ),
        "folds": fold_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--opened-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=6)
    args = parser.parse_args()
    payload = build(
        task_path=args.task,
        opened_result_path=args.opened_result,
        output_path=args.output,
        fold_count=args.fold_count,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
