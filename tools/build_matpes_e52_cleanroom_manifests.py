"""Freeze E52 development and secondary-confirmation system rosters."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_sha256(values: set[str]) -> str:
    return hashlib.sha256(
        "".join(f"{value}\n" for value in sorted(values)).encode()
    ).hexdigest()


def build(
    *,
    task_path: Path,
    prior_crossfit_path: Path,
    output_dir: Path,
    development_fold_indices: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir.resolve().is_relative_to(repo_root):
        raise ValueError("E52 manifests must remain outside Git")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    prior = json.loads(prior_crossfit_path.read_text(encoding="utf-8"))
    task_sha = _sha256(task_path)
    if prior.get("task_sha256") != task_sha:
        raise ValueError("prior cross-fit manifest does not match task")
    folds_by_index = {int(fold["fold_index"]): fold for fold in prior["folds"]}
    if set(development_fold_indices) - set(folds_by_index):
        raise ValueError("requested development fold is unavailable")
    selected_folds = [folds_by_index[index] for index in development_fold_indices]
    development_systems = {
        str(system) for fold in selected_folds for system in fold["query_systems"]
    }
    if sum(len(fold["query_systems"]) for fold in selected_folds) != len(
        development_systems
    ):
        raise ValueError("development folds overlap")
    all_systems = set(task["development_systems"])
    confirmation_systems = all_systems - development_systems
    if not confirmation_systems:
        raise ValueError("secondary confirmation roster is empty")

    development_path = output_dir / "matpes-e52-cleanroom-development-crossfit.json"
    confirmation_path = output_dir / "matpes-e52-secondary-confirmation-crossfit.json"
    manifest_path = output_dir / "matpes-e52-cleanroom-split-manifest.json"
    for path in (development_path, confirmation_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    development_folds = [
        {
            **fold,
            "source_fold_index": int(fold["fold_index"]),
            "fold_index": new_index,
            "fit_system_count": len(development_systems - set(fold["query_systems"])),
        }
        for new_index, fold in enumerate(selected_folds)
    ]
    development = {
        "schema_version": 1,
        "status": "e52_cleanroom_development_crossfit_frozen",
        "release_id": task["release_id"],
        "task_sha256": task_sha,
        "eligible_system_count": len(development_systems),
        "eligible_systems": sorted(development_systems),
        "fold_count": len(development_folds),
        "folds": development_folds,
        "assignment_rule": "reuse prior outcome-independent folds 1--5; exclude all other systems",
        "assignment_uses_target_outcomes": False,
        "excluded_secondary_confirmation_systems": sorted(confirmation_systems),
    }
    confirmation = {
        "schema_version": 1,
        "status": "e52_secondary_confirmation_execution_split_frozen",
        "release_id": task["release_id"],
        "task_sha256": task_sha,
        "eligible_system_count": len(all_systems),
        "eligible_systems": sorted(all_systems),
        "fold_count": 1,
        "folds": [
            {
                "fold_index": 0,
                "query_systems": sorted(confirmation_systems),
                "system_count": len(confirmation_systems),
                "fit_system_count": len(development_systems),
            }
        ],
        "fit_systems_are_exactly_development_roster": True,
        "assignment_rule": "complement of frozen 230-system development roster",
        "assignment_uses_target_outcomes": False,
        "evidence_scope": (
            "secondary held-out rerun under the frozen E52 protocol; systems were exposed in "
            "historical experiments and are not an untouched confirmation panel"
        ),
    }
    development_path.write_text(
        json.dumps(development, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    confirmation_path.write_text(
        json.dumps(confirmation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = {
        "schema_version": 1,
        "status": "e52_cleanroom_split_complete",
        "task_sha256": task_sha,
        "prior_crossfit_sha256": _sha256(prior_crossfit_path),
        "development_system_count": len(development_systems),
        "development_system_set_sha256": _set_sha256(development_systems),
        "development_manifest_path": str(development_path.resolve()),
        "development_manifest_sha256": _sha256(development_path),
        "secondary_confirmation_system_count": len(confirmation_systems),
        "secondary_confirmation_system_set_sha256": _set_sha256(confirmation_systems),
        "secondary_confirmation_manifest_path": str(confirmation_path.resolve()),
        "secondary_confirmation_manifest_sha256": _sha256(confirmation_path),
        "target_outcomes_used_for_split": False,
        "secondary_confirmation_is_untouched": False,
    }
    manifest_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--prior-crossfit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--development-fold-indices", type=int, nargs="+", default=(1, 2, 3, 4, 5))
    args = parser.parse_args()
    result = build(
        task_path=args.task,
        prior_crossfit_path=args.prior_crossfit,
        output_dir=args.output_dir,
        development_fold_indices=tuple(args.development_fold_indices),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
