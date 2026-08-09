"""Build fold-specific E52 tasks that shrink only the visible query-system pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def _pair_set_sha256(pair_ids: set[str]) -> str:
    return hashlib.sha256(
        "".join(f"{pair_id}\n" for pair_id in sorted(pair_ids)).encode()
    ).hexdigest()


def build(
    *,
    task_path: Path,
    vault_path: Path,
    development_crossfit_path: Path,
    output_dir: Path,
    fractions: tuple[float, ...] = (0.70, 0.85),
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir.resolve().is_relative_to(repo_root):
        raise ValueError("query-pool-shift tasks must remain outside Git")
    normalized = tuple(sorted(set(float(value) for value in fractions)))
    if not normalized or any(value <= 0.0 or value >= 1.0 for value in normalized):
        raise ValueError("query-pool fractions must be unique values in (0, 1)")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    crossfit = json.loads(development_crossfit_path.read_text(encoding="utf-8"))
    if crossfit.get("task_sha256") != _sha256(task_path):
        raise ValueError("development cross-fit manifest does not match task")
    if len(crossfit["folds"]) != 5:
        raise ValueError("E52 query-pool shift requires five development folds")
    development_systems = set(crossfit["eligible_systems"])
    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task["development_pairs"]:
        system = str(row["chemical_system"])
        if system in development_systems:
            rows_by_system[system].append(row)
    if set(rows_by_system) != development_systems:
        raise ValueError("development systems are not exactly available in task")
    outcomes = {str(row["pair_id"]): row for row in vault["target_outcomes"]}
    release_id = str(task["release_id"])
    ranked = {
        system: sorted(
            rows,
            key=lambda row: _stable_hash(
                release_id,
                "e52-query-visible-pool-v1",
                system,
                str(row["pair_id"]),
            ),
        )
        for system, rows in rows_by_system.items()
    }
    retired = {
        "development_pairs",
        "development_systems",
        "development_initial_phase_entries",
        "system_summary",
        "selected_pair_id_set_sha256",
        "selection_rule",
        "status",
        "pool_shift",
    }
    planned: list[Path] = [output_dir / "matpes-e52-query-pool-shift-manifest.json"]
    for fraction in normalized:
        tag = f"{round(100 * fraction):03d}"
        for fold_index in range(5):
            prefix = output_dir / f"pool-{tag}-fold{fold_index + 1}"
            planned.extend(
                (
                    prefix.with_name(prefix.name + "-task.json"),
                    prefix.with_name(prefix.name + "-vault.json"),
                    prefix.with_name(prefix.name + "-crossfit.json"),
                )
            )
    existing = [path for path in planned if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, Any] = {}
    for fraction in normalized:
        tag = f"{round(100 * fraction):03d}"
        for fold_index, source_fold in enumerate(crossfit["folds"]):
            query_systems = set(source_fold["query_systems"])
            fit_systems = development_systems - query_systems
            selected_by_system = {
                system: (
                    ranked[system][: max(1, math.ceil(len(ranked[system]) * fraction))]
                    if system in query_systems
                    else ranked[system]
                )
                for system in sorted(development_systems)
            }
            if min(len(selected_by_system[system]) for system in query_systems) < 12:
                raise ValueError("query-pool shift violates the 12-candidate runner gate")
            selected_rows = [
                row
                for system in sorted(development_systems)
                for row in selected_by_system[system]
            ]
            selected_ids = {str(row["pair_id"]) for row in selected_rows}
            if not selected_ids <= set(outcomes):
                raise ValueError("derived task cannot join the source vault")
            selected_checksum = _pair_set_sha256(selected_ids)
            stem = f"pool-{tag}-fold{fold_index + 1}"
            task_output = output_dir / f"{stem}-task.json"
            vault_output = output_dir / f"{stem}-vault.json"
            crossfit_output = output_dir / f"{stem}-crossfit.json"
            variant_task = {
                **{key: value for key, value in task.items() if key not in retired},
                "status": "e52_query_visible_pool_shift_task",
                "development_systems": sorted(development_systems),
                "development_pairs": selected_rows,
                "development_initial_phase_entries": {
                    system: task["development_initial_phase_entries"][system]
                    for system in sorted(development_systems)
                },
                "selected_pair_id_set_sha256": selected_checksum,
                "selection_rule": (
                    "fit-system rows unchanged; query-system rows are a nested SHA256 prefix; "
                    "no target outcome used"
                ),
                "pool_shift": {
                    "protocol_id": "e52-query-visible-pool-v1",
                    "query_pool_fraction": fraction,
                    "fit_pool_fraction": 1.0,
                    "query_systems": sorted(query_systems),
                    "fit_systems": sorted(fit_systems),
                    "outcome_used_for_selection": False,
                    "parent_task_sha256": _sha256(task_path),
                },
                "system_summary": {
                    system: {
                        "role": "query" if system in query_systems else "fit",
                        "original_candidate_count": len(ranked[system]),
                        "selected_candidate_count": len(selected_by_system[system]),
                    }
                    for system in sorted(development_systems)
                },
            }
            variant_vault = {
                **{
                    key: value
                    for key, value in vault.items()
                    if key not in {"target_outcomes", "selected_pair_id_set_sha256", "status"}
                },
                "status": "e52_query_visible_pool_shift_oracle_vault",
                "selected_pair_id_set_sha256": selected_checksum,
                "target_outcomes": [outcomes[pair_id] for pair_id in sorted(selected_ids)],
            }
            task_output.write_text(
                json.dumps(variant_task, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            vault_output.write_text(
                json.dumps(variant_vault, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            variant_crossfit = {
                "schema_version": 1,
                "status": "e52_query_visible_pool_shift_crossfit",
                "release_id": task["release_id"],
                "task_sha256": _sha256(task_output),
                "eligible_system_count": len(development_systems),
                "eligible_systems": sorted(development_systems),
                "fold_count": 1,
                "folds": [
                    {
                        "fold_index": 0,
                        "source_fold_index": source_fold.get("source_fold_index", fold_index + 1),
                        "query_systems": sorted(query_systems),
                        "system_count": len(query_systems),
                        "fit_system_count": len(fit_systems),
                    }
                ],
                "assignment_rule": "frozen E52 development fold; query pool only is reduced",
                "assignment_uses_target_outcomes": False,
                "parent_development_crossfit_sha256": _sha256(development_crossfit_path),
            }
            crossfit_output.write_text(
                json.dumps(variant_crossfit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            variants[stem] = {
                "query_pool_fraction": fraction,
                "query_system_count": len(query_systems),
                "fit_system_count": len(fit_systems),
                "fit_rows_preserved": all(
                    len(selected_by_system[system]) == len(ranked[system])
                    for system in fit_systems
                ),
                "task_path": str(task_output.resolve()),
                "task_sha256": _sha256(task_output),
                "vault_path": str(vault_output.resolve()),
                "vault_sha256": _sha256(vault_output),
                "crossfit_path": str(crossfit_output.resolve()),
                "crossfit_sha256": _sha256(crossfit_output),
            }
    result = {
        "schema_version": 1,
        "status": "e52_query_visible_pool_shift_tasks_complete",
        "source_task_sha256": _sha256(task_path),
        "source_vault_sha256": _sha256(vault_path),
        "development_crossfit_sha256": _sha256(development_crossfit_path),
        "selection_uses_target_outcomes": False,
        "fit_rows_preserved": True,
        "variants": variants,
    }
    planned[0].write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--development-crossfit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.70, 0.85))
    args = parser.parse_args()
    result = build(
        task_path=args.task,
        vault_path=args.vault,
        development_crossfit_path=args.development_crossfit,
        output_dir=args.output_dir,
        fractions=tuple(args.fractions),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
