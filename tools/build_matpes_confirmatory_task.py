"""Build an outcome-independent fresh MatPES confirmatory split.

The input task may be the all-eligible development-schema task produced by
``build_matpes_protocol_task.py``.  Selection uses only observable pool size,
original-parent uniqueness and a release/system hash; oracle values are copied
only after the system set has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def _system_stratum(system: str) -> str:
    count = len(system.split("-"))
    return "binary" if count == 2 else "ternary" if count == 3 else "quaternary_or_higher"


def build(
    *,
    all_task_path: Path,
    all_vault_path: Path,
    development_task_path: Path,
    task_output: Path,
    vault_output: Path,
    max_systems_per_stratum: int = 8,
    max_systems: int | None = None,
    minimum_candidates: int = 16,
    minimum_parents: int = 1,
) -> dict[str, Any]:
    for path in (task_output, vault_output):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
    repo_root = Path(__file__).resolve().parents[1]
    if any(path.resolve().is_relative_to(repo_root) for path in (task_output, vault_output)):
        raise ValueError("confirmatory task and vault must remain outside Git")

    all_task = json.loads(all_task_path.read_text(encoding="utf-8"))
    all_vault = json.loads(all_vault_path.read_text(encoding="utf-8"))
    development_task = json.loads(development_task_path.read_text(encoding="utf-8"))
    pair_key = "confirmatory_pairs" if "confirmatory_pairs" in all_task else "development_pairs"
    initial_key = (
        "confirmatory_initial_phase_entries"
        if "confirmatory_initial_phase_entries" in all_task
        else "development_initial_phase_entries"
    )
    rows = list(all_task[pair_key])
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("all-eligible task contains duplicate pair IDs")
    outcomes = {row["pair_id"]: row for row in all_vault["target_outcomes"]}
    if set(outcomes) != {row["pair_id"] for row in rows}:
        raise ValueError("all-eligible task/vault join is not exact")
    excluded_systems = set(development_task.get("development_systems", ()))
    excluded_ids = {
        row["pair_id"] for row in development_task.get("development_pairs", ())
    }
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["chemical_system"] not in excluded_systems and row["pair_id"] not in excluded_ids:
            by_system[row["chemical_system"]].append(row)
    eligible: dict[str, list[dict[str, Any]]] = {}
    for system, system_rows in by_system.items():
        # The original MP parent is the independent configuration identity.
        # Keep exactly one row per parent, with a deterministic observable ID
        # tie-break, before applying the candidate/parent gates.
        by_parent: dict[str, dict[str, Any]] = {}
        for row in sorted(system_rows, key=lambda value: value["pair_id"]):
            parent = str(row.get("original_mp_id", row["pair_id"]))
            by_parent.setdefault(parent, row)
        deduplicated = list(by_parent.values())
        parents = set(by_parent)
        if len(deduplicated) >= minimum_candidates and len(parents) >= minimum_parents:
            eligible[system] = deduplicated
    selected: list[str] = []
    selection_by_stratum: dict[str, list[str]] = {}
    for stratum in ("binary", "ternary", "quaternary_or_higher"):
        systems = sorted(
            (system for system in eligible if _system_stratum(system) == stratum),
            key=lambda system: _stable_hash(
                str(all_task["release_id"]), "confirmatory-fresh-v1", system
            ),
        )[:max_systems_per_stratum]
        selection_by_stratum[stratum] = systems
        selected.extend(systems)
    selected = sorted(selected, key=lambda system: _stable_hash(str(all_task["release_id"]), system))
    if max_systems is not None:
        selected = selected[:max_systems]
    if not selected:
        raise ValueError("no fresh exact systems satisfy the confirmatory pool gate")

    selected_rows = [row for system in selected for row in eligible[system]]
    selected_ids = {row["pair_id"] for row in selected_rows}
    selected_id_checksum = hashlib.sha256(
        "".join(f"{pair_id}\n" for pair_id in sorted(selected_ids)).encode()
    ).hexdigest()
    initial_entries = {
        system: all_task[initial_key].get(
            system,
            [
                {
                    "entry_id": f"reference-{element}",
                    "composition": {element: 1.0},
                    "corrected_total_energy_ev": 0.0,
                }
                for element in system.split("-")
            ],
        )
        for system in selected
    }
    task = {
        **all_task,
        "schema_version": max(int(all_task.get("schema_version", 1)), 2),
        "status": "confirmatory_fresh_task_frozen_selection",
        "confirmatory_systems": selected,
        "confirmatory_pairs": selected_rows,
        "confirmatory_initial_phase_entries": initial_entries,
        "selected_pair_id_set_sha256": selected_id_checksum,
        "selection_rule": (
            "exclude every development system and pair; retain exact-system pools with "
            "candidate/parent gates; SHA256(release, confirmatory-fresh-v1, system) "
            "within binary/ternary/quaternary strata; no outcome used"
        ),
        "development_exclusion": {
            "task_sha256": _sha256(development_task_path),
            "systems": sorted(excluded_systems),
            "pair_count": len(excluded_ids),
        },
        "selection_by_stratum": selection_by_stratum,
    }
    vault = {
        "schema_version": max(int(all_vault.get("schema_version", 1)), 2),
        "release_id": all_vault["release_id"],
        "status": "confirmatory_sealed_oracle_vault",
        "selected_pair_id_set_sha256": selected_id_checksum,
        "target_outcomes": [
            {**outcomes[pair_id], "split": "confirmatory"}
            for pair_id in sorted(selected_ids)
        ],
    }
    task_output.parent.mkdir(parents=True, exist_ok=True)
    vault_output.parent.mkdir(parents=True, exist_ok=True)
    task_output.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vault_output.write_text(json.dumps(vault, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "task_sha256": _sha256(task_output),
        "vault_sha256": _sha256(vault_output),
        "development_exclusion_sha256": _sha256(development_task_path),
        "selected_systems": selected,
        "selected_pair_count": len(selected_rows),
        "selection_by_stratum": selection_by_stratum,
        "oracle_values_used_for_selection": False,
    }
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-task", type=Path, required=True)
    parser.add_argument("--all-vault", type=Path, required=True)
    parser.add_argument("--development-task", type=Path, required=True)
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--vault-output", type=Path, required=True)
    parser.add_argument("--max-systems-per-stratum", type=int, default=8)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--minimum-candidates", type=int, default=16)
    parser.add_argument("--minimum-parents", type=int, default=1)
    args = parser.parse_args()
    build(
        all_task_path=args.all_task,
        all_vault_path=args.all_vault,
        development_task_path=args.development_task,
        task_output=args.task_output,
        vault_output=args.vault_output,
        max_systems_per_stratum=args.max_systems_per_stratum,
        max_systems=args.max_systems,
        minimum_candidates=args.minimum_candidates,
        minimum_parents=args.minimum_parents,
    )


if __name__ == "__main__":
    main()
