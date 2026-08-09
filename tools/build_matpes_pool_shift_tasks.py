"""Build nested, outcome-independent MatPES candidate-pool variants for E52-C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _pair_set_sha256(pair_ids: set[str]) -> str:
    return hashlib.sha256(
        "".join(f"{pair_id}\n" for pair_id in sorted(pair_ids)).encode()
    ).hexdigest()


def build(
    *,
    task_path: Path,
    vault_path: Path,
    output_dir: Path,
    fractions: tuple[float, ...] = (0.70, 0.85, 1.0),
    crossfit_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Write exact-join task/vault variants without using target outcomes to select IDs."""

    repo_root = Path(__file__).resolve().parents[1]
    if output_dir.resolve().is_relative_to(repo_root):
        raise ValueError("pool-shift tasks and vaults must remain outside Git")
    normalized = tuple(sorted(set(float(value) for value in fractions)))
    if not normalized or any(value <= 0.0 or value > 1.0 for value in normalized):
        raise ValueError("pool fractions must be unique values in (0, 1]")
    if normalized[-1] != 1.0:
        raise ValueError("pool-shift roster must include the 100% reference pool")

    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    pair_key = "development_pairs" if "development_pairs" in task else "confirmatory_pairs"
    split = "development" if pair_key == "development_pairs" else "confirmatory"
    system_key = f"{split}_systems"
    initial_key = f"{split}_initial_phase_entries"
    rows = list(task[pair_key])
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("input task contains duplicate pair IDs")
    outcomes = {str(row["pair_id"]): row for row in vault["target_outcomes"]}
    if set(outcomes) != set(pair_ids):
        raise ValueError("input task/vault join is not exact")

    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_system[str(row["chemical_system"])].append(row)
    release_id = str(task["release_id"])
    ranked = {
        system: sorted(
            system_rows,
            key=lambda row: _stable_hash(
                release_id,
                "e52-pool-shift-v1",
                system,
                str(row["pair_id"]),
            ),
        )
        for system, system_rows in by_system.items()
    }

    crossfit_manifest = None
    if crossfit_manifest_path is not None:
        crossfit_manifest = json.loads(crossfit_manifest_path.read_text(encoding="utf-8"))
        if crossfit_manifest.get("task_sha256") != _sha256(task_path):
            raise ValueError("cross-fit manifest does not match the source task")
        unavailable = set(crossfit_manifest["eligible_systems"]) - set(by_system)
        if unavailable:
            raise ValueError(f"cross-fit manifest contains unavailable systems: {unavailable}")

    planned_paths: list[Path] = []
    for fraction in normalized:
        tag = f"{round(100 * fraction):03d}"
        planned_paths.extend(
            (
                output_dir / f"matpes-e52-pool-{tag}-task.json",
                output_dir / f"matpes-e52-pool-{tag}-vault.json",
            )
        )
        if crossfit_manifest is not None:
            planned_paths.append(output_dir / f"matpes-e52-pool-{tag}-crossfit.json")
    manifest_path = output_dir / "matpes-e52-pool-shift-manifest.json"
    planned_paths.append(manifest_path)
    existing = [path for path in planned_paths if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite {existing[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Any] = {}
    previous_ids: set[str] = set()
    retired = {
        pair_key,
        system_key,
        "system_summary",
        "selected_pair_id_set_sha256",
        "selection_rule",
        "status",
        "pool_shift",
    }
    for fraction in normalized:
        tag = f"{round(100 * fraction):03d}"
        selected_by_system = {
            system: system_rows[: max(1, math.ceil(len(system_rows) * fraction))]
            for system, system_rows in ranked.items()
        }
        selected_rows = [
            row for system in sorted(selected_by_system) for row in selected_by_system[system]
        ]
        selected_ids = {str(row["pair_id"]) for row in selected_rows}
        if previous_ids and not previous_ids <= selected_ids:
            raise AssertionError("pool variants are not nested")
        previous_ids = selected_ids
        selected_checksum = _pair_set_sha256(selected_ids)
        task_output = output_dir / f"matpes-e52-pool-{tag}-task.json"
        vault_output = output_dir / f"matpes-e52-pool-{tag}-vault.json"
        variant_task = {
            **{key: value for key, value in task.items() if key not in retired},
            "status": "e52_outcome_independent_pool_shift_task",
            system_key: sorted(selected_by_system),
            pair_key: selected_rows,
            initial_key: {
                system: task[initial_key][system] for system in sorted(selected_by_system)
            },
            "selected_pair_id_set_sha256": selected_checksum,
            "selection_rule": (
                "nested prefix after SHA256(release_id, e52-pool-shift-v1, "
                "chemical_system, pair_id); ceil(original_count*fraction); no target outcome"
            ),
            "pool_shift": {
                "protocol_id": "e52-pool-shift-v1",
                "fraction": fraction,
                "parent_task_sha256": _sha256(task_path),
                "outcome_used_for_selection": False,
            },
            "system_summary": {
                system: {
                    "original_candidate_count": len(ranked[system]),
                    "selected_candidate_count": len(selected_by_system[system]),
                }
                for system in sorted(selected_by_system)
            },
        }
        variant_vault = {
            **{
                key: value
                for key, value in vault.items()
                if key not in {"target_outcomes", "selected_pair_id_set_sha256", "status"}
            },
            "status": "e52_pool_shift_oracle_vault",
            "selected_pair_id_set_sha256": selected_checksum,
            "target_outcomes": [outcomes[pair_id] for pair_id in sorted(selected_ids)],
        }
        task_output.write_text(
            json.dumps(variant_task, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        vault_output.write_text(
            json.dumps(variant_vault, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        crossfit_output = None
        crossfit_sha256 = None
        if crossfit_manifest is not None:
            crossfit_output = output_dir / f"matpes-e52-pool-{tag}-crossfit.json"
            variant_crossfit = {
                **crossfit_manifest,
                "status": "e52_pool_shift_crossfit_roster",
                "task_sha256": _sha256(task_output),
                "parent_crossfit_manifest_sha256": _sha256(crossfit_manifest_path),
                "pool_shift_fraction": fraction,
                "assignment_uses_target_outcomes": False,
            }
            crossfit_output.write_text(
                json.dumps(variant_crossfit, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            crossfit_sha256 = _sha256(crossfit_output)
        variants[tag] = {
            "fraction": fraction,
            "system_count": len(selected_by_system),
            "pair_count": len(selected_rows),
            "pair_id_set_sha256": selected_checksum,
            "task_path": str(task_output.resolve()),
            "task_sha256": _sha256(task_output),
            "vault_path": str(vault_output.resolve()),
            "vault_sha256": _sha256(vault_output),
            "crossfit_manifest_path": (
                None if crossfit_output is None else str(crossfit_output.resolve())
            ),
            "crossfit_manifest_sha256": crossfit_sha256,
        }

    if previous_ids != set(pair_ids):
        raise AssertionError("100% pool does not reproduce the input pair set")
    manifest = {
        "schema_version": 1,
        "status": "e52_pool_shift_tasks_complete",
        "source_task_sha256": _sha256(task_path),
        "source_vault_sha256": _sha256(vault_path),
        "selection_uses_target_outcomes": False,
        "variants": variants,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, default=None)
    parser.add_argument("--fractions", type=float, nargs="+", default=(0.70, 0.85, 1.0))
    args = parser.parse_args()
    result = build(
        task_path=args.task,
        vault_path=args.vault,
        output_dir=args.output_dir,
        fractions=tuple(args.fractions),
        crossfit_manifest_path=args.crossfit_manifest,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
