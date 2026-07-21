"""Fit and seal a MatPES transport model on a registered fit-system split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from run_matpes_protocol_closed_loop_exploratory import fit_transport_model_for_task


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(
    *,
    task_path: Path,
    vault_path: Path,
    output_path: Path,
    exclude_task_path: Path | None = None,
    transport_family: str = "ridge_random_intercept",
    ridge_penalty: float = 1.0,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("transport freeze must remain outside Git")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    pair_key = "development_pairs" if "development_pairs" in task else "confirmatory_pairs"
    task_rows = task[pair_key]
    outcomes = {row["pair_id"]: row for row in vault["target_outcomes"]}
    if set(outcomes) != {row["pair_id"] for row in task_rows}:
        raise ValueError("transport fit task/vault join is not exact")
    excluded: set[str] = set()
    if exclude_task_path is not None:
        excluded_task = json.loads(exclude_task_path.read_text(encoding="utf-8"))
        excluded.update(excluded_task.get("development_systems", ()))
        excluded.update(excluded_task.get("confirmatory_systems", ()))
    systems = tuple(sorted({row["chemical_system"] for row in task_rows} - excluded))
    if len(systems) < 2:
        raise ValueError("transport freeze requires at least two disjoint fit systems")
    model = fit_transport_model_for_task(
        task=task,
        outcome_rows=outcomes,
        fit_systems=systems,
        ridge_penalty=ridge_penalty,
        transport_family=transport_family,  # type: ignore[arg-type]
        pairs_key=pair_key,
    )
    payload = {
        "schema_version": 1,
        "status": "frozen_transport_model_v1",
        "task_sha256": _sha256(task_path),
        "vault_sha256": _sha256(vault_path),
        "exclude_task_sha256": None if exclude_task_path is None else _sha256(exclude_task_path),
        "fit_system_ids": list(model.fit_system_ids),
        "fit_row_count": model.fit_row_count,
        "model_checksum": model.identity_checksum,
        "model": model.model_dump(mode="json"),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["payload_sha256"] = hashlib.sha256(canonical).hexdigest()
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-task", type=Path, default=None)
    parser.add_argument(
        "--transport-family",
        choices=("ridge_random_intercept", "hierarchical_matern52_frozen_structure"),
        default="ridge_random_intercept",
    )
    parser.add_argument("--ridge-penalty", type=float, default=1.0)
    args = parser.parse_args()
    freeze(
        task_path=args.task,
        vault_path=args.vault,
        output_path=args.output,
        exclude_task_path=args.exclude_task,
        transport_family=args.transport_family,
        ridge_penalty=args.ridge_penalty,
    )


if __name__ == "__main__":
    main()
