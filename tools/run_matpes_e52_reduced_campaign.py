"""Run the reduced E52 objective, pool-shift, and two-step audit campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

OBJECTIVE_POLICIES = (
    "posterior_mean_target_margin",
    "delta_hull_active_search",
)
EQUIVALENCE_POLICIES = (
    "delta_hull_anchored_rollout",
    "protocol_hull_knowledge_gradient",
)


@dataclass(frozen=True)
class Unit:
    command: tuple[str, ...]
    output: Path
    log: Path
    identity: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_command(
    *,
    runner: Path,
    task: Path,
    vault: Path,
    output: Path,
    crossfit: Path,
    fold_index: int,
    budget: int,
    posterior_sample_count: int,
    fantasy_count: int,
    timeout: float,
    policies: tuple[str, ...],
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(runner),
        "--task",
        str(task),
        "--development-vault",
        str(vault),
        "--output",
        str(output),
        "--query-budget",
        str(budget),
        "--maximum-budget",
        "6",
        "--minimum-candidates",
        "12",
        "--seed",
        "20260809",
        "--posterior-sample-count",
        str(posterior_sample_count),
        "--fantasy-count",
        str(fantasy_count),
        "--hull-backend",
        "fixed_composition",
        "--transport-family",
        "hierarchical_matern52_frozen_structure",
        "--rollout-selection-timeout-seconds",
        str(timeout),
        "--crossfit-manifest",
        str(crossfit),
        "--fold-index",
        str(fold_index),
        "--policies",
        *policies,
    )


def _unit(
    *,
    runner: Path,
    task: Path,
    vault: Path,
    crossfit: Path,
    output: Path,
    fold_index: int,
    pool: str,
    budget: int,
    posterior_sample_count: int,
    fantasy_count: int,
    timeout: float,
    policies: tuple[str, ...],
    stage: str,
) -> Unit:
    identity = {
        "protocol": "E52-reduced-development-v2",
        "stage": stage,
        "task_sha256": _sha256(task),
        "vault_sha256": _sha256(vault),
        "crossfit_manifest_sha256": _sha256(crossfit),
        "pool": pool,
        "fold_index": fold_index,
        "budget": budget,
        "seed": 20260809,
        "posterior_sample_count": posterior_sample_count,
        "fantasy_count": fantasy_count,
        "policies": policies,
    }
    return Unit(
        command=_base_command(
            runner=runner,
            task=task,
            vault=vault,
            output=output,
            crossfit=crossfit,
            fold_index=fold_index,
            budget=budget,
            posterior_sample_count=posterior_sample_count,
            fantasy_count=fantasy_count,
            timeout=timeout,
            policies=policies,
        ),
        output=output,
        log=output.with_suffix(".log"),
        identity=identity,
    )


def build_units(
    *,
    full_pool_root: Path,
    query_shift_root: Path,
    equivalence_manifest: Path,
    output_root: Path,
    runner: Path,
    stages: tuple[str, ...],
    posterior_sample_count: int,
    equivalence_sample_count: int,
    fantasy_count: int,
    max_timeout: float,
) -> list[Unit]:
    units: list[Unit] = []
    full_task = full_pool_root / "matpes-e52-pool-100-task.json"
    full_vault = full_pool_root / "matpes-e52-pool-100-vault.json"
    full_crossfit = full_pool_root / "matpes-e52-pool-100-crossfit.json"
    if "objective" in stages:
        for pool in ("070", "085", "100"):
            for fold in range(5):
                if pool == "100":
                    task = full_task
                    vault = full_vault
                    crossfit = full_crossfit
                    fold_index = fold
                else:
                    stem = f"pool-{pool}-fold{fold + 1}"
                    task = query_shift_root / f"{stem}-task.json"
                    vault = query_shift_root / f"{stem}-vault.json"
                    crossfit = query_shift_root / f"{stem}-crossfit.json"
                    fold_index = 0
                output = output_root / "objective" / pool / f"fold{fold + 1}-b6.json"
                units.append(
                    _unit(
                        runner=runner,
                        task=task,
                        vault=vault,
                        crossfit=crossfit,
                        output=output,
                        fold_index=fold_index,
                        pool=pool,
                        budget=6,
                        posterior_sample_count=posterior_sample_count,
                        fantasy_count=fantasy_count,
                        timeout=max_timeout,
                        policies=OBJECTIVE_POLICIES,
                        stage="objective_prefix_curve",
                    )
                )
    if "equivalence" in stages:
        output = output_root / "equivalence" / "two-step-fast-b2.json"
        units.append(
            _unit(
                runner=runner,
                task=full_task,
                vault=full_vault,
                crossfit=equivalence_manifest,
                output=output,
                fold_index=0,
                pool="100",
                budget=2,
                posterior_sample_count=equivalence_sample_count,
                fantasy_count=fantasy_count,
                timeout=max_timeout,
                policies=EQUIVALENCE_POLICIES,
                stage="two_step_equivalence_fast",
            )
        )
    return units


def _run_unit(unit: Unit) -> str:
    if unit.output.exists():
        payload = json.loads(unit.output.read_text(encoding="utf-8"))
        if payload.get("task_sha256") != unit.identity["task_sha256"]:
            raise ValueError(f"existing output has wrong task identity: {unit.output}")
        if tuple(payload.get("active_policies", ())) != tuple(unit.identity["policies"]):
            raise ValueError(f"existing output has wrong policy roster: {unit.output}")
        return f"resume-skip={unit.output}"
    failure = unit.output.with_suffix(".failure.json")
    if failure.exists():
        raise RuntimeError(f"registered unit already failed: {failure}")
    unit.output.parent.mkdir(parents=True, exist_ok=True)
    with unit.log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(unit.command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        failure.write_text(
            json.dumps(
                {
                    "status": "failed_incomplete",
                    "identity": unit.identity,
                    "command": unit.command,
                    "returncode": completed.returncode,
                    "log": str(unit.log),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(completed.returncode, unit.command)
    return f"complete={unit.output}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-pool-root", type=Path, required=True)
    parser.add_argument("--query-shift-root", type=Path, required=True)
    parser.add_argument("--equivalence-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
    )
    parser.add_argument(
        "--stages", nargs="+", choices=("objective", "equivalence"), default=("objective", "equivalence")
    )
    parser.add_argument("--posterior-sample-count", type=int, default=1024)
    parser.add_argument("--equivalence-sample-count", type=int, default=128)
    parser.add_argument("--fantasy-count", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--selection-timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    repo_root = Path(__file__).resolve().parents[1]
    if args.output_root.resolve().is_relative_to(repo_root):
        raise ValueError("E52 outputs must remain outside Git")
    stages = tuple(dict.fromkeys(args.stages))
    units = build_units(
        full_pool_root=args.full_pool_root,
        query_shift_root=args.query_shift_root,
        equivalence_manifest=args.equivalence_manifest,
        output_root=args.output_root,
        runner=args.runner,
        stages=stages,
        posterior_sample_count=args.posterior_sample_count,
        equivalence_sample_count=args.equivalence_sample_count,
        fantasy_count=args.fantasy_count,
        max_timeout=args.selection_timeout_seconds,
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(_run_unit, unit): unit.output for unit in units}
        for future in as_completed(futures):
            output = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:  # noqa: BLE001 - persist independent failures
                failures.append(f"{output}: {error}")
                print(f"failure={output}: {error}", flush=True)
    if failures:
        raise RuntimeError("E52 reduced campaign failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
