"""Run the resumable E52 matched-baseline and pool-shift development campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

POLICIES = (
    "posterior_mean_target_margin",
    "delta_hull_active_search",
    "protocol_hull_knowledge_gradient",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_unit(*, command: list[str], output: Path, log: Path, identity: dict[str, Any]) -> str:
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("task_sha256") != identity["task_sha256"]:
            raise ValueError(f"existing output has wrong task identity: {output}")
        if tuple(payload.get("active_policies", ())) != tuple(identity["policies"]):
            raise ValueError(f"existing output has wrong policy roster: {output}")
        return f"resume-skip={output}"
    failure = output.with_suffix(".failure.json")
    if failure.exists():
        raise RuntimeError(f"registered unit already failed: {failure}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        failure.write_text(
            json.dumps(
                {
                    "status": "failed_incomplete",
                    "identity": identity,
                    "command": command,
                    "returncode": completed.returncode,
                    "log": str(log),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise subprocess.CalledProcessError(completed.returncode, command)
    return f"complete={output}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-task-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
    )
    parser.add_argument("--pools", nargs="+", choices=("070", "085", "100"), default=("070", "085", "100"))
    parser.add_argument("--budgets", type=int, nargs="+", default=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--posterior-sample-count", type=int, default=1024)
    parser.add_argument("--fantasy-count", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--selection-timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    pools = tuple(args.pools)
    budgets = tuple(args.budgets)
    folds = tuple(args.folds)
    if len(set(pools)) != len(pools):
        raise ValueError("pool tags must be unique")
    if tuple(sorted(set(budgets))) != budgets or any(value not in range(1, 7) for value in budgets):
        raise ValueError("budgets must be an ordered unique subset of 1..6")
    if tuple(sorted(set(folds))) != folds or any(value not in range(5) for value in folds):
        raise ValueError("folds must be an ordered unique subset of 0..4")
    if args.posterior_sample_count < 4 or args.fantasy_count < 1 or args.max_workers < 1:
        raise ValueError("invalid numerical or worker setting")
    repo_root = Path(__file__).resolve().parents[1]
    if args.output_root.resolve().is_relative_to(repo_root):
        raise ValueError("E52 outputs must remain outside Git")

    units: list[tuple[list[str], Path, Path, dict[str, Any]]] = []
    for pool in pools:
        task = args.pool_task_root / f"matpes-e52-pool-{pool}-task.json"
        vault = args.pool_task_root / f"matpes-e52-pool-{pool}-vault.json"
        crossfit = args.pool_task_root / f"matpes-e52-pool-{pool}-crossfit.json"
        manifest = json.loads(crossfit.read_text(encoding="utf-8"))
        if len(manifest["folds"]) != 5:
            raise ValueError(f"E52 development cross-fit must contain five folds: {crossfit}")
        task_sha = _sha256(task)
        if manifest.get("task_sha256") != task_sha:
            raise ValueError(f"cross-fit task mismatch: {crossfit}")
        for budget in budgets:
            for fold in folds:
                output = args.output_root / f"pool-{pool}" / f"e52-pool-{pool}-fold{fold + 1}-b{budget}.json"
                log = output.with_suffix(".log")
                identity = {
                    "protocol": "E52-cleanroom-development-v1",
                    "task_sha256": task_sha,
                    "vault_sha256": _sha256(vault),
                    "crossfit_manifest_sha256": _sha256(crossfit),
                    "pool": pool,
                    "fold": fold,
                    "budget": budget,
                    "seed": 20260809,
                    "posterior_sample_count": args.posterior_sample_count,
                    "fantasy_count": args.fantasy_count,
                    "policies": POLICIES,
                }
                command = [
                    sys.executable,
                    str(args.runner),
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
                    str(args.posterior_sample_count),
                    "--fantasy-count",
                    str(args.fantasy_count),
                    "--hull-backend",
                    "fixed_composition",
                    "--transport-family",
                    "hierarchical_matern52_frozen_structure",
                    "--rollout-selection-timeout-seconds",
                    str(args.selection_timeout_seconds),
                    "--crossfit-manifest",
                    str(crossfit),
                    "--fold-index",
                    str(fold),
                    "--policies",
                    *POLICIES,
                ]
                units.append((command, output, log, identity))

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(
                _run_unit, command=command, output=output, log=log, identity=identity
            ): output
            for command, output, log, identity in units
        }
        for future in as_completed(futures):
            output = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:  # noqa: BLE001 - persist all independent unit failures
                failures.append(f"{output}: {error}")
                print(f"failure={output}: {error}", flush=True)
    if failures:
        raise RuntimeError("E52 campaign failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
