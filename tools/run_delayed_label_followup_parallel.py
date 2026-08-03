"""Run E32-A units in parallel without changing the scientific protocol.

This scheduler is deliberately separate from the registered sequential suite.
It launches one unchanged closed-loop runner per independent (fold, budget)
unit, preserves the frozen policy/settings, and records a distinct executor
identity in the external recovery root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

POLICIES = (
    "source_margin",
    "posterior_mean_target_margin",
    "posterior_current_hull_probability",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "delta_hull_anchored_rollout",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(args: argparse.Namespace, runner: Path, worker: Path) -> dict[str, Any]:
    return {
        "protocol": "docs/DELAYED_LABEL_FOLLOWUP_PROTOCOL_E32.md",
        "executor": "parallel_unit_scheduler_v1",
        "task_sha256": _sha256(args.task),
        "vault_sha256": _sha256(args.vault),
        "crossfit_manifest_sha256": _sha256(args.crossfit_manifest),
        "runner_sha256": _sha256(runner),
        "policy_worker_sha256": _sha256(worker),
        "acquisition_sha256": _sha256(
            runner.parents[1] / "src" / "matmem" / "protocol_acquisition.py"
        ),
        "seed": 20270720,
        "policies": list(POLICIES),
        "budgets": list(args.budgets),
        "folds": list(args.folds),
        "max_systems": args.max_systems,
        "posterior_sample_count": args.posterior_sample_count,
        "hull_backend": "fixed_composition",
        "transport_family": "hierarchical_matern52_frozen_structure",
        "rollout_selection_timeout_seconds": args.rollout_selection_timeout_seconds,
        "max_workers": args.max_workers,
        "blas_threads_per_unit": 1,
    }


def _command(args: argparse.Namespace, output: Path, fold: int, budget: int) -> list[str]:
    return [
        sys.executable,
        str(args.runner),
        "--task",
        str(args.task),
        "--development-vault",
        str(args.vault),
        "--output",
        str(output),
        "--max-systems",
        str(args.max_systems),
        "--query-budget",
        str(budget),
        "--maximum-budget",
        "6",
        "--minimum-candidates",
        "12",
        "--seed",
        "20270720",
        "--posterior-sample-count",
        str(args.posterior_sample_count),
        "--rollout-selection-timeout-seconds",
        str(args.rollout_selection_timeout_seconds),
        "--hull-backend",
        "fixed_composition",
        "--transport-family",
        "hierarchical_matern52_frozen_structure",
        "--crossfit-manifest",
        str(args.crossfit_manifest),
        "--fold-index",
        str(fold),
        "--policies",
        *POLICIES,
    ]


def _run_unit(
    *,
    command: list[str],
    output: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("task_sha256") != identity["task_sha256"]:
            raise ValueError(f"existing output has wrong task identity: {output}")
        return {"status": "resume_skip", "output": str(output)}
    failure = output.with_suffix(".failure.json")
    if failure.exists():
        return {"status": "existing_failure", "failure": str(failure)}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            command,
            check=True,
            env={
                **__import__("os").environ,
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
        )
    except subprocess.CalledProcessError as error:
        failure.write_text(
            json.dumps(
                {
                    "status": "failed_incomplete",
                    "identity": identity,
                    "command": command,
                    "returncode": error.returncode,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {"status": "failed", "failure": str(failure), "returncode": error.returncode}
    return {"status": "complete", "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=(1, 2, 3, 4, 5, 6))
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--max-systems", type=int, default=1000)
    parser.add_argument("--posterior-sample-count", type=int, default=128)
    parser.add_argument("--rollout-selection-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-workers", type=int, default=16)
    args = parser.parse_args()
    args.budgets = tuple(args.budgets)
    args.folds = tuple(args.folds)
    if args.budgets != tuple(sorted(set(args.budgets))) or any(
        budget not in range(1, 7) for budget in args.budgets
    ):
        raise ValueError("budgets must be an ordered subset of 1..6")
    if args.folds != tuple(sorted(set(args.folds))) or any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must be an ordered subset of 0..4")
    if args.max_workers < 1 or args.posterior_sample_count < 16:
        raise ValueError("parallel runtime limits are invalid")
    if args.rollout_selection_timeout_seconds <= 0:
        raise ValueError("rollout timeout must be positive")
    manifest = json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
    if not set(args.folds) <= set(range(len(manifest["folds"]))):
        raise ValueError("requested fold is outside the cross-fit manifest")
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity = _identity(args, args.runner, Path(__file__).resolve().parents[1] / "src" / "matmem" / "protocol_policy_worker.py")
    (args.output_root / "e32_parallel_executor_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    jobs: list[tuple[int, int, Path, list[str]]] = []
    for budget in args.budgets:
        for fold in args.folds:
            output = args.output_root / f"e32-fold{fold + 1}-b{budget}-main.json"
            jobs.append((budget, fold, output, _command(args, output, fold, budget)))
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        pending: dict[Future[dict[str, Any]], tuple[int, int]] = {
            executor.submit(_run_unit, command=command, output=output, identity=identity): (
                budget,
                fold,
            )
            for budget, fold, output, command in jobs
        }
        for future in as_completed(pending):
            budget, fold = pending[future]
            result = future.result()
            print(json.dumps({"budget": budget, "fold": fold, **result}, sort_keys=True), flush=True)
            if result["status"] == "failed":
                failures.append(str(result["failure"]))
    if failures:
        raise SystemExit(f"parallel E32 recovery had failures: {failures}")


if __name__ == "__main__":
    main()
