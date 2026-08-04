"""Run the registered Hull-ENS P0 units in parallel.

Only independent (fold, budget) units are parallelized.  The closed-loop
runner, posterior, reveal semantics and hull backend are unchanged from the
registered MatPES development task; this scheduler creates a new method
identity and refuses to overwrite existing outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "hull_ens",
    "safe_hull_ens",
)
METHOD_SEED = 20270804


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    runner = args.runner
    return {
        "protocol": "docs/HULL_ENS_PROTOCOL_V1.md",
        "executor": "hull_ens_parallel_scheduler_v1",
        "task_sha256": _sha256(args.task),
        "vault_sha256": _sha256(args.vault),
        "crossfit_manifest_sha256": _sha256(args.crossfit_manifest),
        "runner_sha256": _sha256(runner),
        "policy_worker_sha256": _sha256(repo_root / "src" / "matmem" / "protocol_policy_worker.py"),
        "acquisition_sha256": _sha256(repo_root / "src" / "matmem" / "protocol_acquisition.py"),
        "policy_registry_sha256": _sha256(repo_root / "src" / "matmem" / "policy_registry.py"),
        "seed": METHOD_SEED,
        "policies": list(POLICIES),
        "budgets": list(args.budgets),
        "folds": list(args.folds),
        "max_systems": args.max_systems,
        "posterior_sample_count": args.posterior_sample_count,
        "fantasy_count": args.fantasy_count,
        "hull_candidate_workers": args.hull_candidate_workers,
        "hull_backend": "fixed_composition",
        "transport_family": "hierarchical_matern52_frozen_structure",
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
        str(METHOD_SEED),
        "--posterior-sample-count",
        str(args.posterior_sample_count),
        "--fantasy-count",
        str(args.fantasy_count),
        "--hull-candidate-workers",
        str(args.hull_candidate_workers),
        "--rollout-selection-timeout-seconds",
        "1800",
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
    *, command: list[str], output: Path, identity: dict[str, Any]
) -> dict[str, Any]:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing output: {output}")
    failure = output.with_suffix(".failure.json")
    if failure.exists():
        raise RuntimeError(f"refusing to reuse failed output identity: {failure}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            command,
            check=True,
            env={
                **os.environ,
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
    parser.add_argument("--fantasy-count", type=int, default=8)
    parser.add_argument("--hull-candidate-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=20)
    args = parser.parse_args()
    args.budgets = tuple(args.budgets)
    args.folds = tuple(args.folds)
    if args.budgets != tuple(sorted(set(args.budgets))) or any(
        budget not in range(1, 7) for budget in args.budgets
    ):
        raise ValueError("budgets must be an ordered subset of 1..6")
    if args.folds != tuple(sorted(set(args.folds))) or any(fold not in range(5) for fold in args.folds):
        raise ValueError("folds must be an ordered subset of 0..4")
    if args.max_systems < 1 or args.posterior_sample_count < 16 or args.fantasy_count < 2:
        raise ValueError("Hull-ENS runtime limits are invalid")
    if args.max_workers < 1 or args.hull_candidate_workers < 1:
        raise ValueError("worker counts must be positive")
    manifest = json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
    if not set(args.folds) <= set(range(len(manifest["folds"]))):
        raise ValueError("requested fold is outside the cross-fit manifest")
    args.output_root.mkdir(parents=True, exist_ok=True)
    identity = _identity(args)
    (args.output_root / "hull_ens_protocol_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    jobs: list[tuple[int, int, Path, list[str]]] = []
    for budget in args.budgets:
        for fold in args.folds:
            output = args.output_root / f"hull-ens-fold{fold + 1}-b{budget}-main.json"
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
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001 - scheduler must report every unit
                result = {"status": "failed", "error": str(error)}
            print(json.dumps({"budget": budget, "fold": fold, **result}, sort_keys=True), flush=True)
            if result["status"] == "failed":
                failures.append(json.dumps(result, sort_keys=True))
    if failures:
        raise SystemExit(f"Hull-ENS parallel run had failures: {failures}")


if __name__ == "__main__":
    main()
