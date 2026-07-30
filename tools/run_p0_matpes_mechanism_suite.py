"""Resumable launcher for the locked retrospective MatPES P0 mechanism suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

CORE_POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "independent_confirmation_source_rollout",
)
ABLATION_POLICIES = (
    "source_margin",
    "posterior_mean_target_margin",
    "ridge_margin",
    "ridge_uncertainty",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "diagonal_ic_sarr",
    "independent_mc_ic_sarr",
    "independent_confirmation_source_rollout",
)
RANDOM_SEEDS = tuple(range(20270721, 20270726))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(command: list[str], *, output: Path, identity: dict[str, object]) -> None:
    if output.exists():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("task_sha256") != identity["task_sha256"]:
            raise ValueError(f"existing output has wrong task identity: {output}")
        print(f"resume-skip={output}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        failure = output.with_suffix(".failure.json")
        failure.write_text(
            json.dumps(
                {"status": "failed_incomplete", "identity": identity, "command": command, "returncode": error.returncode},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--crossfit-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"))
    parser.add_argument("--tier", choices=("core", "ablation"), required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=None)
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    args = parser.parse_args()
    budgets = tuple(args.budgets) if args.budgets is not None else ((1, 2, 3, 4, 5, 6) if args.tier == "core" else (6,))
    if tuple(budgets) != tuple(sorted(set(budgets))) or any(
        budget not in {1, 2, 3, 4, 5, 6} for budget in budgets
    ):
        raise ValueError("P0 budgets must be a unique ordered subset of 1..6")
    if args.tier == "ablation" and budgets != (6,):
        raise ValueError("the registered component ablation runs only at B=6")
    manifest = json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
    if not set(args.folds) <= set(range(len(manifest["folds"]))):
        raise ValueError("requested fold index is outside the frozen manifest")
    identity = {
        "protocol": "docs/P0_MECHANISM_SUITE_PROTOCOL.md",
        "task_sha256": _sha256(args.task),
        "vault_sha256": _sha256(args.vault),
        "crossfit_manifest_sha256": _sha256(args.crossfit_manifest),
        "seed": 20270720,
        "tier": args.tier,
        "policies": CORE_POLICIES if args.tier == "core" else ABLATION_POLICIES,
        "random_seeds": () if args.tier == "core" else RANDOM_SEEDS,
    }
    policies = CORE_POLICIES if args.tier == "core" else ABLATION_POLICIES
    for budget in budgets:
        for fold in args.folds:
            main_output = args.output_root / f"matpes-p0v2-{args.tier}-fold{fold + 1}-b{budget}-main.json"
            command = [
                sys.executable,
                str(args.runner),
                "--task", str(args.task),
                "--development-vault", str(args.vault),
                "--output", str(main_output),
                "--query-budget", str(budget),
                "--maximum-budget", "6",
                "--minimum-candidates", "12",
                "--seed", "20270720",
                "--posterior-sample-count", "1024",
                "--hull-backend", "fixed_composition",
                "--transport-family", "hierarchical_matern52_frozen_structure",
                "--crossfit-manifest", str(args.crossfit_manifest),
                "--fold-index", str(fold),
                "--policies", *policies,
            ]
            _run(command, output=main_output, identity=identity)
            if args.tier == "core":
                continue
            for seed in RANDOM_SEEDS:
                random_output = args.output_root / f"matpes-p0v2-ablation-fold{fold + 1}-b{budget}-random-seed{seed}.json"
                random_command = [
                    sys.executable,
                    str(args.runner),
                    "--task", str(args.task),
                    "--development-vault", str(args.vault),
                    "--output", str(random_output),
                    "--query-budget", str(budget),
                    "--maximum-budget", "6",
                    "--minimum-candidates", "12",
                    "--seed", str(seed),
                    "--posterior-sample-count", "1024",
                    "--hull-backend", "fixed_composition",
                    "--transport-family", "hierarchical_matern52_frozen_structure",
                    "--crossfit-manifest", str(args.crossfit_manifest),
                    "--fold-index", str(fold),
                    "--policies", "random",
                ]
                _run(random_command, output=random_output, identity=identity)


if __name__ == "__main__":
    main()
