"""Run the registered E32 delayed-label objective/lookahead suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

POLICIES = (
    "source_margin",
    "posterior_mean_target_margin",
    "posterior_current_hull_probability",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "delta_hull_anchored_rollout",
    "independent_confirmation_source_rollout",
)


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
        raise


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
    parser.add_argument("--posterior-sample-count", type=int, default=1024)
    args = parser.parse_args()
    budgets = tuple(args.budgets)
    folds = tuple(args.folds)
    if budgets != tuple(sorted(set(budgets))) or any(
        budget not in range(1, 7) for budget in budgets
    ):
        raise ValueError("E32 budgets must be an ordered subset of 1..6")
    if folds != tuple(sorted(set(folds))) or any(fold not in range(5) for fold in folds):
        raise ValueError("E32 folds must be an ordered subset of 0..4")
    if args.max_systems < 1 or args.posterior_sample_count < 16:
        raise ValueError("E32 runtime limits are invalid")
    manifest = json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
    if not set(folds) <= set(range(len(manifest["folds"]))):
        raise ValueError("requested fold is outside the cross-fit manifest")
    repo_root = Path(__file__).resolve().parents[1]
    identity = {
        "protocol": "docs/DELAYED_LABEL_FOLLOWUP_PROTOCOL_E32.md",
        "task_sha256": _sha256(args.task),
        "vault_sha256": _sha256(args.vault),
        "crossfit_manifest_sha256": _sha256(args.crossfit_manifest),
        "runner_sha256": _sha256(args.runner),
        "policy_worker_sha256": _sha256(repo_root / "src" / "matmem" / "protocol_policy_worker.py"),
        "acquisition_sha256": _sha256(repo_root / "src" / "matmem" / "protocol_acquisition.py"),
        "seed": 20270720,
        "policies": POLICIES,
        "budgets": budgets,
        "folds": folds,
        "max_systems": args.max_systems,
        "posterior_sample_count": args.posterior_sample_count,
        "hull_backend": "fixed_composition",
        "transport_family": "hierarchical_matern52_frozen_structure",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "e32_protocol_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for budget in budgets:
        for fold in folds:
            output = args.output_root / f"e32-fold{fold + 1}-b{budget}-main.json"
            command = [
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
            _run(command, output=output, identity=identity)


if __name__ == "__main__":
    main()
