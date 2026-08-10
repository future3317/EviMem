"""Run the frozen E53 matched-adjudicator MatPES campaign."""

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

POLICIES = (
    "posterior_mean_target_margin",
    "matched_local_hull_probability",
    "delta_hull_active_search",
)
PROTOCOL = "E53-objective-isolation-v1"
SEED = 20260810


@dataclass(frozen=True)
class Unit:
    command: tuple[str, ...]
    output: Path
    log: Path
    identity: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(
    *,
    runner: Path,
    task: Path,
    vault: Path,
    output: Path,
    crossfit: Path,
    fold_index: int,
    posterior_sample_count: int,
    selection_timeout_seconds: float,
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
        "6",
        "--maximum-budget",
        "6",
        "--minimum-candidates",
        "12",
        "--seed",
        str(SEED),
        "--posterior-sample-count",
        str(posterior_sample_count),
        "--fantasy-count",
        "1",
        "--hull-backend",
        "fixed_composition",
        "--transport-family",
        "hierarchical_matern52_frozen_structure",
        "--rollout-selection-timeout-seconds",
        str(selection_timeout_seconds),
        "--crossfit-manifest",
        str(crossfit),
        "--fold-index",
        str(fold_index),
        "--policies",
        *POLICIES,
    )


def build_units(
    *,
    full_pool_root: Path,
    split_manifest_root: Path,
    output_root: Path,
    runner: Path,
    stage: str,
    posterior_sample_count: int,
    selection_timeout_seconds: float,
) -> list[Unit]:
    """Build one explicitly selected E53 stage without opening outcomes."""

    repo_root = Path(__file__).resolve().parents[1]
    if output_root.resolve().is_relative_to(repo_root):
        raise ValueError("E53 outputs must remain outside Git")
    if stage not in {"development", "secondary"}:
        raise ValueError("E53 stage must be development or secondary")
    if posterior_sample_count < 4:
        raise ValueError("E53 needs at least four posterior samples")

    task = full_pool_root / "matpes-e52-pool-100-task.json"
    vault = full_pool_root / "matpes-e52-pool-100-vault.json"
    development_crossfit = full_pool_root / "matpes-e52-pool-100-crossfit.json"
    secondary_crossfit = (
        split_manifest_root / "matpes-e52-secondary-confirmation-crossfit.json"
    )
    required = (task, vault, development_crossfit, secondary_crossfit)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing frozen E53 input(s): " + ", ".join(missing))

    specifications: list[tuple[str, int, Path, Path, bool]]
    if stage == "development":
        specifications = [
            (
                "development_crossfit",
                fold,
                development_crossfit,
                output_root / "development" / f"fold{fold + 1}-b6.json",
                False,
            )
            for fold in range(5)
        ]
    else:
        specifications = [
            (
                "secondary_heldout_matpes_rerun",
                0,
                secondary_crossfit,
                output_root / "secondary" / "heldout-b6.json",
                False,
            )
        ]

    units: list[Unit] = []
    for identity_stage, fold, crossfit, output, untouched in specifications:
        identity = {
            "protocol": PROTOCOL,
            "stage": identity_stage,
            "task_sha256": _sha256(task),
            "vault_sha256": _sha256(vault),
            "crossfit_manifest_sha256": _sha256(crossfit),
            "fold_index": fold,
            "budget": 6,
            "seed": SEED,
            "posterior_sample_count": posterior_sample_count,
            "policies": POLICIES,
            "secondary_is_untouched": untouched,
        }
        units.append(
            Unit(
                command=_command(
                    runner=runner,
                    task=task,
                    vault=vault,
                    output=output,
                    crossfit=crossfit,
                    fold_index=fold,
                    posterior_sample_count=posterior_sample_count,
                    selection_timeout_seconds=selection_timeout_seconds,
                ),
                output=output,
                log=output.with_suffix(".log"),
                identity=identity,
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
        raise RuntimeError(f"registered E53 unit already failed: {failure}")
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
    parser.add_argument("--split-manifest-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
    )
    parser.add_argument("--stage", choices=("development", "secondary"), default="development")
    parser.add_argument("--posterior-sample-count", type=int, default=1024)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--selection-timeout-seconds", type=float, default=7200.0)
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    units = build_units(
        full_pool_root=args.full_pool_root,
        split_manifest_root=args.split_manifest_root,
        output_root=args.output_root,
        runner=args.runner,
        stage=args.stage,
        posterior_sample_count=args.posterior_sample_count,
        selection_timeout_seconds=args.selection_timeout_seconds,
    )
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(units))) as executor:
        futures = {executor.submit(_run_unit, unit): unit.output for unit in units}
        for future in as_completed(futures):
            output = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:  # noqa: BLE001 - preserve independent failures
                failures.append(f"{output}: {error}")
                print(f"failure={output}: {error}", flush=True)
    if failures:
        raise RuntimeError("E53 campaign failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
