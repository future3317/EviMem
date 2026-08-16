"""Run the frozen E54 CAL-style hull-entropy MatPES campaign.

This launcher is intentionally separate from the historical E52/E53 launchers.
It reuses the unified secure closed-loop runner and records only unit manifests;
raw JSON traces stay outside the repository.
"""

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
    "delta_hull_active_search",
    "cal_style_hull_entropy",
)
PROTOCOL = "E54-cal-style-hull-entropy-v1"
SEED = 20260810


@dataclass(frozen=True)
class Unit:
    command: tuple[str, ...]
    output: Path
    log: Path
    identity: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(
    *,
    runner: Path,
    task: Path,
    vault: Path,
    output: Path,
    crossfit: Path,
    fold_index: int,
    posterior_sample_count: int,
    fantasy_count: int,
    hull_candidate_workers: int,
    selection_timeout_seconds: float,
    policies: tuple[str, ...],
    seed: int,
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
        str(seed),
        "--posterior-sample-count",
        str(posterior_sample_count),
        "--fantasy-count",
        str(fantasy_count),
        "--hull-candidate-workers",
        str(hull_candidate_workers),
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
        *policies,
    )


def build_units(
    *,
    full_pool_root: Path,
    split_manifest_root: Path,
    secondary_task: Path | None,
    secondary_vault: Path | None,
    output_root: Path,
    runner: Path,
    stage: str,
    posterior_sample_count: int = 200,
    fantasy_count: int = 10,
    hull_candidate_workers: int = 1,
    selection_timeout_seconds: float = 7200.0,
    policies: tuple[str, ...] = POLICIES,
    protocol: str = PROTOCOL,
    seed: int = SEED,
) -> list[Unit]:
    """Build one explicitly selected E54 stage without opening outcomes."""

    repo_root = Path(__file__).resolve().parents[1]
    if output_root.resolve().is_relative_to(repo_root):
        raise ValueError("E54 outputs must remain outside Git")
    if stage not in {"development", "secondary"}:
        raise ValueError("E54 stage must be development or secondary")
    if not policies or len(set(policies)) != len(policies):
        raise ValueError("campaign policy roster must be nonempty and unique")
    if posterior_sample_count < 4 or fantasy_count < 1 or hull_candidate_workers < 1:
        raise ValueError("E54 numerical settings are too small")

    development_task = full_pool_root / "matpes-e52-pool-100-task.json"
    development_vault = full_pool_root / "matpes-e52-pool-100-vault.json"
    development_crossfit = full_pool_root / "matpes-e52-pool-100-crossfit.json"
    secondary_crossfit = (
        split_manifest_root / "matpes-e52-secondary-confirmation-crossfit.json"
    )
    specifications: list[tuple[str, int, Path, Path, bool]]
    if stage == "development":
        required = (development_task, development_vault, development_crossfit)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing frozen E54 development input(s): " + ", ".join(missing))
        task = development_task
        vault = development_vault
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
        if secondary_task is None or secondary_vault is None:
            raise ValueError("secondary stage requires an explicit secondary task and vault")
        required = (secondary_task, secondary_vault, secondary_crossfit)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing frozen E54 secondary input(s): " + ", ".join(missing))
        task = secondary_task
        vault = secondary_vault
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
    for identity_stage, fold, crossfit, output, secondary_is_untouched in specifications:
        identity = {
            "protocol": protocol,
            "stage": identity_stage,
            "task_sha256": _sha256(task),
            "vault_sha256": _sha256(vault),
            "crossfit_manifest_sha256": _sha256(crossfit),
            "fold_index": fold,
            "budget": 6,
            "seed": seed,
            "posterior_sample_count": posterior_sample_count,
            "fantasy_count": fantasy_count,
            "hull_candidate_workers": hull_candidate_workers,
            "hull_backend": "fixed_composition",
            "policies": policies,
            "secondary_is_untouched": secondary_is_untouched,
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
                    fantasy_count=fantasy_count,
                    hull_candidate_workers=hull_candidate_workers,
                    selection_timeout_seconds=selection_timeout_seconds,
                    policies=policies,
                    seed=seed,
                ),
                output=output,
                log=output.with_suffix(".log"),
                identity=identity,
            )
        )
    return units


def _validate_existing(unit: Unit, payload: dict[str, Any]) -> None:
    if payload.get("task_sha256") != unit.identity["task_sha256"]:
        raise ValueError(f"existing output has wrong task identity: {unit.output}")
    if payload.get("oracle_vault_sha256") != unit.identity["vault_sha256"]:
        raise ValueError(f"existing output has wrong vault identity: {unit.output}")
    if tuple(payload.get("active_policies", ())) != tuple(unit.identity["policies"]):
        raise ValueError(f"existing output has wrong policy roster: {unit.output}")
    config = payload.get("config", {})
    for key in (
        "query_budget",
        "posterior_sample_count",
        "fantasy_count",
        "hull_candidate_workers",
        "hull_backend",
    ):
        if config.get(key) != {
            "query_budget": 6,
            "posterior_sample_count": unit.identity["posterior_sample_count"],
            "fantasy_count": unit.identity["fantasy_count"],
            "hull_candidate_workers": unit.identity["hull_candidate_workers"],
            "hull_backend": "fixed_composition",
        }[key]:
            raise ValueError(f"existing output has wrong {key}: {unit.output}")


def _run_unit(unit: Unit) -> str:
    if unit.output.exists():
        _validate_existing(unit, json.loads(unit.output.read_text(encoding="utf-8")))
        return f"resume-skip={unit.output}"
    failure = unit.output.with_suffix(".failure.json")
    if failure.exists():
        raise RuntimeError(f"registered E54 unit already failed: {failure}")
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
    _validate_existing(unit, json.loads(unit.output.read_text(encoding="utf-8")))
    return f"complete={unit.output}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-pool-root", type=Path, required=True)
    parser.add_argument("--split-manifest-root", type=Path, required=True)
    parser.add_argument("--secondary-task", type=Path)
    parser.add_argument("--secondary-vault", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
    )
    parser.add_argument("--stage", choices=("development", "secondary"), default="development")
    parser.add_argument("--posterior-sample-count", type=int, default=200)
    parser.add_argument("--fantasy-count", type=int, default=10)
    parser.add_argument("--hull-candidate-workers", type=int, default=1)
    parser.add_argument("--max-workers", type=int, default=5)
    parser.add_argument("--selection-timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--policies", nargs="+", default=POLICIES)
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.max_workers < 1:
        raise ValueError("max-workers must be positive")
    units = build_units(
        full_pool_root=args.full_pool_root,
        split_manifest_root=args.split_manifest_root,
        secondary_task=args.secondary_task,
        secondary_vault=args.secondary_vault,
        output_root=args.output_root,
        runner=args.runner,
        stage=args.stage,
        posterior_sample_count=args.posterior_sample_count,
        fantasy_count=args.fantasy_count,
        hull_candidate_workers=args.hull_candidate_workers,
        selection_timeout_seconds=args.selection_timeout_seconds,
        policies=tuple(args.policies),
        protocol=args.protocol,
        seed=args.seed,
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
        raise RuntimeError("E54 campaign failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
