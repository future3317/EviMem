"""Launch the write-once E55 Delta-Hull and CAL numerical convergence campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROTOCOL = "e55-numerical-convergence-v1"
SEED = 20260810
DELTA_POSTERIOR_SAMPLE_COUNTS = (64, 128, 256, 512, 1024)
CAL_POSTERIOR_SAMPLE_COUNTS = (100, 200, 400)
CAL_FANTASY_COUNTS = (5, 10, 20)
DELTA_FANTASY_COUNT = 10
DELTA_WORKERS = 1
CAL_WORKERS = 8
DELTA_SELECTION_TIMEOUT_SECONDS = 7200.0
CAL_SELECTION_TIMEOUT_SECONDS = 21600.0
STAGES = ("delta", "cal")


@dataclass(frozen=True)
class Unit:
    command: tuple[str, ...]
    output: Path
    log: Path
    identity: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _assert_outside_git(path: Path) -> None:
    current = path.resolve()
    while True:
        if current.name == ".git" or (current / ".git").exists():
            raise ValueError("E55 outputs must remain outside Git")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _command(
    *,
    runner: Path,
    task: Path,
    vault: Path,
    output: Path,
    crossfit: Path,
    fold_index: int,
    sample_count: int,
    fantasy_count: int,
    workers: int,
    timeout: float,
    policy: str,
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
        str(sample_count),
        "--fantasy-count",
        str(fantasy_count),
        "--hull-candidate-workers",
        str(workers),
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
        policy,
    )


def _read_crossfit(path: Path, task_sha256: str, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("task_sha256") != task_sha256:
        raise ValueError(f"{label} does not match the E52 task")
    if int(payload.get("fold_count", 0)) != 5 or len(payload.get("folds", [])) != 5:
        raise ValueError(f"{label} must contain five folds")
    folds = sorted(payload["folds"], key=lambda fold: int(fold["fold_index"]))
    if [int(fold["fold_index"]) for fold in folds] != list(range(5)):
        raise ValueError(f"{label} fold indices must be 0 through 4")
    return payload


def _validate_delta_crossfit(payload: dict[str, Any]) -> None:
    eligible = [str(system) for system in payload.get("eligible_systems", ())]
    if len(eligible) != 230 or len(set(eligible)) != 230:
        raise ValueError("E52 Delta cross-fit requires 230 explicit unique eligible systems")
    eligible_set = set(eligible)
    folds = sorted(payload["folds"], key=lambda fold: int(fold["fold_index"]))
    query_sets: list[set[str]] = []
    for fold in folds:
        query_systems = [str(system) for system in fold.get("query_systems", ())]
        fit_systems = [str(system) for system in fold.get("fit_systems", ())]
        if len(query_systems) != 46 or len(set(query_systems)) != 46:
            raise ValueError("E52 Delta folds require 46 unique query systems")
        if len(fit_systems) != 184 or len(set(fit_systems)) != 184:
            raise ValueError("E52 Delta folds require 184 unique fit systems")
        query_set = set(query_systems)
        if not query_set <= eligible_set:
            raise ValueError("E52 Delta fold query systems are outside eligible systems")
        if set(fit_systems) != eligible_set - query_set:
            raise ValueError("E52 Delta fit systems are not the query complement")
        query_sets.append(query_set)
    if any(left & right for index, left in enumerate(query_sets) for right in query_sets[index + 1 :]):
        raise ValueError("E52 Delta query folds must be pairwise disjoint")
    if set().union(*query_sets) != eligible_set:
        raise ValueError("E52 Delta query folds must cover eligible systems exactly")


def _runner_manifest(
    *,
    path: Path,
    task_sha256: str,
    source_manifest_sha256: str,
    fold: dict[str, Any],
) -> Path:
    """Write the one-fold runner manifest preserving E55's original fit roster."""
    query_systems = [str(system) for system in fold["query_systems"]]
    fit_systems = [str(system) for system in fold["fit_systems"]]
    if len(query_systems) != 3 or len(fit_systems) != 184:
        raise ValueError("E55 CAL folds require three query and 184 fit systems")
    if len(set(query_systems)) != 3 or len(set(fit_systems)) != 184:
        raise ValueError("E55 CAL fold systems must be unique")
    if set(query_systems) & set(fit_systems):
        raise ValueError("E55 CAL fit and query rosters overlap")
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "task_sha256": task_sha256,
        "source_e55_manifest_sha256": source_manifest_sha256,
        "eligible_systems": sorted((*query_systems, *fit_systems)),
        "fold_count": 1,
        "folds": [{"fold_index": 0, "query_systems": query_systems}],
    }
    try:
        _write_json_exclusive(path, payload)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"existing E55 runner manifest has wrong identity: {path}")
    return path


def _unit(
    *,
    stage: str,
    runner: Path,
    task: Path,
    vault: Path,
    crossfit: Path,
    source_crossfit_sha256: str,
    source_crossfit_key: str,
    fold_index: int,
    runner_fold_index: int,
    query_systems: list[str],
    fit_systems: list[str],
    output: Path,
    sample_count: int,
    fantasy_count: int,
    workers: int,
    timeout: float,
    policy: str,
) -> Unit:
    identity = {
        "protocol": PROTOCOL,
        "stage": stage,
        "task_sha256": _sha256(task),
        "vault_sha256": _sha256(vault),
        "runner_path": str(runner),
        "runner_sha256": _sha256(runner),
        source_crossfit_key: source_crossfit_sha256,
        "runner_crossfit_manifest_sha256": _sha256(crossfit),
        "fold_index": fold_index,
        "runner_fold_index": runner_fold_index,
        "query_systems": query_systems,
        "query_system_count": len(query_systems),
        "fit_systems": fit_systems,
        "fit_system_count": len(fit_systems),
        "fit_system_order_semantics": "set equality; unified runner consumes sorted complement",
        "budget": 6,
        "maximum_budget": 6,
        "minimum_candidates": 12,
        "seed": SEED,
        "posterior_sample_count": sample_count,
        "fantasy_count": fantasy_count,
        "hull_candidate_workers": workers,
        "hull_backend": "fixed_composition",
        "transport_family": "hierarchical_matern52_frozen_structure",
        "selection_timeout_seconds": timeout,
        "policy": policy,
        "output": str(output),
        "log": str(output.with_suffix(".log")),
    }
    return Unit(
        command=_command(
            runner=runner,
            task=task,
            vault=vault,
            output=output,
            crossfit=crossfit,
            fold_index=runner_fold_index,
            sample_count=sample_count,
            fantasy_count=fantasy_count,
            workers=workers,
            timeout=timeout,
            policy=policy,
        ),
        output=output,
        log=output.with_suffix(".log"),
        identity=identity,
    )


def build_units(
    *,
    full_pool_root: Path,
    cal_manifest: Path,
    output_root: Path,
    runner: Path,
    stages: tuple[str, ...],
) -> list[Unit]:
    """Build the frozen E55 units without opening task outcomes."""
    _assert_outside_git(output_root)
    stages = tuple(dict.fromkeys(stages))
    if not stages or set(stages) - set(STAGES):
        raise ValueError("E55 stages must be one or both of: delta, cal")
    task = full_pool_root / "matpes-e52-pool-100-task.json"
    vault = full_pool_root / "matpes-e52-pool-100-vault.json"
    development_crossfit = full_pool_root / "matpes-e52-pool-100-crossfit.json"
    for path in (task, vault, development_crossfit, cal_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    canonical_runner = Path(__file__).with_name(
        "run_matpes_protocol_closed_loop_exploratory.py"
    ).resolve()
    if runner.resolve() != canonical_runner:
        raise ValueError(f"E55 requires the canonical runner: {canonical_runner}")
    runner = canonical_runner
    task_sha256 = _sha256(task)
    development = _read_crossfit(
        development_crossfit, task_sha256, label="E52 development cross-fit manifest"
    )
    _validate_delta_crossfit(development)
    cal = _read_crossfit(cal_manifest, task_sha256, label="E55 CAL manifest")
    development_sha256 = _sha256(development_crossfit)
    cal_sha256 = _sha256(cal_manifest)
    if cal.get("development_crossfit_sha256") != development_sha256:
        raise ValueError("E55 CAL manifest does not match the original development cross-fit")

    units: list[Unit] = []
    development_folds = sorted(development["folds"], key=lambda fold: int(fold["fold_index"]))
    if "delta" in stages:
        for fold in development_folds:
            fold_index = int(fold["fold_index"])
            query_systems = [str(system) for system in fold["query_systems"]]
            fit_systems = [str(system) for system in fold["fit_systems"]]
            if len(query_systems) != 46 or len(fit_systems) != 184:
                raise ValueError("E52 Delta folds require 46 query and 184 fit systems")
            for sample_count in DELTA_POSTERIOR_SAMPLE_COUNTS:
                output = output_root / "delta" / (
                    f"delta-fold{fold_index + 1}-m{sample_count}-k{DELTA_FANTASY_COUNT}-b6.json"
                )
                units.append(
                    _unit(
                        stage="delta",
                        runner=runner,
                        task=task,
                        vault=vault,
                        crossfit=development_crossfit,
                        source_crossfit_sha256=development_sha256,
                        source_crossfit_key="crossfit_manifest_sha256",
                        fold_index=fold_index,
                        runner_fold_index=fold_index,
                        query_systems=query_systems,
                        fit_systems=fit_systems,
                        output=output,
                        sample_count=sample_count,
                        fantasy_count=DELTA_FANTASY_COUNT,
                        workers=DELTA_WORKERS,
                        timeout=DELTA_SELECTION_TIMEOUT_SECONDS,
                        policy="delta_hull_active_search",
                    )
                )
    if "cal" in stages:
        for fold in sorted(cal["folds"], key=lambda value: int(value["fold_index"])):
            fold_index = int(fold["fold_index"])
            runner_crossfit = _runner_manifest(
                path=output_root / "cal-runner-manifests" / f"fold{fold_index + 1}.json",
                task_sha256=task_sha256,
                source_manifest_sha256=cal_sha256,
                fold=fold,
            )
            query_systems = [str(system) for system in fold["query_systems"]]
            fit_systems = [str(system) for system in fold["fit_systems"]]
            for sample_count in CAL_POSTERIOR_SAMPLE_COUNTS:
                for fantasy_count in CAL_FANTASY_COUNTS:
                    output = output_root / "cal" / (
                        f"cal-fold{fold_index + 1}-m{sample_count}-k{fantasy_count}-b6.json"
                    )
                    units.append(
                        _unit(
                            stage="cal",
                            runner=runner,
                            task=task,
                            vault=vault,
                            crossfit=runner_crossfit,
                            source_crossfit_sha256=cal_sha256,
                            source_crossfit_key="e55_manifest_sha256",
                            fold_index=fold_index,
                            runner_fold_index=0,
                            query_systems=query_systems,
                            fit_systems=fit_systems,
                            output=output,
                            sample_count=sample_count,
                            fantasy_count=fantasy_count,
                            workers=CAL_WORKERS,
                            timeout=CAL_SELECTION_TIMEOUT_SECONDS,
                            policy="cal_style_hull_entropy",
                        )
                    )
    return units


def _validate_existing(unit: Unit, payload: dict[str, Any]) -> None:
    identity = unit.identity
    if payload.get("status") != "exploratory_development_systems_only_not_confirmatory":
        raise ValueError(f"existing output has wrong status: {unit.output}")
    expected = {
        "script_sha256": identity["runner_sha256"],
        "task_sha256": identity["task_sha256"],
        "oracle_vault_sha256": identity["vault_sha256"],
        "active_policies": [identity["policy"]],
        "transport_fit_system_count": identity["fit_system_count"],
        "transport_fit_and_query_systems_disjoint": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"existing output has wrong {key}: {unit.output}")
    query_systems = [str(system) for system in payload.get("query_systems", ())]
    if len(query_systems) != identity["query_system_count"] or set(query_systems) != set(
        identity["query_systems"]
    ):
        raise ValueError(f"existing output has wrong query systems: {unit.output}")
    if set(payload.get("transport_fit_systems", ())) != set(identity["fit_systems"]):
        raise ValueError(f"existing output has wrong transport_fit_systems: {unit.output}")
    expected_config = {
        "query_budget": identity["budget"],
        "maximum_budget": identity["maximum_budget"],
        "minimum_candidates": identity["minimum_candidates"],
        "seed": identity["seed"],
        "posterior_sample_count": identity["posterior_sample_count"],
        "fantasy_count": identity["fantasy_count"],
        "hull_candidate_workers": identity["hull_candidate_workers"],
        "hull_backend": identity["hull_backend"],
        "transport_family": identity["transport_family"],
        "rollout_selection_timeout_seconds": identity["selection_timeout_seconds"],
        "crossfit_manifest_sha256": identity["runner_crossfit_manifest_sha256"],
        "crossfit_fold_index": identity["runner_fold_index"],
    }
    config = payload.get("config", {})
    for key, value in expected_config.items():
        if config.get(key) != value:
            raise ValueError(f"existing output has wrong {key}: {unit.output}")
    systems = payload.get("systems")
    if not isinstance(systems, dict) or set(systems) != set(identity["query_systems"]):
        raise ValueError(f"existing output has missing or duplicate systems: {unit.output}")
    terminal_metrics = (
        "final_causal_confirmed_discoveries",
        "oracle_pool_confirmed_discoveries",
        "oracle_pool_discovery_ceiling",
        "oracle_pool_discovery_gap_to_ceiling",
        "invalidated_causal_discoveries_by_oracle_pool_hull",
        "oracle_pool_final_labels_by_pair_id",
        "trace_checksum",
        "event_log_sha256",
        "wall_seconds",
    )
    for system in identity["query_systems"]:
        system_payload = systems[system]
        if system_payload.get("budget") != 6:
            raise ValueError(f"existing output has wrong system budget: {unit.output}")
        if not isinstance(system_payload.get("transport_element_support"), bool):
            raise ValueError(f"existing output lacks transport support: {unit.output}")
        strategies = system_payload.get("strategies")
        if not isinstance(strategies, dict) or set(strategies) != {identity["policy"]}:
            raise ValueError(f"existing output has wrong system policy roster: {unit.output}")
        strategy = strategies[identity["policy"]]
        selected_ids = strategy.get("selected_pair_ids")
        rounds = strategy.get("policy_decision_rounds")
        if not isinstance(selected_ids, list) or len(selected_ids) != 6 or len(set(selected_ids)) != 6:
            raise ValueError(f"existing output has wrong selected_pair_ids: {unit.output}")
        if not isinstance(rounds, list) or [round_.get("round_index") for round_ in rounds] != list(
            range(1, 7)
        ):
            raise ValueError(f"existing output has wrong policy decision rounds: {unit.output}")
        if any(key not in strategy for key in terminal_metrics):
            raise ValueError(f"existing output lacks terminal metrics: {unit.output}")


def _write_failure(unit: Unit, returncode: int | None, reason: str) -> None:
    failure = unit.output.with_suffix(".failure.json")
    payload = {
        "status": "failed_incomplete",
        "identity": unit.identity,
        "command": list(unit.command),
        "returncode": returncode,
        "reason": reason,
        "log": str(unit.log),
    }
    try:
        _write_json_exclusive(failure, payload)
    except FileExistsError:
        existing = json.loads(failure.read_text(encoding="utf-8"))
        if existing.get("status") != "failed_incomplete" or existing.get("identity") != unit.identity:
            raise ValueError(f"existing E55 failure marker has wrong identity: {failure}")


def _run_unit(unit: Unit) -> str:
    failure = unit.output.with_suffix(".failure.json")
    if failure.exists():
        existing = json.loads(failure.read_text(encoding="utf-8"))
        if existing.get("status") != "failed_incomplete" or existing.get("identity") != unit.identity:
            raise ValueError(f"existing E55 failure marker has wrong identity: {failure}")
        raise RuntimeError(f"registered E55 unit already failed: {failure}")
    if unit.output.exists():
        try:
            _validate_existing(unit, json.loads(unit.output.read_text(encoding="utf-8")))
        except Exception as error:
            try:
                _write_failure(unit, None, f"output identity validation failed: {error}")
            except FileExistsError:
                pass
            raise
        return f"resume-skip={unit.output}"
    if unit.log.exists():
        raise FileExistsError(f"refusing to overwrite E55 log: {unit.log}")
    unit.output.parent.mkdir(parents=True, exist_ok=True)
    with unit.log.open("x", encoding="utf-8") as handle:
        completed = subprocess.run(unit.command, stdout=handle, stderr=subprocess.STDOUT)
    if completed.returncode:
        _write_failure(unit, completed.returncode, "runner returned nonzero")
        raise subprocess.CalledProcessError(completed.returncode, unit.command)
    if not unit.output.is_file():
        _write_failure(unit, 0, "runner exited without an output")
        raise RuntimeError(f"runner exited without an output: {unit.output}")
    try:
        _validate_existing(unit, json.loads(unit.output.read_text(encoding="utf-8")))
    except Exception as error:
        try:
            _write_failure(unit, 0, f"output identity validation failed: {error}")
        except FileExistsError:
            pass
        raise
    return f"complete={unit.output}"


def _write_top_level_manifest(path: Path, units: list[Unit]) -> None:
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "runner_path": units[0].identity["runner_path"] if units else None,
        "runner_sha256": units[0].identity["runner_sha256"] if units else None,
        "unit_count": len(units),
        "units": [
            {
                "command": list(unit.command),
                "output": str(unit.output),
                "log": str(unit.log),
                "identity": unit.identity,
            }
            for unit in units
        ],
    }
    try:
        _write_json_exclusive(path, payload)
    except FileExistsError:
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"existing E55 unit manifest has wrong identity: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-pool-root", type=Path, required=True)
    parser.add_argument("--cal-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
    )
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    parser.add_argument("--max-workers", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.max_workers <= 5:
        raise ValueError("max-workers must be between 1 and 5")
    units = build_units(
        full_pool_root=args.full_pool_root,
        cal_manifest=args.cal_manifest,
        output_root=args.output_root,
        runner=args.runner,
        stages=tuple(dict.fromkeys(args.stages)),
    )
    _write_top_level_manifest(args.output_root / "e55-unit-manifest.json", units)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=min(args.max_workers, len(units))) as executor:
        futures = {executor.submit(_run_unit, unit): unit.output for unit in units}
        for future in as_completed(futures):
            output = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as error:  # noqa: BLE001 - retain every independent failure
                failures.append(f"{output}: {error}")
                print(f"failure={output}: {error}", flush=True)
    if failures:
        raise RuntimeError("E55 convergence failures:\n" + "\n".join(failures))


if __name__ == "__main__":
    main()
