"""Audit write-once E55 numerical-convergence artifacts without rerunning them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

DELTA_GRID = {(fold, samples, 10) for fold in range(5) for samples in (64, 128, 256, 512, 1024)}
CAL_GRID = {
    (fold, samples, fantasies)
    for fold in range(5)
    for samples in (100, 200, 400)
    for fantasies in (5, 10, 20)
}
REFERENCE = {"delta": (1024, 10), "cal": (400, 20)}
POLICIES = {"delta": "delta_hull_active_search", "cal": "cal_style_hull_entropy"}
FROZEN_IDENTITY_FIELDS = (
    "task_sha256",
    "vault_sha256",
    "runner_path",
    "runner_sha256",
    "budget",
    "maximum_budget",
    "minimum_candidates",
    "seed",
    "hull_backend",
    "transport_family",
)


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("runtime traces must be nonempty and finite")
    return {
        "population": "per_system_policy_wall_seconds_from_traces",
        "system_count": int(len(array)),
        "median": float(np.median(array)),
        "iqr": [float(np.quantile(array, 0.25)), float(np.quantile(array, 0.75))],
    }


def _option(command: list[str], name: str) -> str:
    if command.count(name) != 1:
        raise ValueError(f"E55 command must contain exactly one {name}")
    index = command.index(name)
    if index + 1 == len(command):
        raise ValueError(f"E55 command lacks value for {name}")
    return str(command[index + 1])


def _json_source(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing E55 {label} path: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid E55 {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"invalid E55 {label}: {path}")
    return payload


def _source_folds(
    payload: dict[str, Any],
    *,
    count: int,
    label: str,
    eligible: set[str] | None = None,
    expected_fit_by_fold: dict[int, set[str]] | None = None,
) -> dict[int, tuple[set[str], set[str]]]:
    folds = payload.get("folds")
    if payload.get("fold_count") != 5 or not isinstance(folds, list) or len(folds) != 5:
        raise ValueError(f"E55 {label} must contain five folds")
    result = {}
    query_sets = []
    for fold in folds:
        index = int(fold.get("fold_index", -1))
        query = {str(system) for system in fold.get("query_systems", ())}
        if "fit_systems" in fold:
            fit = {str(system) for system in fold["fit_systems"]}
        elif expected_fit_by_fold is not None and fold.get("fit_system_count") == len(
            expected_fit_by_fold.get(index, set())
        ):
            fit = expected_fit_by_fold[index]
        elif eligible is not None and fold.get("fit_system_count") == len(eligible) - len(query):
            fit = eligible - query
        else:
            fit = set()
        if index in result or len(query) != count or len(fit) != 184:
            raise ValueError(f"E55 {label} has wrong query systems or fit roster")
        result[index] = (query, fit)
        query_sets.append(query)
    if set(result) != set(range(5)) or any(
        left & right
        for position, left in enumerate(query_sets)
        for right in query_sets[position + 1 :]
    ):
        raise ValueError(f"E55 {label} query systems overlap across folds")
    return result


def _audit_sources(
    input_root: Path, units: list[dict[str, Any]]
) -> tuple[dict[str, str], dict[str, dict[int, tuple[set[str], set[str]]]]]:
    task_paths: set[Path] = set()
    vault_paths: set[Path] = set()
    runner_paths: set[Path] = set()
    delta_crossfit_paths: set[Path] = set()
    identities: list[dict[str, Any]] = []
    for unit in units:
        identity = unit.get("identity")
        command = unit.get("command")
        if not isinstance(identity, dict) or not isinstance(command, list) or len(command) < 2:
            raise ValueError("E55 unit lacks command identity")
        stage = identity.get("stage")
        task_path = Path(_option(command, "--task")).resolve()
        vault_path = Path(_option(command, "--development-vault")).resolve()
        runner_path = Path(str(command[1])).resolve()
        crossfit_path = Path(_option(command, "--crossfit-manifest")).resolve()
        task_paths.add(task_path)
        vault_paths.add(vault_path)
        runner_paths.add(runner_path)
        if stage == "delta":
            delta_crossfit_paths.add(crossfit_path)
        expected_options = {
            "--output": identity.get("output", unit.get("output")),
            "--query-budget": identity.get("budget"),
            "--maximum-budget": identity.get("maximum_budget"),
            "--minimum-candidates": identity.get("minimum_candidates"),
            "--seed": identity.get("seed"),
            "--posterior-sample-count": identity.get("posterior_sample_count"),
            "--fantasy-count": identity.get("fantasy_count"),
            "--hull-candidate-workers": identity.get("hull_candidate_workers"),
            "--hull-backend": identity.get("hull_backend"),
            "--transport-family": identity.get("transport_family"),
            "--rollout-selection-timeout-seconds": identity.get(
                "selection_timeout_seconds"
            ),
            "--fold-index": identity.get("runner_fold_index", identity.get("fold_index")),
            "--policies": identity.get("policy"),
        }
        for option, expected in expected_options.items():
            if _option(command, option) != str(expected):
                raise ValueError(f"E55 command has wrong {option}")
        if runner_path != Path(str(identity.get("runner_path"))).resolve():
            raise ValueError("E55 command has wrong runner path")
        if _sha256(task_path) != identity.get("task_sha256"):
            raise ValueError("E55 task path/hash mismatch")
        if _sha256(vault_path) != identity.get("vault_sha256"):
            raise ValueError("E55 vault path/hash mismatch")
        if _sha256(runner_path) != identity.get("runner_sha256"):
            raise ValueError("E55 runner path/hash mismatch")
        if _sha256(crossfit_path) != identity.get("runner_crossfit_manifest_sha256"):
            raise ValueError("E55 command cross-fit path/hash mismatch")
        identities.append(identity)
    if any(len(paths) != 1 for paths in (task_paths, vault_paths, runner_paths, delta_crossfit_paths)):
        raise ValueError("E55 source paths change across units")
    task_path = next(iter(task_paths))
    crossfit_path = next(iter(delta_crossfit_paths))
    crossfit = _json_source(crossfit_path, "E52 cross-fit")
    task_sha256 = _sha256(task_path)
    if crossfit.get("task_sha256") != task_sha256:
        raise ValueError("E55 E52 cross-fit does not match task")
    eligible = {str(system) for system in crossfit.get("eligible_systems", ())}
    delta_folds = _source_folds(
        crossfit, count=46, label="E52 cross-fit", eligible=eligible
    )
    if len(eligible) != 230 or set().union(*(query for query, _ in delta_folds.values())) != eligible:
        raise ValueError("E55 E52 cross-fit does not cover 230 unique query systems")
    if any(fit != eligible - query for query, fit in delta_folds.values()):
        raise ValueError("E55 E52 cross-fit has wrong fit complement")
    cal_path = input_root / "e55-cal-manifest.json"
    cal = _json_source(cal_path, "CAL manifest")
    if cal.get("task_sha256") != task_sha256 or cal.get(
        "development_crossfit_sha256"
    ) != _sha256(crossfit_path):
        raise ValueError("E55 CAL manifest source hashes mismatch")
    cal_folds = _source_folds(
        cal,
        count=3,
        label="CAL manifest",
        expected_fit_by_fold={fold: fit for fold, (_, fit) in delta_folds.items()},
    )
    cal_sha256 = _sha256(cal_path)
    for identity in identities:
        stage = str(identity["stage"])
        fold = int(identity["fold_index"])
        query, fit = (delta_folds if stage == "delta" else cal_folds)[fold]
        if set(identity.get("query_systems", ())) != query or int(
            identity.get("query_system_count", -1)
        ) != len(query):
            raise ValueError(f"E55 {stage} unit has wrong query systems")
        if set(identity.get("fit_systems", ())) != fit or int(
            identity.get("fit_system_count", -1)
        ) != len(fit):
            raise ValueError(f"E55 {stage} unit has wrong fit systems")
        if stage == "delta" and identity.get("crossfit_manifest_sha256") != _sha256(
            crossfit_path
        ):
            raise ValueError("E55 Delta source cross-fit hash mismatch")
        if stage == "cal" and identity.get("e55_manifest_sha256") != cal_sha256:
            raise ValueError("E55 CAL manifest hash mismatch")
        if stage == "cal":
            runner_path = Path(
                _option(
                    next(unit["command"] for unit in units if unit["identity"] is identity),
                    "--crossfit-manifest",
                )
            ).resolve()
            runner = _json_source(runner_path, "CAL runner manifest")
            runner_folds = runner.get("folds")
            if (
                runner.get("task_sha256") != task_sha256
                or runner.get("source_e55_manifest_sha256") != cal_sha256
                or runner.get("fold_count") != 1
                or not isinstance(runner_folds, list)
                or len(runner_folds) != 1
                or int(runner_folds[0].get("fold_index", -1)) != 0
                or set(runner_folds[0].get("query_systems", ())) != query
                or set(runner.get("eligible_systems", ())) != query | fit
            ):
                raise ValueError("E55 CAL runner manifest content mismatch")
    return (
        {
            "task_path": str(task_path),
            "task_sha256": task_sha256,
            "vault_path": str(next(iter(vault_paths))),
            "vault_sha256": _sha256(next(iter(vault_paths))),
            "runner_path": str(next(iter(runner_paths))),
            "runner_sha256": _sha256(next(iter(runner_paths))),
            "development_crossfit_path": str(crossfit_path),
            "development_crossfit_sha256": _sha256(crossfit_path),
            "cal_manifest_path": str(cal_path.resolve()),
            "cal_manifest_sha256": cal_sha256,
        },
        {"delta": delta_folds, "cal": cal_folds},
    )


def _paired(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("paired terminal T values must be nonempty and finite")
    rng = np.random.default_rng(20260810)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    bootstrap = array[indices].mean(axis=1)
    return {
        "system_count": int(len(array)),
        "paired_mean_difference": float(array.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wins": int(np.sum(array > 0)),
        "ties": int(np.sum(array == 0)),
        "losses": int(np.sum(array < 0)),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    ordered = values[order]
    while start < len(values):
        stop = start + 1
        while stop < len(values) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def _score_vector(event: dict[str, Any]) -> tuple[list[str], np.ndarray] | None:
    diagnostic = event.get("selection_diagnostics")
    if not isinstance(diagnostic, dict):
        return None
    candidate_ids = diagnostic.get("candidate_pair_ids")
    scores = next(
        (
            diagnostic.get(key)
            for key in (
                "candidate_scores",
                "final_stability_probabilities",
                "cal_scores",
                "two_step_scores",
            )
            if isinstance(diagnostic.get(key), dict)
        ),
        None,
    )
    if not isinstance(candidate_ids, list) or not isinstance(scores, dict):
        return None
    if set(candidate_ids) != set(scores) or len(candidate_ids) < 2:
        return None
    return [str(pair_id) for pair_id in candidate_ids], np.asarray(
        [float(scores[pair_id]) for pair_id in candidate_ids], dtype=float
    )


def _validate_unit(identity: dict[str, Any], payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stage = identity.get("stage")
    if stage not in POLICIES:
        raise ValueError("unknown E55 stage")
    expected_policy = POLICIES[stage]
    if payload.get("status") != "exploratory_development_systems_only_not_confirmatory":
        raise ValueError("wrong E55 status")
    expected = {
        "script_sha256": identity.get("runner_sha256"),
        "task_sha256": identity.get("task_sha256"),
        "oracle_vault_sha256": identity.get("vault_sha256"),
        "active_policies": [expected_policy],
        "transport_fit_system_count": identity.get("fit_system_count"),
        "transport_fit_and_query_systems_disjoint": True,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            if key == "active_policies":
                raise ValueError("wrong E55 policy roster")
            raise ValueError(f"wrong E55 {key}")
    if set(payload.get("query_systems", ())) != set(identity.get("query_systems", ())):
        raise ValueError("wrong E55 query roster")
    if set(payload.get("transport_fit_systems", ())) != set(identity.get("fit_systems", ())):
        raise ValueError("wrong E55 fit roster")
    config = payload.get("config", {})
    config_keys = {
        "query_budget": "budget",
        "maximum_budget": "maximum_budget",
        "minimum_candidates": "minimum_candidates",
        "seed": "seed",
        "posterior_sample_count": "posterior_sample_count",
        "fantasy_count": "fantasy_count",
        "hull_candidate_workers": "hull_candidate_workers",
        "hull_backend": "hull_backend",
        "transport_family": "transport_family",
        "rollout_selection_timeout_seconds": "selection_timeout_seconds",
        "crossfit_manifest_sha256": "runner_crossfit_manifest_sha256",
    }
    for key, identity_key in config_keys.items():
        if config.get(key) != identity.get(identity_key):
            raise ValueError(f"wrong E55 {key}")
    systems = payload.get("systems")
    if not isinstance(systems, dict) or set(systems) != set(identity["query_systems"]):
        raise ValueError("wrong E55 system roster")
    rows: dict[str, dict[str, Any]] = {}
    for system, result in systems.items():
        if result.get("budget") != 6:
            raise ValueError("wrong E55 terminal budget")
        strategies = result.get("strategies")
        if not isinstance(strategies, dict) or set(strategies) != {expected_policy}:
            raise ValueError("wrong E55 policy roster")
        row = strategies[expected_policy]
        selected = row.get("selected_pair_ids")
        if not isinstance(selected, list) or len(selected) != 6 or len(set(selected)) != 6:
            raise ValueError("wrong E55 selected actions")
        rounds = row.get("policy_decision_rounds")
        if not isinstance(rounds, list) or [event.get("round_index") for event in rounds] != list(range(1, 7)):
            raise ValueError("wrong E55 decision rounds")
        event_actions = [event.get("selected_pair_id") for event in rounds]
        if event_actions != selected:
            raise ValueError("E55 event actions do not reconcile with selected_pair_ids")
        if any(not event.get("pre_reveal_state_checksum") for event in rounds):
            raise ValueError("E55 decision round lacks pre-reveal state checksum")
        if "oracle_pool_confirmed_discoveries" not in row or "wall_seconds" not in row:
            raise ValueError("missing E55 terminal trace metrics")
        rows[str(system)] = row
    return rows


def _matched_diagnostics(
    current: dict[str, dict[str, Any]], reference: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    agreements: list[float] = []
    first_actions: list[float] = []
    errors: list[float] = []
    rank_correlations: list[float] = []
    score_states = 0
    for system in sorted(current):
        current_events = current[system].get("policy_decision_rounds", [])
        reference_events = reference[system].get("policy_decision_rounds", [])
        for index, (event, ref_event) in enumerate(zip(current_events, reference_events, strict=True)):
            if event.get("pre_reveal_state_checksum") != ref_event.get(
                "pre_reveal_state_checksum"
            ):
                break
            action_agreement = event.get("selected_pair_id") == ref_event.get("selected_pair_id")
            agreements.append(float(action_agreement))
            if index == 0:
                first_actions.append(float(action_agreement))
            vector = _score_vector(event)
            ref_vector = _score_vector(ref_event)
            if vector is None or ref_vector is None or vector[0] != ref_vector[0]:
                continue
            score_states += 1
            centered = vector[1] - vector[1].mean() - (ref_vector[1] - ref_vector[1].mean())
            errors.extend(np.abs(centered).tolist())
            ranks = _rank(vector[1])
            ref_ranks = _rank(ref_vector[1])
            if not (np.all(ranks == ranks[0]) or np.all(ref_ranks == ref_ranks[0])):
                rank_correlations.append(float(np.corrcoef(ranks, ref_ranks)[0, 1]))
    action = {
        "common_state_count": len(agreements),
        "agreement_rate": None if not agreements else float(np.mean(agreements)),
        "first_action_system_count": len(first_actions),
        "first_action_agreement_rate": None if not first_actions else float(np.mean(first_actions)),
    }
    diagnostics = {
        "score_available": bool(errors),
        "rank_available": bool(rank_correlations),
        "score_bearing_state_count": score_states,
        "rank_comparable_state_count": len(rank_correlations),
        "mean_centered_absolute_score_error": None if not errors else float(np.mean(errors)),
        "mean_spearman_rank_correlation": None if not rank_correlations else float(np.mean(rank_correlations)),
    }
    return action, diagnostics


def summarize_e55(input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest_path = input_root / "e55-unit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "e55-numerical-convergence-v1":
        raise ValueError("wrong E55 manifest protocol")
    units = manifest.get("units")
    if not isinstance(units, list) or manifest.get("unit_count") != len(units):
        raise ValueError("wrong E55 manifest unit count")
    source_audit, _ = _audit_sources(input_root, units)
    expected_grids = {"delta": DELTA_GRID, "cal": CAL_GRID}
    loaded: dict[str, dict[tuple[int, int, int], dict[str, dict[str, Any]]]] = {"delta": {}, "cal": {}}
    frozen_identity: dict[str, Any] | None = None
    input_hashes = {str(manifest_path): _sha256(manifest_path)}
    for unit in units:
        identity = unit.get("identity")
        if not isinstance(identity, dict):
            raise ValueError("E55 unit lacks identity")
        stage = identity.get("stage")
        expected_frozen = {
            "budget": 6,
            "maximum_budget": 6,
            "minimum_candidates": 12,
            "seed": 20260810,
            "hull_backend": "fixed_composition",
            "transport_family": "hierarchical_matern52_frozen_structure",
            "hull_candidate_workers": 1 if stage == "delta" else 8,
            "selection_timeout_seconds": 7200.0 if stage == "delta" else 21600.0,
            "policy": POLICIES.get(str(stage)),
        }
        if any(identity.get(key) != value for key, value in expected_frozen.items()):
            raise ValueError("wrong frozen E55 config")
        current_frozen_identity = {key: identity.get(key) for key in FROZEN_IDENTITY_FIELDS}
        if frozen_identity is None:
            frozen_identity = current_frozen_identity
        elif current_frozen_identity != frozen_identity:
            raise ValueError("frozen E55 identity mismatch")
        key = (int(identity.get("fold_index", -1)), int(identity.get("posterior_sample_count", -1)), int(identity.get("fantasy_count", -1)))
        if stage not in expected_grids or key not in expected_grids[stage]:
            raise ValueError("unexpected E55 unit grid")
        if key in loaded[stage]:
            raise ValueError("duplicate E55 unit")
        path = Path(unit["output"])
        failure = path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"registered E55 unit failed: {failure}")
        if not path.is_file():
            raise FileNotFoundError(f"missing E55 output: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        loaded[stage][key] = _validate_unit(identity, payload)
        input_hashes[str(path)] = _sha256(path)
    for stage, grid in expected_grids.items():
        if set(loaded[stage]) != grid:
            label = "Delta" if stage == "delta" else "CAL"
            raise ValueError(f"expected {len(grid)} {label} units")
    stages: dict[str, Any] = {}
    for stage, units_by_key in loaded.items():
        ref_samples, ref_fantasies = REFERENCE[stage]
        configurations: dict[str, Any] = {}
        for samples, fantasies in sorted({key[1:] for key in units_by_key}):
            current = {
                system: row
                for fold in range(5)
                for system, row in units_by_key[(fold, samples, fantasies)].items()
            }
            reference = {
                system: row
                for fold in range(5)
                for system, row in units_by_key[(fold, ref_samples, ref_fantasies)].items()
            }
            if set(current) != set(reference):
                raise ValueError("E55 reference system roster mismatch")
            action, diagnostics = _matched_diagnostics(current, reference)
            terminal = [
                float(current[system]["oracle_pool_confirmed_discoveries"])
                for system in sorted(current)
            ]
            reference_t = [
                float(reference[system]["oracle_pool_confirmed_discoveries"])
                for system in sorted(current)
            ]
            configurations[f"m{samples}-k{fantasies}"] = {
                "reference": f"m{ref_samples}-k{ref_fantasies}",
                "action_agreement": action,
                "score_rank_diagnostics": diagnostics,
                "terminal_T": {
                    "system_count": len(terminal),
                    "mean": float(np.mean(terminal)),
                    "reference_paired": _paired(np.asarray(terminal) - np.asarray(reference_t)),
                },
                "runtime": _quantiles([float(row["wall_seconds"]) for row in current.values()]),
            }
        stages[stage] = {"reference": f"m{ref_samples}-k{ref_fantasies}", "configurations": configurations}
    result = {
        "schema_version": 1,
        "status": "complete_e55_numerical_convergence_audit",
        "input_root": str(input_root),
        "input_sha256": input_hashes,
        "unit_counts": {stage: len(rows) for stage, rows in loaded.items()},
        "frozen_identity": frozen_identity,
        "source_audit": source_audit,
        "stages": stages,
    }
    _write_json_exclusive(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_e55(args.input_root, args.output)
    print(json.dumps({"status": result["status"], "unit_counts": result["unit_counts"]}))


if __name__ == "__main__":
    main()
