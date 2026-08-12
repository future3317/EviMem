"""Read the frozen E32 budget artifacts and summarize rollout versus Delta-Hull."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

POLICIES = (
    "source_margin",
    "posterior_mean_target_margin",
    "posterior_current_hull_probability",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "delta_hull_anchored_rollout",
)
IDENTITY_FIELDS = ("task_sha256", "oracle_vault_sha256", "script_sha256")
STATUS = "exploratory_development_systems_only_not_confirmatory"
TASK_SHA256 = "f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df"
VAULT_SHA256 = "a272d3a2ce6286443ae6fce35726a688751a37284e3df362c5d1f70e2fcb9952"
CROSSFIT_SHA256 = "a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa"


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


def _paired(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired effects require finite exact-system values")
    bootstrap_rng = np.random.default_rng(20260803)
    indices = bootstrap_rng.integers(0, len(values), size=(20_000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(20260804)
    signs = sign_rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    randomized = np.abs(np.sum(signs * values[None, :], axis=1))
    return {
        "system_count": int(len(values)),
        "paired_mean_difference": float(values.mean()),
        "paired_bootstrap_95ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))],
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "losses": int(np.sum(values < 0)),
        "two_sided_sign_flip_p": float((np.sum(randomized >= abs(float(values.sum()))) + 1) / (len(randomized) + 1)),
    }


def _runtime(values: np.ndarray) -> dict[str, Any]:
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("runtime traces must be finite")
    return {
        "population": "per_system_policy_wall_seconds_from_traces",
        "system_count": int(len(values)),
        "median": float(np.median(values)),
        "iqr": [float(np.quantile(values, 0.25)), float(np.quantile(values, 0.75))],
    }


def _read_identity(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing E32 {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "protocol": "docs/DELAYED_LABEL_FOLLOWUP_PROTOCOL_E32.md",
        "task_sha256": TASK_SHA256,
        "vault_sha256": VAULT_SHA256,
        "crossfit_manifest_sha256": CROSSFIT_SHA256,
        "seed": 20270720,
        "policies": list(POLICIES),
        "budgets": list(range(1, 7)),
        "folds": list(range(5)),
        "max_systems": 1000,
        "posterior_sample_count": 128,
        "hull_backend": "fixed_composition",
        "transport_family": "hierarchical_matern52_frozen_structure",
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f"wrong E32 {label}")
    if any(
        not isinstance(payload.get(key), str) or not payload[key]
        for key in ("runner_sha256", "policy_worker_sha256", "acquisition_sha256")
    ):
        raise ValueError(f"wrong E32 {label} code hashes")
    return payload


def _audit_identities(input_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    protocol_path = input_root / "e32_protocol_identity.json"
    executor_path = input_root / "e32_parallel_executor_identity.json"
    protocol = _read_identity(protocol_path, "protocol identity")
    executor = _read_identity(executor_path, "executor identity")
    for key, value in protocol.items():
        if executor.get(key) != value:
            raise ValueError("E32 protocol identity and executor identity mismatch")
    expected_executor = {
        "executor": "parallel_unit_scheduler_v1",
        "rollout_selection_timeout_seconds": 900.0,
        "max_workers": 16,
        "blas_threads_per_unit": 1,
    }
    if any(executor.get(key) != value for key, value in expected_executor.items()):
        raise ValueError("wrong E32 executor identity")
    return protocol, executor, {
        str(protocol_path): _sha256(protocol_path),
        str(executor_path): _sha256(executor_path),
    }


def _load(
    path: Path,
    budget: int,
    fold: int,
    protocol_identity: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        failure = path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"registered E32 output failed: {failure}")
        raise FileNotFoundError(f"registered E32 output is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != STATUS:
        raise ValueError("wrong E32 status")
    if tuple(payload.get("active_policies", ())) != POLICIES:
        raise ValueError("wrong E32 policy roster")
    config = payload.get("config", {})
    if config.get("query_budget") != budget:
        raise ValueError("wrong E32 budget")
    expected_identity = {
        "task_sha256": protocol_identity["task_sha256"],
        "oracle_vault_sha256": protocol_identity["vault_sha256"],
        "script_sha256": protocol_identity["runner_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("E32 frozen identity mismatch")
    expected_config = {
        "maximum_budget": 6,
        "minimum_candidates": 12,
        "seed": protocol_identity["seed"],
        "posterior_sample_count": protocol_identity["posterior_sample_count"],
        "fantasy_count": 16,
        "rollout_selection_timeout_seconds": 900.0,
        "hull_backend": protocol_identity["hull_backend"],
        "transport_family": protocol_identity["transport_family"],
        "crossfit_manifest_sha256": protocol_identity["crossfit_manifest_sha256"],
        "crossfit_fold_index": fold,
    }
    for key, value in expected_config.items():
        if config.get(key) != value:
            message = "fold" if key == "crossfit_fold_index" else "frozen identity"
            raise ValueError(f"wrong E32 {message}: {key}")
    systems = payload.get("systems")
    query_systems = payload.get("query_systems")
    if (
        not isinstance(systems, dict)
        or not isinstance(query_systems, list)
        or len(query_systems) != 46
        or len(set(query_systems)) != 46
        or set(query_systems) != set(systems)
    ):
        raise ValueError("wrong E32 query roster")
    fit_systems = payload.get("transport_fit_systems")
    if (
        not isinstance(fit_systems, list)
        or len(fit_systems) != 184
        or len(set(fit_systems)) != 184
        or set(fit_systems) & set(query_systems)
        or payload.get("transport_fit_system_count") != 184
        or payload.get("transport_fit_and_query_systems_disjoint") is not True
    ):
        raise ValueError("wrong E32 fit roster")
    for system, result in systems.items():
        if result.get("budget") != budget:
            raise ValueError(f"wrong E32 system budget: {system}")
        strategies = result.get("strategies", {})
        if set(strategies) != set(POLICIES):
            raise ValueError("wrong E32 policy roster")
        for strategy in strategies.values():
            actions = strategy.get("selected_pair_ids")
            events = strategy.get("policy_decision_rounds")
            if not isinstance(actions, list) or len(actions) != budget:
                raise ValueError("wrong E32 action count")
            if not isinstance(events, list) or len(events) != budget:
                raise ValueError("wrong E32 event count")
            if [event.get("round_index") for event in events] != list(range(1, budget + 1)):
                raise ValueError("wrong E32 event count or round indices")
            if [event.get("selected_pair_id") for event in events] != actions:
                raise ValueError("E32 events do not reconcile with selected_pair_ids")
            if any(not event.get("pre_reveal_state_checksum") for event in events):
                raise ValueError("E32 event lacks pre-reveal state checksum")
            if not np.isfinite(float(strategy.get("oracle_pool_confirmed_discoveries", np.nan))):
                raise ValueError("E32 terminal T is missing or non-finite")
            if not np.isfinite(float(strategy.get("wall_seconds", np.nan))):
                raise ValueError("E32 runtime is missing or non-finite")
        if result.get("transport_element_support") is not True:
            actions = {
                tuple(str(pair_id) for pair_id in strategy.get("selected_pair_ids", ()))
                for strategy in strategies.values()
            }
            event_actions = {
                tuple(event["selected_pair_id"] for event in strategy["policy_decision_rounds"])
                for strategy in strategies.values()
            }
            terminal_t = {
                float(strategy.get("oracle_pool_confirmed_discoveries", np.nan))
                for strategy in strategies.values()
            }
            if len(actions) != 1 or len(event_actions) != 1 or len(terminal_t) != 1:
                raise ValueError("unsupported fallback must preserve common policy actions and T")
    return payload


def _rows(payloads: list[dict[str, Any]], policy: str) -> tuple[dict[str, dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    unsupported = 0
    for payload in payloads:
        for system, result in payload["systems"].items():
            if system in rows:
                raise ValueError(f"E32 chemical system occurs twice: {system}")
            rows[system] = result["strategies"][policy]
            unsupported += result.get("transport_element_support") is not True
    return rows, unsupported


def _area(curve: np.ndarray) -> np.ndarray:
    if curve.shape[0] != 6:
        raise ValueError("expected independent B=1 through B=6 artifacts")
    return 0.5 * curve[0] + curve[1:5].sum(axis=0) + 0.5 * curve[5]


def summarize_e32(input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    protocol_identity, executor_identity, input_hashes = _audit_identities(input_root)
    by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    unsupported_counts: dict[int, int] = {}
    fold_rosters: dict[int, tuple[str, ...]] = {}
    fold_fit_rosters: dict[int, set[str]] = {}
    for budget in range(1, 7):
        payloads = []
        budget_rosters: list[set[str]] = []
        for fold in range(5):
            path = input_root / f"e32-fold{fold + 1}-b{budget}-main.json"
            payload = _load(path, budget, fold, protocol_identity)
            payloads.append(payload)
            input_hashes[str(path)] = _sha256(path)
            roster = tuple(str(system) for system in payload["query_systems"])
            fit_roster = {str(system) for system in payload["transport_fit_systems"]}
            if fold in fold_rosters and roster != fold_rosters[fold]:
                raise ValueError("E32 fold query roster changes across budgets")
            if fold in fold_fit_rosters and fit_roster != fold_fit_rosters[fold]:
                raise ValueError("E32 fold fit roster changes across budgets")
            fold_rosters.setdefault(fold, roster)
            fold_fit_rosters.setdefault(fold, fit_roster)
            budget_rosters.append(set(roster))
        if any(
            left & right
            for position, left in enumerate(budget_rosters)
            for right in budget_rosters[position + 1 :]
        ):
            raise ValueError("E32 chemical system occurs twice across folds")
        all_systems = set().union(*budget_rosters)
        if any(fold_fit_rosters[fold] != all_systems - budget_rosters[fold] for fold in range(5)):
            raise ValueError("wrong E32 fit roster: not the 230-system fold complement")
        delta, unsupported = _rows(payloads, "delta_hull_active_search")
        rollout, _ = _rows(payloads, "delta_hull_anchored_rollout")
        if len(delta) != 230 or set(delta) != set(rollout):
            raise ValueError("E32 requires 230 unique systems with paired policy identities")
        by_budget[budget] = {"delta": delta, "rollout": rollout}
        unsupported_counts[budget] = unsupported
    systems = sorted(by_budget[1]["delta"])
    if any(set(rows["delta"]) != set(systems) or set(rows["rollout"]) != set(systems) for rows in by_budget.values()):
        raise ValueError("E32 system roster changes across independent budgets")
    budgets: dict[str, Any] = {}
    delta_t: list[np.ndarray] = []
    rollout_t: list[np.ndarray] = []
    for budget in range(1, 7):
        delta = by_budget[budget]["delta"]
        rollout = by_budget[budget]["rollout"]
        delta_values = np.asarray([float(delta[system]["oracle_pool_confirmed_discoveries"]) for system in systems])
        rollout_values = np.asarray([float(rollout[system]["oracle_pool_confirmed_discoveries"]) for system in systems])
        delta_runtime = np.asarray([float(delta[system]["wall_seconds"]) for system in systems])
        rollout_runtime = np.asarray([float(rollout[system]["wall_seconds"]) for system in systems])
        delta_t.append(delta_values)
        rollout_t.append(rollout_values)
        if budget == 1:
            continue
        disagreements = np.asarray(
            [
                np.any(np.asarray(delta[system]["selected_pair_ids"], dtype=object) != np.asarray(rollout[system]["selected_pair_ids"], dtype=object))
                for system in systems
            ]
        )
        rates = np.asarray(
            [
                np.mean(np.asarray(delta[system]["selected_pair_ids"], dtype=object) != np.asarray(rollout[system]["selected_pair_ids"], dtype=object))
                for system in systems
            ]
        )
        budgets[str(budget)] = {
            "delta_hull_terminal_T_mean": float(delta_values.mean()),
            "anchored_rollout_terminal_T_mean": float(rollout_values.mean()),
            "paired_terminal_T": _paired(rollout_values - delta_values),
            "action_disagreement": {
                "population": "all 230 systems; unsupported transport fallbacks retained",
                "unsupported_transport_fallback_system_count": unsupported_counts[budget],
                "systems_with_any_disagreement": int(disagreements.sum()),
                "mean_per_system_round_fraction": float(rates.mean()),
            },
            "runtime": {
                "population": "per_system_policy_wall_seconds_from_traces",
                "delta_hull": _runtime(delta_runtime),
                "anchored_rollout": _runtime(rollout_runtime),
                "paired_mean_difference_rollout_minus_delta": float((rollout_runtime - delta_runtime).mean()),
            },
        }
    delta_area = _area(np.stack(delta_t))
    rollout_area = _area(np.stack(rollout_t))
    result = {
        "schema_version": 1,
        "status": "complete_e32_read_only_rollout_curve",
        "input_root": str(input_root),
        "input_sha256": input_hashes,
        "protocol_identity": protocol_identity,
        "executor_identity": executor_identity,
        "fold_rosters": {str(fold): list(roster) for fold, roster in fold_rosters.items()},
        "system_count": len(systems),
        "inference": {
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20260803,
            "sign_flip_draws": 100_000,
            "sign_flip_seed": 20260804,
            "statistical_unit": "exact_chemical_system",
        },
        "budgets": budgets,
        "integrated_b0_to_b6_terminal_T": _paired(rollout_area - delta_area),
    }
    _write_json_exclusive(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_e32(args.input_root, args.output)
    print(json.dumps({"status": result["status"], "system_count": result["system_count"]}))


if __name__ == "__main__":
    main()
