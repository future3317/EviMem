"""Read the frozen E32 budget artifacts and summarize rollout versus Delta-Hull."""

from __future__ import annotations

import argparse
import hashlib
import json
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
CONFIG_FIELDS = (
    "seed",
    "posterior_sample_count",
    "fantasy_count",
    "hull_backend",
    "transport_family",
    "crossfit_manifest_sha256",
)


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


def _load(path: Path, budget: int, identity: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        failure = path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"registered E32 output failed: {failure}")
        raise FileNotFoundError(f"registered E32 output is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload.get("active_policies", ())) != POLICIES:
        raise ValueError("wrong E32 policy roster")
    config = payload.get("config", {})
    if config.get("query_budget") != budget:
        raise ValueError("wrong E32 budget")
    current_identity = {key: payload.get(key) for key in IDENTITY_FIELDS}
    current_identity.update({key: config.get(key) for key in CONFIG_FIELDS})
    if identity is not None and current_identity != identity:
        raise ValueError("E32 frozen identity mismatch")
    systems = payload.get("systems")
    if not isinstance(systems, dict):
        raise ValueError("missing E32 systems")
    for system, result in systems.items():
        if result.get("budget") != budget:
            raise ValueError(f"wrong E32 system budget: {system}")
        strategies = result.get("strategies", {})
        if set(strategies) != set(POLICIES):
            raise ValueError("wrong E32 policy roster")
        if result.get("transport_element_support") is False:
            actions = {
                tuple(str(pair_id) for pair_id in strategy.get("selected_pair_ids", ()))
                for strategy in strategies.values()
            }
            terminal_t = {
                float(strategy.get("oracle_pool_confirmed_discoveries", np.nan))
                for strategy in strategies.values()
            }
            if len(actions) != 1 or len(terminal_t) != 1:
                raise ValueError("unsupported fallback must preserve common policy actions and T")
    return payload, current_identity


def _rows(payloads: list[dict[str, Any]], policy: str) -> tuple[dict[str, dict[str, Any]], int]:
    rows: dict[str, dict[str, Any]] = {}
    unsupported = 0
    for payload in payloads:
        for system, result in payload["systems"].items():
            if system in rows:
                raise ValueError(f"E32 chemical system occurs twice: {system}")
            rows[system] = result["strategies"][policy]
            unsupported += result.get("transport_element_support") is False
    return rows, unsupported


def _area(curve: np.ndarray) -> np.ndarray:
    if curve.shape[0] != 6:
        raise ValueError("expected independent B=1 through B=6 artifacts")
    return 0.5 * curve[0] + curve[1:5].sum(axis=0) + 0.5 * curve[5]


def summarize_e32(input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    identity: dict[str, Any] | None = None
    by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    input_hashes: dict[str, str] = {}
    unsupported_counts: dict[int, int] = {}
    for budget in range(1, 7):
        payloads = []
        for fold in range(1, 6):
            path = input_root / f"e32-fold{fold}-b{budget}-main.json"
            payload, identity = _load(path, budget, identity)
            payloads.append(payload)
            input_hashes[str(path)] = _sha256(path)
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
        "frozen_identity": identity,
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
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
