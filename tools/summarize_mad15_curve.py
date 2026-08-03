"""Summarize a frozen MAD-1.5 B=0..6 acquisition curve."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "oracle_pool_confirmed_discoveries",
    "final_causal_confirmed_discoveries",
    "oracle_pool_discovery_ceiling",
    "wall_seconds",
)


def _ci(values: np.ndarray, *, seed: int, repetitions: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _sign_flip(values: np.ndarray, *, seed: int, repetitions: int) -> float:
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray((-1.0, 1.0)), size=(repetitions, len(values)))
    null_means = (signs * values[None, :]).mean(axis=1)
    observed = abs(float(values.mean()))
    return float((np.count_nonzero(np.abs(null_means) >= observed) + 1) / (repetitions + 1))


def _difference_summary(values: np.ndarray, *, seed: int, bootstrap: int, sign_flips: int) -> dict[str, Any]:
    return {
        "mean": float(values.mean()),
        "bootstrap_95pct": _ci(values, seed=seed, repetitions=bootstrap),
        "sign_flip_pvalue_mc": _sign_flip(values, seed=seed + 1, repetitions=sign_flips),
        "wins": int(np.count_nonzero(values > 0)),
        "ties": int(np.count_nonzero(values == 0)),
        "losses": int(np.count_nonzero(values < 0)),
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    *,
    input_paths: tuple[Path, ...],
    output_path: Path,
    manifest_path: Path,
    bootstrap_repetitions: int = 20_000,
    sign_flip_repetitions: int = 100_000,
    seed: int = 20260730,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if len(input_paths) != 6:
        raise ValueError("the registered curve requires exactly B=1..6 inputs")
    payloads = [_load(path) for path in input_paths]
    budgets = [int(payload["config"].get("query_budget", 0)) for payload in payloads]
    if budgets != list(range(1, 7)):
        raise ValueError(f"curve inputs must be ordered B=1..6, got {budgets}")
    manifest = _load(manifest_path)
    expected_systems = tuple(manifest["selected_systems"])
    if any(tuple(payload["query_systems"]) != expected_systems for payload in payloads):
        raise ValueError("curve inputs do not use the frozen system manifest")
    task_shas = {payload["task_sha256"] for payload in payloads}
    vault_shas = {payload["oracle_vault_sha256"] for payload in payloads}
    model_shas = {payload["transport_model_checksum"] for payload in payloads}
    if len(task_shas) != 1 or len(vault_shas) != 1 or len(model_shas) != 1:
        raise ValueError("curve inputs do not share task, vault and transport checksums")
    policy_names = tuple(payloads[0]["active_policies"])
    if policy_names != ("source_margin", "independent_confirmation_source_rollout"):
        raise ValueError("curve primary comparison must be source margin versus frozen IC-SARR")
    if any(tuple(payload["active_policies"]) != policy_names for payload in payloads):
        raise ValueError("curve inputs use different active policies")
    systems = list(expected_systems)
    curves: dict[str, dict[str, list[float]]] = {policy: {metric: [0.0] for metric in METRICS} for policy in policy_names}
    differences: dict[str, list[dict[str, Any]]] = {}
    for budget, payload in enumerate(payloads, start=1):
        for policy in policy_names:
            rows = payload["systems"]
            for metric in METRICS:
                curves[policy][metric].append(
                    float(np.mean([rows[system]["strategies"][policy][metric] for system in systems]))
                )
        source_rows = payload["systems"]
        values = []
        for system in systems:
            source = source_rows[system]["strategies"]["source_margin"]
            ic = source_rows[system]["strategies"]["independent_confirmation_source_rollout"]
            values.append(
                {
                    "chemical_system": system,
                    "oracle_diff": float(ic["oracle_pool_confirmed_discoveries"])
                    - float(source["oracle_pool_confirmed_discoveries"]),
                    "final_causal_diff": float(ic["final_causal_confirmed_discoveries"])
                    - float(source["final_causal_confirmed_discoveries"]),
                    "wall_seconds_diff": float(ic["wall_seconds"]) - float(source["wall_seconds"]),
                    "source_headroom": float(source["oracle_pool_discovery_ceiling"])
                    - float(source["oracle_pool_confirmed_discoveries"]),
                }
            )
        differences[str(budget)] = values

    curve_summary: dict[str, Any] = {}
    for policy in policy_names:
        curve_summary[policy] = {
            metric: values for metric, values in curves[policy].items()
        }
    paired_by_budget: dict[str, Any] = {}
    for budget, values in differences.items():
        oracle = np.asarray([row["oracle_diff"] for row in values], dtype=float)
        final_causal = np.asarray([row["final_causal_diff"] for row in values], dtype=float)
        wall = np.asarray([row["wall_seconds_diff"] for row in values], dtype=float)
        headroom = np.asarray([row["source_headroom"] for row in values], dtype=float)
        paired_by_budget[budget] = {
            "oracle_difference_ic_minus_source": _difference_summary(
                oracle, seed=seed + int(budget), bootstrap=bootstrap_repetitions, sign_flips=sign_flip_repetitions
            ),
            "final_causal_difference_ic_minus_source": _difference_summary(
                final_causal, seed=seed + 100 + int(budget), bootstrap=bootstrap_repetitions, sign_flips=sign_flip_repetitions
            ),
            "mean_wall_seconds_difference_ic_minus_source": float(wall.mean()),
            "mean_source_headroom": float(headroom.mean()),
            "recovered_headroom_fraction": float(oracle.mean() / headroom.mean()) if headroom.mean() else None,
            "incremental_confirmations_per_incremental_second": float(oracle.mean() / wall.mean()) if wall.mean() > 0 else None,
        }
    oracle_auc = {
        policy: float(np.trapezoid(np.asarray(curves[policy]["oracle_pool_confirmed_discoveries"])))
        for policy in policy_names
    }
    final_auc = {
        policy: float(np.trapezoid(np.asarray(curves[policy]["final_causal_confirmed_discoveries"])))
        for policy in policy_names
    }
    wall_auc = {
        policy: float(np.trapezoid(np.asarray(curves[policy]["wall_seconds"])))
        for policy in policy_names
    }
    oracle_auc_diffs = np.asarray(
        [
            differences["1"][index]["oracle_diff"] / 2
            + sum(
                (differences[str(budget)][index]["oracle_diff"] + differences[str(budget + 1)][index]["oracle_diff"])
                / 2
                for budget in range(1, 6)
            )
            for index in range(len(systems))
        ],
        dtype=float,
    )
    final_auc_diffs = np.asarray(
        [
            differences["1"][index]["final_causal_diff"] / 2
            + sum(
                (differences[str(budget)][index]["final_causal_diff"] + differences[str(budget + 1)][index]["final_causal_diff"])
                / 2
                for budget in range(1, 6)
            )
            for index in range(len(systems))
        ],
        dtype=float,
    )
    result = {
        "status": "development_only_mad15_curve_summary",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256(manifest_path),
        "input_paths": [str(path.resolve()) for path in input_paths],
        "input_sha256": {str(path): _sha256(path) for path in input_paths},
        "task_sha256": next(iter(task_shas)),
        "oracle_vault_sha256": next(iter(vault_shas)),
        "transport_model_checksum": next(iter(model_shas)),
        "system_count": len(systems),
        "budgets": list(range(7)),
        "policies": list(policy_names),
        "bootstrap_repetitions": bootstrap_repetitions,
        "sign_flip_repetitions": sign_flip_repetitions,
        "seed": seed,
        "curves": curve_summary,
        "paired_by_budget": paired_by_budget,
        "curve_level": {
            "oracle_auc": oracle_auc,
            "final_causal_auc": final_auc,
            "wall_seconds_auc": wall_auc,
            "oracle_auc_difference_ic_minus_source": _difference_summary(
                oracle_auc_diffs, seed=seed + 1000, bootstrap=bootstrap_repetitions, sign_flips=sign_flip_repetitions
            ),
            "final_causal_auc_difference_ic_minus_source": _difference_summary(
                final_auc_diffs, seed=seed + 1100, bootstrap=bootstrap_repetitions, sign_flips=sign_flip_repetitions
            ),
            "mean_wall_seconds_auc_difference_ic_minus_source": wall_auc[policy_names[1]] - wall_auc[policy_names[0]],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=20_000)
    parser.add_argument("--sign-flip-repetitions", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()
    result = run(
        input_paths=tuple(args.input),
        output_path=args.output,
        manifest_path=args.manifest,
        bootstrap_repetitions=args.bootstrap_repetitions,
        sign_flip_repetitions=args.sign_flip_repetitions,
        seed=args.seed,
    )
    print(json.dumps(result["curve_level"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
