"""Derive E52 B=1..6 objective curves from frozen B=6 trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

TARGET = "posterior_mean_target_margin"
DELTA = "delta_hull_active_search"
POOLS = ("070", "085", "100")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sign_flip_p(values: np.ndarray, *, seed: int, draw_count: int = 200_000) -> float:
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 1.0
    observed = abs(float(np.mean(nonzero)))
    if len(nonzero) <= 20:
        masks = np.arange(1 << len(nonzero), dtype=np.uint64)[:, None]
        bits = (masks >> np.arange(len(nonzero), dtype=np.uint64)) & 1
        signs = 2.0 * bits.astype(float) - 1.0
        null = np.abs(np.mean(signs * nonzero, axis=1))
        return float(np.mean(null >= observed - 1e-15))
    rng = np.random.default_rng(seed)
    exceed = 0
    remaining = draw_count
    while remaining:
        size = min(10_000, remaining)
        signs = rng.choice((-1.0, 1.0), size=(size, len(nonzero)))
        exceed += int(np.sum(np.abs(np.mean(signs * nonzero, axis=1)) >= observed))
        remaining -= size
    return float((exceed + 1) / (draw_count + 1))


def _bootstrap_interval(values: np.ndarray, *, seed: int, count: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(count, len(values)))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _prefix_value(strategy: dict[str, Any], budget: int) -> int:
    selected = tuple(str(value) for value in strategy["selected_pair_ids"])
    if len(selected) != 6 or len(set(selected)) != 6:
        raise ValueError("objective prefix derivation requires a unique B=6 trace")
    labels = strategy["oracle_pool_final_labels_by_pair_id"]
    if not set(selected) <= set(labels):
        raise ValueError("selected action is missing a complete-pool label")
    return sum(bool(labels[pair_id]) for pair_id in selected[:budget])


def summarize(
    *,
    input_root: Path,
    output: Path,
    expected_system_count: int = 230,
    bootstrap_count: int = 20_000,
    seed: int = 20260809,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("E52 summaries must remain outside Git")
    curves: dict[str, Any] = {}
    input_hashes: dict[str, str] = {}
    pool_rows: dict[str, dict[str, dict[str, list[int]]]] = {}
    for pool in POOLS:
        paths = sorted((input_root / pool).glob("fold*-b6.json"))
        if len(paths) != 5:
            raise ValueError(f"pool {pool} requires five complete B=6 folds")
        rows: dict[str, dict[str, list[int]]] = {}
        task_hashes: set[str] = set()
        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            input_hashes[str(path.resolve())] = _sha256(path)
            if tuple(payload["active_policies"]) != (TARGET, DELTA):
                raise ValueError(f"unexpected policy roster in {path}")
            task_hashes.add(str(payload["task_sha256"]))
            for system, result in payload["systems"].items():
                if system in rows:
                    raise ValueError(f"duplicate system {system} in pool {pool}")
                if int(result["budget"]) != 6:
                    raise ValueError(f"non-B=6 result in {path}")
                rows[system] = {
                    policy: [
                        _prefix_value(result["strategies"][policy], budget)
                        for budget in range(1, 7)
                    ]
                    for policy in (TARGET, DELTA)
                }
        if len(rows) != expected_system_count:
            raise ValueError(
                f"pool {pool} has {len(rows)} systems, expected {expected_system_count}"
            )
        if pool == "100" and len(task_hashes) != 1:
            raise ValueError("100% folds must share one task")
        if pool != "100" and len(task_hashes) != 5:
            raise ValueError("query-only shifted folds must have fold-specific tasks")
        pool_rows[pool] = rows

    reference_systems = set(pool_rows["100"])
    if any(set(rows) != reference_systems for rows in pool_rows.values()):
        raise ValueError("pool-shift system rosters do not match")
    for pool_index, pool in enumerate(POOLS):
        rows = pool_rows[pool]
        budgets: dict[str, Any] = {}
        for budget in range(1, 7):
            target = np.asarray([rows[system][TARGET][budget - 1] for system in sorted(rows)])
            delta = np.asarray([rows[system][DELTA][budget - 1] for system in sorted(rows)])
            difference = delta - target
            lower, upper = _bootstrap_interval(
                difference,
                seed=seed + 100 * pool_index + budget,
                count=bootstrap_count,
            )
            budgets[str(budget)] = {
                "target_margin_mean_T": float(np.mean(target)),
                "delta_hull_mean_T": float(np.mean(delta)),
                "paired_delta_T": float(np.mean(difference)),
                "bootstrap_95": {"lower": lower, "upper": upper},
                "sign_flip_p": _sign_flip_p(
                    difference.astype(float), seed=seed + 1000 * pool_index + budget
                ),
                "wins": int(np.sum(difference > 0)),
                "ties": int(np.sum(difference == 0)),
                "losses": int(np.sum(difference < 0)),
            }
        curves[pool] = {
            "system_count": len(rows),
            "budgets": budgets,
            "delta_hull_minus_target_margin_T_auc_b1_b6": float(
                sum(budgets[str(budget)]["paired_delta_T"] for budget in range(1, 7))
            ),
        }
    result = {
        "schema_version": 1,
        "status": "e52_objective_prefix_summary_complete",
        "prefix_source": "frozen B=6 trajectories",
        "primary_utility": "complete-pool terminal confirmations T",
        "bootstrap_unit": "exact chemical system",
        "bootstrap_count": bootstrap_count,
        "input_sha256": input_hashes,
        "curves": curves,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-count", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    result = summarize(
        input_root=args.input_root,
        output=args.output,
        bootstrap_count=args.bootstrap_count,
        seed=args.seed,
    )
    print(json.dumps({"status": result["status"], "curves": result["curves"]}, indent=2))


if __name__ == "__main__":
    main()
