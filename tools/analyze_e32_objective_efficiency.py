"""Analyze the same-posterior objective contrast in the completed E32-A suite.

This is evaluator-only post-processing.  It reads the completed E32-A policy
outputs, preserves exact chemical systems as the resampling unit, and never
reruns a policy or recomputes an action.  The primary contrast is
Delta-Hull minus posterior-mean target-margin greedy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

POLICIES = (
    "posterior_mean_target_margin",
    "delta_hull_active_search",
)
T_FIELD = "oracle_pool_confirmed_discoveries"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired(values: np.ndarray, *, bootstrap_seed: int = 20260803) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("paired values must be nonempty and finite")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(20260804)
    signs = sign_rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    observed = abs(float(values.sum()))
    randomized = np.abs(np.sum(signs * values[None, :], axis=1))
    return {
        "system_count": int(len(values)),
        "paired_mean_difference": float(values.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "losses": int(np.sum(values < 0)),
        "two_sided_sign_flip_p": float(
            (np.sum(randomized >= observed) + 1) / (len(randomized) + 1)
        ),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": 20_000,
        "sign_flip_seed": 20260804,
        "sign_flip_draws": 100_000,
        "resampling_unit": "exact_chemical_system; all budgets retained within each draw",
    }


def _load(path: Path, budget: int) -> dict[str, Any]:
    if not path.exists():
        failure = path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"registered E32 output failed: {failure}")
        raise FileNotFoundError(f"registered E32 output is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "source_margin",
        "posterior_mean_target_margin",
        "posterior_current_hull_probability",
        "delta_hull_active_search",
        "ungated_source_rollout",
        "source_rollout_delta_hull",
        "delta_hull_anchored_rollout",
    }
    if set(payload.get("active_policies", ())) != expected:
        raise ValueError(f"unexpected E32 policy roster in {path}")
    if payload.get("config", {}).get("query_budget") != budget:
        raise ValueError(f"wrong budget in {path}")
    for system, result in payload.get("systems", {}).items():
        if int(result["budget"]) != budget:
            raise ValueError(f"wrong system budget for {system} in {path}")
        if set(result["strategies"]) != expected:
            raise ValueError(f"incomplete strategy roster for {system} in {path}")
    return payload


def _rows(payloads: list[dict[str, Any]], policy: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    for payload in payloads:
        for system, result in payload["systems"].items():
            if system in rows:
                raise ValueError(f"chemical system occurs twice: {system}")
            rows[system] = float(result["strategies"][policy][T_FIELD])
    if not rows:
        raise ValueError("no E32 system rows")
    return rows


def _auc(values: np.ndarray) -> np.ndarray:
    """Trapezoidal AUC over B=0,...,6, with zero reward at B=0."""
    values = np.asarray(values, dtype=float)
    if values.shape[0] != 6:
        raise ValueError("expected six nonzero budget rows")
    return 0.5 * values[0] + values[1:-1].sum(axis=0) + 0.5 * values[-1]


def analyze(*, input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    values: dict[str, list[np.ndarray]] = {policy: [] for policy in POLICIES}
    systems_by_budget: list[list[str]] = []
    for budget in range(1, 7):
        payloads = [
            _load(input_root / f"e32-fold{fold + 1}-b{budget}-main.json", budget=budget)
            for fold in range(5)
        ]
        rows_by_policy = {policy: _rows(payloads, policy) for policy in POLICIES}
        systems = sorted(rows_by_policy[POLICIES[0]])
        if any(set(rows) != set(systems) for rows in rows_by_policy.values()):
            raise ValueError(f"policy system rosters differ at B={budget}")
        systems_by_budget.append(systems)
        for policy in POLICIES:
            values[policy].append(np.asarray([rows_by_policy[policy][s] for s in systems]))

    if any(systems != systems_by_budget[0] for systems in systems_by_budget[1:]):
        raise ValueError("chemical-system roster changes across budgets")
    system_roster = systems_by_budget[0]
    target = np.stack(values["posterior_mean_target_margin"])
    delta = np.stack(values["delta_hull_active_search"])
    budget_differences = delta - target
    auc_differences = _auc(delta) - _auc(target)
    budget_summary = []
    for budget, difference in enumerate(budget_differences, start=1):
        stats = _paired(difference)
        stats["budget"] = budget
        budget_summary.append(stats)

    quantiles = {
        str(q): float(np.quantile(auc_differences, q))
        for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    }
    result = {
        "schema_version": 1,
        "status": "complete_e32_a_objective_efficiency_evaluator_analysis",
        "input_root": str(input_root),
        "input_sha256": {
            str(path): _sha256(path) for path in sorted(input_root.glob("e32-*.json"))
        },
        "policies": list(POLICIES),
        "budgets": list(range(1, 7)),
        "system_count": len(system_roster),
        "resampling_unit": "exact_chemical_system; each bootstrap draw retains all six budgets",
        "contrast": "delta_hull_active_search_minus_posterior_mean_target_margin",
        "paired_budget_curve": budget_summary,
        "auc": {
            "grid": [0, 1, 2, 3, 4, 5, 6],
            "difference": _paired(auc_differences),
            "system_difference_quantiles": quantiles,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(input_root=args.input_root, output=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "system_count": result["system_count"],
                "auc": result["auc"]["difference"],
            }
        )
    )


if __name__ == "__main__":
    main()
