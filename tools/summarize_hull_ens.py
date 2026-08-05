"""Summarize complete Hull-ENS development outputs without overwriting them."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

POLICIES = ("source_margin", "delta_hull_active_search", "hull_ens", "safe_hull_ens")
PAIR_CONTRASTS = {
    "hull_ens_minus_delta_hull": ("hull_ens", "delta_hull_active_search"),
    "safe_hull_ens_minus_delta_hull": ("safe_hull_ens", "delta_hull_active_search"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_interval(values: np.ndarray, *, seed: int) -> list[float]:
    if len(values) == 0:
        raise ValueError("cannot bootstrap an empty paired contrast")
    rng = np.random.default_rng(seed)
    means = np.empty(20_000, dtype=np.float64)
    for start in range(0, len(means), 1_000):
        indices = rng.integers(0, len(values), size=(1_000, len(values)))
        means[start : start + 1_000] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def _sign_flip_pvalue(values: np.ndarray, *, seed: int) -> float:
    if len(values) == 0:
        raise ValueError("cannot sign-flip an empty paired contrast")
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    draws = 100_000
    for start in range(0, draws, 1_000):
        signs = rng.choice(np.asarray((-1.0, 1.0)), size=(1_000, len(values)))
        exceed += int(np.count_nonzero(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
    return float((exceed + 1) / (draws + 1))


def _contrast_summary(values: np.ndarray, *, seed: int) -> dict[str, Any]:
    return {
        "n_systems": int(len(values)),
        "mean": float(values.mean()),
        "bootstrap_95_ci": _bootstrap_interval(values, seed=seed),
        "sign_flip_p_two_sided": _sign_flip_pvalue(values, seed=seed + 1),
        "wins": int(np.count_nonzero(values > 1e-12)),
        "ties": int(np.count_nonzero(np.abs(values) <= 1e-12)),
        "losses": int(np.count_nonzero(values < -1e-12)),
    }


def _metric(row: dict[str, Any], policy: str, name: str) -> float:
    value = row["strategies"][policy][name]
    return float(value)


def summarize(
    *, input_root: Path, output_path: Path, expected_system_count: int = 230
) -> dict[str, Any]:
    identity_path = input_root / "hull_ens_protocol_identity.json"
    if not identity_path.exists():
        raise ValueError(f"missing executor identity: {identity_path}")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    expected = {
        f"hull-ens-fold{fold}-b{budget}-main.json"
        for fold in range(1, 6)
        for budget in range(1, 7)
    }
    paths = sorted(input_root.glob("hull-ens-fold*-b*-main.json"))
    if {path.name for path in paths} != expected:
        raise ValueError(f"Hull-ENS roster is incomplete: found {len(paths)} files")
    if any(path.with_suffix(".failure.json").exists() for path in paths):
        raise ValueError("Hull-ENS output root contains a failure marker")
    records: dict[int, dict[str, dict[str, Any]]] = {}
    all_system_ids: set[str] = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        if tuple(config.get("policies", ())) != POLICIES:
            raise ValueError(f"wrong policy roster in {path}")
        if not str(payload.get("status", "")).startswith(("complete", "exploratory_development")):
            raise ValueError(f"output is not complete: {path}")
        budget = int(config["query_budget"])
        if budget not in range(1, 7):
            raise ValueError(f"unexpected budget in {path}")
        systems = payload.get("systems")
        if not isinstance(systems, dict):
            raise ValueError(f"missing systems object in {path}")
        budget_records = records.setdefault(budget, {})
        for system_id, system_record in systems.items():
            if system_id in budget_records:
                raise ValueError(f"duplicate system/budget record: {system_id}, B={budget}")
            if set(system_record.get("strategies", {})) != set(POLICIES):
                raise ValueError(f"policy coverage mismatch for {system_id}, B={budget}")
            budget_records[str(system_id)] = system_record
            all_system_ids.add(str(system_id))
    if len(all_system_ids) != expected_system_count:
        raise ValueError(
            f"expected {expected_system_count} exact systems, found {len(all_system_ids)}"
        )
    per_budget: dict[str, Any] = {}
    for budget in range(1, 7):
        by_system = records[budget]
        if set(by_system) != all_system_ids:
            raise ValueError(f"budget {budget} does not cover the same exact systems")
        metrics: dict[str, Any] = {}
        for policy in POLICIES:
            metrics[policy] = {
                name: float(
                    np.mean([_metric(row, policy, name) for row in by_system.values()])
                )
                for name in (
                    "oracle_pool_confirmed_discoveries",
                    "final_causal_confirmed_discoveries",
                    "causal_discoveries",
                    "wall_seconds",
                )
            }
        contrasts: dict[str, Any] = {}
        for label, (left, right) in PAIR_CONTRASTS.items():
            contrasts[label] = {}
            for metric_name in (
                "oracle_pool_confirmed_discoveries",
                "final_causal_confirmed_discoveries",
                "causal_discoveries",
                "wall_seconds",
            ):
                values = np.asarray(
                    [
                        _metric(row, left, metric_name) - _metric(row, right, metric_name)
                        for row in by_system.values()
                    ],
                    dtype=np.float64,
                )
                contrasts[label][metric_name] = _contrast_summary(
                    values, seed=20270804 + budget * 100 + len(metric_name)
                )
        per_budget[str(budget)] = {"means": metrics, "direct_contrasts": contrasts}
    summary = {
        "status": "complete_hull_ens_development_summary",
        "protocol": identity["protocol"],
        "executor_identity": identity,
        "input_root": str(input_root),
        "input_file_count": len(paths),
        "exact_system_count": len(all_system_ids),
        "budgets": per_budget,
    }
    if output_path.exists():
        raise ValueError(f"refusing to overwrite summary: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-system-count", type=int, default=230)
    args = parser.parse_args()
    print(json.dumps(summarize(**vars(args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
