"""Summarize frozen all-budget IC-SARR versus Delta-Hull contrasts.

This is post-processing only.  It reads the complete P0-v3 core records from
the external result root, computes paired exact-system intervals at each
budget, and writes a derived JSON outside Git.  It never runs a policy and
refuses partial schedules or overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from summarize_p0_matpes_mechanism_suite import _paired_summary

CORE_POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "independent_confirmation_source_rollout",
)
IC_SARR = CORE_POLICIES[-1]
METRICS = {
    "D": "causal_discoveries",
    "F": "final_causal_confirmed_discoveries",
    "T": "oracle_pool_confirmed_discoveries",
    "wall_seconds": "wall_seconds",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_core(root: Path) -> tuple[dict[int, dict[str, dict[str, Any]]], list[Path]]:
    by_budget: dict[int, dict[str, dict[str, Any]]] = {}
    paths: list[Path] = []
    for budget in range(1, 7):
        fold_paths = sorted(root.glob(f"matpes-p0v2-core-fold*-b{budget}-main.json"))
        expected = [f"matpes-p0v2-core-fold{fold}-b{budget}-main.json" for fold in range(1, 6)]
        if [path.name for path in fold_paths] != expected:
            raise ValueError(f"expected five complete fold files for B={budget}, found {[p.name for p in fold_paths]}")
        systems: dict[str, dict[str, Any]] = {}
        for path in fold_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if tuple(payload.get("active_policies", ())) != CORE_POLICIES:
                raise ValueError(f"unexpected policy roster in {path}")
            if payload.get("status") != "exploratory_development_systems_only_not_confirmatory":
                raise ValueError(f"unexpected status in {path}: {payload.get('status')}")
            for system, result in payload.get("systems", {}).items():
                if system in systems:
                    raise ValueError(f"system occurs in multiple folds at B={budget}: {system}")
                if int(result.get("budget", -1)) != budget:
                    raise ValueError(f"wrong budget for {system} in {path}")
                strategies = result.get("strategies", {})
                if set(strategies) != set(CORE_POLICIES):
                    raise ValueError(f"unexpected strategies for {system} in {path}")
                systems[system] = strategies
            paths.append(path)
        if len(systems) != 230:
            raise ValueError(f"expected 230 systems at B={budget}, found {len(systems)}")
        by_budget[budget] = systems
    reference = set(by_budget[1])
    if any(set(rows) != reference for rows in by_budget.values()):
        raise ValueError("budget curves do not use one identical exact-system roster")
    return by_budget, paths


def _contrast(rows: dict[str, dict[str, Any]], policy: str, baseline: str) -> dict[str, Any]:
    systems = sorted(rows)
    result: dict[str, Any] = {}
    for name, field in METRICS.items():
        deltas = np.asarray(
            [float(rows[s][policy][field]) - float(rows[s][baseline][field]) for s in systems],
            dtype=float,
        )
        policy_values = np.asarray([float(rows[s][policy][field]) for s in systems], dtype=float)
        baseline_values = np.asarray([float(rows[s][baseline][field]) for s in systems], dtype=float)
        result[name] = {
            "policy_mean": float(policy_values.mean()),
            "baseline_mean": float(baseline_values.mean()),
            "direct_paired": _paired_summary(deltas),
        }
    return result


def summarize(*, input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("derived summaries must remain outside Git")
    by_budget, paths = _load_core(input_root)
    first_payload = json.loads(
        (input_root / "matpes-p0v2-core-fold1-b1-main.json").read_text(encoding="utf-8")
    )
    identity_keys = (
        "task_sha256",
        "oracle_vault_sha256",
        "development_vault_sha256",
        "script_sha256",
        "posterior_sampler",
        "selected_action_is_only_reveal",
    )
    result = {
        "schema_version": 1,
        "status": "complete_all_budget_direct_paired_comparisons",
        "input_root": str(input_root),
        "input_sha256": {str(path): _sha256(path) for path in paths},
        "system_count": 230,
        "budgets": {
            str(budget): {
                "ic_sarr_vs_delta_hull": _contrast(rows, IC_SARR, "delta_hull_active_search")
            }
            for budget, rows in by_budget.items()
        },
        "inference": {
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20260730,
            "sign_flip_draws": 100_000,
            "sign_flip_seed": 20260731,
            "sign_flip_method": "deterministic_monte_carlo",
            "statistical_unit": "exact chemical system",
        },
        "frozen_protocol_identity": {key: first_payload.get(key) for key in identity_keys},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(input_root=args.input_root, output=args.output)
    print(f"output={args.output.resolve()}")
    print(json.dumps({"status": result["status"], "budgets": list(result["budgets"])}, indent=2))


if __name__ == "__main__":
    main()
