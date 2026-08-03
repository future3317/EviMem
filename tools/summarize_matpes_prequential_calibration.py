"""Summarize prequential energy-posterior diagnostics in frozen P0-v3 records.

The P0-v3 runner stores energy-level diagnostics for each system and policy.
This tool reports equal-system means by budget and records explicitly that
hull-membership probability calibration was not logged by the frozen runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "independent_confirmation_source_rollout",
)
FIELDS = {
    "coverage_90": "prequential_posterior_energy_90pct_coverage",
    "rmse_ev_per_atom": "prequential_posterior_energy_rmse_ev_per_atom",
    "mae_ev_per_atom": "prequential_posterior_energy_mae_ev_per_atom",
    "gaussian_nll": "prequential_posterior_energy_gaussian_nll",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(root: Path) -> tuple[dict[int, list[dict[str, Any]]], list[Path]]:
    all_rows: dict[int, list[dict[str, Any]]] = {}
    paths: list[Path] = []
    for budget in range(1, 7):
        fold_paths = sorted(root.glob(f"matpes-p0v2-core-fold*-b{budget}-main.json"))
        expected = [f"matpes-p0v2-core-fold{fold}-b{budget}-main.json" for fold in range(1, 6)]
        if [path.name for path in fold_paths] != expected:
            raise ValueError(f"incomplete calibration roster at B={budget}")
        rows: list[dict[str, Any]] = []
        for path in fold_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if tuple(payload.get("active_policies", ())) != POLICIES:
                raise ValueError(f"unexpected policy roster in {path}")
            for system, result in payload["systems"].items():
                for policy in POLICIES:
                    metrics = result["strategies"][policy]
                    row = {"system": system, "policy": policy}
                    for name, field in FIELDS.items():
                        value = metrics.get(field)
                        row[name] = None if value is None else float(value)
                    rows.append(row)
            paths.append(path)
        if len(rows) != 230 * len(POLICIES):
            raise ValueError(f"expected 690 rows at B={budget}, found {len(rows)}")
        all_rows[budget] = rows
    return all_rows, paths


def summarize(*, input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("derived summaries must remain outside Git")
    all_rows, paths = _load(input_root)
    by_budget: dict[str, dict[str, Any]] = {}
    for budget, rows in all_rows.items():
        by_policy: dict[str, Any] = {}
        for policy in POLICIES:
            policy_rows = [row for row in rows if row["policy"] == policy]
            by_policy[policy] = {
                name: {
                    "mean_over_systems": float(np.mean(values)) if values else None,
                    "median_over_systems": float(np.median(values)) if values else None,
                    "system_count": len(values),
                    "missing_system_count": len(policy_rows) - len(values),
                }
                for name in FIELDS
                for values in [[row[name] for row in policy_rows if row[name] is not None]]
            }
        by_budget[str(budget)] = by_policy
    result = {
        "schema_version": 1,
        "status": "complete_prequential_energy_calibration_summary",
        "input_root": str(input_root),
        "input_sha256": {str(path): _sha256(path) for path in paths},
        "system_count": 230,
        "policy_roster": list(POLICIES),
        "budgets": by_budget,
        "hull_membership_probability_calibration": {
            "status": "not_recorded_in_frozen_p0_v3_outputs",
            "note": "The runner stores energy-level prequential diagnostics and decision traces, but not per-candidate posterior hull-membership probabilities with realized labels.",
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
    result = summarize(input_root=args.input_root, output=args.output)
    print(f"output={args.output.resolve()}")
    print(json.dumps({"status": result["status"], "budgets": list(result["budgets"])}, indent=2))


if __name__ == "__main__":
    main()
