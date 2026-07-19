"""Execute the frozen exact-system WBM grid with preregistered trace reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
SRC_ROOT = TOOLS_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmem import (  # noqa: E402
    PRIMARY_STRATEGIES,
    PrequentialRoundMetrics,
    aggregate_prequential_prefix,
    frozen_grid_cells,
    paired_system_bootstrap,
)

BOOTSTRAP_SEED = 20270717
BOOTSTRAP_ITERATIONS = 10000
PREQUENTIAL_METRICS = (
    "boundary_weighted_causal_crps",
    "boundary_weighted_causal_brier",
    "boundary_weighted_causal_log_loss",
    "residual_rmse_ev_per_atom",
    "residual_gaussian_nll",
    "boundary_weighted_false_stable_cost",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _physical_groups() -> tuple[dict[str, Any], ...]:
    return (
        *(
            {
                "name": f"primary-k{capacity}-b12",
                "budget": 12,
                "capacity": capacity,
                "strategies": PRIMARY_STRATEGIES,
            }
            for capacity in (1, 2, 4)
        ),
        {
            "name": "full-history-b12",
            "budget": 12,
            "capacity": 0,
            "strategies": ("full_history",),
        },
        {
            "name": "joint-risk-k2-b8",
            "budget": 8,
            "capacity": 2,
            "strategies": ("joint_posterior_risk_one_swap",),
        },
        {
            "name": "joint-risk-k4-b12",
            "budget": 12,
            "capacity": 4,
            "strategies": ("joint_posterior_risk_one_swap",),
        },
    )


def _run_physical_group(args: argparse.Namespace, group: dict[str, Any]) -> Path:
    output = args.output_dir / "physical" / group["name"]
    command = [
        sys.executable,
        str(TOOLS_DIR / "run_wbm_calibration_engineering_pilot.py"),
        "--gate-audit",
        str(args.gate_audit),
        "--license-manifest",
        str(args.license_manifest),
        "--pool-manifest",
        str(args.grid_manifest),
        "--parity-audit",
        str(args.parity_audit),
        "--soap-cache",
        str(args.soap_cache),
        "--cleaned-ids",
        str(args.cleaned_ids),
        "--raw-cse-root",
        str(args.raw_cse_root),
        "--ppd",
        str(args.ppd),
        "--output-dir",
        str(output),
        "--budget",
        str(group["budget"]),
        "--capacity",
        str(group["capacity"]),
        "--acquisition",
        "frozen",
        "--audit-budget-prefix",
    ]
    for strategy in group["strategies"]:
        command.extend(("--strategy", strategy))
    subprocess.run(command, check=True)
    return output / "summary.json"


def _group_for_cell(strategy: str, capacity: int | None, canonical_budget: int) -> str:
    if strategy in PRIMARY_STRATEGIES:
        return f"primary-k{capacity}-b12"
    if strategy == "full_history":
        return "full-history-b12"
    return f"joint-risk-k{capacity}-b{canonical_budget}"


def _stratum_means(rows: list[dict[str, Any]], metric: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["stratum"]].append(float(row["prequential"][metric]))
    return {key: sum(values) / len(values) for key, values in sorted(grouped.items())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--license-manifest", type=Path, required=True)
    parser.add_argument("--grid-manifest", type=Path, required=True)
    parser.add_argument("--parity-audit", type=Path, required=True)
    parser.add_argument("--soap-cache", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--raw-cse-root", type=Path, required=True)
    parser.add_argument("--ppd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for value in vars(args).values():
        if isinstance(value, Path) and value.resolve().is_relative_to(repo_root):
            raise ValueError("real WBM grid inputs and outputs must remain outside Git")
    if args.output_dir.exists():
        raise FileExistsError("frozen grid output directory is immutable")
    manifest = json.loads(args.grid_manifest.read_text(encoding="utf-8"))
    expected_cells = [item.model_dump(mode="json") for item in frozen_grid_cells()]
    if manifest["selection"]["grid"]["cells"] != expected_cells:
        raise ValueError("grid manifest does not match the code-frozen execution plan")
    args.output_dir.mkdir(parents=True)
    summaries = {
        group["name"]: _run_physical_group(args, group)
        for group in _physical_groups()
    }
    physical = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in summaries.items()
    }
    pools = manifest["selection"]["pools"]
    run_index = {
        (group_name, run["pool"], run["strategy"]): run
        for group_name, summary in physical.items()
        for run in summary["runs"]
    }
    rows: list[dict[str, Any]] = []
    for cell in frozen_grid_cells():
        group_name = _group_for_cell(
            cell.strategy, cell.capacity, cell.canonical_budget
        )
        for system_name, pool in sorted(pools.items()):
            run = run_index[(group_name, system_name, cell.strategy)]
            round_rows = tuple(
                PrequentialRoundMetrics.model_validate(item)
                for item in run["prequential_rounds"]
            )
            prefix = aggregate_prequential_prefix(round_rows, cell.budget)
            rows.append(
                {
                    "system": system_name,
                    "stratum": pool["chemical_complexity_stratum"],
                    "candidate_count": pool["candidate_count"],
                    **cell.model_dump(mode="json"),
                    "prequential": prefix,
                    "selected_query_ids": run["selected_query_ids"][: cell.budget],
                    "canonical_trace_checksum": run["trace_checksum"],
                    "canonical_wall_seconds": (
                        run["wall_seconds"] if cell.physical_execution else None
                    ),
                    "physical_group": group_name,
                }
            )
    comparisons: list[dict[str, Any]] = []
    for budget in (4, 8, 12):
        for capacity in (1, 2, 4):
            if capacity >= budget:
                continue
            cell_rows = [
                item
                for item in rows
                if item["budget"] == budget and item["capacity"] == capacity
            ]
            for baseline in ("gp_variance_one_swap", "full_history"):
                for metric in PREQUENTIAL_METRICS:
                    dacc = {
                        item["system"]: float(item["prequential"][metric])
                        for item in cell_rows
                        if item["strategy"] == "decision_coreset"
                    }
                    baseline_values = {
                        item["system"]: float(item["prequential"][metric])
                        for item in rows
                        if item["budget"] == budget
                        and item["strategy"] == baseline
                        and (baseline == "full_history" or item["capacity"] == capacity)
                    }
                    comparison = paired_system_bootstrap(
                        dacc,
                        baseline_values,
                        seed=BOOTSTRAP_SEED,
                        iterations=BOOTSTRAP_ITERATIONS,
                    )
                    comparisons.append(
                        {
                            "budget": budget,
                            "capacity": capacity,
                            "baseline": baseline,
                            "metric": metric,
                            **comparison,
                            "dacc_stratum_means": _stratum_means(
                                [
                                    item
                                    for item in cell_rows
                                    if item["strategy"] == "decision_coreset"
                                ],
                                metric,
                            ),
                        }
                    )
    report = {
        "schema_version": "wbm-frozen-grid-results-v1",
        "scope": "fixed_historical_pipeline_engineering_grid_not_claim_grade",
        "grid_manifest_sha256": _sha256(args.grid_manifest),
        "physical_batch_count": len(summaries),
        "physical_trace_count_per_system": 15,
        "reported_cell_count_per_system": 37,
        "system_count": len(pools),
        "system_cell_rows": rows,
        "paired_system_clustered_bootstrap": comparisons,
        "noninferiority_margins": None,
        "paper_go_status": "blocked_until_disjoint_calibration_margins_are_frozen",
        "physical_summaries": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in summaries.items()
        },
        "guardrails": [
            "exact chemical systems are never mixed",
            "lower-budget labels reuse only the identical strategy/capacity trace prefix",
            "oracle outcomes enter only evaluator-side prequential metrics",
            "system-clustered bootstrap treats each exact system as one unit",
            "survival and exhaustive subset audits are absent",
        ],
    }
    output = args.output_dir / "grid_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"grid_summary={output}")


if __name__ == "__main__":
    main()
