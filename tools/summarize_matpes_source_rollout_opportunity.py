"""Validate and summarize a frozen Source-Rollout opportunity-cost audit.

The tool is deliberately read-only with respect to task and oracle artifacts:
it receives only the audit replay and its precommitted state plan.  Its output
is a descriptive diagnostic, not a rule that retunes the policy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty value group")
    return {
        "count": len(values),
        "mean": fmean(values),
        "median": median(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def summarize(*, audit_path: Path, plan_path: Path, output_path: Path) -> dict[str, Any]:
    """Return a deterministic, development-only opportunity-cost summary."""
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("audit summary must remain outside Git")
    if output_path.exists():
        raise FileExistsError(output_path)

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if audit.get("status") != "development_high_precision_replay":
        raise ValueError("audit is not a high-precision development replay")
    if audit.get("evaluation_systems_accessed") is not False:
        raise ValueError("audit must not access evaluation systems")
    if plan.get("evaluation_systems_accessed") is not False:
        raise ValueError("frozen plan must be development-only")
    if audit.get("plan_sha256") != _sha256(plan_path):
        raise ValueError("audit plan checksum disagrees")
    if audit.get("task_sha256") != plan.get("task_sha256"):
        raise ValueError("audit task checksum disagrees with the frozen plan")
    if audit.get("sarr_sha256") != plan.get("sarr_result_sha256"):
        raise ValueError("audit SARR checksum disagrees with the frozen plan")

    planned = {
        (row["chemical_system"], int(row["round_index"])): tuple(row["reasons"])
        for row in plan["states"]
    }
    observed: dict[tuple[str, int], dict[str, Any]] = {}
    for row in audit.get("states", []):
        key = (row["chemical_system"], int(row["round_index"]))
        if key in observed:
            raise ValueError(f"duplicate audited state: {key}")
        observed[key] = row
    if set(observed) != set(planned):
        raise ValueError("audited states do not exactly equal the frozen plan")
    if audit.get("state_count") != len(planned) or plan.get("state_count") != len(planned):
        raise ValueError("state count disagrees with the frozen plan")

    numeric_fields = (
        "selected_opportunity_cost",
        "source_opportunity_cost",
        "best_second_gap",
        "selected_high_precision_advantage",
        "selected_high_precision_lower_bound",
    )
    all_values: dict[str, list[float]] = defaultdict(list)
    by_reason: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    selected_non_source = 0
    selected_negative_advantage = 0
    source_positive_opportunity = 0
    for key, row in observed.items():
        values = {name: float(row[name]) for name in numeric_fields}
        for name, value in values.items():
            all_values[name].append(value)
        for reason in planned[key]:
            for name, value in values.items():
                by_reason[reason][name].append(value)
        if row["selected_pair_id"] != row["source_pair_id"]:
            selected_non_source += 1
            if values["selected_high_precision_advantage"] < 0.0:
                selected_negative_advantage += 1
        if values["source_opportunity_cost"] > 0.0:
            source_positive_opportunity += 1

    output = {
        "schema_version": 1,
        "status": "development_high_precision_opportunity_summary",
        "audit_sha256": _sha256(audit_path),
        "plan_sha256": _sha256(plan_path),
        "task_sha256": audit["task_sha256"],
        "sarr_sha256": audit["sarr_sha256"],
        "evaluation_systems_accessed": False,
        "state_count": len(planned),
        "overall": {name: _summary(values) for name, values in sorted(all_values.items())},
        "by_reason": {
            reason: {name: _summary(values) for name, values in sorted(groups.items())}
            for reason, groups in sorted(by_reason.items())
        },
        "counts": {
            "selected_non_source": selected_non_source,
            "selected_non_source_negative_high_precision_advantage": selected_negative_advantage,
            "source_positive_opportunity_cost": source_positive_opportunity,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(audit_path=args.audit, plan_path=args.plan, output_path=args.output)
    print(f"output={args.output} states={result['state_count']}")


if __name__ == "__main__":
    main()
