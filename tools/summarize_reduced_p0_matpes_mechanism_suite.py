"""Summarize the frozen reduced P0 MatPES mechanism ablation.

The reduced amendment is intentionally separate from the full-suite
summarizer: it has a smaller, explicitly registered policy roster and a
different external raw-result root. This utility refuses partial schedules and
never writes a summary inside the Git repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from summarize_p0_matpes_mechanism_suite import (
    CORE_POLICIES,
    METRICS,
    _expected_path,
    _load,
    _policy_summary,
    _sha256,
    _system_rows,
)

REDUCED_POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "diagonal_ic_sarr",
    "independent_confirmation_source_rollout",
)


def _reduced_path(root: Path, fold: int) -> Path:
    return root / f"matpes-p0v3-reduced-ablation-fold{fold}-b6.json"


def summarize(*, core_root: Path, reduced_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("P0 summaries must remain outside Git")

    core = {
        budget: [
            _load(
                _expected_path(core_root, "core", fold, budget, "main"),
                expected_policies=CORE_POLICIES,
                budget=budget,
            )
            for fold in range(1, 6)
        ]
        for budget in range(1, 7)
    }
    reduced = [
        _load(
            _reduced_path(reduced_root, fold),
            expected_policies=REDUCED_POLICIES,
            budget=6,
        )
        for fold in range(1, 6)
    ]

    source_by_budget = {
        budget: _system_rows(payloads, "source_margin") for budget, payloads in core.items()
    }
    curve = {
        str(budget): {
            policy: _policy_summary(
                policy_rows=_system_rows(payloads, policy),
                source_rows=source_by_budget[budget],
            )
            for policy in CORE_POLICIES
        }
        for budget, payloads in core.items()
    }
    reduced_source = _system_rows(reduced, "source_margin")
    reduced_audit = {
        policy: _policy_summary(
            policy_rows=_system_rows(reduced, policy),
            source_rows=reduced_source,
        )
        for policy in REDUCED_POLICIES
    }
    payload = {
        "schema_version": 1,
        "status": "complete_retrospective_crossfitted_development_reduced_mechanism_summary",
        "core_root": str(core_root),
        "reduced_root": str(reduced_root),
        "input_sha256": {
            "core_top_level_json": {
                str(path): _sha256(path) for path in sorted(core_root.glob("*.json"))
            },
            "reduced_top_level_json": {
                str(path): _sha256(path) for path in sorted(reduced_root.glob("*.json"))
            },
        },
        "bootstrap": {"replicates": 20_000, "seed": 20260730},
        "sign_flip": {"draws": 100_000, "seed": 20260731},
        "core_curve": curve,
        "reduced_b6_targeted_mechanism_audit": reduced_audit,
        "reduced_policy_roster": list(REDUCED_POLICIES),
        "metrics": list(METRICS),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-root", type=Path, required=True)
    parser.add_argument("--reduced-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(core_root=args.core_root, reduced_root=args.reduced_root, output=args.output)
    print(f"output={args.output.resolve()}")
    print(json.dumps({"status": result["status"], "curve_budgets": sorted(result["core_curve"])}, indent=2))


if __name__ == "__main__":
    main()
