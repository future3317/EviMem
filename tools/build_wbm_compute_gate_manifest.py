"""Build the oracle-blind WBM long-archive compute-relevance panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_wbm_small_pool_manifest import (  # noqa: E402
    ObservableCandidate,
    _read_cleaned_ids,
    read_observable_candidates,
)

RELEASE_ID = "WBM-2021.68-cleaned-256963-compute-relevance-v1"
BUDGETS = (12, 24, 40)
MINIMUM_CANDIDATE_COUNT = max(BUDGETS) + 1
DEFAULT_SYSTEM_COUNT = 3
AMDAHL_TARGET_SPEEDUP = 1.10
AMDAHL_MINIMUM_GP_FRACTION = 1.0 - 1.0 / AMDAHL_TARGET_SPEEDUP


def _rank(value: str) -> str:
    return hashlib.sha256(f"{RELEASE_ID}|{value}".encode()).hexdigest()


def _sha256_lines(values: list[str]) -> str:
    return "sha256:" + hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def select_compute_gate_systems(
    candidates: list[ObservableCandidate],
    *,
    system_count: int = DEFAULT_SYSTEM_COUNT,
) -> dict[str, Any]:
    """Select longest exact systems using only observable pool topology."""

    if system_count < 1:
        raise ValueError("compute-gate system count must be positive")
    by_system: dict[tuple[str, ...], list[ObservableCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_system[candidate.chemical_system].append(candidate)
    eligible = {
        system: rows
        for system, rows in by_system.items()
        if len(rows) >= MINIMUM_CANDIDATE_COUNT
    }
    ranked = sorted(
        eligible,
        key=lambda system: (-len(eligible[system]), _rank("-".join(system)), system),
    )
    selected = ranked[:system_count]
    if not selected:
        raise ValueError(
            f"no exact WBM system has at least {MINIMUM_CANDIDATE_COUNT} candidates"
        )
    pools: dict[str, Any] = {}
    selected_ids: list[str] = []
    for system in selected:
        rows = sorted(
            eligible[system],
            key=lambda item: (_rank(item.query_id), item.query_id),
        )
        name = "-".join(system)
        ids = [item.query_id for item in rows]
        selected_ids.extend(ids)
        pools[name] = {
            "chemical_system": list(system),
            "chemical_complexity_stratum": (
                "binary" if len(system) == 2 else "ternary" if len(system) == 3 else "higher"
            ),
            "candidate_count": len(rows),
            "candidate_order_rule": "SHA256(release_id || query_id), then query_id",
            "candidates": [asdict(item) for item in rows],
        }
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("compute-gate exact systems must have disjoint candidate IDs")
    return {
        "release_id": RELEASE_ID,
        "selection_information": (
            "cleaned membership, exact chemical system, candidate count, initial "
            "structure bytes, and IDs only; no energy, residual, label, or predictor score"
        ),
        "selection_rule": (
            "descending exact-system candidate count; SHA256 release/system tie-break; "
            "use every cleaned candidate in each selected system"
        ),
        "minimum_candidate_count": MINIMUM_CANDIDATE_COUNT,
        "eligible_system_count": len(eligible),
        "requested_system_count": system_count,
        "selected_system_count": len(selected),
        "selected_candidate_count": len(selected_ids),
        "selected_candidate_id_checksum": _sha256_lines(sorted(selected_ids)),
        "budgets": list(BUDGETS),
        "amdahl_gate": {
            "target_ideal_speedup": AMDAHL_TARGET_SPEEDUP,
            "minimum_real_trace_gp_numerical_fraction": AMDAHL_MINIMUM_GP_FRACTION,
            "decision": (
                "stop WBM end-to-end compute-Pareto claims if the B40 GP numerical "
                "fraction is below this threshold"
            ),
        },
        "pools": pools,
        "guardrails": [
            "exact chemical systems are never mixed",
            "selection is oracle blind and outcome independent",
            "B12 and B24 are prefixes of the same frozen B40 action sequence",
            "compute timing is not a causal calibration comparison",
            "AKSC implementation remains blocked until this gate is evaluated",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cse-root", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--system-count", type=int, default=DEFAULT_SYSTEM_COUNT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("compute-gate manifest must remain outside the repository")
    candidates = read_observable_candidates(
        cse_root=args.cse_root,
        structures_root=args.structures_root,
        cleaned_ids=_read_cleaned_ids(args.cleaned_ids),
    )
    manifest = {
        "schema_version": "wbm-long-archive-compute-gate-v1",
        "scope": "oracle_blind_compute_relevance_preregistration_no_method_comparison",
        "selection": select_compute_gate_systems(
            candidates,
            system_count=args.system_count,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"eligible={manifest['selection']['eligible_system_count']} "
        f"selected={manifest['selection']['selected_system_count']} "
        f"candidates={manifest['selection']['selected_candidate_count']}"
    )
    for name, pool in manifest["selection"]["pools"].items():
        print(f"{name}: {pool['candidate_count']}")


if __name__ == "__main__":
    main()
