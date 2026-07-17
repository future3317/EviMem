"""Build the oracle-blind exact-system manifest for the frozen WBM grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
SRC_ROOT = TOOLS_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from build_wbm_small_pool_manifest import (  # noqa: E402
    ObservableCandidate,
    _read_cleaned_ids,
    read_observable_candidates,
)

from matmem import frozen_grid_cells  # noqa: E402

RELEASE_ID = "WBM-2021.68-cleaned-256963-frozen-grid-v1"
MIN_SYSTEM_SIZE = 16
MAX_SYSTEMS_PER_STRATUM = 8
CALIBRATION_FRACTION = 0.05
STRATA = ("binary", "ternary", "quaternary_or_higher")


def _sha256_lines(values: list[str]) -> str:
    return "sha256:" + hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _rank(release_id: str, value: str) -> str:
    return hashlib.sha256(f"{release_id}|{value}".encode()).hexdigest()


def _stratum(system: tuple[str, ...]) -> str:
    if len(system) == 2:
        return "binary"
    if len(system) == 3:
        return "ternary"
    return "quaternary_or_higher"


def select_frozen_grid_systems(
    candidates: list[ObservableCandidate],
    *,
    release_id: str = RELEASE_ID,
) -> dict[str, Any]:
    """Select up to eight systems/stratum using only composition and release ID."""

    by_system: dict[tuple[str, ...], list[ObservableCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_system[candidate.chemical_system].append(candidate)
    all_systems = sorted(by_system)
    if not all_systems:
        raise ValueError("frozen grid selection requires candidate systems")
    calibration_count = math.ceil(CALIBRATION_FRACTION * len(all_systems))
    calibration = set(
        sorted(
            all_systems,
            key=lambda system: (_rank(release_id, "-".join(system)), system),
        )[:calibration_count]
    )
    eligible = {
        system: rows
        for system, rows in by_system.items()
        if system not in calibration and len(rows) >= MIN_SYSTEM_SIZE
    }
    selected_systems: list[tuple[str, ...]] = []
    stratum_report: dict[str, Any] = {}
    for stratum in STRATA:
        available = [system for system in eligible if _stratum(system) == stratum]
        chosen = sorted(
            available,
            key=lambda system: (_rank(release_id, "-".join(system)), system),
        )[:MAX_SYSTEMS_PER_STRATUM]
        selected_systems.extend(chosen)
        stratum_report[stratum] = {
            "eligible_system_count": len(available),
            "selected_system_count": len(chosen),
            "selected_systems": ["-".join(item) for item in chosen],
        }
    if not selected_systems:
        raise ValueError("no evaluation system meets the frozen minimum size")
    pools: dict[str, Any] = {}
    selected_ids: list[str] = []
    for system in selected_systems:
        rows = sorted(
            eligible[system],
            key=lambda item: (_rank(release_id, item.query_id), item.query_id),
        )
        name = "-".join(system)
        ids = [item.query_id for item in rows]
        selected_ids.extend(ids)
        pools[name] = {
            "chemical_system": list(system),
            "chemical_complexity_stratum": _stratum(system),
            "candidate_count": len(rows),
            "candidate_order_rule": "SHA256(release_id || query_id), then query_id",
            "exact_structure_duplicate_count": len(rows)
            - len({item.exact_structure_sha256 for item in rows}),
            "candidates": [asdict(item) for item in rows],
        }
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("frozen exact-system pools must have disjoint candidate IDs")
    cells = frozen_grid_cells()
    execution_keys = sorted({item.execution_key for item in cells})
    return {
        "release_id": release_id,
        "selection_information": "composition, cleaned membership, structure bytes, IDs only",
        "calibration_fraction": CALIBRATION_FRACTION,
        "calibration_system_count": len(calibration),
        "calibration_system_checksum": _sha256_lines(
            sorted("-".join(item) for item in calibration)
        ),
        "minimum_exact_system_candidate_count": MIN_SYSTEM_SIZE,
        "maximum_systems_per_stratum": MAX_SYSTEMS_PER_STRATUM,
        "strata": stratum_report,
        "selected_system_count": len(selected_systems),
        "selected_candidate_count": len(selected_ids),
        "selected_candidate_id_checksum": _sha256_lines(sorted(selected_ids)),
        "pools": pools,
        "grid": {
            "reported_cell_count_per_system": len(cells),
            "physical_trace_count_per_system": len(execution_keys),
            "trace_reuse_rule": (
                "run the canonical maximum-budget trace once and derive lower-budget "
                "prefixes only for the identical strategy and capacity"
            ),
            "cells": [item.model_dump(mode="json") for item in cells],
        },
        "guardrails": [
            "no exact chemical systems are mixed",
            "every selected system uses all cleaned candidates in frozen hash order",
            "selection does not read predictor residuals, energies, hull labels, or outcomes",
            "joint posterior risk runs only at B8/K2 and B12/K4",
            "survival acquisition is absent",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cse-root", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("frozen grid manifest must remain outside the repository")
    candidates = read_observable_candidates(
        cse_root=args.cse_root,
        structures_root=args.structures_root,
        cleaned_ids=_read_cleaned_ids(args.cleaned_ids),
    )
    manifest = {
        "schema_version": "wbm-frozen-exact-system-grid-v1",
        "scope": "oracle_blind_frozen_grid_not_yet_claim_grade",
        "selection": select_frozen_grid_systems(candidates),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    selection = manifest["selection"]
    print(
        f"systems={selection['selected_system_count']} "
        f"candidates={selection['selected_candidate_count']} "
        f"physical_traces_per_system={selection['grid']['physical_trace_count_per_system']}"
    )


if __name__ == "__main__":
    main()
