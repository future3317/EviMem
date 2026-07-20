"""Freeze disjoint exact-system WBM calibration pools without reading outcomes."""

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

RELEASE_ID = "WBM-2021.68-cleaned-calibration-v1"
MIN_SYSTEM_SIZE = 16
SYSTEMS_PER_STRATUM = 4
STRATA = ("binary", "ternary")


def _sha256_lines(values: list[str]) -> str:
    return "sha256:" + hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def _rank(value: str) -> str:
    return hashlib.sha256(f"{RELEASE_ID}|{value}".encode()).hexdigest()


def _stratum(system: tuple[str, ...]) -> str | None:
    if len(system) == 2:
        return "binary"
    if len(system) == 3:
        return "ternary"
    return None


def _systems_from_manifest(path: Path) -> set[tuple[str, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        tuple(sorted(pool["chemical_system"]))
        for pool in payload["selection"]["pools"].values()
    }


def select_calibration_systems(
    candidates: list[ObservableCandidate],
    *,
    excluded_systems: set[tuple[str, ...]],
) -> dict[str, Any]:
    """Select eight visible-only exact systems under the frozen hash rule."""

    by_system: dict[tuple[str, ...], list[ObservableCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_system[candidate.chemical_system].append(candidate)
    pools: dict[str, Any] = {}
    report: dict[str, Any] = {}
    for stratum in STRATA:
        eligible = sorted(
            (
                system
                for system, rows in by_system.items()
                if _stratum(system) == stratum
                and len(rows) >= MIN_SYSTEM_SIZE
                and system not in excluded_systems
            ),
            key=lambda system: (_rank("-".join(system)), system),
        )
        chosen = eligible[:SYSTEMS_PER_STRATUM]
        if len(chosen) != SYSTEMS_PER_STRATUM:
            raise ValueError(f"need {SYSTEMS_PER_STRATUM} eligible {stratum} systems")
        report[stratum] = {
            "eligible_system_count": len(eligible),
            "selected_systems": ["-".join(system) for system in chosen],
        }
        for system in chosen:
            rows = sorted(
                by_system[system],
                key=lambda item: (_rank(item.query_id), item.query_id),
            )
            name = "-".join(system)
            pools[name] = {
                "chemical_system": list(system),
                "chemical_complexity_stratum": stratum,
                "candidate_count": len(rows),
                "candidate_order_rule": "SHA256(release_id || query_id), then query_id",
                "exact_structure_duplicate_count": len(rows)
                - len({item.exact_structure_sha256 for item in rows}),
                "candidates": [asdict(item) for item in rows],
            }
    selected_ids = [
        candidate["query_id"]
        for pool in pools.values()
        for candidate in pool["candidates"]
    ]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("calibration pools must have disjoint candidate IDs")
    return {
        "release_id": RELEASE_ID,
        "selection_information": "composition, cleaned membership, structure bytes, IDs only",
        "minimum_exact_system_candidate_count": MIN_SYSTEM_SIZE,
        "systems_per_stratum": SYSTEMS_PER_STRATUM,
        "strata": report,
        "excluded_system_count": len(excluded_systems),
        "excluded_system_checksum": _sha256_lines(
            sorted("-".join(system) for system in excluded_systems)
        ),
        "selected_system_count": len(pools),
        "selected_candidate_count": len(selected_ids),
        "selected_candidate_id_checksum": _sha256_lines(sorted(selected_ids)),
        "pools": pools,
        "guardrails": [
            "calibration systems are exact chemical systems",
            "calibration systems are disjoint from evaluation and GP-development systems",
            "selection does not read predictions, residuals, energies, hull labels, or outcomes",
            "all cleaned candidates in each selected system are retained",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cse-root", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(repo_root):
        parser.error("calibration manifest must remain outside the repository")
    if args.output.exists():
        raise FileExistsError("calibration manifest is immutable")
    excluded = _systems_from_manifest(args.evaluation_manifest)
    excluded.update(_systems_from_manifest(args.development_manifest))
    candidates = read_observable_candidates(
        cse_root=args.cse_root,
        structures_root=args.structures_root,
        cleaned_ids=_read_cleaned_ids(args.cleaned_ids),
    )
    payload = {
        "schema_version": "wbm-disjoint-calibration-exact-systems-v1",
        "scope": "oracle_blind_gp_and_margin_calibration_only",
        "selection": select_calibration_systems(
            candidates, excluded_systems=excluded
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"systems={payload['selection']['selected_system_count']} "
        f"candidates={payload['selection']['selected_candidate_count']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
