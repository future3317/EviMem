"""Build oracle-blind, exact-chemical-system WBM small pools outside Git.

The frozen 256-candidate exact-system design is infeasible: no cleaned WBM
system has 64 candidates. This tool implements only the explicitly labelled
16-candidate exploratory amendment. It reads composition and structure fields,
never WBM energy, hull label, or predictor value.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from build_wbm_cleaned_id_manifest import (  # noqa: E402
    STEP3_ANOMALOUS_STRUCTURE_IDS,
    _fix_step3_alignment,
    _source_to_wbm_id,
)

RELEASE_ID = "wbm-small-exact-system-pilot-v1"
CALIBRATION_FRACTION = 0.05
POOL_SIZE = 16
# Exact 4+ element WBM systems contain at most seven cleaned candidates. The
# small-pool pilot is therefore deliberately limited to two- and three-element
# systems, not padded with unrelated systems.
STRATA = ((2, "small"), (2, "large"), (3, "small"), (3, "large"))


@dataclass(frozen=True)
class ObservableCandidate:
    query_id: str
    chemical_system: tuple[str, ...]
    composition: tuple[tuple[str, float], ...]
    exact_structure_sha256: str


def _checksum(values: list[str]) -> str:
    return "sha256:" + hashlib.sha256(("\n".join(sorted(values)) + "\n").encode()).hexdigest()


def _rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()


def _structure_checksum(structure: object) -> str:
    payload = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _read_cleaned_ids(path: Path) -> set[str]:
    return {item.strip() for item in path.read_text(encoding="utf-8").splitlines() if item.strip()}


def read_observable_candidates(
    *, cse_root: Path, structures_root: Path, cleaned_ids: set[str]
) -> list[ObservableCandidate]:
    """Read only policy-visible CSE fields and apply pinned source-ID alignment."""

    candidates: list[ObservableCandidate] = []
    anomalies = set(STEP3_ANOMALOUS_STRUCTURE_IDS)
    for step in range(1, 6):
        cse_path = cse_root / f"step_{step}.json.bz2"
        structures_path = structures_root / f"wbm-structures-step-{step}.json.bz2"
        cse_payload = json.loads(bz2.decompress(cse_path.read_bytes()))
        structures = json.loads(bz2.decompress(structures_path.read_bytes()))
        entries = cse_payload.get("entries")
        if not isinstance(entries, list) or not isinstance(structures, dict):
            raise ValueError(f"invalid WBM source files for step {step}")
        source_ids = [item for item in structures if item not in anomalies]
        if len(entries) != len(source_ids):
            raise ValueError(f"WBM source alignment mismatch in step {step}")
        for source_id, entry in zip(source_ids, entries, strict=True):
            query_id = _fix_step3_alignment(_source_to_wbm_id(source_id))
            if query_id not in cleaned_ids:
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("composition"), dict):
                raise ValueError(f"invalid WBM composition for {query_id}")
            composition = tuple(sorted((str(key), float(value)) for key, value in entry["composition"].items()))
            candidates.append(
                ObservableCandidate(
                    query_id=query_id,
                    chemical_system=tuple(item[0] for item in composition),
                    composition=composition,
                    exact_structure_sha256=_structure_checksum(entry.get("structure")),
                )
            )
    if len({item.query_id for item in candidates}) != len(candidates):
        raise ValueError("aligned cleaned WBM IDs are not unique")
    if {item.query_id for item in candidates} != cleaned_ids:
        raise ValueError("observable candidate IDs do not exactly match cleaned IDs")
    return candidates


def deduplicate_exact_structures(candidates: list[ObservableCandidate]) -> tuple[list[ObservableCandidate], int]:
    """Retain the lexicographically first row for byte-identical CSE structures."""

    groups: dict[str, list[ObservableCandidate]] = defaultdict(list)
    for candidate in candidates:
        groups[candidate.exact_structure_sha256].append(candidate)
    retained = [min(items, key=lambda item: item.query_id) for items in groups.values()]
    return sorted(retained, key=lambda item: item.query_id), len(candidates) - len(retained)


def select_small_pools(
    candidates: list[ObservableCandidate],
    *, pool_size: int = POOL_SIZE,
    calibration_fraction: float = CALIBRATION_FRACTION,
) -> dict[str, object]:
    """Select eight systems without labels or prediction values, or fail visibly."""

    if pool_size < 1 or not 0 < calibration_fraction < 1:
        raise ValueError("pool_size must be positive and calibration_fraction must be in (0, 1)")
    by_system: dict[tuple[str, ...], list[ObservableCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_system[candidate.chemical_system].append(candidate)
    all_systems = sorted(by_system)
    calibration_count = math.ceil(calibration_fraction * len(all_systems))
    calibration_systems = set(
        sorted(all_systems, key=lambda system: (_rank(RELEASE_ID, "-".join(system)), system))[
            :calibration_count
        ]
    )
    evaluation = {system: rows for system, rows in by_system.items() if system not in calibration_systems}
    eligible = {system: rows for system, rows in evaluation.items() if len(rows) >= pool_size}
    selected: dict[str, object] = {}
    stratum_profile: dict[str, object] = {}
    for element_count, size in STRATA:
        systems = [system for system in eligible if len(system) == element_count]
        counts = [len(eligible[system]) for system in systems]
        key = f"{element_count}_{size}"
        if not counts:
            raise ValueError(f"no eligible systems in stratum {key}")
        median = statistics.median(counts)
        if size == "small":
            stratum_systems = [system for system in systems if len(eligible[system]) <= median]
        else:
            stratum_systems = [system for system in systems if len(eligible[system]) > median]
        if len(stratum_systems) < 2:
            raise ValueError(f"fewer than two systems in stratum {key}")
        chosen_systems = sorted(
            stratum_systems, key=lambda system: (_rank(RELEASE_ID, "-".join(system)), system)
        )[:2]
        stratum_profile[key] = {
            "eligible_system_count": len(systems), "median_candidate_count": median,
            "stratum_system_count": len(stratum_systems), "selected_systems": ["-".join(item) for item in chosen_systems],
        }
        for system in chosen_systems:
            rows = sorted(
                eligible[system],
                key=lambda item: (
                    _rank(RELEASE_ID, item.exact_structure_sha256), item.query_id
                ),
            )[:pool_size]
            selected["-".join(system)] = {
                "chemical_system": list(system), "candidate_count_before_selection": len(eligible[system]),
                "candidates": [asdict(item) for item in rows],
            }
    selected_ids = [item["query_id"] for pool in selected.values() for item in pool["candidates"]]
    expected_pool_count = 2 * len(STRATA)
    if (
        len(selected) != expected_pool_count
        or len(selected_ids) != expected_pool_count * pool_size
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise ValueError("selected small pools are not disjoint fixed-size pools")
    return {
        "release_id": RELEASE_ID, "pool_size": pool_size,
        "calibration_fraction": calibration_fraction, "calibration_system_count": len(calibration_systems),
        "calibration_system_checksum": _checksum(["-".join(item) for item in calibration_systems]),
        "exact_system_count": len(all_systems), "eligible_evaluation_system_count": len(eligible),
        "strata": stratum_profile, "pools": selected,
        "selected_candidate_id_checksum": _checksum(selected_ids),
        "mp_overlap_exclusion": "not implemented; this is an exploratory amendment, not the formal canonicalized pool protocol",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cse-root", type=Path, required=True)
    parser.add_argument("--structures-root", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[1]):
        parser.error("pool manifest must remain outside the repository")
    raw = read_observable_candidates(
        cse_root=args.cse_root, structures_root=args.structures_root, cleaned_ids=_read_cleaned_ids(args.cleaned_ids)
    )
    deduplicated, duplicate_count = deduplicate_exact_structures(raw)
    manifest = {
        "scope": "exploratory_wbm_small_exact_system_pool_oracle_blind",
        "raw_cleaned_candidate_count": len(raw), "exact_serialized_structure_duplicates_removed": duplicate_count,
        "selection": select_small_pools(deduplicated),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"pools={len(manifest['selection']['pools'])} pool_size={manifest['selection']['pool_size']}")
    print(f"selected_id_checksum={manifest['selection']['selected_candidate_id_checksum']}")


if __name__ == "__main__":
    main()
