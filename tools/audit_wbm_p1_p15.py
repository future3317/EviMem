"""Audit engineering P1 parity and frozen-pool P1.5 support outside Git.

This tool is deliberately outcome-reading but policy-free.  It may classify a
frozen pool as informative or underpowered; it must never select, replace, or
rank pools.  Candidate truth is a fixed historical-pipeline WBM replay, not an
assertion that an independent official corrected-energy table was reproduced.
"""

from __future__ import annotations

import argparse
import bz2
import gc
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from audit_wbm_official_artifacts import _load_ppd_read_only  # noqa: E402
from build_wbm_candidate_parity_audit import (  # noqa: E402
    CROSS_ENVIRONMENT_TOLERANCE_EV_PER_ATOM,
    STEP_COUNTS,
    STRICT_TOLERANCE_EV_PER_ATOM,
    _repair_historical_composition_interface,
)

NEAR_HULL_EV_PER_ATOM = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_external(path: Path, repo_root: Path) -> None:
    if path.resolve().is_relative_to(repo_root):
        raise ValueError("WBM gate inputs and outputs must remain outside the repository")


def _pool_index(payload: dict[str, Any]) -> tuple[dict[str, str], set[tuple[str, ...]]]:
    by_id: dict[str, str] = {}
    systems: set[tuple[str, ...]] = set()
    for pool_name, pool in payload["selection"]["pools"].items():
        system = tuple(sorted(pool["chemical_system"]))
        systems.add(system)
        for candidate in pool["candidates"]:
            query_id = candidate["query_id"]
            if query_id in by_id:
                raise ValueError(f"duplicate frozen query ID: {query_id}")
            by_id[query_id] = pool_name
    if len(by_id) != 128 or len(systems) != 8:
        raise ValueError("engineering amendment requires eight disjoint 16-candidate pools")
    return by_id, systems


def validate_engineering_p1(
    *,
    pool_payload: dict[str, Any],
    parity_payload: dict[str, Any],
    soap_payload: dict[str, Any],
    soap_query_ids: list[str],
    soap_vectors: np.ndarray,
    soap_cache_sha256: str,
) -> dict[str, Any]:
    """Validate the explicitly limited historical-replay engineering P1 gate."""

    pool_by_id, _ = _pool_index(pool_payload)
    rows = parity_payload["rows"]
    parity_by_id = {row["query_id"]: row for row in rows}
    if len(parity_by_id) != len(rows) or set(parity_by_id) != set(pool_by_id):
        raise ValueError("parity rows do not exactly match the frozen pool IDs")
    if set(soap_payload["pool_by_id"]) != set(pool_by_id):
        raise ValueError("SOAP manifest does not exactly match the frozen pool IDs")
    if set(soap_query_ids) != set(pool_by_id) or len(soap_query_ids) != len(set(soap_query_ids)):
        raise ValueError("SOAP cache IDs do not uniquely match the frozen pool IDs")
    if soap_vectors.shape[0] != len(pool_by_id) or soap_vectors.ndim != 2:
        raise ValueError("SOAP cache has an invalid matrix shape")
    norms = np.linalg.norm(soap_vectors, axis=1)
    if not np.all(np.isfinite(soap_vectors)) or not np.allclose(norms, 1.0, atol=1e-10):
        raise ValueError("SOAP vectors must be finite and unit-normalized")
    if soap_payload["cache_sha256"] != soap_cache_sha256:
        raise ValueError("SOAP cache checksum differs from its manifest")

    form_diffs = [
        abs(
            row["modern_corrected_formation_energy_ev_per_atom"]
            - row["parity_corrected_formation_energy_ev_per_atom"]
        )
        for row in rows
    ]
    hull_diffs = [
        abs(
            row["initial_e_above_hull_modern_ev_per_atom"]
            - row["initial_e_above_hull_parity_ev_per_atom"]
        )
        for row in rows
    ]
    label_mismatches = sum(
        row["stable_label_modern"] != row["stable_label_parity"] for row in rows
    )
    phase_mismatches = sum(
        row["phase_membership_modern"] != row["phase_membership_parity"]
        for row in rows
    )
    passed = bool(
        max(form_diffs, default=math.inf) <= CROSS_ENVIRONMENT_TOLERANCE_EV_PER_ATOM
        and max(hull_diffs, default=math.inf)
        <= CROSS_ENVIRONMENT_TOLERANCE_EV_PER_ATOM
        and label_mismatches == 0
        and phase_mismatches == 0
    )
    return {
        "engineering_p1_passed": passed,
        "claim_scope": "fixed_historical_pipeline_wbm_replay",
        "official_energy_reproduction_claim_permitted": False,
        "identity_level": "byte_identical_relaxed_cse_structure_engineering_only",
        "claim_grade_identity_passed": False,
        "claim_grade_identity_blockers": [
            "primitive_conventional_cell_invariant_matching_pending",
            "prototype_clustering_pending",
            "wbm_mp_structural_overlap_pending",
        ],
        "candidate_count": len(rows),
        "maximum_cross_environment_corrected_formation_difference_ev_per_atom": max(
            form_diffs, default=math.inf
        ),
        "maximum_cross_environment_initial_hull_difference_ev_per_atom": max(
            hull_diffs, default=math.inf
        ),
        "stable_label_mismatch_count": label_mismatches,
        "phase_membership_mismatch_count": phase_mismatches,
        "soap_record_count": len(soap_query_ids),
        "soap_vector_dimension": int(soap_vectors.shape[1]),
        "soap_maximum_norm_error": float(np.max(np.abs(norms - 1.0))),
    }


def _read_cleaned_ids(path: Path) -> set[str]:
    return {
        item.strip()
        for item in path.read_text(encoding="utf-8").splitlines()
        if item.strip()
    }


def _load_exact_system_universe(
    raw_root: Path,
    cleaned_ids: set[str],
    systems: set[tuple[str, ...]],
) -> dict[tuple[str, ...], list[Any]]:
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    selected: dict[tuple[str, ...], list[Any]] = defaultdict(list)
    for step, expected_count in enumerate(STEP_COUNTS, start=1):
        with bz2.open(raw_root / f"step_{step}.json.bz2", "rt", encoding="utf-8") as handle:
            entries = json.load(handle).get("entries")
        if not isinstance(entries, list) or len(entries) != expected_count:
            raise ValueError(f"unexpected WBM step-{step} entry count")
        for index, raw_entry in enumerate(entries, start=1):
            query_id = f"wbm-{step}-{index}"
            if query_id not in cleaned_ids:
                continue
            system = tuple(sorted(str(item) for item in raw_entry["composition"]))
            if system not in systems:
                continue
            copied = json.loads(json.dumps(raw_entry))
            copied["entry_id"] = query_id
            parameters = copied.get("parameters")
            if not isinstance(parameters, dict) or "is_hubbard" not in parameters:
                raise ValueError(f"missing WBM calculation parameters for {query_id}")
            parameters["run_type"] = "GGA+U" if parameters["is_hubbard"] else "GGA"
            selected[system].append(ComputedStructureEntry.from_dict(copied))
        del entries
        gc.collect()
    if set(selected) != systems:
        raise ValueError("one or more frozen exact systems have no cleaned WBM universe")
    return selected


def _correct_exact_system_universe(
    raw_by_system: dict[tuple[str, ...], list[Any]],
) -> dict[tuple[str, ...], list[Any]]:
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

    corrected_by_system: dict[tuple[str, ...], list[Any]] = {}
    for system, entries in raw_by_system.items():
        corrected = MaterialsProject2020Compatibility().process_entries(
            entries, clean=True, verbose=False
        )
        if len(corrected) != len(entries):
            rejected = sorted(
                {str(item.entry_id) for item in entries}
                - {str(item.entry_id) for item in corrected}
            )
            raise ValueError(f"MP2020 rejected exact-system WBM entries: {rejected}")
        corrected_by_system[system] = corrected
    return corrected_by_system


def build_oracle_support(
    *,
    pool_payload: dict[str, Any],
    parity_payload: dict[str, Any],
    raw_root: Path,
    cleaned_ids: set[str],
    ppd: Any,
) -> dict[str, Any]:
    """Build full exact-system oracle hulls without changing the frozen pools."""

    from pymatgen.analysis.phase_diagram import PhaseDiagram
    pool_by_id, systems = _pool_index(pool_payload)
    raw_by_system = _load_exact_system_universe(raw_root, cleaned_ids, systems)
    corrected_by_system = _correct_exact_system_universe(raw_by_system)

    initial_by_system = {
        system: [
            entry
            for entry in ppd.all_entries
            if set(str(element) for element in entry.composition.elements).issubset(system)
        ]
        for system in systems
    }
    parity_by_id = {row["query_id"]: row for row in parity_payload["rows"]}
    pool_reports = []
    for pool_name, pool in sorted(pool_payload["selection"]["pools"].items()):
        system = tuple(sorted(pool["chemical_system"]))
        universe = corrected_by_system[system]
        by_id = {str(entry.entry_id): entry for entry in universe}
        diagram = PhaseDiagram([*initial_by_system[system], *universe])
        margins = {
            query_id: float(diagram.get_e_above_hull(by_id[query_id], allow_negative=True))
            for query_id, owner in pool_by_id.items()
            if owner == pool_name
        }
        stable_ids = sorted(
            query_id
            for query_id, margin in margins.items()
            if margin <= STRICT_TOLERANCE_EV_PER_ATOM
        )
        near_ids = sorted(
            query_id
            for query_id, margin in margins.items()
            if margin <= NEAR_HULL_EV_PER_ATOM
        )
        compositions = {
            str(by_id[query_id].composition.reduced_composition)
            for query_id in margins
        }
        initial_stable = sum(
            bool(parity_by_id[query_id]["stable_label_parity"])
            for query_id in margins
        )
        values = sorted(margins.values())
        pool_reports.append(
            {
                "pool": pool_name,
                "chemical_system": list(system),
                "selected_candidate_count": len(margins),
                "full_exact_system_wbm_universe_count": len(universe),
                "unique_reduced_composition_count": len(compositions),
                "initial_mp_stable_count": initial_stable,
                "oracle_final_stable_count": len(stable_ids),
                "oracle_final_stable_id_checksum": "sha256:"
                + hashlib.sha256(("\n".join(stable_ids) + "\n").encode()).hexdigest(),
                "oracle_final_near_hull_50mev_count": len(near_ids),
                "oracle_margin_ev_per_atom": {
                    "minimum": values[0],
                    "median": float(np.median(values)),
                    "maximum": values[-1],
                },
                "prototype_family_count": None,
                "wbm_mp_overlap_count": None,
                "claim_grade_identity_note": (
                    "not measured by engineering P1.5; required before claim-grade run"
                ),
            }
        )
    total_stable = sum(item["oracle_final_stable_count"] for item in pool_reports)
    stable_pools = sum(item["oracle_final_stable_count"] > 0 for item in pool_reports)
    total_near = sum(item["oracle_final_near_hull_50mev_count"] for item in pool_reports)
    engineering_support = total_stable > 0 and stable_pools >= 2 and total_near > 0
    return {
        "engineering_p1_5_support_present": engineering_support,
        "threshold_status": (
            "retrospective_engineering_diagnostic_not_a_preregistered_claim_grade_threshold"
        ),
        "pool_replacement_permitted": False,
        "zero_positive_pools_retained": True,
        "claim_grade_p1_5_passed": False,
        "claim_grade_blocker": (
            "system count and identity clusters must be frozen from a declared precision target"
        ),
        "oracle_final_stable_candidate_count": total_stable,
        "pools_with_oracle_final_stable_candidate": stable_pools,
        "oracle_final_near_hull_50mev_candidate_count": total_near,
        "pools": pool_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--parity-audit", type=Path, required=True)
    parser.add_argument("--soap-manifest", type=Path, required=True)
    parser.add_argument("--soap-cache", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--raw-cse-root", type=Path, required=True)
    parser.add_argument("--ppd", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for value in vars(args).values():
        if isinstance(value, Path):
            _require_external(value, repo_root)

    pool_payload = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    parity_payload = json.loads(args.parity_audit.read_text(encoding="utf-8"))
    soap_payload = json.loads(args.soap_manifest.read_text(encoding="utf-8"))
    with np.load(args.soap_cache, allow_pickle=False) as cache:
        soap_query_ids = [str(item) for item in cache["query_ids"]]
        soap_vectors = np.asarray(cache["vectors"], dtype=float)
    p1 = validate_engineering_p1(
        pool_payload=pool_payload,
        parity_payload=parity_payload,
        soap_payload=soap_payload,
        soap_query_ids=soap_query_ids,
        soap_vectors=soap_vectors,
        soap_cache_sha256=_sha256(args.soap_cache),
    )
    _repair_historical_composition_interface()
    p15 = build_oracle_support(
        pool_payload=pool_payload,
        parity_payload=parity_payload,
        raw_root=args.raw_cse_root,
        cleaned_ids=_read_cleaned_ids(args.cleaned_ids),
        ppd=_load_ppd_read_only(args.ppd),
    )
    report = {
        "schema_version": 1,
        "scope": "engineering_p1_and_frozen_pool_p1_5_no_policy_execution",
        "pool_manifest_sha256": _sha256(args.pool_manifest),
        "parity_audit_sha256": _sha256(args.parity_audit),
        "soap_manifest_sha256": _sha256(args.soap_manifest),
        "soap_cache_sha256": _sha256(args.soap_cache),
        "cleaned_ids_sha256": _sha256(args.cleaned_ids),
        "ppd_sha256": _sha256(args.ppd),
        "p1": p1,
        "p1_5": p15,
        "execution_effect": {
            "engineering_runner_smoke": bool(
                p1["engineering_p1_passed"]
                and p15["engineering_p1_5_support_present"]
            ),
            "comparative_claim_grade_matrix": False,
            "made": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"engineering_p1_passed={p1['engineering_p1_passed']}")
    print(
        "engineering_p1_5_support_present="
        f"{p15['engineering_p1_5_support_present']}"
    )
    print("comparative_claim_grade_matrix=False")
    print(f"report={args.output.resolve()}")


if __name__ == "__main__":
    main()
