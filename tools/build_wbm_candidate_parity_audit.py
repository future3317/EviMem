"""Build or merge the frozen 128-candidate WBM parity audit outside Git.

Build mode runs in either the historical ``wbm-parity`` environment or the
modern adapter environment. Merge mode combines both environment outputs with
official summary/prediction fields. This tool never executes an acquisition
policy and never changes the frozen pool manifest.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gc
import gzip
import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any

from audit_wbm_official_artifacts import _load_ppd_read_only

STEP_COUNTS = (61_848, 52_800, 79_205, 40_328, 23_308)
STRICT_TOLERANCE_EV_PER_ATOM = 1e-8
OFFICIAL_ROUNDING_TOLERANCE_EV_PER_ATOM = 5.1e-4
CROSS_ENVIRONMENT_TOLERANCE_EV_PER_ATOM = 1e-6
QUERY_ID = re.compile(r"wbm-(?P<step>[1-5])-(?P<index>[1-9][0-9]*)$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_external(path: Path, repo_root: Path) -> None:
    if path.resolve().is_relative_to(repo_root):
        raise ValueError("parity inputs and outputs must remain outside the repository")


def _pool_rows(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    pools = payload["selection"]["pools"]
    rows: dict[str, dict[str, Any]] = {}
    for exact_chemsys, pool in pools.items():
        for candidate in pool["candidates"]:
            query_id = candidate["query_id"]
            if query_id in rows:
                raise ValueError(f"duplicate frozen pool query ID: {query_id}")
            rows[query_id] = {
                "query_id": query_id,
                "exact_chemsys": exact_chemsys,
                "canonical_structure_id": (
                    "byte-identical:" + candidate["exact_structure_sha256"]
                ),
                "prototype_cluster_id": None,
                "mp_overlap_group": None,
            }
    selection = payload["selection"]
    declared_count = selection.get("selected_candidate_count")
    if declared_count is not None and len(rows) != declared_count:
        raise ValueError(
            f"frozen pool has {len(rows)} candidates, manifest declares {declared_count}"
        )
    pool_size = selection.get("pool_size")
    if pool_size is not None and len(rows) != int(pool_size) * len(pools):
        raise ValueError("fixed-size frozen pool manifest has an inconsistent row count")
    if not rows:
        raise ValueError("frozen pool manifest has no candidate rows")
    return rows


def _load_raw_entries(raw_root: Path, query_ids: set[str]) -> dict[str, dict[str, Any]]:
    by_step: dict[int, dict[int, str]] = {}
    for query_id in query_ids:
        match = QUERY_ID.fullmatch(query_id)
        if match is None:
            raise ValueError(f"invalid WBM query ID: {query_id}")
        step, index = int(match["step"]), int(match["index"])
        by_step.setdefault(step, {})[index - 1] = query_id
    result: dict[str, dict[str, Any]] = {}
    for step, indices in sorted(by_step.items()):
        with bz2.open(raw_root / f"step_{step}.json.bz2", "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("entries")
        if not isinstance(entries, list) or len(entries) != STEP_COUNTS[step - 1]:
            raise ValueError(f"unexpected WBM step-{step} entry count")
        for index, query_id in indices.items():
            raw = json.loads(json.dumps(entries[index]))
            raw["entry_id"] = query_id
            parameters = raw.get("parameters")
            if not isinstance(parameters, dict) or "is_hubbard" not in parameters:
                raise ValueError(f"missing WBM calculation parameters for {query_id}")
            parameters["run_type"] = "GGA+U" if parameters["is_hubbard"] else "GGA"
            result[query_id] = raw
        del payload, entries
        gc.collect()
    return result


def _official_summary(path: Path, query_ids: set[str]) -> dict[str, dict[str, float]]:
    selected: dict[str, dict[str, float]] = {}
    source_id = re.compile(r"step_(?P<step>[1-5])_(?P<index>[0-9]+)$")
    with path.open("rt", encoding="utf-8") as handle:
        for line in handle:
            fields = line.split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 8:
                raise ValueError(
                    "candidate parity requires the explicit-ID WBM summary"
                )
            if fields[7] == "None":
                continue
            match = source_id.fullmatch(fields[7])
            if match is None:
                raise ValueError("official WBM summary has an invalid source ID")
            query_id = f"wbm-{int(match['step'])}-{int(match['index']) + 1}"
            if query_id not in query_ids:
                continue
            if query_id in selected:
                raise ValueError(f"duplicate official summary ID: {query_id}")
            selected[query_id] = {
                "official_raw_total_energy_ev": float(fields[3]),
                "official_raw_formation_energy_ev_per_atom": float(fields[4]),
                "official_legacy_e_above_hull_ev_per_atom": float(fields[5]),
            }
    if set(selected) != query_ids:
        raise ValueError("official summary does not cover the frozen pool")
    return selected


def _official_predictions(path: Path, query_ids: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            query_id = row["material_id"]
            if query_id in query_ids:
                if query_id in result:
                    raise ValueError(f"duplicate official prediction: {query_id}")
                result[query_id] = float(row["e_form_per_atom"])
    if set(result) != query_ids:
        raise ValueError("official predictions do not cover the frozen pool")
    return result


def _repair_historical_composition_interface() -> None:
    from pymatgen.core.composition import Composition

    if not hasattr(Composition, "_natoms"):
        Composition._natoms = property(  # type: ignore[attr-defined]
            lambda self: (
                self.__dict__["_n_atoms"]
                if "_n_atoms" in self.__dict__
                else self.__dict__["_natoms"]
            ),
            lambda self, value: self.__dict__.__setitem__("_natoms", value),
        )


def build(args: argparse.Namespace) -> None:
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedEntry, ComputedStructureEntry

    pool_rows = _pool_rows(args.pool_manifest)
    query_ids = set(pool_rows)
    if args.compact_entries is not None:
        compact = json.loads(args.compact_entries.read_text(encoding="utf-8"))
        if compact.get("environment_label") != args.environment_label:
            raise ValueError("compact entry environment label differs from build")
        corrected_by_id = {
            item["query_id"]: ComputedEntry(
                item["composition"],
                item["corrected_total_energy_ev"],
                entry_id=item["query_id"],
            )
            for item in compact["entries"]
        }
        energy_metadata = {
            item["query_id"]: {
                "uncorrected_total_energy_ev": item["uncorrected_total_energy_ev"],
                "correction_ev_per_atom": item["correction_ev_per_atom"],
            }
            for item in compact["entries"]
        }
    else:
        if args.raw_cse_root is None:
            raise ValueError("build requires --raw-cse-root or --compact-entries")
        raw = _load_raw_entries(args.raw_cse_root, query_ids)
        entries = [
            ComputedStructureEntry.from_dict(raw[query_id]) for query_id in sorted(raw)
        ]
        corrected = MaterialsProject2020Compatibility().process_entries(
            entries,
            clean=True,
            verbose=False,
        )
        corrected_by_id = {
            str(entry.entry_id): ComputedEntry(
                entry.composition,
                entry.energy,
                entry_id=str(entry.entry_id),
            )
            for entry in corrected
        }
        energy_metadata = {
            str(entry.entry_id): {
                "uncorrected_total_energy_ev": float(entry.uncorrected_energy),
                "correction_ev_per_atom": float(entry.correction_per_atom),
            }
            for entry in corrected
        }
        del raw, entries, corrected
        gc.collect()
    if set(corrected_by_id) != query_ids:
        rejected = sorted(query_ids - set(corrected_by_id))
        raise ValueError(f"MP2020 rejected frozen candidates: {rejected}")
    if args.compact_only:
        compact_report = {
            "schema_version": 1,
            "scope": "environment_specific_compact_corrected_entries_no_policy_execution",
            "environment_label": args.environment_label,
            "pymatgen_version": importlib.metadata.version("pymatgen"),
            "pool_manifest_sha256": _sha256(args.pool_manifest),
            "entries": [
                {
                    "query_id": query_id,
                    "composition": entry.composition.as_dict(),
                    "corrected_total_energy_ev": float(entry.energy),
                    "uncorrected_total_energy_ev": energy_metadata[query_id][
                        "uncorrected_total_energy_ev"
                    ],
                    "correction_ev_per_atom": energy_metadata[query_id][
                        "correction_ev_per_atom"
                    ],
                }
                for query_id, entry in sorted(corrected_by_id.items())
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(compact_report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"environment={args.environment_label}")
        print(f"compact_candidate_count={len(corrected_by_id)}")
        print(f"output={args.output.resolve()}")
        return
    if args.ppd is None or args.summary is None or args.predictions is None:
        raise ValueError("full build requires PPD, summary, and predictions")
    _repair_historical_composition_interface()
    ppd = _load_ppd_read_only(args.ppd)
    official = _official_summary(args.summary, query_ids)
    predictions = _official_predictions(args.predictions, query_ids)
    rows = []
    for query_id in sorted(query_ids):
        entry = corrected_by_id[query_id]
        formation = float(ppd.get_form_energy_per_atom(entry))
        raw_formation = formation - energy_metadata[query_id]["correction_ev_per_atom"]
        e_above_hull = float(ppd.get_e_above_hull(entry, allow_negative=True))
        rows.append(
            {
                **pool_rows[query_id],
                "official_prediction_ev_per_atom": predictions[query_id],
                **official[query_id],
                "computed_corrected_total_energy_ev": float(entry.energy),
                "replayed_raw_formation_energy_ev_per_atom": raw_formation,
                "computed_correction_ev_per_atom": energy_metadata[query_id][
                    "correction_ev_per_atom"
                ],
                "computed_corrected_formation_energy_ev_per_atom": formation,
                "computed_initial_e_above_hull_ev_per_atom": e_above_hull,
                "computed_stable_label": e_above_hull <= STRICT_TOLERANCE_EV_PER_ATOM,
                "computed_phase_membership": e_above_hull <= STRICT_TOLERANCE_EV_PER_ATOM,
            }
        )
    report = {
        "schema_version": 1,
        "scope": "candidate_level_parity_only_no_policy_execution",
        "environment_label": args.environment_label,
        "pymatgen_version": importlib.metadata.version("pymatgen"),
        "pool_manifest_sha256": _sha256(args.pool_manifest),
        "ppd_sha256": _sha256(args.ppd),
        "summary_sha256": _sha256(args.summary),
        "prediction_sha256": _sha256(args.predictions),
        "strict_tolerance_ev_per_atom": STRICT_TOLERANCE_EV_PER_ATOM,
        "official_rounding_tolerance_ev_per_atom": (
            OFFICIAL_ROUNDING_TOLERANCE_EV_PER_ATOM
        ),
        "candidate_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"environment={args.environment_label}")
    print(f"candidate_count={len(rows)}")
    print(f"output={args.output.resolve()}")


def _reason(row: dict[str, Any]) -> str:
    parity_form = row["parity_corrected_formation_energy_ev_per_atom"]
    parity_raw_form = row["parity_replayed_raw_formation_energy_ev_per_atom"]
    official_raw_form = row["official_raw_formation_energy_ev_per_atom"]
    parity_hull = row["initial_e_above_hull_parity_ev_per_atom"]
    modern_form = row["modern_corrected_formation_energy_ev_per_atom"]
    modern_hull = row["initial_e_above_hull_modern_ev_per_atom"]
    reasons = ["official_compiled_corrected_artifact_unavailable"]
    if (
        abs(parity_raw_form - official_raw_form)
        > OFFICIAL_ROUNDING_TOLERANCE_EV_PER_ATOM
    ):
        reasons.append("raw_energy_or_elemental_reference_mismatch")
    if (
        abs(modern_form - parity_form) > CROSS_ENVIRONMENT_TOLERANCE_EV_PER_ATOM
        or abs(modern_hull - parity_hull) > CROSS_ENVIRONMENT_TOLERANCE_EV_PER_ATOM
    ):
        reasons.append("environment_version_difference")
    if min(abs(parity_hull), abs(modern_hull)) <= (
        OFFICIAL_ROUNDING_TOLERANCE_EV_PER_ATOM
    ):
        reasons.append("boundary_ambiguous")
    return ";".join(reasons)


def merge(args: argparse.Namespace) -> None:
    modern = json.loads(args.modern_input.read_text(encoding="utf-8"))
    parity = json.loads(args.parity_input.read_text(encoding="utf-8"))
    modern_rows = {row["query_id"]: row for row in modern["rows"]}
    parity_rows = {row["query_id"]: row for row in parity["rows"]}
    if set(modern_rows) != set(parity_rows) or not modern_rows:
        raise ValueError("modern and parity candidate sets differ")
    rows = []
    for query_id in sorted(parity_rows):
        current, historic = modern_rows[query_id], parity_rows[query_id]
        row = {
            "query_id": query_id,
            "exact_chemsys": historic["exact_chemsys"],
            "canonical_structure_id": historic["canonical_structure_id"],
            "prototype_cluster_id": historic["prototype_cluster_id"],
            "mp_overlap_group": historic["mp_overlap_group"],
            "official_prediction_ev_per_atom": historic[
                "official_prediction_ev_per_atom"
            ],
            "modern_corrected_formation_energy_ev_per_atom": current[
                "computed_corrected_formation_energy_ev_per_atom"
            ],
            "parity_corrected_formation_energy_ev_per_atom": historic[
                "computed_corrected_formation_energy_ev_per_atom"
            ],
            "official_corrected_formation_energy_ev_per_atom": None,
            "official_raw_formation_energy_ev_per_atom": historic[
                "official_raw_formation_energy_ev_per_atom"
            ],
            "parity_replayed_raw_formation_energy_ev_per_atom": historic[
                "replayed_raw_formation_energy_ev_per_atom"
            ],
            "initial_e_above_hull_modern_ev_per_atom": current[
                "computed_initial_e_above_hull_ev_per_atom"
            ],
            "initial_e_above_hull_parity_ev_per_atom": historic[
                "computed_initial_e_above_hull_ev_per_atom"
            ],
            "initial_e_above_hull_official_ev_per_atom": None,
            "official_legacy_e_above_hull_ev_per_atom": historic[
                "official_legacy_e_above_hull_ev_per_atom"
            ],
            "stable_label_modern": current["computed_stable_label"],
            "stable_label_parity": historic["computed_stable_label"],
            "stable_label_official": None,
            "phase_membership_modern": current["computed_phase_membership"],
            "phase_membership_parity": historic["computed_phase_membership"],
            "phase_membership_official": None,
        }
        row["difference_reason"] = _reason(row)
        rows.append(row)
    report = {
        "schema_version": 1,
        "scope": "candidate_level_parity_only_no_policy_execution",
        "formal_p1_passed": False,
        "formal_p1_blockers": [
            "human license manifest remains pending",
            "official compiled corrected candidate-level summary is unavailable",
            "prototype clustering and MP overlap audit remain pending",
            "claim-grade canonical structure identity remains pending",
        ],
        "modern_environment": modern["pymatgen_version"],
        "parity_environment": parity["pymatgen_version"],
        "strict_tolerance_ev_per_atom": STRICT_TOLERANCE_EV_PER_ATOM,
        "cross_environment_tolerance_ev_per_atom": (
            CROSS_ENVIRONMENT_TOLERANCE_EV_PER_ATOM
        ),
        "official_rounding_tolerance_ev_per_atom": (
            OFFICIAL_ROUNDING_TOLERANCE_EV_PER_ATOM
        ),
        "candidate_count": len(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"candidate_count={len(rows)}")
    print("formal_p1_passed=False")
    print(f"output={args.output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--environment-label", required=True)
    build_parser.add_argument("--pool-manifest", type=Path, required=True)
    build_parser.add_argument("--raw-cse-root", type=Path)
    build_parser.add_argument("--compact-entries", type=Path)
    build_parser.add_argument("--compact-only", action="store_true")
    build_parser.add_argument("--summary", type=Path)
    build_parser.add_argument("--ppd", type=Path)
    build_parser.add_argument("--predictions", type=Path)
    build_parser.add_argument("--output", type=Path, required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--modern-input", type=Path, required=True)
    merge_parser.add_argument("--parity-input", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    paths = [value for value in vars(args).values() if isinstance(value, Path)]
    for path in paths:
        _require_external(path, repo_root)
    if args.command == "build":
        build(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
