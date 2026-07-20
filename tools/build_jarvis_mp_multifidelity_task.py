"""Build an oracle-isolated JARVIS--MP multi-fidelity task outside Git.

The low-fidelity observable is a JARVIS OptB88vdW formation energy and relaxed
structure.  The high-fidelity target is a frozen Materials Project pure-GGA
entry under MP2020 corrections. Database references are only candidate joins:
every retained pair must also pass an explicit pymatgen StructureMatcher gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ijson
import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.entries.computed_entries import ComputedEntry

JARVIS_JSONL_SHA256 = "4aab73e44140282757a1c083e6791f37080b3a4d4ed0bad2dc93ec2b8b2bd9c6"
JARVIS_ZIP_SHA256 = "d4c64660e9e1fa45c82bd8868a96ec10162195eed69636445972c05550d8d0d6"
MP_CSE_SHA256 = "553d6272f049a8f4ec26e503b89751e2616dd3af53d086545f6ea00f317a361f"
MP_ID = re.compile(r"mp-\d+")
RELEASE_ID = "jarvis-dft-3d-2022-12-12__mp-cse-2023-02-07__v1"


@dataclass(frozen=True)
class JarvisMeta:
    jid: str
    mp_id: str
    formula: str
    formation_energy_ev_per_atom: float


@dataclass(frozen=True)
class TargetMeta:
    key: str
    mp_id: str
    entry_id: str
    chemical_system: str
    composition: dict[str, float]
    corrected_total_energy_ev: float
    run_type: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _id_checksum(values: list[str] | tuple[str, ...] | set[str]) -> str:
    payload = "\n".join(sorted(set(values))) + "\n"
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _require_external(path: Path, repo_root: Path) -> None:
    if path.resolve().is_relative_to(repo_root):
        raise ValueError("datasets and task outputs must remain outside the repository")


def _float_mapping(values: dict[str, Any]) -> dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _chemical_system(composition: Composition) -> str:
    return "-".join(sorted(element.symbol for element in composition.elements))


def _stratum(chemical_system: str) -> str | None:
    size = len(chemical_system.split("-"))
    if size == 2:
        return "binary"
    if size == 3:
        return "ternary"
    if size >= 4:
        return "quaternary_plus"
    return None


def _jarvis_structure(record: dict[str, Any]) -> Structure:
    atoms = record.get("atoms")
    if not isinstance(atoms, dict):
        raise ValueError("JARVIS record lacks atoms")
    elements, coordinates = atoms.get("elements"), atoms.get("coords")
    lattice = atoms.get("lattice_mat")
    if not isinstance(elements, list) or not isinstance(coordinates, list):
        raise ValueError("JARVIS atoms require elements and coordinates")
    if len(elements) != len(coordinates) or not elements:
        raise ValueError("JARVIS atom arrays have inconsistent lengths")
    return Structure(
        Lattice(np.asarray(lattice, dtype=float)),
        elements,
        np.asarray(coordinates, dtype=float),
        coords_are_cartesian=bool(atoms.get("cartesian", False)),
    )


def _structure_hash(record: dict[str, Any]) -> str:
    encoded = json.dumps(
        record["atoms"], sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _descriptor(structure: Structure) -> tuple[float, ...]:
    """Outcome-free composition and low-fidelity geometry descriptor."""

    composition = structure.composition.fractional_composition
    fractions = [float(composition[Element.from_Z(z)]) for z in range(1, 119)]
    lengths = np.asarray(structure.lattice.abc, dtype=float)
    angles = np.asarray(structure.lattice.angles, dtype=float) / 180.0
    geometry = [
        math.log1p(float(structure.volume / len(structure))),
        math.log1p(float(structure.density)),
        float(lengths.max() / lengths.min()),
        *angles.tolist(),
    ]
    values = tuple(fractions + geometry)
    if len(values) != 124 or not all(math.isfinite(value) for value in values):
        raise ValueError("invalid multi-fidelity structure descriptor")
    return values


def _read_jarvis_metadata(path: Path) -> tuple[dict[str, list[JarvisMeta]], dict[str, Any]]:
    by_mp: dict[str, list[JarvisMeta]] = defaultdict(list)
    jid_counts: Counter[str] = Counter()
    functionals: Counter[str] = Counter()
    row_count = 0
    for line in path.open("r", encoding="utf-8"):
        row = json.loads(line)
        row_count += 1
        jid = str(row.get("jid", "")).strip()
        reference = str(row.get("reference", "")).strip()
        functional = str(row.get("func", "")).strip()
        energy = float(row.get("formation_energy_peratom"))
        if not jid or not functional or not math.isfinite(energy):
            raise ValueError("JARVIS row has invalid identity, protocol, or energy")
        jid_counts[jid] += 1
        functionals[functional] += 1
        if MP_ID.fullmatch(reference):
            by_mp[reference].append(
                JarvisMeta(
                    jid=jid,
                    mp_id=reference,
                    formula=str(row["formula"]),
                    formation_energy_ev_per_atom=energy,
                )
            )
    if any(count != 1 for count in jid_counts.values()):
        raise ValueError("JARVIS JIDs are not unique")
    if set(functionals) != {"OptB88vdW"}:
        raise ValueError(f"JARVIS source is not a single frozen protocol: {functionals}")
    return by_mp, {
        "row_count": row_count,
        "unique_jid_count": len(jid_counts),
        "mp_reference_row_count": sum(len(rows) for rows in by_mp.values()),
        "unique_mp_reference_count": len(by_mp),
        "duplicate_mp_reference_rows": sum(len(rows) - 1 for rows in by_mp.values()),
        "functional_counts": dict(functionals),
    }


def _mp_material_join(cse_path: Path, mp_ids: set[str]) -> tuple[dict[str, str], dict[str, Any]]:
    joined: dict[str, str] = {}
    total = 0
    with gzip.open(cse_path, "rb") as handle:
        for key, material_id in ijson.kvitems(handle, "material_id"):
            total += 1
            material_id = str(material_id)
            if material_id in mp_ids:
                joined[str(key)] = material_id
    return joined, {
        "material_id_mapping_count": total,
        "joined_mapping_count": len(joined),
        "joined_unique_mp_id_count": len(set(joined.values())),
        "unmatched_jarvis_mp_id_count": len(mp_ids - set(joined.values())),
    }


def _target_metadata(
    cse_path: Path,
    key_to_mp: dict[str, str],
    jarvis_by_mp: dict[str, list[JarvisMeta]],
) -> tuple[dict[str, TargetMeta], dict[str, Any]]:
    targets: dict[str, TargetMeta] = {}
    run_types: Counter[str] = Counter()
    composition_mismatch = 0
    with gzip.open(cse_path, "rb") as handle:
        for key, row in ijson.kvitems(handle, "entry"):
            key = str(key)
            mp_id = key_to_mp.get(key)
            if mp_id is None:
                continue
            parameters = row.get("parameters") or {}
            run_type = str(
                parameters.get("run_type")
                or (row.get("data") or {}).get("run_type")
                or ""
            )
            run_types[run_type] += 1
            composition_dict = _float_mapping(row["composition"])
            composition = Composition(composition_dict)
            source_formulas = {
                Composition(item.formula).reduced_formula for item in jarvis_by_mp[mp_id]
            }
            if composition.reduced_formula not in source_formulas:
                composition_mismatch += 1
                continue
            if run_type != "GGA":
                continue
            corrected_energy = float(row["energy"]) + float(row.get("correction", 0.0))
            if not math.isfinite(corrected_energy):
                raise ValueError("MP CSE contains a non-finite corrected energy")
            targets[key] = TargetMeta(
                key=key,
                mp_id=mp_id,
                entry_id=str(row["entry_id"]),
                chemical_system=_chemical_system(composition),
                composition=composition_dict,
                corrected_total_energy_ev=corrected_energy,
                run_type=run_type,
            )
    system_counts = Counter(item.chemical_system for item in targets.values())
    return targets, {
        "joined_run_type_counts": dict(run_types),
        "composition_mismatch_count": composition_mismatch,
        "pure_gga_composition_matched_count": len(targets),
        "exact_system_count": len(system_counts),
        "systems_ge_16": sum(count >= 16 for count in system_counts.values()),
    }


def _load_selected_jarvis_rows(path: Path, mp_ids: set[str]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for line in path.open("r", encoding="utf-8"):
        row = json.loads(line)
        reference = str(row.get("reference", "")).strip()
        if reference in mp_ids:
            rows[reference].append(row)
    return rows


def _load_selected_mp_structures(
    cse_path: Path, target_keys: set[str]
) -> dict[str, Structure]:
    structures: dict[str, Structure] = {}
    with gzip.open(cse_path, "rb") as handle:
        for key, row in ijson.kvitems(handle, "entry"):
            key = str(key)
            if key in target_keys:
                structure_dict = json.loads(
                    json.dumps(row["structure"], default=float)
                )
                structures[key] = Structure.from_dict(structure_dict)
    if set(structures) != target_keys:
        raise ValueError("selected MP structures are incomplete")
    return structures


def _match_pairs(
    targets: dict[str, TargetMeta],
    target_structures: dict[str, Structure],
    jarvis_rows: dict[str, list[dict[str, Any]]],
    matcher: StructureMatcher,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matched: list[dict[str, Any]] = []
    no_structure_match = 0
    source_composition_mismatch = 0
    multiple_candidate_ids = 0
    for key in sorted(targets):
        target = targets[key]
        target_structure = target_structures[key]
        candidates = jarvis_rows[target.mp_id]
        if len(candidates) > 1:
            multiple_candidate_ids += 1
        ordered = sorted(
            candidates,
            key=lambda row: _stable_hash(RELEASE_ID, "source-candidate", target.mp_id, row["jid"]),
        )
        chosen: tuple[dict[str, Any], Structure, tuple[float, float] | None] | None = None
        for row in ordered:
            source_structure = _jarvis_structure(row)
            if (
                source_structure.composition.reduced_formula
                != target_structure.composition.reduced_formula
            ):
                source_composition_mismatch += 1
                continue
            rms = matcher.get_rms_dist(target_structure, source_structure)
            if rms is not None:
                chosen = (row, source_structure, rms)
                break
        if chosen is None:
            no_structure_match += 1
            continue
        source_row, source_structure, rms = chosen
        pair_id = f"{target.mp_id}__{source_row['jid']}"
        canonical_id = "jarvis-mp-structure:" + _stable_hash(RELEASE_ID, pair_id)
        matched.append(
            {
                "pair_id": pair_id,
                "canonical_structure_id": canonical_id,
                "mp_material_id": target.mp_id,
                "mp_entry_id": target.entry_id,
                "mp_entry_key": target.key,
                "jarvis_id": str(source_row["jid"]),
                "chemical_system": target.chemical_system,
                "composition": target.composition,
                "source_formula": str(source_row["formula"]),
                "source_formation_energy_ev_per_atom": float(
                    source_row["formation_energy_peratom"]
                ),
                "source_structure_sha256": _structure_hash(source_row),
                "source_descriptor": _descriptor(source_structure),
                "_source_structure": source_structure.as_dict(),
                "structure_match_rms": float(rms[0]) if rms is not None else None,
                "structure_match_max_distance": float(rms[1]) if rms is not None else None,
                "target_composition": target.composition,
                "target_corrected_total_energy_ev": target.corrected_total_energy_ev,
            }
        )
    return matched, {
        "matched_pair_count": len(matched),
        "no_structure_match_count": no_structure_match,
        "source_composition_mismatch_attempt_count": source_composition_mismatch,
        "mp_ids_with_multiple_jarvis_candidates": multiple_candidate_ids,
    }


CHGNET_MODEL_NAME = "0.3.0"
CHGNET_MODEL_SHA256 = "d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"


def _attach_source_embeddings(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach frozen CHGNet crystal features from policy-visible source structures."""

    import chgnet
    import chgnet.model.model as chgnet_model_module
    from chgnet.model.model import CHGNet

    checkpoint = (
        Path(chgnet_model_module.module_dir)
        / "../pretrained/0.3.0/chgnet_0.3.0_e29f68s314m37.pth.tar"
    ).resolve()
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != CHGNET_MODEL_SHA256:
        raise ValueError("frozen CHGNet representation checkpoint checksum mismatch")
    structures = [Structure.from_dict(row["_source_structure"]) for row in rows]
    model = CHGNet.load(model_name=CHGNET_MODEL_NAME, use_device="cpu", verbose=False)
    predictions = model.predict_structure(
        structures,
        task="e",
        return_crystal_feas=True,
        batch_size=16,
    )
    if not isinstance(predictions, list) or len(predictions) != len(rows):
        raise RuntimeError("CHGNet representation batch returned an unexpected shape")
    dimensions: set[int] = set()
    for row, prediction in zip(rows, predictions, strict=True):
        embedding = tuple(float(value) for value in prediction["crystal_fea"])
        if not embedding or not all(math.isfinite(value) for value in embedding):
            raise ValueError("CHGNet source embedding contains invalid values")
        dimensions.add(len(embedding))
        row["source_environment_embedding"] = embedding
    if len(dimensions) != 1:
        raise ValueError("CHGNet source embedding dimension is not fixed")
    return {
        "encoder": "CHGNet frozen crystal_fea",
        "model_name": CHGNET_MODEL_NAME,
        "package_version": chgnet.__version__,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "dimension": dimensions.pop(),
        "device": "cpu",
        "structure_source": "policy-visible JARVIS low-fidelity relaxed structure",
        "target_structure_used": False,
        "outcomes_used": False,
    }


def _choose_systems(
    system_counts: Counter[str],
    *,
    calibration_per_stratum: dict[str, int],
    evaluation_per_stratum: dict[str, int],
    excluded_systems: set[str] | None = None,
    selection_salt: str | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    excluded = excluded_systems or set()
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for system in system_counts:
        if system in excluded:
            continue
        stratum = _stratum(system)
        if stratum is not None:
            by_stratum[stratum].append(system)
    calibration: dict[str, list[str]] = {}
    evaluation: dict[str, list[str]] = {}
    for stratum in ("binary", "ternary", "quaternary_plus"):
        ordered = sorted(by_stratum[stratum], key=lambda system: (
            _stable_hash(RELEASE_ID, "system", system)
            if selection_salt is None
            else _stable_hash(RELEASE_ID, selection_salt, "system", system)
        ))
        n_cal = calibration_per_stratum[stratum]
        n_eval = evaluation_per_stratum[stratum]
        if len(ordered) < n_cal + n_eval:
            raise ValueError(
                f"insufficient {stratum} systems after structure matching: "
                f"available={len(ordered)}, required={n_cal + n_eval}"
            )
        calibration[stratum] = ordered[:n_cal]
        evaluation[stratum] = ordered[n_cal : n_cal + n_eval]
    return calibration, evaluation


def _partition_all_eligible_systems(
    system_counts: Counter[str],
    *,
    excluded_systems: set[str],
    selection_salt: str,
    evaluation_stride: int,
    evaluation_offset: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Use every eligible fresh system with a deterministic hash-fold split."""

    if evaluation_stride < 2 or not 0 <= evaluation_offset < evaluation_stride:
        raise ValueError("evaluation hash fold requires stride >=2 and a valid offset")
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for system in system_counts:
        if system in excluded_systems:
            continue
        if (stratum := _stratum(system)) is not None:
            by_stratum[stratum].append(system)
    calibration: dict[str, list[str]] = {}
    evaluation: dict[str, list[str]] = {}
    for stratum in ("binary", "ternary", "quaternary_plus"):
        ordered = sorted(
            by_stratum[stratum],
            key=lambda system: _stable_hash(
                RELEASE_ID, selection_salt, "system", system
            ),
        )
        evaluation[stratum] = [
            system
            for index, system in enumerate(ordered)
            if index % evaluation_stride == evaluation_offset
        ]
        calibration[stratum] = [
            system
            for index, system in enumerate(ordered)
            if index % evaluation_stride != evaluation_offset
        ]
    if not any(evaluation.values()) or not any(calibration.values()):
        raise ValueError("fresh hash-fold split produced an empty task partition")
    return calibration, evaluation


def _select_pairs(
    matched: list[dict[str, Any]],
    systems: dict[str, list[str]],
    *,
    max_pairs_per_system: int,
) -> list[dict[str, Any]]:
    wanted = {system for values in systems.values() for system in values}
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in matched:
        if row["chemical_system"] in wanted:
            by_system[row["chemical_system"]].append(row)
    selected: list[dict[str, Any]] = []
    for system in sorted(wanted):
        ordered = sorted(
            by_system[system],
            key=lambda row: _stable_hash(RELEASE_ID, "pair", row["pair_id"]),
        )
        selected.extend(ordered[:max_pairs_per_system])
    return selected


def _phase_entries_for_systems(
    cse_path: Path,
    systems: set[str],
    excluded_entry_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    element_sets = {system: frozenset(system.split("-")) for system in systems}
    result: dict[str, list[dict[str, Any]]] = {system: [] for system in systems}
    with gzip.open(cse_path, "rb") as handle:
        for _, row in ijson.kvitems(handle, "entry"):
            entry_id = str(row["entry_id"])
            if entry_id in excluded_entry_ids:
                continue
            composition = Composition(_float_mapping(row["composition"]))
            elements = frozenset(element.symbol for element in composition.elements)
            supported = [
                system for system, allowed in element_sets.items() if elements <= allowed
            ]
            if not supported:
                continue
            energy = float(row["energy"]) + float(row.get("correction", 0.0))
            compact = {
                "entry_id": entry_id,
                "composition": composition.as_dict(),
                "corrected_total_energy_ev": energy,
            }
            for system in supported:
                result[system].append(compact)
    return result


def _attach_hull_oracles(
    selected: list[dict[str, Any]],
    phase_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagrams: dict[str, PhaseDiagram] = {}
    for system, rows in phase_rows.items():
        entries = [
            ComputedEntry(
                row["composition"],
                row["corrected_total_energy_ev"],
                entry_id=row["entry_id"],
            )
            for row in rows
        ]
        diagrams[system] = PhaseDiagram(entries)
    oracle_rows: list[dict[str, Any]] = []
    for row in selected:
        entry = ComputedEntry(
            row["target_composition"],
            row["target_corrected_total_energy_ev"],
            entry_id=row["mp_entry_id"],
        )
        diagram = diagrams[row["chemical_system"]]
        oracle_rows.append(
            {
                "pair_id": row["pair_id"],
                "mp_material_id": row["mp_material_id"],
                "mp_entry_id": row["mp_entry_id"],
                "chemical_system": row["chemical_system"],
                "composition": row["target_composition"],
                "target_corrected_total_energy_ev": row[
                    "target_corrected_total_energy_ev"
                ],
                "target_formation_energy_ev_per_atom": float(
                    diagram.get_form_energy_per_atom(entry)
                ),
                "initial_e_above_hull_ev_per_atom": float(
                    diagram.get_e_above_hull(entry, allow_negative=True)
                ),
            }
        )
    return oracle_rows, {
        "system_phase_counts": {
            system: len(rows) for system, rows in sorted(phase_rows.items())
        },
        "initial_phase_entry_count_with_system_duplicates": sum(
            len(rows) for rows in phase_rows.values()
        ),
    }


def _public_pair(row: dict[str, Any], split: str) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if not key.startswith("target_") and key not in {"mp_entry_key"}
    } | {"split": split}


def build(args: argparse.Namespace) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in (args.jarvis_jsonl, args.jarvis_zip, args.mp_cse, args.output_dir):
        _require_external(path, repo_root)
    expected_hashes = {
        args.jarvis_jsonl: JARVIS_JSONL_SHA256,
        args.jarvis_zip: JARVIS_ZIP_SHA256,
        args.mp_cse: MP_CSE_SHA256,
    }
    for path, expected in expected_hashes.items():
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"frozen artifact checksum mismatch: {path}")
    output_files = [
        args.output_dir / "task-manifest.json",
        args.output_dir / "oracle-vault.json",
        args.output_dir / "build-audit.json",
    ]
    if any(path.exists() for path in output_files):
        raise FileExistsError("task output already exists; never overwrite a frozen build")

    jarvis_by_mp, jarvis_audit = _read_jarvis_metadata(args.jarvis_jsonl)
    key_to_mp, join_audit = _mp_material_join(args.mp_cse, set(jarvis_by_mp))
    targets, target_audit = _target_metadata(args.mp_cse, key_to_mp, jarvis_by_mp)
    pre_counts = Counter(item.chemical_system for item in targets.values())
    eligible_systems = {
        system for system, count in pre_counts.items() if count >= args.min_system_size
    }
    eligible_targets = {
        key: item for key, item in targets.items() if item.chemical_system in eligible_systems
    }
    eligible_mp_ids = {item.mp_id for item in eligible_targets.values()}
    jarvis_rows = _load_selected_jarvis_rows(args.jarvis_jsonl, eligible_mp_ids)
    target_structures = _load_selected_mp_structures(
        args.mp_cse, set(eligible_targets)
    )
    matcher = StructureMatcher(
        ltol=args.ltol,
        stol=args.stol,
        angle_tol=args.angle_tol,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
    )
    matched, match_audit = _match_pairs(
        eligible_targets, target_structures, jarvis_rows, matcher
    )
    matched_counts = Counter(row["chemical_system"] for row in matched)
    matched_counts = Counter(
        {system: count for system, count in matched_counts.items() if count >= args.min_system_size}
    )
    excluded_systems: set[str] = set()
    if args.excluded_systems_file is not None:
        exclusion_payload = json.loads(
            args.excluded_systems_file.read_text(encoding="utf-8")
        )
        excluded_systems = {
            str(system).strip()
            for system in exclusion_payload["excluded_exact_systems"]
        }
        if not all(excluded_systems):
            raise ValueError("excluded exact systems must be non-empty")
    calibration_per_stratum = {
        "binary": args.calibration_binary,
        "ternary": args.calibration_ternary,
        "quaternary_plus": args.calibration_quaternary_plus,
    }
    evaluation_per_stratum = {
        "binary": args.evaluation_binary,
        "ternary": args.evaluation_ternary,
        "quaternary_plus": args.evaluation_quaternary_plus,
    }
    if args.use_all_eligible_systems:
        if args.selection_salt is None:
            raise ValueError("all-system hash partition requires a selection salt")
        calibration_systems, evaluation_systems = _partition_all_eligible_systems(
            matched_counts,
            excluded_systems=excluded_systems,
            selection_salt=args.selection_salt,
            evaluation_stride=args.evaluation_stride,
            evaluation_offset=args.evaluation_offset,
        )
    else:
        calibration_systems, evaluation_systems = _choose_systems(
            matched_counts,
            calibration_per_stratum=calibration_per_stratum,
            evaluation_per_stratum=evaluation_per_stratum,
            excluded_systems=excluded_systems,
            selection_salt=args.selection_salt,
        )
    calibration_rows = _select_pairs(
        matched,
        calibration_systems,
        max_pairs_per_system=args.max_pairs_per_system,
    )
    evaluation_rows = _select_pairs(
        matched,
        evaluation_systems,
        max_pairs_per_system=args.max_pairs_per_system,
    )
    selected = calibration_rows + evaluation_rows
    if len({row["pair_id"] for row in selected}) != len(selected):
        raise ValueError("selected pair IDs are not unique")
    representation_audit = _attach_source_embeddings(selected)
    representation_manifest = {
        key: value
        for key, value in representation_audit.items()
        if key != "checkpoint_path"
    }
    calibration_system_set = {
        system for values in calibration_systems.values() for system in values
    }
    eval_system_set = {
        system for values in evaluation_systems.values() for system in values
    }
    excluded_entry_ids = {row["mp_entry_id"] for row in selected}
    phase_rows = _phase_entries_for_systems(
        args.mp_cse,
        calibration_system_set | eval_system_set,
        excluded_entry_ids,
    )
    oracle_rows, phase_audit = _attach_hull_oracles(selected, phase_rows)
    split_by_pair = {
        **{row["pair_id"]: "calibration" for row in calibration_rows},
        **{row["pair_id"]: "evaluation" for row in evaluation_rows},
    }
    oracle_rows = [row | {"split": split_by_pair[row["pair_id"]]} for row in oracle_rows]

    task = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "scope": "real_same-structure_multi-protocol_multi-fidelity_pilot",
        "policy_visible_target_outcomes": False,
        "source_protocol": {
            "functional": "OptB88vdW",
            "hubbard_u_ev": {},
            "pseudopotential_set": "JARVIS-DFT-PAW-PBE",
            "correction_scheme": "JARVIS-formation-energy-reference-v2022-12-12",
            "relaxation_protocol": "JARVIS-DFT-OptB88vdW-relaxation",
            "calculation_code": "VASP-JARVIS-DFT",
        },
        "target_protocol": {
            "functional": "PBE-GGA",
            "hubbard_u_ev": {},
            "pseudopotential_set": "MP-VASP-PAW-PBE-frozen-2023-02-07",
            "correction_scheme": "MaterialsProject2020Compatibility",
            "relaxation_protocol": "MaterialsProject-GGA-relaxation",
            "calculation_code": "VASP-Materials-Project",
        },
        "structure_matcher": {
            "ltol": args.ltol,
            "stol": args.stol,
            "angle_tol": args.angle_tol,
            "primitive_cell": True,
            "scale": True,
            "attempt_supercell": False,
        },
        "selection": {
            "minimum_matched_pairs_per_exact_system": args.min_system_size,
            "maximum_pairs_per_system": args.max_pairs_per_system,
            "system_order": "SHA256(release_id|system|exact_chemical_system)",
            "pair_order": "SHA256(release_id|pair|pair_id)",
            "outcome_independent": True,
            "selection_salt": args.selection_salt,
            "use_all_eligible_systems": args.use_all_eligible_systems,
            "evaluation_stride": (
                args.evaluation_stride if args.use_all_eligible_systems else None
            ),
            "evaluation_offset": (
                args.evaluation_offset if args.use_all_eligible_systems else None
            ),
            "excluded_exact_systems": sorted(excluded_systems),
            "calibration_systems": calibration_systems,
            "evaluation_systems": evaluation_systems,
        },
        "descriptor": {
            "source": "policy-visible JARVIS low-fidelity relaxed structure",
            "dimension": 124,
            "fields": "118 elemental fractions, log1p(volume/atom), log1p(density), lattice anisotropy, three normalized angles",
        },
        "environment_representation": representation_manifest,
        "calibration_pairs": [
            _public_pair(row, "calibration") for row in calibration_rows
        ],
        "evaluation_pairs": [
            _public_pair(row, "evaluation") for row in evaluation_rows
        ],
        "evaluation_initial_phase_entries": {
            system: phase_rows[system] for system in sorted(eval_system_set)
        },
        "calibration_initial_phase_entries": {
            system: phase_rows[system] for system in sorted(calibration_system_set)
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    task_path, vault_path, audit_path = output_files
    task_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    task_sha = _sha256(task_path)
    vault = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "task_manifest_sha256": task_sha,
        "access_contract": {
            "calibration": "available only while fitting frozen base and transport models",
            "evaluation": "available only to reveal boundary and evaluator",
        },
        "target_outcomes": oracle_rows,
    }
    vault_path.write_text(
        json.dumps(vault, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    vault_sha = _sha256(vault_path)
    audit = {
        "schema_version": 1,
        "scope": "data_provenance_join_structure_match_and_task_freeze_no_policy_result",
        "artifacts": {
            "jarvis_jsonl": {
                "path": str(args.jarvis_jsonl.resolve()),
                "sha256": JARVIS_JSONL_SHA256,
            },
            "jarvis_official_zip": {
                "path": str(args.jarvis_zip.resolve()),
                "sha256": JARVIS_ZIP_SHA256,
                "source": "JARVIS-DFT Figshare file 38521619",
                "license": "CC BY 4.0",
            },
            "mp_cse": {
                "path": str(args.mp_cse.resolve()),
                "sha256": MP_CSE_SHA256,
                "source": "Matbench Discovery registry Figshare file 40344436",
                "use_status": "local research use; redistribution remains subject to MP terms",
            },
        },
        "jarvis": jarvis_audit,
        "mp_join": join_audit,
        "target_filter": target_audit,
        "structure_matching": match_audit,
        "environment_representation": representation_audit,
        "matched_exact_systems_ge_minimum": len(matched_counts),
        "excluded_exact_system_count": len(excluded_systems),
        "matched_system_count_by_stratum": dict(
            Counter(
                stratum
                for system in matched_counts
                if (stratum := _stratum(system)) is not None
            )
        ),
        "selected_calibration_pair_count": len(calibration_rows),
        "selected_evaluation_pair_count": len(evaluation_rows),
        "selected_pair_id_checksum": _id_checksum(
            [row["pair_id"] for row in selected]
        ),
        "calibration_evaluation_system_overlap": sorted(
            calibration_system_set & eval_system_set
        ),
        "phase_diagrams": phase_audit,
        "task_manifest_sha256": task_sha,
        "oracle_vault_sha256": vault_sha,
        "technical_gate_passed": True,
        "paper_claim_authorized": False,
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"technical_gate_passed={audit['technical_gate_passed']}")
    print(f"matched_pair_count={len(matched)}")
    print(f"calibration_pairs={len(calibration_rows)}")
    print(f"evaluation_pairs={len(evaluation_rows)}")
    print(f"task_manifest={task_path.resolve()}")
    print(f"oracle_vault={vault_path.resolve()}")
    print(f"audit={audit_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jarvis-jsonl", type=Path, required=True)
    parser.add_argument("--jarvis-zip", type=Path, required=True)
    parser.add_argument("--mp-cse", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-system-size", type=int, default=16)
    parser.add_argument("--max-pairs-per-system", type=int, default=24)
    parser.add_argument("--calibration-binary", type=int, default=4)
    parser.add_argument("--calibration-ternary", type=int, default=4)
    parser.add_argument("--calibration-quaternary-plus", type=int, default=2)
    parser.add_argument("--evaluation-binary", type=int, default=4)
    parser.add_argument("--evaluation-ternary", type=int, default=4)
    parser.add_argument("--evaluation-quaternary-plus", type=int, default=2)
    parser.add_argument("--ltol", type=float, default=0.2)
    parser.add_argument("--stol", type=float, default=0.3)
    parser.add_argument("--angle-tol", type=float, default=5.0)
    parser.add_argument("--excluded-systems-file", type=Path)
    parser.add_argument("--selection-salt")
    parser.add_argument("--use-all-eligible-systems", action="store_true")
    parser.add_argument("--evaluation-stride", type=int, default=4)
    parser.add_argument("--evaluation-offset", type=int, default=0)
    args = parser.parse_args()
    if args.min_system_size < 4 or args.max_pairs_per_system < args.min_system_size:
        raise ValueError("task sizes require max_pairs_per_system >= min_system_size >= 4")
    build(args)


if __name__ == "__main__":
    main()
