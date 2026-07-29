"""Build an oracle-isolated MAD-1.5 PBE-to-r2SCAN task.

The first MAD task uses the dataset's atomization energy as the energy scale.
It is deliberately called an atomization-hull proxy: MAD-1.5 does not provide
MatPES formation energies, and total energy must not be silently relabeled as
formation energy.  Atomization energies are relative to the isolated-atom
reference, so the initial reference entries are zero for each element.

Raw files, task outputs and the oracle vault must remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Element

ELEMENT_FRACTION_ORDER = tuple(
    Element.from_Z(atomic_number).symbol for atomic_number in range(1, 119)
)
R2SCAN_FILES = {
    "train": "mad-1.5-r2scan-train.xyz",
    "val": "mad-1.5-r2scan-val.xyz",
    "test": "mad-1.5-r2scan-test.xyz",
    "llpr_rejected": "mad-1.5-r2scan-llpr-rejected.xyz",
}
BULK_SUBSETS = {
    "mc3d",
    "mc3d_rattled",
    "mc3d_random",
    "mc3d_random_ext",
    "mc3d_random_extended",
}
HEADER_RE = {
    "frame_id": re.compile(r"(?:^|\s)frame_id=(\d+)(?:\s|$)"),
    "subset": re.compile(r"(?:^|\s)subset=([^\s]+)(?:\s|$)"),
    "pbc": re.compile(r'(?:^|\s)pbc="([^"]+)"'),
    "lattice": re.compile(r'(?:^|\s)Lattice="([^"]+)"'),
    "atomization": re.compile(r"(?:^|\s)atomization_energy=([^\s]+)(?:\s|$)"),
    "stress": re.compile(r'(?:^|\s)stress="([^"]+)"'),
}


@dataclass(frozen=True, slots=True)
class MADRecord:
    subset: str
    frame_id: int
    split: str
    atom_count: int
    species: tuple[str, ...]
    composition: dict[str, float]
    pbc: tuple[str, ...]
    lattice: tuple[float, ...]
    structure_hash: str
    atomization_energy: float
    descriptor: tuple[float, ...] | None

    @property
    def key(self) -> tuple[str, int]:
        return self.subset, self.frame_id

    @property
    def chemical_system(self) -> str:
        return "-".join(sorted(self.composition))

    @property
    def atomization_energy_per_atom(self) -> float:
        return self.atomization_energy / self.atom_count

    @property
    def bulk_3d(self) -> bool:
        return (
            self.subset in BULK_SUBSETS and self.pbc == ("T", "T", "T") and len(self.lattice) == 9
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def _float_bytes(values: tuple[float, ...]) -> bytes:
    return b"".join(struct.pack("<d", value) for value in values)


def _geometry_hash(
    species: tuple[str, ...], positions: tuple[float, ...], lattice: tuple[float, ...]
) -> str:
    digest = hashlib.sha256()
    digest.update("\0".join(species).encode())
    digest.update(_float_bytes(positions))
    digest.update(_float_bytes(lattice))
    return digest.hexdigest()


def _descriptor(
    *,
    atom_count: int,
    species: tuple[str, ...],
    lattice: tuple[float, ...],
    positions: tuple[float, ...],
    forces: tuple[float, ...],
    stress: tuple[float, ...],
    atomization_energy: float,
) -> tuple[float, ...]:
    del positions  # Geometry identity is hashed; the descriptor stays compact.
    lattice_matrix = np.asarray(lattice, dtype=float).reshape(3, 3)
    singular = np.linalg.svd(lattice_matrix, compute_uv=False)
    volume = abs(float(np.linalg.det(lattice_matrix)))
    force_array = np.asarray(forces, dtype=float).reshape(atom_count, 3)
    force_norms = np.linalg.norm(force_array, axis=1)
    stress_norm = float(np.linalg.norm(np.asarray(stress, dtype=float)))
    counts = Counter(species)
    fractions = {element: count / atom_count for element, count in counts.items()}
    atomic_numbers = np.asarray([Element(element).Z for element in fractions], dtype=float)
    weights = np.asarray(list(fractions.values()), dtype=float)
    mean_z = float(np.sum(weights * atomic_numbers))
    std_z = float(np.sqrt(np.sum(weights * (atomic_numbers - mean_z) ** 2)))
    descriptor = (
        atomization_energy / atom_count,
        math.log1p(atom_count),
        math.log(max(volume / atom_count, 1e-8)),
        float(np.mean(force_norms)),
        float(np.std(force_norms)),
        float(np.max(force_norms)),
        math.log1p(stress_norm),
        math.log(max(float(np.max(singular) / max(float(np.min(singular)), 1e-8)), 1.0)),
        mean_z / 100.0,
        std_z / 100.0,
        float(np.ptp(atomic_numbers)) / 100.0,
        *(fractions.get(symbol, 0.0) for symbol in ELEMENT_FRACTION_ORDER),
    )
    if not all(math.isfinite(value) for value in descriptor):
        raise ValueError("MAD source descriptor contains a non-finite value")
    return descriptor


def _header_float(pattern: re.Pattern[str], header: str, *, name: str) -> float:
    match = pattern.search(header)
    if not match:
        raise ValueError(f"MAD header has no {name}")
    value = float(match.group(1))
    if not math.isfinite(value):
        raise ValueError(f"MAD header has non-finite {name}")
    return value


def _parse_frame(
    handle, path: Path, split: str, frame_number: int, *, with_descriptor: bool
) -> MADRecord | None:
    count_line = handle.readline()
    if not count_line:
        return None
    while count_line and not count_line.strip():
        count_line = handle.readline()
    if not count_line:
        return None
    try:
        atom_count = int(count_line.strip())
    except ValueError as exc:
        raise ValueError(f"invalid atom count in {path} frame {frame_number}") from exc
    header = handle.readline()
    if not header:
        raise ValueError(f"missing MAD header in {path} frame {frame_number}")
    frame_match = HEADER_RE["frame_id"].search(header)
    subset_match = HEADER_RE["subset"].search(header)
    pbc_match = HEADER_RE["pbc"].search(header)
    lattice_match = HEADER_RE["lattice"].search(header)
    if not frame_match or not subset_match or not pbc_match:
        raise ValueError(f"incomplete MAD identity header in {path} frame {frame_number}")
    lattice = (
        tuple(float(value) for value in lattice_match.group(1).split()) if lattice_match else ()
    )
    if lattice and len(lattice) != 9:
        raise ValueError(f"MAD lattice is not 3x3 in {path} frame {frame_number}")
    species: list[str] = []
    positions: list[float] = []
    forces: list[float] = []
    for atom_number in range(atom_count):
        fields = handle.readline().split()
        if len(fields) < 7:
            raise ValueError(f"malformed MAD atom line in {path} frame {frame_number}")
        species.append(fields[0])
        positions.extend(float(value) for value in fields[1:4])
        forces.extend(float(value) for value in fields[4:7])
    composition_counts = Counter(species)
    composition = {element: float(count) for element, count in sorted(composition_counts.items())}
    atomization = _header_float(HEADER_RE["atomization"], header, name="atomization_energy")
    stress_match = HEADER_RE["stress"].search(header)
    stress = tuple(float(value) for value in stress_match.group(1).split()) if stress_match else ()
    if stress and len(stress) != 9:
        raise ValueError(f"MAD stress is not 3x3 in {path} frame {frame_number}")
    descriptor = None
    if (
        with_descriptor
        and subset_match.group(1) in BULK_SUBSETS
        and pbc_match.group(1).split() == ["T", "T", "T"]
    ):
        if len(lattice) != 9:
            raise ValueError(f"bulk MAD frame has no valid lattice in {path} frame {frame_number}")
        descriptor = _descriptor(
            atom_count=atom_count,
            species=tuple(species),
            lattice=lattice,
            positions=tuple(positions),
            forces=tuple(forces),
            stress=stress or (0.0,) * 9,
            atomization_energy=atomization,
        )
    return MADRecord(
        subset=subset_match.group(1),
        frame_id=int(frame_match.group(1)),
        split=split,
        atom_count=atom_count,
        species=tuple(species),
        composition=composition,
        pbc=tuple(pbc_match.group(1).split()),
        lattice=lattice,
        structure_hash=_geometry_hash(tuple(species), tuple(positions), lattice),
        atomization_energy=atomization,
        descriptor=descriptor,
    )


def _read_file(
    path: Path, split: str, *, with_descriptor: bool
) -> dict[tuple[str, int], MADRecord]:
    records: dict[tuple[str, int], MADRecord] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        frame_number = 0
        while True:
            record = _parse_frame(
                handle, path, split, frame_number, with_descriptor=with_descriptor
            )
            if record is None:
                break
            if record.key in records:
                raise ValueError(f"duplicate MAD key {record.key} in {path}")
            records[record.key] = record
            frame_number += 1
    return records


def _pair_rows(
    source: dict[tuple[str, int], MADRecord], target_root: Path
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    by_system: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for split, filename in R2SCAN_FILES.items():
        target_records = _read_file(target_root / filename, split, with_descriptor=False)
        for key, target in target_records.items():
            source_record = source.get(key)
            if source_record is None or not source_record.bulk_3d or not target.bulk_3d:
                continue
            if source_record.structure_hash != target.structure_hash:
                continue
            if (
                source_record.atom_count != target.atom_count
                or source_record.species != target.species
            ):
                continue
            if source_record.descriptor is None:
                raise AssertionError("MAD source descriptor was not constructed")
            pair_id = f"mad15:{source_record.subset}:{source_record.frame_id}"
            pair = {
                "pair_id": pair_id,
                "chemical_system": source_record.chemical_system,
                "composition": source_record.composition,
                "source_structure_sha256": source_record.structure_hash,
                "source_formation_energy_ev_per_atom": source_record.atomization_energy_per_atom,
                "source_environment_embedding": list(source_record.descriptor),
                "source_local_environment_embedding": None,
                "upstream_pbe_split": source_record.split,
                "upstream_r2scan_split": split,
                "energy_semantics": "atomization_energy_ev_per_atom",
            }
            oracle = {
                "pair_id": pair_id,
                "source_structure_sha256": source_record.structure_hash,
                "chemical_system": source_record.chemical_system,
                "composition": target.composition,
                "target_corrected_total_energy_ev": target.atomization_energy,
                "target_formation_energy_ev_per_atom": target.atomization_energy_per_atom,
                "split": "development",
                "upstream_r2scan_split": split,
                "energy_semantics": "atomization_energy_ev_per_atom",
            }
            by_system[source_record.chemical_system].append((pair, oracle))
    return by_system


def run(
    *,
    root: Path,
    audit_path: Path,
    task_output: Path,
    vault_output: Path,
    minimum_candidates: int = 8,
    minimum_compositions: int = 3,
    max_systems: int | None = None,
    max_candidates_per_system: int = 64,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if any(path.resolve().is_relative_to(repo_root) for path in (task_output, vault_output)):
        raise ValueError("MAD task and vault must remain outside Git")
    if task_output.exists() or vault_output.exists():
        raise FileExistsError("MAD task builder cannot overwrite outputs")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("target_values_used") is True:
        raise ValueError("pair audit must not use target values for selection")
    source = _read_file(root / "mad-1.5-pbe.xyz", "pbe", with_descriptor=True)
    by_system = _pair_rows(source, root)
    eligible = sorted(
        system
        for system, rows in by_system.items()
        if len(rows) >= minimum_candidates
        and len({json.dumps(pair["composition"], sort_keys=True) for pair, _ in rows})
        >= minimum_compositions
    )
    if max_systems is not None:
        eligible = sorted(
            eligible,
            key=lambda system: _stable_hash("MAD-1.5-protocol-shift-v1", system),
        )[:max_systems]
    if not eligible:
        raise ValueError("MAD task has no eligible systems")

    task_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    initial_entries: dict[str, list[dict[str, Any]]] = {}
    system_summary: dict[str, Any] = {}
    for system in sorted(eligible):
        rows = sorted(
            by_system[system],
            key=lambda item: _stable_hash("MAD-1.5-protocol-shift-v1", system, item[0]["pair_id"]),
        )[:max_candidates_per_system]
        task_rows.extend(pair for pair, _ in rows)
        oracle_rows.extend(oracle for _, oracle in rows)
        initial_entries[system] = [
            {
                "entry_id": f"isolated-atom-reference-{element}",
                "composition": {element: 1.0},
                "corrected_total_energy_ev": 0.0,
            }
            for element in system.split("-")
        ]
        system_summary[system] = {
            "selected_candidate_count": len(rows),
            "available_candidate_count": len(by_system[system]),
            "selected_composition_count": len(
                {json.dumps(pair["composition"], sort_keys=True) for pair, _ in rows}
            ),
        }

    selected_ids = sorted(row["pair_id"] for row in task_rows)
    selected_checksum = hashlib.sha256(
        "".join(f"{item}\n" for item in selected_ids).encode()
    ).hexdigest()
    source_protocol = {
        "functional": "PBE",
        "pseudopotential_set": "FHI-aims 2020 species defaults",
        "correction_scheme": "MAD-1.5 atomization energy relative to isolated atoms",
        "relaxation_protocol": "shared fixed geometry; no pairwise relaxation",
        "calculation_code": "FHI-aims 2020",
    }
    target_protocol = {**source_protocol, "functional": "r2SCAN"}
    task = {
        "schema_version": 1,
        "release_id": "MAD-1.5-PBE-r2SCAN-atomization-hull-v1",
        "representation_id": "observable-mad15-pbe-atomization-composition-v1",
        "status": "external_protocol_shift_development_not_holdout",
        "energy_semantics": "atomization_energy_ev_per_atom; isolated-atom reference zero",
        "hull_semantics": "atomization-energy convex-hull proxy, not solid-state formation hull",
        "source_protocol": source_protocol,
        "target_protocol": target_protocol,
        "pair_audit_sha256": _sha256(audit_path),
        "selected_pair_id_set_sha256": selected_checksum,
        "selection_rule": (
            "3D bulk and structure-exact pairing; minimum candidate/composition gates; "
            "stable hash selection; no target energy used for selection"
        ),
        "descriptor": {
            "name": "observable_pbe_atomization_geometry_force_stress_element_fraction_v1",
            "dimension": len(task_rows[0]["source_environment_embedding"]),
            "uses_target_outcome": False,
            "element_fraction_order": ELEMENT_FRACTION_ORDER,
        },
        "development_systems": sorted(eligible),
        "development_pairs": task_rows,
        "development_initial_phase_entries": initial_entries,
        "system_summary": system_summary,
    }
    vault = {
        "schema_version": 1,
        "release_id": task["release_id"],
        "status": "development_oracle_vault",
        "selected_pair_id_set_sha256": selected_checksum,
        "target_outcomes": oracle_rows,
    }
    task_output.parent.mkdir(parents=True, exist_ok=True)
    vault_output.parent.mkdir(parents=True, exist_ok=True)
    task_output.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vault_output.write_text(json.dumps(vault, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "task_path": str(task_output.resolve()),
        "vault_path": str(vault_output.resolve()),
        "selected_system_count": len(eligible),
        "selected_pair_count": len(task_rows),
        "task_sha256": _sha256(task_output),
        "vault_sha256": _sha256(vault_output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--vault-output", type=Path, required=True)
    parser.add_argument("--minimum-candidates", type=int, default=8)
    parser.add_argument("--minimum-compositions", type=int, default=3)
    parser.add_argument("--max-systems", type=int, default=None)
    parser.add_argument("--max-candidates-per-system", type=int, default=64)
    args = parser.parse_args()
    run(
        root=args.root,
        audit_path=args.audit,
        task_output=args.task_output,
        vault_output=args.vault_output,
        minimum_candidates=args.minimum_candidates,
        minimum_compositions=args.minimum_compositions,
        max_systems=args.max_systems,
        max_candidates_per_system=args.max_candidates_per_system,
    )


if __name__ == "__main__":
    main()
