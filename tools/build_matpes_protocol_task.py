"""Build an oracle-isolated MatPES PBE-to-r2SCAN development task."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Element

from matmem.matpes_data import (
    MATPES_PBE_STEM,
    MATPES_R2SCAN_STEM,
    MATPES_SPLITS,
    MatPESCompactConfiguration,
    compact_matpes_configuration,
    iter_matpes_jsonl,
)

RELEASE_ID = "MatPES-PBE-r2SCAN-2025.2-same-configuration-v1"
REPRESENTATION_ID = "observable-pbe-composition-aware-v2"
ELEMENT_FRACTION_ORDER = tuple(
    Element.from_Z(atomic_number).symbol for atomic_number in range(1, 119)
)


@dataclass(frozen=True, slots=True)
class SourceRecord:
    compact: MatPESCompactConfiguration
    composition: dict[str, float]
    descriptor: tuple[float, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def _finite_or_zero(value: Any) -> tuple[float, float]:
    if value is None:
        return 0.0, 1.0
    parsed = float(value)
    if not math.isfinite(parsed):
        return 0.0, 1.0
    return parsed, 0.0


def source_descriptor(row: dict[str, Any]) -> tuple[float, ...]:
    """Create a compact, policy-visible descriptor from the PBE calculation.

    The descriptor intentionally uses no r2SCAN outcome or post-query geometry.
    Its modest dimension avoids repeating the underdetermined 64-dimensional
    local ridge used in the JARVIS development task.
    """

    nsites = int(row["nsites"])
    volume_per_atom = float(row["volume"]) / nsites
    bandgap, bandgap_missing = _finite_or_zero(row.get("bandgap"))
    forces = np.asarray(row.get("forces", ()), dtype=np.float64)
    if forces.shape != (nsites, 3) or not np.isfinite(forces).all():
        force_norms = np.zeros(nsites, dtype=np.float64)
    else:
        force_norms = np.linalg.norm(forces, axis=1)
    stress = np.asarray(row.get("stress", ()), dtype=np.float64).reshape(-1)
    stress_norm = float(np.linalg.norm(stress)) if np.isfinite(stress).all() else 0.0
    lattice = np.asarray(row["structure"]["lattice"]["matrix"], dtype=np.float64)
    singular = np.linalg.svd(lattice, compute_uv=False)
    anisotropy = float(np.max(singular) / max(float(np.min(singular)), 1e-8))
    composition = {str(symbol): float(amount) for symbol, amount in row["composition"].items()}
    composition_total = float(sum(composition.values()))
    if (
        not composition
        or not math.isfinite(composition_total)
        or composition_total <= 0
        or any(not math.isfinite(amount) or amount <= 0 for amount in composition.values())
    ):
        raise ValueError("MatPES PBE descriptor requires a valid composition")
    fractions = {symbol: amount / composition_total for symbol, amount in composition.items()}
    atomic_numbers = np.asarray([Element(symbol).Z for symbol in fractions], dtype=np.float64)
    atomic_weights = np.asarray(list(fractions.values()), dtype=np.float64)
    atomic_number_mean = float(np.sum(atomic_weights * atomic_numbers))
    atomic_number_std = float(
        np.sqrt(np.sum(atomic_weights * (atomic_numbers - atomic_number_mean) ** 2))
    )
    formation = row.get("formation_energy_per_atom")
    if formation is None:
        raise ValueError("MatPES source descriptor requires PBE formation energy")
    cohesive, _ = _finite_or_zero(row.get("cohesive_energy_per_atom"))
    symmetry = row.get("symmetry") or {}
    descriptor = (
        cohesive,
        math.log1p(nsites),
        math.log(max(volume_per_atom, 1e-8)),
        float(row["density"]),
        bandgap,
        bandgap_missing,
        float(symmetry.get("number", 0)) / 230.0,
        float(np.mean(force_norms)),
        float(np.std(force_norms)),
        float(np.max(force_norms)),
        math.log1p(stress_norm),
        math.log(max(anisotropy, 1.0)),
        atomic_number_mean / 100.0,
        atomic_number_std / 100.0,
        float(np.ptp(atomic_numbers)) / 100.0,
        *(fractions.get(symbol, 0.0) for symbol in ELEMENT_FRACTION_ORDER),
    )
    if not all(math.isfinite(value) for value in descriptor):
        raise ValueError("MatPES PBE descriptor contains a non-finite value")
    return descriptor


def _same_configuration(
    source: MatPESCompactConfiguration,
    target: MatPESCompactConfiguration,
) -> bool:
    return (
        source.nsites == target.nsites
        and source.chemsys == target.chemsys
        and source.composition_key == target.composition_key
        and source.original_mp_id == target.original_mp_id
        and source.rounded_geometry_sha256 == target.rounded_geometry_sha256
    )


def _read_audit(path: Path) -> dict[str, Any]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    decision = audit.get("decision", {})
    if not decision.get("same_configuration_protocol_task_supported"):
        raise ValueError("MatPES pair audit does not authorize a protocol task")
    return audit


def run(
    *,
    pbe_root: Path,
    r2scan_root: Path,
    audit_path: Path,
    task_output: Path,
    vault_output: Path,
    max_systems: int,
    max_candidates_per_system: int,
    minimum_candidates_per_system: int,
    minimum_parents_per_system: int,
) -> dict[str, Any]:
    if task_output.exists() or vault_output.exists():
        raise FileExistsError("MatPES task builder cannot overwrite outputs")
    repo_root = Path(__file__).resolve().parents[1]
    if any(path.resolve().is_relative_to(repo_root) for path in (task_output, vault_output)):
        raise ValueError("MatPES task and oracle vault must remain outside Git")
    if (
        max_systems < 1
        or max_candidates_per_system < 2
        or minimum_candidates_per_system < 2
        or minimum_candidates_per_system > max_candidates_per_system
        or minimum_parents_per_system < 1
    ):
        raise ValueError("invalid MatPES task size configuration")
    audit = _read_audit(audit_path)

    pbe_index: dict[str, SourceRecord] = {}
    for split in MATPES_SPLITS:
        path = pbe_root / f"{MATPES_PBE_STEM}-{split}.jsonl"
        for row in iter_matpes_jsonl(path):
            if row.get("formation_energy_per_atom") is None:
                continue
            compact = compact_matpes_configuration(row, split=split)
            identifier = str(row["matpes_id"])
            if identifier in pbe_index:
                raise ValueError("duplicate PBE matpes_id while building task")
            pbe_index[identifier] = SourceRecord(
                compact=compact,
                composition={
                    str(element): float(amount) for element, amount in row["composition"].items()
                },
                descriptor=source_descriptor(row),
            )

    by_system: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    split_pairs: Counter[str] = Counter()
    for split in MATPES_SPLITS:
        path = r2scan_root / f"{MATPES_R2SCAN_STEM}-{split}.jsonl"
        for row in iter_matpes_jsonl(path):
            if row.get("formation_energy_per_atom") is None:
                continue
            identifier = str(row["matpes_id"])
            source = pbe_index.get(identifier)
            if source is None:
                continue
            target = compact_matpes_configuration(row, split=split)
            if not _same_configuration(source.compact, target):
                raise ValueError(f"audited MatPES pair changed identity: {identifier}")
            elements = tuple(target.chemsys.split("-"))
            if len(elements) < 2:
                continue
            pair = {
                "pair_id": identifier,
                "chemical_system": target.chemsys,
                "composition": source.composition,
                "source_structure_sha256": source.compact.exact_geometry_sha256,
                "source_formation_energy_ev_per_atom": (
                    source.compact.formation_energy_ev_per_atom
                ),
                "source_environment_embedding": source.descriptor,
                "original_mp_id": source.compact.original_mp_id,
                "upstream_pbe_split": source.compact.split,
                "upstream_r2scan_split": target.split,
            }
            oracle = {
                "pair_id": identifier,
                "source_structure_sha256": source.compact.exact_geometry_sha256,
                "chemical_system": target.chemsys,
                "composition": source.composition,
                "target_corrected_total_energy_ev": (
                    target.formation_energy_ev_per_atom * target.nsites
                ),
                "target_formation_energy_ev_per_atom": (target.formation_energy_ev_per_atom),
                "split": "development",
            }
            by_system[target.chemsys].append((pair, oracle))
            split_pairs[f"{source.compact.split}->{target.split}"] += 1

    eligible = []
    for system, rows in by_system.items():
        parent_count = len({pair["original_mp_id"] for pair, _ in rows})
        if (
            len(rows) >= minimum_candidates_per_system
            and parent_count >= minimum_parents_per_system
        ):
            eligible.append(system)
    selected_systems = sorted(
        eligible, key=lambda system: _stable_hash(RELEASE_ID, "development", system)
    )[:max_systems]
    if not selected_systems:
        raise ValueError("no MatPES exact system satisfies the development-pool gate")

    task_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []
    initial_entries: dict[str, list[dict[str, Any]]] = {}
    system_summary: dict[str, Any] = {}
    for system in selected_systems:
        rows = sorted(
            by_system[system],
            key=lambda item: _stable_hash(RELEASE_ID, system, item[0]["pair_id"]),
        )[:max_candidates_per_system]
        task_rows.extend(pair for pair, _ in rows)
        oracle_rows.extend(oracle for _, oracle in rows)
        elements = system.split("-")
        initial_entries[system] = [
            {
                "entry_id": f"reference-{element}",
                "composition": {element: 1.0},
                "corrected_total_energy_ev": 0.0,
            }
            for element in elements
        ]
        system_summary[system] = {
            "selected_candidate_count": len(rows),
            "available_candidate_count": len(by_system[system]),
            "selected_original_mp_parent_count": len({pair["original_mp_id"] for pair, _ in rows}),
        }

    selected_ids = sorted(row["pair_id"] for row in task_rows)
    selected_id_checksum = hashlib.sha256(
        "".join(f"{identifier}\n" for identifier in selected_ids).encode()
    ).hexdigest()
    source_protocol = {
        "functional": "PBE",
        "pseudopotential_set": "MatPES VASP PAW",
        "correction_scheme": "MatPES-2025.2 formation_energy_per_atom",
        "relaxation_protocol": "shared pre-generated MatPES configuration",
        "calculation_code": "VASP",
    }
    target_protocol = {
        **source_protocol,
        "functional": "r2SCAN",
    }
    task = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "representation_id": REPRESENTATION_ID,
        "status": "exploratory_development_task_not_confirmatory",
        "source_protocol": source_protocol,
        "target_protocol": target_protocol,
        "pair_audit_sha256": _sha256(audit_path),
        "pair_audit_pair_set_sha256": audit["pairing"]["same_configuration_pair_id_set_sha256"],
        "selected_pair_id_set_sha256": selected_id_checksum,
        "selection_rule": (
            "formation-label availability, exact-system and original-parent count gates, "
            "then SHA256(release, system[, pair_id]); no target value used"
        ),
        "descriptor": {
            "name": "observable_pbe_scalar_structure_element_fraction_v2",
            "dimension": len(task_rows[0]["source_environment_embedding"]),
            "uses_target_outcome": False,
            "element_fraction_order": ELEMENT_FRACTION_ORDER,
            "composition_weighted_element_statistics": True,
        },
        "development_systems": selected_systems,
        "development_pairs": task_rows,
        "development_initial_phase_entries": initial_entries,
        "system_summary": system_summary,
        "upstream_split_pair_counts_before_selection": dict(sorted(split_pairs.items())),
    }
    vault = {
        "schema_version": 1,
        "release_id": RELEASE_ID,
        "status": "development_oracle_vault",
        "selected_pair_id_set_sha256": selected_id_checksum,
        "target_outcomes": oracle_rows,
    }
    task_output.parent.mkdir(parents=True, exist_ok=True)
    vault_output.parent.mkdir(parents=True, exist_ok=True)
    task_output.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    vault_output.write_text(json.dumps(vault, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "task_path": str(task_output.resolve()),
        "task_sha256": _sha256(task_output),
        "vault_path": str(vault_output.resolve()),
        "vault_sha256": _sha256(vault_output),
        "selected_pair_count": len(task_rows),
        "selected_system_count": len(selected_systems),
        "selected_systems": selected_systems,
        "system_summary": system_summary,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pbe-root", type=Path, required=True)
    parser.add_argument("--r2scan-root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--task-output", type=Path, required=True)
    parser.add_argument("--vault-output", type=Path, required=True)
    parser.add_argument("--max-systems", type=int, default=12)
    parser.add_argument("--max-candidates-per-system", type=int, default=64)
    parser.add_argument("--minimum-candidates-per-system", type=int, default=32)
    parser.add_argument("--minimum-parents-per-system", type=int, default=8)
    args = parser.parse_args()
    run(
        pbe_root=args.pbe_root,
        r2scan_root=args.r2scan_root,
        audit_path=args.audit,
        task_output=args.task_output,
        vault_output=args.vault_output,
        max_systems=args.max_systems,
        max_candidates_per_system=args.max_candidates_per_system,
        minimum_candidates_per_system=args.minimum_candidates_per_system,
        minimum_parents_per_system=args.minimum_parents_per_system,
    )


if __name__ == "__main__":
    main()
