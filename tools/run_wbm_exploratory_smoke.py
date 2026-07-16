"""Run a deliberately limited, exploratory WBM data-path smoke experiment.

This is not the preregistered WBM policy matrix. It uses one deterministic
step-1 pool, the frozen official CHGNet prediction file, MP2020-corrected WBM
energies and the frozen MP PPD only to check that a prediction ranking and the
zero-cost FIFO/on-demand exact emulator run on real records.  It does not use
SOAP, canonical duplicate exclusion, dynamic hull updates, or report a
persistence advantage.
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audit_wbm_official_artifacts import _load_ppd_read_only  # noqa: E402

from evimem.matmem import (  # noqa: E402
    BaseBoundaryAcquisition,
    CandidatePoolItem,
    HullSnapshot,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    ProtocolCertificate,
    SourceProvenance,
    run_fifo_exact_emulation,
)

RELEASE_ID = "wbm-exploratory-smoke-v1"
MP_RELEASE = "MP-2022.10.28"
PHASE_CHECKSUM = "sha256:3bddfcdd656b673213a40227e6dab058254b5c3ee4248eb22e72b26f688a637f"
AS_OF = datetime(2023, 2, 7, tzinfo=UTC)
OBSERVED_AT = datetime(2023, 12, 21, tzinfo=UTC)


def _hash_rank(namespace: str, value: str) -> str:
    return hashlib.sha256(f"{namespace}|{value}".encode()).hexdigest()


def _load_step_one_records(cse_path: Path, cleaned_ids: set[str]) -> dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]]:
    payload = json.loads(bz2.decompress(cse_path.read_bytes()))
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("WBM step-1 CSE file does not contain entries")
    groups: dict[tuple[str, ...], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for index, raw_entry in enumerate(entries, start=1):
        query_id = f"wbm-1-{index}"
        if query_id not in cleaned_ids:
            continue
        if not isinstance(raw_entry, dict) or not isinstance(raw_entry.get("composition"), dict):
            raise ValueError(f"invalid raw WBM CSE record for {query_id}")
        system = tuple(sorted(raw_entry["composition"]))
        groups[system].append((query_id, raw_entry))
    return groups


def _load_predictions(path: Path) -> dict[str, float]:
    predictions: dict[str, float] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            material_id = row["material_id"]
            if material_id in predictions:
                raise ValueError(f"duplicate official prediction ID: {material_id}")
            predictions[material_id] = float(row["e_form_per_atom"])
    return predictions


def _compatibility_process(entries: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, Any]]:
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.entries.computed_entries import ComputedStructureEntry

    converted = []
    for query_id, raw in entries:
        record = json.loads(json.dumps(raw))
        record["entry_id"] = query_id
        record["parameters"]["run_type"] = "GGA+U" if record["parameters"]["is_hubbard"] else "GGA"
        converted.append(ComputedStructureEntry.from_dict(record))
    corrected = MaterialsProject2020Compatibility().process_entries(
        converted, clean=True, verbose=False
    )
    by_id = {str(entry.entry_id): entry for entry in corrected}
    if len(by_id) != len(entries):
        raise ValueError("MP2020 processing rejected at least one exploratory candidate")
    return [(query_id, by_id[query_id]) for query_id, _ in entries]


def _repair_historical_composition_interface() -> None:
    """Bridge the saved PPD's historic composition state in pymatgen 2023.5.10."""

    from pymatgen.core.composition import Composition

    if not hasattr(Composition, "_natoms"):
        # New compositions assign ``_natoms`` during construction, whereas the
        # published PPD's unpickled compositions retain ``_n_atoms``.  Support
        # both spellings without changing either serialized input.
        Composition._natoms = property(  # type: ignore[attr-defined]
            lambda self: self._n_atoms,
            lambda self, value: setattr(self, "_n_atoms", value),
        )


def _candidate_items(
    entries: list[tuple[str, Any]],
    predictions: dict[str, float],
    ppd: Any,
    system: tuple[str, ...],
) -> tuple[list[CandidatePoolItem], list[float]]:
    protocol = ProtocolCertificate(
        functional="PBE",
        pseudopotential_set="PAW",
        correction_scheme="MaterialsProject2020Compatibility(pymatgen-2023.5.10-default)",
        relaxation_protocol="WBM-2021-relaxed",
        calculation_code="VASP",
    )
    items: list[CandidatePoolItem] = []
    labels: list[float] = []
    for query_id, entry in entries:
        if query_id not in predictions:
            raise ValueError(f"missing official CHGNet prediction for {query_id}")
        reference = float(ppd.get_hull_energy_per_atom(entry.composition))
        above_hull = float(ppd.get_e_above_hull(entry, allow_negative=True))
        formation_energy = reference + above_hull
        structure_hash = "sha256:" + hashlib.sha256(
            json.dumps(entry.structure.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        hull = HullSnapshot(
            snapshot_id=f"{MP_RELEASE}:{query_id}", chemical_system=system,
            reference_hull_energy_ev_per_atom=reference, phase_set_checksum=PHASE_CHECKSUM,
            known_through=AS_OF, built_at=AS_OF, source_version=f"MaterialsProject:{MP_RELEASE}",
        )
        identity = MaterialIdentity(
            exact_calculation_id=query_id, canonical_structure_id=f"exploratory-unresolved:{query_id}",
            composition_family="-".join(system), prototype_family=None,
        )
        query = MaterialQuery(
            query_id=query_id, structure_hash=structure_hash, identity=identity,
            composition=entry.composition.reduced_formula, embedding=(1.0, 0.0), protocol=protocol,
            hull_snapshot=hull, base_predicted_formation_energy_ev_per_atom=predictions[query_id],
            as_of=AS_OF,
        )
        card = MaterialMemoryCard(
            card_id=f"wbm-card:{query_id}", material_id=query_id, structure_hash=structure_hash,
            identity=identity, composition=query.composition, embedding=query.embedding, protocol=protocol,
            provenance=SourceProvenance(source_name="WBM", source_version="2021.68", record_locator=query_id, retrieved_at=OBSERVED_AT),
            formation_energy_ev_per_atom=formation_energy,
            base_predicted_formation_energy_ev_per_atom=predictions[query_id],
            oracle_residual_ev_per_atom=formation_energy - predictions[query_id], hull_snapshot=hull,
            recorded_hull_distance_ev_per_atom=above_hull, observed_at=OBSERVED_AT,
        )
        items.append(CandidatePoolItem(query=query, oracle_card=card))
        labels.append(float(above_hull <= 1e-8))
    return items, labels


def _discoveries_at_budgets(order: list[int], labels: list[float], budgets: tuple[int, ...]) -> dict[str, int]:
    return {str(budget): int(sum(labels[index] for index in order[:budget])) for budget in budgets}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-cse-root", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pool-size", type=int, default=64)
    parser.add_argument("--budget", type=int, default=24)
    parser.add_argument("--random-seeds", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.budget <= args.pool_size or args.random_seeds < 1:
        parser.error("require 1 <= budget <= pool-size and positive random-seeds")
    if args.output.resolve().is_relative_to(ROOT):
        parser.error("exploratory output must remain outside the repository")

    cleaned_ids = {
        line.strip() for line in args.cleaned_ids.read_text(encoding="utf-8").splitlines() if line.strip()
    }
    groups = _load_step_one_records(args.raw_cse_root / "step_1.json.bz2", cleaned_ids)
    eligible = [(system, records) for system, records in groups.items() if len(records) >= args.pool_size]
    if not eligible:
        raise ValueError("no step-1 chemical system has enough cleaned candidates for this smoke pool")
    system, records = min(eligible, key=lambda item: (_hash_rank(RELEASE_ID, "-".join(item[0])), item[0]))
    ranked_records = sorted(records, key=lambda item: (_hash_rank(RELEASE_ID, item[0]), item[0]))[: args.pool_size]
    predictions = _load_predictions(args.artifact_dir / "2023-12-21-chgnet-0.3.0-discovery.csv.gz")
    _repair_historical_composition_interface()
    ppd = _load_ppd_read_only(args.artifact_dir / "2023-02-07-ppd-mp.pkl.gz")
    corrected = _compatibility_process(ranked_records)
    items, labels = _candidate_items(corrected, predictions, ppd, system)
    budgets = tuple(sorted({min(8, args.budget), min(16, args.budget), args.budget}))
    prediction_order = sorted(
        range(len(items)),
        key=lambda index: (
            items[index].query.base_hull_distance_ev_per_atom, items[index].query.query_id,
        ),
    )
    random_discoveries = []
    for seed in range(args.random_seeds):
        order = sorted(range(len(items)), key=lambda index: (_hash_rank(str(seed), items[index].query.query_id), items[index].query.query_id))
        random_discoveries.append(_discoveries_at_budgets(order, labels, budgets))
    exact = run_fifo_exact_emulation(
        items, BaseBoundaryAcquisition, capacity=min(8, args.budget), oracle_budget=args.budget,
        causal_hull_updates=False,
    )
    report = {
        "scope": "exploratory_wbm_smoke_not_preregistered_policy_evidence",
        "limitations": [
            "step-1-only deterministic pool; no canonical duplicate or MP-overlap exclusion",
            "no SOAP, residual acquisition, CAW-Joint, or dynamic causal hull",
            "initial-hull labels only; final-hull metrics are deliberately not reported",
            "persistent/on-demand comparison is an exact-emulation invariant, not an effect estimate",
        ],
        "pool": {"chemical_system": list(system), "candidate_count": len(items), "stable_initial_hull_count": int(sum(labels))},
        "oracle_budget": args.budget,
        "frozen_prediction": {"model": "CHGNet 0.3.0 official discovery file", "unit": "eV/atom"},
        "prediction_ranking_initial_hull_discoveries": _discoveries_at_budgets(prediction_order, labels, budgets),
        "random_initial_hull_discoveries": random_discoveries,
        "zero_cost_fifo_on_demand_exact_emulation": exact.model_dump(mode="json"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"system={'-'.join(system)} candidates={len(items)} stable_initial={int(sum(labels))}")
    print(f"prediction_discoveries={report['prediction_ranking_initial_hull_discoveries']}")
    print(f"exact_emulation={exact.passed} checksum={exact.persistent_checksum}")


if __name__ == "__main__":
    main()
