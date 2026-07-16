"""Run a small, gate-checked WBM calibration-coreset engineering pilot.

This is not the claim-grade policy matrix.  It consumes the fixed historical
replay, frozen 8x16 pools and full exact-system oracle universes, runs behind
the sole secure WBM runner, and writes every ledger and result outside Git.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
SRC_ROOT = TOOLS_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_wbm_official_artifacts import _load_ppd_read_only  # noqa: E402
from audit_wbm_p1_p15 import (  # noqa: E402
    _correct_exact_system_universe,
    _load_exact_system_universe,
    _read_cleaned_ids,
)
from build_wbm_candidate_parity_audit import (  # noqa: E402
    _repair_historical_composition_interface,
)

from evimem.matmem import (  # noqa: E402
    AppendOnlyWBMEventLog,
    CalibrationUtilityBuilder,
    CompositionHullState,
    CorrectedPhaseEntry,
    DiversityBoundedMemory,
    FacilityLocationCoresetPlanner,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    HullSnapshot,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    PersistentFIFOEvidence,
    PolicySubprocess,
    ProtocolCompatibilityResolver,
    ReconstructedFIFOEvidence,
    SecureWBMRunner,
    SourceProvenance,
    StreamingCalibrationCoreset,
    StreamingCoresetEvidence,
    WBMOracleRecord,
    WBMOracleVault,
)
from evimem.matmem.protocols import ProtocolCertificate  # noqa: E402

INITIAL_TIME = datetime(2023, 2, 7, tzinfo=UTC)
OBSERVED_TIME = INITIAL_TIME + timedelta(days=1)
SOURCE_VERSION = "fixed-historical-pipeline-WBM-2021.68-MP2020-pymatgen-2023.5.10"


class _DiversityEvidence:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = DiversityBoundedMemory(capacity)

    def active(self, archive: tuple[Any, ...]) -> tuple[Any, ...]:
        del archive
        return self.memory.cards()

    def admit(self, card: Any, query_pool: tuple[MaterialQuery, ...]) -> None:
        self.memory.admit(card, query_pool)


def exact_gram_embedding(vectors: np.ndarray) -> np.ndarray:
    """Losslessly factor the finite-pool SOAP Gram matrix into <=N dimensions."""

    matrix = np.asarray(vectors, dtype=float)
    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    keep = eigenvalues > 1e-12
    coordinates = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
    if not np.allclose(coordinates @ coordinates.T, gram, atol=1e-9):
        raise ValueError("finite-pool SOAP Gram factorization is not lossless")
    return coordinates


def _protocol() -> ProtocolCertificate:
    return ProtocolCertificate(
        functional="PBE",
        pseudopotential_set="Materials-Project-compatible-PAW",
        correction_scheme="MP2020",
        relaxation_protocol="WBM-MPRelaxSet",
        calculation_code="VASP",
    )


def _initial_entries(ppd: Any, system: tuple[str, ...]) -> list[Any]:
    return [
        entry
        for entry in ppd.all_entries
        if set(str(element) for element in entry.composition.elements).issubset(system)
    ]


def _build_queries(
    *,
    pool: dict[str, Any],
    pool_name: str,
    entries_by_id: dict[str, Any],
    predictions: dict[str, float],
    embedding_by_id: dict[str, tuple[float, ...]],
    initial_diagram: Any,
) -> tuple[MaterialQuery, ...]:
    system = tuple(sorted(pool["chemical_system"]))
    state = CompositionHullState(
        initial_diagram, chemical_system=system, source_version="MP-2022.10.28"
    )
    queries = []
    for candidate in pool["candidates"]:
        query_id = candidate["query_id"]
        entry = entries_by_id[query_id]
        placeholder = HullSnapshot(
            snapshot_id=f"initial-placeholder:{query_id}",
            chemical_system=system,
            reference_hull_energy_ev_per_atom=0.0,
            phase_set_checksum=state.phase_set_checksum,
            known_through=INITIAL_TIME,
            built_at=INITIAL_TIME,
            source_version="MP-2022.10.28",
        )
        query = MaterialQuery(
            query_id=query_id,
            structure_hash=candidate["exact_structure_sha256"],
            identity=MaterialIdentity(
                exact_calculation_id=query_id,
                canonical_structure_id=(
                    "byte-identical:" + candidate["exact_structure_sha256"]
                ),
                composition_family=pool_name,
            ),
            composition=entry.composition.reduced_formula,
            embedding=embedding_by_id[query_id],
            protocol=_protocol(),
            hull_snapshot=placeholder,
            base_predicted_formation_energy_ev_per_atom=predictions[query_id],
            as_of=INITIAL_TIME,
        )
        snapshot = state.snapshot(query, round_index=0, built_at=INITIAL_TIME)
        queries.append(query.model_copy(update={"hull_snapshot": snapshot}))
    return tuple(queries)


def _coreset(capacity: int, config: FixedKernelGPConfig) -> StreamingCoresetEvidence:
    resolver = ProtocolCompatibilityResolver()
    planner = FacilityLocationCoresetPlanner(
        capacity,
        CalibrationUtilityBuilder(FixedKernelResidualGP(resolver, config=config)),
        min_admission_gain=1e-12,
    )
    return StreamingCoresetEvidence(StreamingCalibrationCoreset(planner))


def _strategy(
    name: str,
    capacity: int,
    config: FixedKernelGPConfig,
    *,
    acquisition: str = "gp_uncertainty",
) -> tuple[PolicySubprocess, Any]:
    if acquisition not in {"gp_uncertainty", "frozen"}:
        raise ValueError("unsupported engineering acquisition")
    base_policy = PolicySubprocess(acquisition, gp_config=config)
    if name == "fifo":
        return base_policy, PersistentFIFOEvidence(capacity)
    if name == "free_same_fifo":
        return base_policy, ReconstructedFIFOEvidence(capacity)
    if name == "full_history":
        return base_policy, ReconstructedFIFOEvidence(16)
    if name == "diversity":
        return base_policy, _DiversityEvidence(capacity)
    if name == "decision_coreset":
        return base_policy, _coreset(capacity, config)
    if name == "survival_coreset":
        if acquisition != "gp_uncertainty":
            raise ValueError("survival coreset requires GP uncertainty acquisition")
        return (
            PolicySubprocess(
                "survival_coreset",
                gp_config=config,
                proposal_size=16,
                num_fantasies=4,
                survival_weight=1.0,
            ),
            _coreset(capacity, config),
        )
    raise ValueError(f"unknown strategy: {name}")


def _run_one(
    *,
    name: str,
    pool_name: str,
    queries: tuple[MaterialQuery, ...],
    universe: tuple[CorrectedPhaseEntry, ...],
    initial_diagram: Any,
    capacity: int,
    budget: int,
    config: FixedKernelGPConfig,
    acquisition: str,
    log_path: Path,
    ppd: Any,
) -> dict[str, Any]:
    selected_ids = {query.query_id for query in queries}
    universe_by_id = {item.query_id: item for item in universe}
    records = [
        WBMOracleRecord(
            query_id=query.query_id,
            structure_hash=query.structure_hash,
            corrected_total_energy_ev=universe_by_id[query.query_id].corrected_total_energy_ev,
            corrected_formation_energy_ev_per_atom=float(
                ppd.get_form_energy_per_atom(universe_by_id[query.query_id].entry)
            ),
            source_record_locator=f"{SOURCE_VERSION}:{query.query_id}",
            observed_at=OBSERVED_TIME,
        )
        for query in queries
    ]
    phase_entries = {
        item.query_id: item.entry for item in universe if item.query_id in selected_ids
    }
    policy, evidence = _strategy(name, capacity, config, acquisition=acquisition)
    started = time.perf_counter()
    with AppendOnlyWBMEventLog(log_path) as event_log:
        result = SecureWBMRunner(
            queries=queries,
            vault=WBMOracleVault(records, phase_entries, source_version=SOURCE_VERSION),
            hull_state=CompositionHullState(
                initial_diagram,
                chemical_system=queries[0].hull_snapshot.chemical_system,
                source_version="MP-2022.10.28",
            ),
            policy=policy,
            evidence_access=evidence,
            oracle_universe=universe,
            event_log=event_log,
        ).run(oracle_budget=float(budget))
    final_active_ids = set(result.events[-1].active_witness_ids if result.events else ())
    query_by_id = {query.query_id: query for query in queries}
    cards = []
    for card_id in sorted(final_active_ids):
        query_id = card_id.removeprefix("wbm-card:")
        query = query_by_id[query_id]
        formation = float(ppd.get_form_energy_per_atom(universe_by_id[query_id].entry))
        cards.append(
            MaterialMemoryCard(
                card_id=card_id,
                material_id=query_id,
                structure_hash=query.structure_hash,
                identity=query.identity,
                composition=query.composition,
                embedding=query.embedding,
                protocol=query.protocol,
                provenance=SourceProvenance(
                    source_name="WBM",
                    source_version=SOURCE_VERSION,
                    record_locator=f"{SOURCE_VERSION}:{query_id}",
                    retrieved_at=OBSERVED_TIME,
                ),
                formation_energy_ev_per_atom=formation,
                base_predicted_formation_energy_ev_per_atom=(
                    query.base_predicted_formation_energy_ev_per_atom
                ),
                oracle_residual_ev_per_atom=(
                    formation - query.base_predicted_formation_energy_ev_per_atom
                ),
                hull_snapshot=query.hull_snapshot,
                observed_at=OBSERVED_TIME,
            )
        )
    remaining_queries = tuple(
        query for query in queries if query.query_id not in result.selected_query_ids
    )
    calibration = {
        "remaining_candidate_count": len(remaining_queries),
        "final_active_witness_count": len(cards),
        "residual_rmse_ev_per_atom": None,
        "residual_gaussian_nll": None,
        "initial_hull_stability_brier": None,
    }
    if remaining_queries:
        posterior = FixedKernelResidualGP(
            ProtocolCompatibilityResolver(), config=config
        ).fit(cards)
        prediction = posterior.predict(remaining_queries)
        truth = np.asarray(
            [
                float(ppd.get_form_energy_per_atom(universe_by_id[item.query_id].entry))
                - item.base_predicted_formation_energy_ev_per_atom
                for item in remaining_queries
            ]
        )
        mean = np.asarray(prediction.mean_ev_per_atom)
        std = np.maximum(np.asarray(prediction.std_ev_per_atom), 1e-9)
        labels = np.asarray(
            [
                item.hull_distance(
                    float(
                        ppd.get_form_energy_per_atom(
                            universe_by_id[item.query_id].entry
                        )
                    )
                )
                <= item.stability_threshold_ev_per_atom
                for item in remaining_queries
            ],
            dtype=float,
        )
        probabilities = np.asarray(prediction.stable_probability)
        calibration.update(
            {
                "residual_rmse_ev_per_atom": float(np.sqrt(np.mean((truth - mean) ** 2))),
                "residual_gaussian_nll": float(
                    np.mean(
                        0.5 * np.log(2 * np.pi * std**2)
                        + 0.5 * ((truth - mean) / std) ** 2
                    )
                ),
                "initial_hull_stability_brier": float(
                    np.mean((probabilities - labels) ** 2)
                ),
            }
        )
    return {
        "pool": pool_name,
        "strategy": name,
        "acquisition": acquisition,
        "budget": budget,
        "capacity": capacity,
        "wall_seconds": time.perf_counter() - started,
        "calibration": calibration,
        **result.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--license-manifest", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--parity-audit", type=Path, required=True)
    parser.add_argument("--soap-cache", type=Path, required=True)
    parser.add_argument("--cleaned-ids", type=Path, required=True)
    parser.add_argument("--raw-cse-root", type=Path, required=True)
    parser.add_argument("--ppd", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=4)
    parser.add_argument(
        "--acquisition", choices=("gp_uncertainty", "frozen"), default="gp_uncertainty"
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for value in vars(args).values():
        if isinstance(value, Path) and value.resolve().is_relative_to(repo_root):
            raise ValueError("real WBM inputs and outputs must remain outside Git")
    gate = json.loads(args.gate_audit.read_text(encoding="utf-8"))
    license_manifest = json.loads(args.license_manifest.read_text(encoding="utf-8"))
    if gate["execution_effect"]["engineering_runner_smoke"] is not True:
        raise ValueError("engineering P1/P1.5 gate does not authorize a runner smoke")
    if license_manifest["local_research_gate_passed"] is not True:
        raise ValueError("local research license gate has not passed")
    if args.output_dir.exists():
        raise FileExistsError("engineering pilot output directory is immutable")
    (args.output_dir / "ledgers").mkdir(parents=True)

    pool_payload = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    parity_payload = json.loads(args.parity_audit.read_text(encoding="utf-8"))
    predictions = {
        row["query_id"]: float(row["official_prediction_ev_per_atom"])
        for row in parity_payload["rows"]
    }
    with np.load(args.soap_cache, allow_pickle=False) as cache:
        soap_ids = [str(item) for item in cache["query_ids"]]
        soap_vectors = np.asarray(cache["vectors"], dtype=float)
    soap_by_id = dict(zip(soap_ids, soap_vectors, strict=True))

    _repair_historical_composition_interface()
    ppd = _load_ppd_read_only(args.ppd)
    systems = {
        tuple(sorted(pool["chemical_system"]))
        for pool in pool_payload["selection"]["pools"].values()
    }
    raw = _load_exact_system_universe(
        args.raw_cse_root, _read_cleaned_ids(args.cleaned_ids), systems
    )
    corrected = _correct_exact_system_universe(raw)
    config = FixedKernelGPConfig(
        kernel="matern52",
        length_scale=0.35,
        signal_std_ev_per_atom=0.08,
        noise_std_ev_per_atom=0.01,
        jitter=1e-10,
    )
    strategy_names = (
        (
            "fifo",
            "free_same_fifo",
            "full_history",
            "diversity",
            "decision_coreset",
            "survival_coreset",
        )
        if args.acquisition == "gp_uncertainty"
        else ("fifo", "full_history", "diversity", "decision_coreset")
    )
    runs = []
    for pool_name, pool in sorted(pool_payload["selection"]["pools"].items()):
        system = tuple(sorted(pool["chemical_system"]))
        universe_entries = corrected[system]
        by_id = {str(entry.entry_id): entry for entry in universe_entries}
        selected_ids = [item["query_id"] for item in pool["candidates"]]
        compact = exact_gram_embedding(
            np.asarray([soap_by_id[query_id] for query_id in selected_ids])
        )
        embedding_by_id = {
            query_id: tuple(float(value) for value in compact[index])
            for index, query_id in enumerate(selected_ids)
        }
        initial_diagram = __import__(
            "pymatgen.analysis.phase_diagram", fromlist=["PhaseDiagram"]
        ).PhaseDiagram(_initial_entries(ppd, system))
        queries = _build_queries(
            pool=pool,
            pool_name=pool_name,
            entries_by_id=by_id,
            predictions=predictions,
            embedding_by_id=embedding_by_id,
            initial_diagram=initial_diagram,
        )
        universe = tuple(
            CorrectedPhaseEntry(
                query_id=str(entry.entry_id),
                corrected_total_energy_ev=float(entry.energy),
                entry=entry,
            )
            for entry in universe_entries
        )
        for strategy_name in strategy_names:
            runs.append(
                _run_one(
                    name=strategy_name,
                    pool_name=pool_name,
                    queries=queries,
                    universe=universe,
                    initial_diagram=initial_diagram,
                    capacity=args.capacity,
                    budget=args.budget,
                    config=config,
                    acquisition=args.acquisition,
                    log_path=(
                        args.output_dir
                        / "ledgers"
                        / f"{pool_name}-{strategy_name}.jsonl"
                    ),
                    ppd=ppd,
                )
            )
    by_pool_strategy = {(run["pool"], run["strategy"]): run for run in runs}
    parity_mismatches = []
    if "free_same_fifo" in strategy_names:
        for pool_name in pool_payload["selection"]["pools"]:
            persistent = by_pool_strategy[(pool_name, "fifo")]
            reconstructed = by_pool_strategy[(pool_name, "free_same_fifo")]
            if persistent["selected_query_ids"] != reconstructed["selected_query_ids"]:
                parity_mismatches.append(pool_name)
    matched_trace_mismatches = []
    if args.acquisition == "frozen":
        for pool_name in pool_payload["selection"]["pools"]:
            reference = by_pool_strategy[(pool_name, "fifo")]["selected_query_ids"]
            for strategy_name in strategy_names:
                if by_pool_strategy[(pool_name, strategy_name)]["selected_query_ids"] != reference:
                    matched_trace_mismatches.append(f"{pool_name}:{strategy_name}")
    aggregates = {}
    for strategy_name in strategy_names:
        selected = [run for run in runs if run["strategy"] == strategy_name]
        aggregates[strategy_name] = {
            "pool_count": len(selected),
            "oracle_final_true_discoveries": sum(
                run["oracle_final_true_discoveries"] for run in selected
            ),
            "causal_discoveries": sum(run["causal_discoveries"] for run in selected),
            "benchmark_false_confirmations": sum(
                run["benchmark_false_confirmations"] for run in selected
            ),
            "wall_seconds": sum(run["wall_seconds"] for run in selected),
            "mean_remaining_residual_rmse_ev_per_atom": float(
                np.mean(
                    [run["calibration"]["residual_rmse_ev_per_atom"] for run in selected]
                )
            ),
            "mean_remaining_residual_gaussian_nll": float(
                np.mean(
                    [run["calibration"]["residual_gaussian_nll"] for run in selected]
                )
            ),
            "mean_remaining_initial_hull_stability_brier": float(
                np.mean(
                    [run["calibration"]["initial_hull_stability_brier"] for run in selected]
                )
            ),
        }
    report = {
        "schema_version": 1,
        "scope": "small_wbm_calibration_coreset_engineering_pilot_not_claim_grade",
        "budget": args.budget,
        "capacity": args.capacity,
        "acquisition": args.acquisition,
        "pool_count": len(pool_payload["selection"]["pools"]),
        "embedding": "exact finite-pool SOAP Gram factorization",
        "free_same_fifo_exact_action_parity": (
            not parity_mismatches if "free_same_fifo" in strategy_names else None
        ),
        "free_same_fifo_mismatch_pools": parity_mismatches,
        "matched_frozen_acquisition_action_parity": (
            not matched_trace_mismatches if args.acquisition == "frozen" else None
        ),
        "matched_frozen_acquisition_mismatches": matched_trace_mismatches,
        "aggregates": aggregates,
        "runs": runs,
        "interpretation_guardrails": [
            "engineering hyperparameters were not claim-grade calibrated",
            "prototype and WBM-MP overlap clustering remain pending",
            "no paper-level GO can be inferred from this run",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if "free_same_fifo" in strategy_names:
        print(f"free_same_fifo_exact_action_parity={not parity_mismatches}")
    if args.acquisition == "frozen":
        print(f"matched_frozen_acquisition_action_parity={not matched_trace_mismatches}")
    for name, aggregate in aggregates.items():
        print(
            f"{name}: oracle_final={aggregate['oracle_final_true_discoveries']} "
            f"causal={aggregate['causal_discoveries']} "
            f"rmse={aggregate['mean_remaining_residual_rmse_ev_per_atom']:.4f} "
            f"seconds={aggregate['wall_seconds']:.3f}"
        )
    print(f"summary={args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
