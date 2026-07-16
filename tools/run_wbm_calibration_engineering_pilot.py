"""Run a small, gate-checked WBM calibration-coreset engineering pilot.

This is not the claim-grade policy matrix.  It consumes the fixed historical
replay, frozen 8x16 pools and full exact-system oracle universes, runs behind
the sole secure WBM runner, and writes every ledger and result outside Git.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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
    GPVarianceOneSwapMemory,
    HullSnapshot,
    JointPosteriorRiskOneSwapPlanner,
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
    compare_facility_and_joint_objectives,
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


class _FullHistoryEvidence:
    """Expose every revealed witness; this baseline has no finite K label."""

    capacity = 2**31 - 1

    def active(self, archive: tuple[Any, ...]) -> tuple[Any, ...]:
        return archive

    def admit(self, card: Any, query_pool: tuple[MaterialQuery, ...]) -> None:
        del card, query_pool


class _GPVarianceEvidence:
    def __init__(self, capacity: int, config: FixedKernelGPConfig) -> None:
        self.capacity = capacity
        self.memory = GPVarianceOneSwapMemory(
            capacity,
            FixedKernelResidualGP(ProtocolCompatibilityResolver(), config=config),
        )

    def active(self, archive: tuple[Any, ...]) -> tuple[MaterialMemoryCard, ...]:
        del archive
        return self.memory.cards()

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: tuple[MaterialQuery, ...],
    ) -> None:
        self.memory.admit(card, query_pool)


class _ObjectiveFidelityEvidence:
    """Run one selector while auditing both objectives on every same neighborhood."""

    def __init__(
        self,
        capacity: int,
        config: FixedKernelGPConfig,
        *,
        selector: str,
    ) -> None:
        if selector not in {"facility", "joint_risk"}:
            raise ValueError("unknown objective-fidelity selector")
        self.capacity = capacity
        self.selector = selector
        resolver = ProtocolCompatibilityResolver()
        builder = CalibrationUtilityBuilder(
            FixedKernelResidualGP(resolver, config=config)
        )
        self.facility = FacilityLocationCoresetPlanner(
            capacity, builder, min_admission_gain=1e-12
        )
        self.joint = JointPosteriorRiskOneSwapPlanner(
            capacity, builder, min_risk_improvement=1e-12
        )
        self._cards: dict[str, MaterialMemoryCard] = {}
        self.diagnostics: list[dict[str, Any]] = []

    def active(self, archive: tuple[Any, ...]) -> tuple[MaterialMemoryCard, ...]:
        del archive
        return tuple(self._cards.values())

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: tuple[MaterialQuery, ...],
    ) -> object:
        current = self.active(())
        diagnostic = compare_facility_and_joint_objectives(
            current,
            card,
            query_pool,
            self.facility,
            self.joint,
        )
        self.diagnostics.append(
            {
                "admission_index": len(self.diagnostics) + 1,
                **diagnostic.model_dump(mode="json"),
            }
        )
        selected_ids = (
            diagnostic.facility_selected_card_ids
            if self.selector == "facility"
            else diagnostic.joint_risk_selected_card_ids
        )
        if card.card_id in selected_ids:
            candidates = {**self._cards, card.card_id: card}
            self._cards = {card_id: candidates[card_id] for card_id in selected_ids}
        return diagnostic


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
    """Construct the coreset used by the paused survival diagnostic only."""

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
        return base_policy, _FullHistoryEvidence()
    if name == "diversity":
        return base_policy, _DiversityEvidence(capacity)
    if name == "gp_variance_one_swap":
        return base_policy, _GPVarianceEvidence(capacity, config)
    if name == "decision_coreset":
        return base_policy, _ObjectiveFidelityEvidence(
            capacity, config, selector="facility"
        )
    if name == "joint_posterior_risk_one_swap":
        return base_policy, _ObjectiveFidelityEvidence(
            capacity, config, selector="joint_risk"
        )
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


def _material_card(
    query: MaterialQuery,
    formation_energy_ev_per_atom: float,
) -> MaterialMemoryCard:
    return MaterialMemoryCard(
        card_id=f"wbm-card:{query.query_id}",
        material_id=query.query_id,
        structure_hash=query.structure_hash,
        identity=query.identity,
        composition=query.composition,
        embedding=query.embedding,
        protocol=query.protocol,
        provenance=SourceProvenance(
            source_name="WBM",
            source_version=SOURCE_VERSION,
            record_locator=f"{SOURCE_VERSION}:{query.query_id}",
            retrieved_at=OBSERVED_TIME,
        ),
        formation_energy_ev_per_atom=formation_energy_ev_per_atom,
        base_predicted_formation_energy_ev_per_atom=(
            query.base_predicted_formation_energy_ev_per_atom
        ),
        oracle_residual_ev_per_atom=(
            formation_energy_ev_per_atom
            - query.base_predicted_formation_energy_ev_per_atom
        ),
        hull_snapshot=query.hull_snapshot,
        observed_at=OBSERVED_TIME,
    )


def _oracle_weighted_decision_loss(
    builder: CalibrationUtilityBuilder,
    queries: tuple[MaterialQuery, ...],
    cards: tuple[MaterialMemoryCard, ...],
    oracle_formation_by_id: dict[str, float],
) -> float:
    posterior = builder.posterior_template.clone_unfit().fit(cards)
    probabilities = posterior.predict(queries).stable_probability
    weights = builder.boundary_weights(queries)
    stable_cutoff = builder.false_stable_cost / (
        builder.false_stable_cost + builder.false_unstable_cost
    )
    total = 0.0
    for query, probability in zip(queries, probabilities, strict=True):
        predicted_stable = probability >= stable_cutoff
        actual_stable = (
            query.hull_distance(oracle_formation_by_id[query.query_id])
            <= query.stability_threshold_ev_per_atom
        )
        if predicted_stable and not actual_stable:
            total += weights[query.query_id] * builder.false_stable_cost
        elif not predicted_stable and actual_stable:
            total += weights[query.query_id] * builder.false_unstable_cost
    return float(total)


def _offline_subset_audit(
    *,
    history_cards: tuple[MaterialMemoryCard, ...],
    active_cards: tuple[MaterialMemoryCard, ...],
    remaining_queries: tuple[MaterialQuery, ...],
    oracle_formation_by_id: dict[str, float],
    capacity: int,
    config: FixedKernelGPConfig,
) -> dict[str, Any] | None:
    if not remaining_queries or capacity < 1 or len(history_cards) < capacity:
        return None
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(ProtocolCompatibilityResolver(), config=config)
    )
    rows = []
    for subset in itertools.combinations(history_cards, capacity):
        rows.append(
            {
                "selected_card_ids": sorted(card.card_id for card in subset),
                "observable_weighted_joint_risk": builder.weighted_decision_risk(
                    remaining_queries, subset
                ),
                "offline_oracle_weighted_decision_loss": (
                    _oracle_weighted_decision_loss(
                        builder,
                        remaining_queries,
                        subset,
                        oracle_formation_by_id,
                    )
                ),
            }
        )
    observable_best = min(
        rows,
        key=lambda row: (
            row["observable_weighted_joint_risk"],
            row["selected_card_ids"],
        ),
    )
    oracle_best = min(
        rows,
        key=lambda row: (
            row["offline_oracle_weighted_decision_loss"],
            row["selected_card_ids"],
        ),
    )
    active_observable = builder.weighted_decision_risk(
        remaining_queries, active_cards
    )
    active_oracle = _oracle_weighted_decision_loss(
        builder,
        remaining_queries,
        active_cards,
        oracle_formation_by_id,
    )
    equal_capacity = len(active_cards) == capacity
    return {
        "history_size": len(history_cards),
        "capacity": capacity,
        "enumerated_exact_capacity_subset_count": len(rows),
        "active_card_ids": sorted(card.card_id for card in active_cards),
        "active_size_matches_capacity": equal_capacity,
        "active_observable_weighted_joint_risk": active_observable,
        "active_offline_oracle_weighted_decision_loss": active_oracle,
        "observable_optimum": observable_best,
        "offline_oracle_optimum": oracle_best,
        "active_observable_regret": (
            max(
                0.0,
                active_observable
                - float(observable_best["observable_weighted_joint_risk"]),
            )
            if equal_capacity
            else None
        ),
        "active_offline_oracle_loss_regret": (
            max(
                0.0,
                active_oracle
                - float(oracle_best["offline_oracle_weighted_decision_loss"]),
            )
            if equal_capacity
            else None
        ),
    }


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
    query_by_id = {query.query_id: query for query in queries}
    oracle_formation_by_id = {
        query_id: float(ppd.get_form_energy_per_atom(item.entry))
        for query_id, item in universe_by_id.items()
        if query_id in query_by_id
    }
    history_cards = tuple(
        _material_card(query_by_id[query_id], oracle_formation_by_id[query_id])
        for query_id in result.selected_query_ids
    )
    final_active_ids = set(
        result.events[-1].active_witness_ids if result.events else ()
    )
    cards = tuple(
        card for card in history_cards if card.card_id in final_active_ids
    )
    final_hull = CompositionHullState(
        initial_diagram,
        chemical_system=queries[0].hull_snapshot.chemical_system,
        source_version="MP-2022.10.28",
    )
    for query_id in result.selected_query_ids:
        final_hull.add_revealed(universe_by_id[query_id])
    remaining_queries = tuple(
        query.model_copy(
            update={
                "hull_snapshot": final_hull.snapshot(
                    query,
                    round_index=len(result.selected_query_ids) + 1,
                    built_at=OBSERVED_TIME,
                ),
                "as_of": OBSERVED_TIME,
            }
        )
        for query in queries
        if query.query_id not in result.selected_query_ids
    )
    calibration = {
        "remaining_candidate_count": len(remaining_queries),
        "final_active_witness_count": len(cards),
        "residual_rmse_ev_per_atom": None,
        "residual_gaussian_nll": None,
        "causal_hull_stability_brier": None,
        "causal_hull_stability_log_loss": None,
        "asymmetric_weighted_decision_loss": None,
        "observable_weighted_joint_risk": None,
        "gaussian_crps": None,
        "interval_90_coverage": None,
        "interval_90_mean_width_ev_per_atom": None,
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
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        z = (truth - mean) / std
        normal_pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
        normal_cdf = 0.5 * (
            1
            + np.asarray(
                [math.erf(float(value) / np.sqrt(2)) for value in z]
            )
        )
        builder = CalibrationUtilityBuilder(
            FixedKernelResidualGP(ProtocolCompatibilityResolver(), config=config)
        )
        calibration.update(
            {
                "residual_rmse_ev_per_atom": float(
                    np.sqrt(np.mean((truth - mean) ** 2))
                ),
                "residual_gaussian_nll": float(
                    np.mean(
                        0.5 * np.log(2 * np.pi * std**2)
                        + 0.5 * ((truth - mean) / std) ** 2
                    )
                ),
                "causal_hull_stability_brier": float(
                    np.mean((probabilities - labels) ** 2)
                ),
                "causal_hull_stability_log_loss": float(
                    -np.mean(
                        labels * np.log(clipped)
                        + (1 - labels) * np.log(1 - clipped)
                    )
                ),
                "asymmetric_weighted_decision_loss": _oracle_weighted_decision_loss(
                    builder,
                    remaining_queries,
                    cards,
                    oracle_formation_by_id,
                ),
                "observable_weighted_joint_risk": builder.weighted_decision_risk(
                    remaining_queries,
                    cards,
                ),
                "gaussian_crps": float(
                    np.mean(
                        std
                        * (
                            z * (2 * normal_cdf - 1)
                            + 2 * normal_pdf
                            - 1 / np.sqrt(np.pi)
                        )
                    )
                ),
                "interval_90_coverage": float(
                    np.mean(np.abs(truth - mean) <= 1.6448536269514722 * std)
                ),
                "interval_90_mean_width_ev_per_atom": float(
                    np.mean(2 * 1.6448536269514722 * std)
                ),
            }
        )
    subset_audit = (
        _offline_subset_audit(
            history_cards=history_cards,
            active_cards=cards,
            remaining_queries=remaining_queries,
            oracle_formation_by_id=oracle_formation_by_id,
            capacity=capacity,
            config=config,
        )
        if name == "decision_coreset"
        else None
    )
    return {
        "pool": pool_name,
        "strategy": name,
        "acquisition": acquisition,
        "budget": budget,
        "capacity": capacity,
        "wall_seconds": time.perf_counter() - started,
        "calibration": calibration,
        "objective_fidelity_rounds": getattr(evidence, "diagnostics", None),
        "offline_subset_audit": subset_audit,
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
    parser.add_argument(
        "--include-paused-survival",
        action="store_true",
        help="run the frozen negative survival diagnostic without tuning it",
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
    strategy_names = [
        "fifo",
        "full_history",
        "diversity",
        "gp_variance_one_swap",
        "decision_coreset",
        "joint_posterior_risk_one_swap",
    ]
    if args.acquisition == "gp_uncertainty":
        strategy_names.insert(1, "free_same_fifo")
    if args.include_paused_survival:
        strategy_names.append("survival_coreset")
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
    for pool_name in pool_payload["selection"]["pools"]:
        reference_run = by_pool_strategy[(pool_name, "decision_coreset")]
        reference_audit = reference_run["offline_subset_audit"]
        if reference_audit is None:
            continue
        for strategy_name in strategy_names:
            run = by_pool_strategy[(pool_name, strategy_name)]
            if run["selected_query_ids"] != reference_run["selected_query_ids"]:
                continue
            active_ids = (
                run["events"][-1]["active_witness_ids"] if run["events"] else []
            )
            equal_capacity = len(active_ids) == args.capacity
            observable = run["calibration"]["observable_weighted_joint_risk"]
            oracle_loss = run["calibration"]["asymmetric_weighted_decision_loss"]
            run["offline_subset_audit"] = {
                "history_size": reference_audit["history_size"],
                "capacity": args.capacity,
                "enumerated_exact_capacity_subset_count": reference_audit[
                    "enumerated_exact_capacity_subset_count"
                ],
                "active_card_ids": sorted(active_ids),
                "active_size_matches_capacity": equal_capacity,
                "active_observable_weighted_joint_risk": observable,
                "active_offline_oracle_weighted_decision_loss": oracle_loss,
                "observable_optimum": reference_audit["observable_optimum"],
                "offline_oracle_optimum": reference_audit["offline_oracle_optimum"],
                "active_observable_regret": (
                    max(
                        0.0,
                        observable
                        - float(
                            reference_audit["observable_optimum"][
                                "observable_weighted_joint_risk"
                            ]
                        ),
                    )
                    if equal_capacity
                    else None
                ),
                "active_offline_oracle_loss_regret": (
                    max(
                        0.0,
                        oracle_loss
                        - float(
                            reference_audit["offline_oracle_optimum"][
                                "offline_oracle_weighted_decision_loss"
                            ]
                        ),
                    )
                    if equal_capacity
                    else None
                ),
                "reused_from_matched_action_reference": strategy_name
                != "decision_coreset",
            }
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
    metric_names = (
        "residual_rmse_ev_per_atom",
        "residual_gaussian_nll",
        "causal_hull_stability_brier",
        "causal_hull_stability_log_loss",
        "asymmetric_weighted_decision_loss",
        "observable_weighted_joint_risk",
        "gaussian_crps",
        "interval_90_coverage",
        "interval_90_mean_width_ev_per_atom",
    )
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
            **{
                f"mean_remaining_{metric}": float(
                    np.mean([run["calibration"][metric] for run in selected])
                )
                for metric in metric_names
            },
        }
        eligible_audits = [
            run["offline_subset_audit"]
            for run in selected
            if run["offline_subset_audit"] is not None
            and run["offline_subset_audit"]["active_size_matches_capacity"]
        ]
        aggregates[strategy_name]["mean_offline_observable_subset_regret"] = (
            float(
                np.mean(
                    [audit["active_observable_regret"] for audit in eligible_audits]
                )
            )
            if eligible_audits
            else None
        )
        aggregates[strategy_name]["mean_offline_oracle_loss_subset_regret"] = (
            float(
                np.mean(
                    [
                        audit["active_offline_oracle_loss_regret"]
                        for audit in eligible_audits
                    ]
                )
            )
            if eligible_audits
            else None
        )
    objective_fidelity = {}
    for strategy_name in (
        "decision_coreset",
        "joint_posterior_risk_one_swap",
    ):
        records = [
            record
            for run in runs
            if run["strategy"] == strategy_name
            for record in (run["objective_fidelity_rounds"] or [])
        ]
        correlations = [
            record["spearman_facility_vs_negative_joint_risk"]
            for record in records
            if record["spearman_facility_vs_negative_joint_risk"] is not None
        ]
        saturated = [
            record
            for record in records
            if len(record["candidates"]) == args.capacity + 1
        ]
        saturated_correlations = [
            record["spearman_facility_vs_negative_joint_risk"]
            for record in saturated
            if record["spearman_facility_vs_negative_joint_risk"] is not None
        ]
        objective_fidelity[strategy_name] = {
            "round_count": len(records),
            "comparable_spearman_round_count": len(correlations),
            "mean_spearman_facility_vs_negative_joint_risk": (
                float(np.mean(correlations)) if correlations else None
            ),
            "selection_agreement_rate": (
                float(np.mean([record["selections_agree"] for record in records]))
                if records
                else None
            ),
            "mean_facility_joint_risk_regret": (
                float(
                    np.mean(
                        [record["facility_joint_risk_regret"] for record in records]
                    )
                )
                if records
                else None
            ),
            "positive_regret_round_count": sum(
                record["facility_joint_risk_regret"] > 1e-12 for record in records
            ),
            "saturated_round_count": len(saturated),
            "saturated_selection_agreement_rate": (
                float(
                    np.mean([record["selections_agree"] for record in saturated])
                )
                if saturated
                else None
            ),
            "saturated_mean_spearman": (
                float(np.mean(saturated_correlations))
                if saturated_correlations
                else None
            ),
            "saturated_positive_regret_round_count": sum(
                record["facility_joint_risk_regret"] > 1e-12
                for record in saturated
            ),
            "saturated_mean_facility_joint_risk_regret": (
                float(
                    np.mean(
                        [
                            record["facility_joint_risk_regret"]
                            for record in saturated
                        ]
                    )
                )
                if saturated
                else None
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
        "objective_fidelity": objective_fidelity,
        "paused_survival_included": args.include_paused_survival,
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
