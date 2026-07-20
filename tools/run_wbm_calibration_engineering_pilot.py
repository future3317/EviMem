"""Run a small, gate-checked WBM calibration-coreset engineering pilot.

This is not the claim-grade policy matrix.  It consumes the fixed historical
replay, frozen 8x16 pools and full exact-system oracle universes, runs behind
the sole secure WBM runner, and writes every ledger and result outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import psutil

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
SRC_ROOT = TOOLS_DIR.parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from audit_wbm_official_artifacts import _load_ppd_read_only  # noqa: E402
from audit_wbm_p1_p15 import (  # noqa: E402
    _load_exact_system_universe,
    _read_cleaned_ids,
)
from build_wbm_candidate_parity_audit import (  # noqa: E402
    _repair_historical_composition_interface,
)

from matmem import (  # noqa: E402
    AppendOnlyWBMEventLog,
    CalibrationUtilityBuilder,
    CompositionHullState,
    CorrectedPhaseEntry,
    DiversityBoundedMemory,
    ExactArchivePosteriorProjectionPlanner,
    FacilityLocationCoresetPlanner,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    GPVarianceOneSwapMemory,
    HullSnapshot,
    JointPosteriorRiskOneSwapPlanner,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    OracleEnergySource,
    PersistentFIFOEvidence,
    PolicySubprocess,
    PosteriorProjectionOneSwapPlanner,
    PrequentialCausalEvaluator,
    PrequentialRoundMetrics,
    ProperPosteriorDivergence,
    ProtocolCompatibilityResolver,
    ReconstructedFIFOEvidence,
    SecureWBMRunner,
    SourceProvenance,
    StreamingCalibrationCoreset,
    StreamingCoresetEvidence,
    StreamingPosteriorProjectionCoreset,
    StructureArtifactIdentity,
    StructureStage,
    WBMOracleRecord,
    WBMOracleVault,
    WBMStructureSourceField,
    aggregate_prequential_prefix,
    compare_facility_and_joint_objectives,
)
from matmem.protocols import ProtocolCertificate  # noqa: E402

INITIAL_TIME = datetime(2023, 2, 7, tzinfo=UTC)
OBSERVED_TIME = INITIAL_TIME + timedelta(days=1)
SOURCE_VERSION = "fixed-historical-pipeline-WBM-2021.68-MP2020-pymatgen-2023.5.10"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_gate_bindings(
    gate: dict[str, Any],
    *,
    pool_manifest: Path,
    parity_audit: Path,
    soap_cache: Path,
    cleaned_ids: Path,
    ppd: Path,
    compute_relevance_only: bool = False,
) -> None:
    """Bind the runner to the exact audited, query-time-observable inputs."""

    p1 = gate.get("p1", {})
    if compute_relevance_only:
        if p1.get("engineering_p1_passed") is not True:
            raise ValueError("compute relevance requires the engineering P1 provenance gate")
    elif gate.get("execution_effect", {}).get("engineering_runner_smoke") is not True:
        raise ValueError("engineering P1/P1.5 gate does not authorize a runner smoke")
    if p1.get("soap_structure_stage") != StructureStage.INITIAL.value:
        raise ValueError("runner requires an initial-structure-only SOAP gate")
    if p1.get("soap_causal_available_before_query") is not True:
        raise ValueError("runner SOAP features must be observable before oracle reveal")
    if p1.get("soap_structure_source_field") != WBMStructureSourceField.ORIGINAL.value:
        raise ValueError("runner requires the official WBM org initial-structure field")
    expected = {
        "pool_manifest_sha256": _sha256(pool_manifest),
        "parity_audit_sha256": _sha256(parity_audit),
        "soap_cache_sha256": _sha256(soap_cache),
        "cleaned_ids_sha256": _sha256(cleaned_ids),
        "ppd_sha256": _sha256(ppd),
    }
    mismatches = {
        name: {"gate": gate.get(name), "actual": value}
        for name, value in expected.items()
        if gate.get(name) != value
    }
    if mismatches:
        raise ValueError(f"runner inputs differ from the audited gate: {mismatches}")


def _matched_action_reference_strategy(strategy_names: list[str]) -> str:
    if not strategy_names:
        raise ValueError("matched-action audit requires at least one strategy")
    return "fifo" if "fifo" in strategy_names else strategy_names[0]


def _load_calibration_freeze(
    path: Path,
    registered_config_path: Path,
) -> tuple[FixedKernelGPConfig, str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "wbm-gp-and-noninferiority-calibration-freeze-v1":
        raise ValueError("unsupported calibration-freeze manifest schema")
    if payload.get("scope") != "disjoint_calibration_only_no_evaluation_results_accessed":
        raise ValueError("calibration freeze is not evaluation-disjoint")
    if payload.get("evaluation_results_accessed") is not False:
        raise ValueError("calibration freeze accessed evaluation results")
    if payload.get("gp_parameter_status") != "frozen_on_disjoint_calibration_systems_v1":
        raise ValueError("GP parameters are not frozen on disjoint calibration systems")
    if payload.get("full_history_prequential_sanity", {}).get("passed") is not True:
        raise ValueError("calibration full-history sanity did not pass")
    systems = payload.get("calibration_system_ids")
    strata = payload.get("calibration_strata")
    if not isinstance(systems, list) or len(systems) != 8 or len(set(systems)) != 8:
        raise ValueError("calibration freeze must identify eight unique systems")
    if not isinstance(strata, dict) or set(strata) != set(systems):
        raise ValueError("calibration freeze strata do not match its systems")
    if list(strata.values()).count("binary") != 4 or list(strata.values()).count("ternary") != 4:
        raise ValueError("calibration freeze requires four binary and four ternary systems")
    gp = payload.get("gp_config")
    if not isinstance(gp, dict):
        raise ValueError("calibration freeze has no GP configuration")
    encoded = json.dumps(gp, sort_keys=True, separators=(",", ":"))
    gp_sha = "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
    if gp_sha != payload.get("gp_config_sha256"):
        raise ValueError("calibration-freeze GP configuration SHA mismatch")
    if _sha256(registered_config_path) != payload.get("config_sha256"):
        raise ValueError("registered configuration SHA does not match calibration freeze")
    registered = json.loads(registered_config_path.read_text(encoding="utf-8"))
    if registered.get("posterior") != gp:
        raise ValueError("registered posterior differs from calibration-freeze GP configuration")
    config = FixedKernelGPConfig(
        kernel=str(gp["kernel"]),
        length_scale=float(gp["length_scale"]),
        signal_std_ev_per_atom=float(gp["signal_std_ev_per_atom"]),
        noise_std_ev_per_atom=float(gp["noise_std_ev_per_atom"]),
        jitter=float(gp["jitter"]),
    )
    return config, _sha256(path), payload


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


class _PosteriorProjectionEvidence:
    """Online P3C with evaluator-visible archive-exact diagnostics."""

    def __init__(
        self,
        capacity: int,
        config: FixedKernelGPConfig,
        divergence_kind: str,
        *,
        max_decision_regret: float | None = None,
        max_log_divergence: float | None = None,
    ) -> None:
        self.capacity = capacity
        divergence = ProperPosteriorDivergence(kind=divergence_kind)  # type: ignore[arg-type]
        posterior = FixedKernelResidualGP(ProtocolCompatibilityResolver(), config=config)
        self.memory = StreamingPosteriorProjectionCoreset(
            PosteriorProjectionOneSwapPlanner(
                capacity,
                posterior,
                divergence,
                max_decision_regret=max_decision_regret,
                max_log_divergence=max_log_divergence,
            )
        )
        self.archive_planner = ExactArchivePosteriorProjectionPlanner(
            capacity,
            posterior,
            divergence,
            max_decision_regret=max_decision_regret,
            max_log_divergence=max_log_divergence,
        )
        self._archive: tuple[MaterialMemoryCard, ...] = ()
        self.diagnostics: list[dict[str, Any]] = []
        self._pending_factorial: dict[str, Any] | None = None
        self.last_online_retention_seconds = 0.0

    def active(self, archive: tuple[MaterialMemoryCard, ...]) -> tuple[MaterialMemoryCard, ...]:
        self._archive = archive
        return self.memory.cards()

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: tuple[MaterialQuery, ...],
    ) -> object:
        previous_cards = self.memory.cards()
        previous_ids = {item.card_id for item in previous_cards}
        union_cards = (*previous_cards, card)
        online_started = time.perf_counter()
        online = self.memory.admit(card, query_pool)
        self.last_online_retention_seconds = time.perf_counter() - online_started
        updated_archive = (*self._archive, card)
        archive_diagnostic_started = time.perf_counter()
        archive = self.archive_planner.select(
            updated_archive,
            query_pool,
            current_cards=previous_cards,
        )
        archive_candidate_sets = tuple(item.selected_card_ids for item in archive.candidates)
        online_candidate_sets = tuple(item.selected_card_ids for item in online.candidates)
        cards_by_id = {item.card_id: item for item in updated_archive}
        union_archive = self.archive_planner.scorer.score(
            reference_cards=union_cards,
            candidate_sets=archive_candidate_sets,
            cards_by_id=cards_by_id,
            queries=query_pool,
            current_ids=tuple(item.card_id for item in previous_cards),
        )
        archive_online = self.archive_planner.scorer.score(
            reference_cards=updated_archive,
            candidate_sets=online_candidate_sets,
            cards_by_id=cards_by_id,
            queries=query_pool,
            current_ids=tuple(item.card_id for item in previous_cards),
        )
        archive_diagnostic_seconds = time.perf_counter() - archive_diagnostic_started
        online_ids = set(online.selected_card_ids)
        online_under_archive = next(
            item for item in archive.candidates if set(item.selected_card_ids) == online_ids
        )
        selected_residuals = [
            item.oracle_residual_ev_per_atom
            for item in updated_archive
            if item.card_id in online_ids
        ]
        archive_residuals = [item.oracle_residual_ev_per_atom for item in updated_archive]
        reactivated = set(archive.selected_card_ids) - previous_ids - {card.card_id}
        factorial = {
            "union_reference__online_search": online,
            "union_reference__archive_search": union_archive,
            "archive_reference__online_search": archive_online,
            "archive_reference__archive_search": archive,
        }

        def selection_summary(selection: Any) -> dict[str, Any]:
            return {
                "reference_card_ids": selection.reference_card_ids,
                "selected_card_ids": selection.selected_card_ids,
                "selected_proper_divergence": selection.selected_proper_divergence,
                "selected_decision_regret": selection.selected_decision_regret,
                "selected_log_divergence": selection.selected_log_divergence,
                "candidate_count": len(selection.candidates),
                "used_constraint_fallback": selection.used_constraint_fallback,
            }

        self.diagnostics.append(
            {
                "admission_index": len(self.diagnostics) + 1,
                "online": online.model_dump(mode="json"),
                "archive_exact_selected_card_ids": archive.selected_card_ids,
                "archive_exact_candidate_count": len(archive.candidates),
                "online_divergence_under_archive_reference": (
                    online_under_archive.proper_divergence
                ),
                "archive_exact_divergence": archive.selected_proper_divergence,
                "online_vs_archive_optimization_gap": max(
                    0.0,
                    online_under_archive.proper_divergence - archive.selected_proper_divergence,
                ),
                "archive_reactivated_old_card_ids": tuple(sorted(reactivated)),
                "retained_minus_archive_residual_mean": (
                    float(np.mean(selected_residuals) - np.mean(archive_residuals))
                    if selected_residuals
                    else None
                ),
                "reference_search_factorial": {
                    name: selection_summary(selection) for name, selection in factorial.items()
                },
                "timing": {
                    "union_reference_fit_seconds": (
                        online.reference_fit_seconds + online.reference_prediction_seconds
                    ),
                    "online_candidate_projection_seconds": (
                        online.candidate_enumeration_seconds + online.candidate_projection_seconds
                    ),
                    "archive_reference_fit_seconds": (
                        archive.reference_fit_seconds + archive.reference_prediction_seconds
                    ),
                    "archive_subset_enumeration_seconds": (
                        archive.candidate_enumeration_seconds + archive.candidate_projection_seconds
                    ),
                    "cross_factorial_diagnostic_seconds": (archive_diagnostic_seconds),
                    "online_retention_seconds": self.last_online_retention_seconds,
                },
            }
        )
        self._pending_factorial = {
            "union_reference_cards": union_cards,
            "archive_reference_cards": updated_archive,
            "union_cards": union_cards,
            "selected_cards": {
                name: tuple(cards_by_id[card_id] for card_id in selection.selected_card_ids)
                for name, selection in factorial.items()
            },
            "retained_card_ids": online.selected_card_ids,
        }
        return online

    def consume_factorial_context(self) -> dict[str, Any]:
        if self._pending_factorial is None:
            raise RuntimeError("no posterior-projection factorial context is pending")
        context = self._pending_factorial
        self._pending_factorial = None
        return context


class _PrequentialEvidence:
    """Evaluator-side instrumentation around an oracle-blind evidence strategy."""

    def __init__(
        self,
        delegate: Any,
        evaluator: PrequentialCausalEvaluator,
        *,
        record_projection_diagnostics: bool = False,
    ) -> None:
        self.delegate = delegate
        self.evaluator = evaluator
        self.capacity = delegate.capacity
        self.rounds: list[PrequentialRoundMetrics] = []
        self.posterior_snapshots: list[dict[str, Any]] = []
        self.record_projection_diagnostics = record_projection_diagnostics
        self._archive: tuple[MaterialMemoryCard, ...] = ()
        self._process = psutil.Process()
        self._round_started: float | None = None
        self._awaiting_post_admission_active = False

    def active(self, archive: tuple[MaterialMemoryCard, ...]) -> tuple[MaterialMemoryCard, ...]:
        self._archive = archive
        active = self.delegate.active(archive)
        if self._awaiting_post_admission_active:
            if self._round_started is None or not self.rounds:
                raise RuntimeError("prequential round timing state is inconsistent")
            self.rounds[-1] = self.rounds[-1].model_copy(
                update={"round_pipeline_seconds": time.perf_counter() - self._round_started}
            )
            self._round_started = None
            self._awaiting_post_admission_active = False
        else:
            self._round_started = time.perf_counter()
        return active

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: tuple[MaterialQuery, ...],
    ) -> object:
        started = time.perf_counter()
        result = self.delegate.admit(card, query_pool)
        delegate_elapsed = time.perf_counter() - started
        retention_seconds = float(
            getattr(self.delegate, "last_online_retention_seconds", delegate_elapsed)
        )
        updated_archive = (*self._archive, card)
        active = self.delegate.active(updated_archive)
        evaluator_started = time.perf_counter()
        round_metrics = self.evaluator.evaluate(
            round_index=len(self.rounds) + 1,
            queries=query_pool,
            cards=active,
            retention_seconds=retention_seconds,
            parent_rss_bytes=self._process.memory_info().rss,
        )
        evaluator_seconds = time.perf_counter() - evaluator_started
        self.rounds.append(round_metrics)
        if self.record_projection_diagnostics:
            deployed = self.evaluator.evaluate_snapshot(
                name="deployed",
                queries=query_pool,
                cards=active,
            )
            self.posterior_snapshots.append(deployed.model_dump(mode="json"))
            if isinstance(self.delegate, _PosteriorProjectionEvidence):
                context = self.delegate.consume_factorial_context()
                diagnostic_started = time.perf_counter()
                evaluations = {
                    "union_reference": self.evaluator.evaluate_snapshot(
                        name="union_reference",
                        queries=query_pool,
                        cards=context["union_reference_cards"],
                    ),
                    "archive_reference": self.evaluator.evaluate_snapshot(
                        name="archive_reference",
                        queries=query_pool,
                        cards=context["archive_reference_cards"],
                    ),
                    **{
                        name: (
                            deployed
                            if name == "union_reference__online_search"
                            else self.evaluator.evaluate_snapshot(
                                name=name,
                                queries=query_pool,
                                cards=selected_cards,
                            )
                        )
                        for name, selected_cards in context["selected_cards"].items()
                    },
                }
                selection_audit_started = time.perf_counter()
                selection_effect = self.evaluator.selection_effect_records(
                    queries=query_pool,
                    union_cards=context["union_cards"],
                    retained_card_ids=context["retained_card_ids"],
                )
                selection_audit_seconds = time.perf_counter() - selection_audit_started
                diagnostic_seconds = time.perf_counter() - diagnostic_started
                record = self.delegate.diagnostics[-1]
                record["causal_evaluations"] = {
                    name: snapshot.model_dump(mode="json") for name, snapshot in evaluations.items()
                }
                record["selection_effect_records"] = [
                    item.model_dump(mode="json") for item in selection_effect
                ]
                record["timing"].update(
                    {
                        "prequential_evaluator_seconds": evaluator_seconds,
                        "factorial_causal_evaluator_seconds": diagnostic_seconds,
                        "selection_effect_audit_seconds": selection_audit_seconds,
                    }
                )
        self._awaiting_post_admission_active = True
        return result


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
        builder = CalibrationUtilityBuilder(FixedKernelResidualGP(resolver, config=config))
        self.facility = FacilityLocationCoresetPlanner(capacity, builder, min_admission_gain=1e-12)
        self.joint = JointPosteriorRiskOneSwapPlanner(capacity, builder, min_risk_improvement=1e-12)
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


def _rebuild_pool_entries_from_parity(
    raw_by_system: dict[tuple[str, ...], list[Any]],
    *,
    required_ids_by_system: dict[tuple[str, ...], set[str]],
    parity_rows_by_id: dict[str, dict[str, Any]],
    ppd: Any,
) -> dict[tuple[str, ...], list[Any]]:
    """Build hull entries from frozen parity-environment formation energies.

    Modern pymatgen is allowed to decode composition, but it must not silently
    replace the corrected WBM energy frozen by the parity environment. Only
    candidates in the executed pool are rebuilt; unrevealed entries outside
    the fixed pool never enter its causal hull.
    """

    from pymatgen.entries.computed_entries import ComputedEntry

    rebuilt: dict[tuple[str, ...], list[Any]] = {}
    for system, required_ids in required_ids_by_system.items():
        raw_by_id = {str(entry.entry_id): entry for entry in raw_by_system[system]}
        missing_raw = required_ids - set(raw_by_id)
        missing_parity = required_ids - set(parity_rows_by_id)
        if missing_raw or missing_parity:
            raise ValueError(
                "pool/parity entry coverage failed: "
                f"missing_raw={sorted(missing_raw)}, "
                f"missing_parity={sorted(missing_parity)}"
            )
        entries = []
        for query_id in sorted(required_ids):
            composition = raw_by_id[query_id].composition
            target_formation = float(
                parity_rows_by_id[query_id][
                    "parity_corrected_formation_energy_ev_per_atom"
                ]
            )
            elemental_reference_total = sum(
                float(ppd.el_refs[element].energy_per_atom) * float(amount)
                for element, amount in composition.items()
            )
            corrected_total = (
                target_formation * float(composition.num_atoms)
                + elemental_reference_total
            )
            entry = ComputedEntry(
                composition,
                corrected_total,
                entry_id=query_id,
            )
            replayed = float(ppd.get_form_energy_per_atom(entry))
            if not math.isclose(replayed, target_formation, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"parity formation-energy reconstruction failed: {query_id}")
            entries.append(entry)
        rebuilt[system] = entries
    return rebuilt


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
            structure_identity=StructureArtifactIdentity.initial(
                query_id, candidate["exact_structure_sha256"]
            ),
            identity=MaterialIdentity(
                exact_calculation_id=query_id,
                canonical_structure_id=("byte-identical:" + candidate["exact_structure_sha256"]),
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
        return base_policy, _ObjectiveFidelityEvidence(capacity, config, selector="facility")
    if name == "joint_posterior_risk_one_swap":
        return base_policy, _ObjectiveFidelityEvidence(capacity, config, selector="joint_risk")
    projection_settings = {
        "p3c_brier": ("brier", None, None),
        "p3c_log": ("log", None, None),
        "p3c_gaussian_kl": ("gaussian_kl", None, None),
        "p3c_twcrps": ("threshold_weighted_crps", None, None),
        # Zero regret is a theory-fixed reference-decision parity constraint,
        # not a tolerance estimated from evaluation systems.
        "p3c_twcrps_decision_safe": ("threshold_weighted_crps", 0.0, None),
    }
    if name in projection_settings:
        divergence_kind, max_decision_regret, max_log_divergence = projection_settings[name]
        return base_policy, _PosteriorProjectionEvidence(
            capacity,
            config,
            divergence_kind,
            max_decision_regret=max_decision_regret,
            max_log_divergence=max_log_divergence,
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
        structure_identity=query.structure_identity,
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
            formation_energy_ev_per_atom - query.base_predicted_formation_energy_ev_per_atom
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
    active_observable = builder.weighted_decision_risk(remaining_queries, active_cards)
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
                active_observable - float(observable_best["observable_weighted_joint_risk"]),
            )
            if equal_capacity
            else None
        ),
        "active_offline_oracle_loss_regret": (
            max(
                0.0,
                active_oracle - float(oracle_best["offline_oracle_weighted_decision_loss"]),
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
    calibration_freeze_sha256: str | None,
    acquisition: str,
    log_path: Path,
    ppd: Any,
    include_exhaustive_subset_audit: bool = False,
    audit_budget_prefix: bool = False,
    record_projection_diagnostics: bool = False,
) -> dict[str, Any]:
    selected_ids = {query.query_id for query in queries}
    universe_by_id = {item.query_id: item for item in universe}
    records = [
        WBMOracleRecord(
            query_id=query.query_id,
            structure_hash=query.structure_hash,
            energy_source=OracleEnergySource.FROZEN_PARITY_CORRECTED,
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
    oracle_formation_by_id = {
        query_id: float(ppd.get_form_energy_per_atom(item.entry))
        for query_id, item in universe_by_id.items()
        if query_id in selected_ids
    }
    policy, evidence = _strategy(name, capacity, config, acquisition=acquisition)
    prequential = _PrequentialEvidence(
        evidence,
        PrequentialCausalEvaluator(
            CalibrationUtilityBuilder(
                FixedKernelResidualGP(ProtocolCompatibilityResolver(), config=config)
            ),
            oracle_formation_by_id,
        ),
        record_projection_diagnostics=record_projection_diagnostics,
    )
    started = time.perf_counter()
    initial_rss = psutil.Process().memory_info().rss
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
            evidence_access=prequential,
            oracle_universe=universe,
            event_log=event_log,
        ).run(
            oracle_budget=float(budget),
            budget_prefix_checks=(4.0, 8.0, 12.0) if audit_budget_prefix and budget == 12 else (),
        )
    if audit_budget_prefix and budget == 12:
        expected_pairs = 4 + 8
        if len(result.budget_prefix_parity) != expected_pairs or not all(
            item.actions_match for item in result.budget_prefix_parity
        ):
            raise RuntimeError("budget-prefix behavioral parity gate failed")
    query_by_id = {query.query_id: query for query in queries}
    history_cards = tuple(
        _material_card(query_by_id[query_id], oracle_formation_by_id[query_id])
        for query_id in result.selected_query_ids
    )
    final_active_ids = set(result.events[-1].active_witness_ids if result.events else ())
    cards = tuple(card for card in history_cards if card.card_id in final_active_ids)
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
        posterior = FixedKernelResidualGP(ProtocolCompatibilityResolver(), config=config).fit(cards)
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
                    float(ppd.get_form_energy_per_atom(universe_by_id[item.query_id].entry))
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
        normal_cdf = 0.5 * (1 + np.asarray([math.erf(float(value) / np.sqrt(2)) for value in z]))
        builder = CalibrationUtilityBuilder(
            FixedKernelResidualGP(ProtocolCompatibilityResolver(), config=config)
        )
        calibration.update(
            {
                "residual_rmse_ev_per_atom": float(np.sqrt(np.mean((truth - mean) ** 2))),
                "residual_gaussian_nll": float(
                    np.mean(0.5 * np.log(2 * np.pi * std**2) + 0.5 * ((truth - mean) / std) ** 2)
                ),
                "causal_hull_stability_brier": float(np.mean((probabilities - labels) ** 2)),
                "causal_hull_stability_log_loss": float(
                    -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
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
                    np.mean(std * (z * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / np.sqrt(np.pi)))
                ),
                "interval_90_coverage": float(
                    np.mean(np.abs(truth - mean) <= 1.6448536269514722 * std)
                ),
                "interval_90_mean_width_ev_per_atom": float(np.mean(2 * 1.6448536269514722 * std)),
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
        if name == "decision_coreset" and include_exhaustive_subset_audit
        else None
    )
    prequential_rounds = tuple(prequential.rounds)
    prequential_aggregate = aggregate_prequential_prefix(
        prequential_rounds, len(result.selected_query_ids)
    )
    if isinstance(evidence, _PosteriorProjectionEvidence):
        if len(evidence.diagnostics) != len(result.phase_timings):
            raise RuntimeError("P3C diagnostics and hull timings are misaligned")
        for diagnostic, phase_timing in zip(
            evidence.diagnostics, result.phase_timings, strict=True
        ):
            diagnostic["timing"]["hull_update_seconds"] = phase_timing.hull_update_seconds
    final_rss = psutil.Process().memory_info().rss
    return {
        "pool": pool_name,
        "strategy": name,
        "calibration_freeze_sha256": calibration_freeze_sha256,
        "acquisition": acquisition,
        "budget": budget,
        "capacity": capacity,
        "wall_seconds": time.perf_counter() - started,
        "prequential_rounds": [item.model_dump(mode="json") for item in prequential_rounds],
        "prequential_posterior_snapshots": (
            prequential.posterior_snapshots if record_projection_diagnostics else None
        ),
        "prequential": prequential_aggregate,
        "peak_parent_rss_bytes": max(
            initial_rss,
            final_rss,
            int(prequential_aggregate["peak_parent_rss_bytes"] or 0),
        ),
        "calibration": calibration,
        "objective_fidelity_rounds": (
            evidence.diagnostics if isinstance(evidence, _ObjectiveFidelityEvidence) else None
        ),
        "posterior_projection_rounds": (
            evidence.diagnostics if isinstance(evidence, _PosteriorProjectionEvidence) else None
        ),
        "offline_subset_audit": subset_audit,
        "budget_prefix_parity_passed": (
            all(item.actions_match for item in result.budget_prefix_parity)
            if audit_budget_prefix and budget == 12
            else None
        ),
        **result.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-audit", type=Path, required=True)
    parser.add_argument("--license-manifest", type=Path, required=True)
    parser.add_argument("--calibration-freeze-manifest", type=Path)
    parser.add_argument("--registered-config", type=Path)
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
        "--include-exhaustive-subset-audit",
        action="store_true",
        help="repeat the expensive B8/K2 subset diagnostic; disabled for grid runs",
    )
    parser.add_argument(
        "--include-paused-survival",
        action="store_true",
        help="run the frozen negative survival diagnostic without tuning it",
    )
    parser.add_argument(
        "--audit-budget-prefix",
        action="store_true",
        help="hard-check B4/B8 behavior against each canonical B12 trace",
    )
    parser.add_argument(
        "--include-reference-path-diagnostics",
        action="store_true",
        help=(
            "record frozen P1 reference headroom, 2x2 path, NLL attribution, "
            "and selection-effect inputs; never use for the full grid"
        ),
    )
    parser.add_argument(
        "--strategy",
        action="append",
        dest="strategies",
        help="run only this retention strategy; repeat for multiple strategies",
    )
    parser.add_argument(
        "--compute-relevance-only",
        action="store_true",
        help=(
            "run only the preregistered B40 frozen-action full-history timing trace; "
            "P1.5 discovery support is intentionally irrelevant"
        ),
    )
    args = parser.parse_args()
    if args.compute_relevance_only and not (
        args.budget == 40
        and args.capacity == 0
        and args.acquisition == "frozen"
        and args.strategies == ["full_history"]
        and not args.include_exhaustive_subset_audit
        and not args.include_paused_survival
        and not args.include_reference_path_diagnostics
    ):
        raise ValueError(
            "compute relevance is restricted to B40, capacity 0, frozen action, "
            "and the full_history strategy only"
        )
    if args.include_reference_path_diagnostics and (
        args.budget != 8 or args.capacity != 2 or args.acquisition != "frozen"
    ):
        raise ValueError("reference/path diagnostics are frozen to matched-action P1 B=8, K=2")
    repo_root = Path(__file__).resolve().parents[1]
    for name, value in vars(args).items():
        if (
            isinstance(value, Path)
            and name != "registered_config"
            and value.resolve().is_relative_to(repo_root)
        ):
            raise ValueError("real WBM inputs and outputs must remain outside Git")
    gate = json.loads(args.gate_audit.read_text(encoding="utf-8"))
    license_manifest = json.loads(args.license_manifest.read_text(encoding="utf-8"))
    _validate_gate_bindings(
        gate,
        pool_manifest=args.pool_manifest,
        parity_audit=args.parity_audit,
        soap_cache=args.soap_cache,
        cleaned_ids=args.cleaned_ids,
        ppd=args.ppd,
        compute_relevance_only=args.compute_relevance_only,
    )
    if license_manifest["local_research_gate_passed"] is not True:
        raise ValueError("local research license gate has not passed")
    if args.output_dir.exists():
        raise FileExistsError("engineering pilot output directory is immutable")
    if args.audit_budget_prefix and args.calibration_freeze_manifest is None:
        raise ValueError("frozen-grid execution requires a calibration-freeze manifest")
    if (args.calibration_freeze_manifest is None) != (args.registered_config is None):
        raise ValueError(
            "calibration-freeze manifest and registered configuration are required together"
        )
    (args.output_dir / "ledgers").mkdir(parents=True)

    pool_payload = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    parity_payload = json.loads(args.parity_audit.read_text(encoding="utf-8"))
    parity_rows_by_id = {row["query_id"]: row for row in parity_payload["rows"]}
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
    required_ids_by_system = {
        tuple(sorted(pool["chemical_system"])): {
            candidate["query_id"] for candidate in pool["candidates"]
        }
        for pool in pool_payload["selection"]["pools"].values()
    }
    raw = _load_exact_system_universe(
        args.raw_cse_root, _read_cleaned_ids(args.cleaned_ids), systems
    )
    corrected = _rebuild_pool_entries_from_parity(
        raw,
        required_ids_by_system=required_ids_by_system,
        parity_rows_by_id=parity_rows_by_id,
        ppd=ppd,
    )
    calibration_freeze_sha256: str | None = None
    calibration_freeze_payload: dict[str, Any] | None = None
    if args.calibration_freeze_manifest is not None:
        config, calibration_freeze_sha256, calibration_freeze_payload = _load_calibration_freeze(
            args.calibration_freeze_manifest,
            args.registered_config,
        )
    else:
        config = FixedKernelGPConfig(
            kernel="matern52",
            length_scale=0.35,
            signal_std_ev_per_atom=0.08,
            noise_std_ev_per_atom=0.01,
            jitter=1e-10,
        )
    default_strategy_names = [
        "fifo",
        "full_history",
        "diversity",
        "gp_variance_one_swap",
        "decision_coreset",
        "joint_posterior_risk_one_swap",
    ]
    diagnostic_projection_names = (
        "p3c_brier",
        "p3c_log",
        "p3c_gaussian_kl",
        "p3c_twcrps",
        "p3c_twcrps_decision_safe",
    )
    if args.strategies:
        unknown = set(args.strategies) - set(
            (
                *default_strategy_names,
                *diagnostic_projection_names,
                "free_same_fifo",
                "survival_coreset",
            )
        )
        if unknown:
            raise ValueError(f"unknown requested strategies: {sorted(unknown)}")
        strategy_names = list(dict.fromkeys(args.strategies))
    else:
        strategy_names = default_strategy_names
    if args.acquisition == "gp_uncertainty" and not args.strategies:
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
                energy_source=OracleEnergySource.FROZEN_PARITY_CORRECTED,
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
                    calibration_freeze_sha256=calibration_freeze_sha256,
                    acquisition=args.acquisition,
                    log_path=(args.output_dir / "ledgers" / f"{pool_name}-{strategy_name}.jsonl"),
                    ppd=ppd,
                    include_exhaustive_subset_audit=(args.include_exhaustive_subset_audit),
                    audit_budget_prefix=args.audit_budget_prefix,
                    record_projection_diagnostics=(args.include_reference_path_diagnostics),
                )
            )
    by_pool_strategy = {(run["pool"], run["strategy"]): run for run in runs}
    if "decision_coreset" in strategy_names:
        for pool_name in pool_payload["selection"]["pools"]:
            reference_run = by_pool_strategy[(pool_name, "decision_coreset")]
            reference_audit = reference_run["offline_subset_audit"]
            if reference_audit is None:
                continue
            for strategy_name in strategy_names:
                run = by_pool_strategy[(pool_name, strategy_name)]
                if run["selected_query_ids"] != reference_run["selected_query_ids"]:
                    continue
                active_ids = run["events"][-1]["active_witness_ids"] if run["events"] else []
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
                    "reused_from_matched_action_reference": strategy_name != "decision_coreset",
                }
    parity_mismatches = []
    if {"fifo", "free_same_fifo"}.issubset(strategy_names):
        for pool_name in pool_payload["selection"]["pools"]:
            persistent = by_pool_strategy[(pool_name, "fifo")]
            reconstructed = by_pool_strategy[(pool_name, "free_same_fifo")]
            if persistent["selected_query_ids"] != reconstructed["selected_query_ids"]:
                parity_mismatches.append(pool_name)
    matched_trace_mismatches = []
    if args.acquisition == "frozen":
        reference_strategy = _matched_action_reference_strategy(strategy_names)
        for pool_name in pool_payload["selection"]["pools"]:
            reference = by_pool_strategy[(pool_name, reference_strategy)]["selected_query_ids"]
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
    prequential_metric_names = (
        "boundary_weighted_causal_crps",
        "boundary_weighted_causal_brier",
        "boundary_weighted_causal_log_loss",
        "residual_rmse_ev_per_atom",
        "residual_gaussian_nll",
        "boundary_weighted_false_stable_cost",
    )
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
            "peak_parent_rss_bytes": max(run["peak_parent_rss_bytes"] for run in selected),
            "posterior_fit_seconds": sum(
                run["prequential"]["posterior_fit_seconds"] for run in selected
            ),
            "retention_seconds": sum(run["prequential"]["retention_seconds"] for run in selected),
            "prediction_seconds": sum(run["prequential"]["prediction_seconds"] for run in selected),
            "round_pipeline_seconds": sum(
                run["prequential"]["round_pipeline_seconds"] for run in selected
            ),
            **{
                f"mean_remaining_{metric}": float(
                    np.mean([run["calibration"][metric] for run in selected])
                )
                for metric in metric_names
            },
            **{
                f"mean_prequential_{metric}": float(
                    np.mean([run["prequential"][metric] for run in selected])
                )
                for metric in prequential_metric_names
            },
        }
        eligible_audits = [
            run["offline_subset_audit"]
            for run in selected
            if run["offline_subset_audit"] is not None
            and run["offline_subset_audit"]["active_size_matches_capacity"]
        ]
        aggregates[strategy_name]["mean_offline_observable_subset_regret"] = (
            float(np.mean([audit["active_observable_regret"] for audit in eligible_audits]))
            if eligible_audits
            else None
        )
        aggregates[strategy_name]["mean_offline_oracle_loss_subset_regret"] = (
            float(
                np.mean([audit["active_offline_oracle_loss_regret"] for audit in eligible_audits])
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
        saturated = [record for record in records if len(record["candidates"]) == args.capacity + 1]
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
                float(np.mean([record["facility_joint_risk_regret"] for record in records]))
                if records
                else None
            ),
            "positive_regret_round_count": sum(
                record["facility_joint_risk_regret"] > 1e-12 for record in records
            ),
            "saturated_round_count": len(saturated),
            "saturated_selection_agreement_rate": (
                float(np.mean([record["selections_agree"] for record in saturated]))
                if saturated
                else None
            ),
            "saturated_mean_spearman": (
                float(np.mean(saturated_correlations)) if saturated_correlations else None
            ),
            "saturated_positive_regret_round_count": sum(
                record["facility_joint_risk_regret"] > 1e-12 for record in saturated
            ),
            "saturated_mean_facility_joint_risk_regret": (
                float(np.mean([record["facility_joint_risk_regret"] for record in saturated]))
                if saturated
                else None
            ),
        }
    posterior_projection = {}
    for strategy_name in diagnostic_projection_names:
        records = [
            record
            for run in runs
            if run["strategy"] == strategy_name
            for record in (run["posterior_projection_rounds"] or [])
        ]
        if not records:
            continue
        posterior_projection[strategy_name] = {
            "round_count": len(records),
            "mean_union_reference_divergence": float(
                np.mean([record["online"]["selected_proper_divergence"] for record in records])
            ),
            "mean_union_reference_decision_regret": float(
                np.mean([record["online"]["selected_decision_regret"] for record in records])
            ),
            "mean_union_reference_log_divergence": float(
                np.mean([record["online"]["selected_log_divergence"] for record in records])
            ),
            "constraint_fallback_round_count": sum(
                record["online"]["used_constraint_fallback"] for record in records
            ),
            "mean_online_vs_archive_optimization_gap": float(
                np.mean([record["online_vs_archive_optimization_gap"] for record in records])
            ),
            "rounds_with_archive_reactivation": sum(
                bool(record["archive_reactivated_old_card_ids"]) for record in records
            ),
            "mean_retained_minus_archive_residual_mean": float(
                np.mean(
                    [
                        record["retained_minus_archive_residual_mean"]
                        for record in records
                        if record["retained_minus_archive_residual_mean"] is not None
                    ]
                )
            ),
        }
    report = {
        "schema_version": 1,
        "scope": (
            "wbm_long_archive_compute_relevance_only_no_policy_or_discovery_claim"
            if args.compute_relevance_only
            else "small_wbm_calibration_coreset_engineering_pilot_not_claim_grade"
        ),
        "budget": args.budget,
        "capacity": args.capacity,
        "acquisition": args.acquisition,
        "pool_count": len(pool_payload["selection"]["pools"]),
        "embedding": "exact finite-pool SOAP Gram factorization",
        "oracle_energy_source": "parity_corrected_formation_energy_ev_per_atom",
        "parity_audit_sha256": _sha256(args.parity_audit),
        "calibration_freeze_sha256": calibration_freeze_sha256,
        "gp_parameter_status": (
            calibration_freeze_payload["gp_parameter_status"]
            if calibration_freeze_payload is not None
            else "engineering_unfrozen"
        ),
        "free_same_fifo_exact_action_parity": (
            not parity_mismatches if {"fifo", "free_same_fifo"}.issubset(strategy_names) else None
        ),
        "free_same_fifo_mismatch_pools": parity_mismatches,
        "matched_frozen_acquisition_action_parity": (
            not matched_trace_mismatches if args.acquisition == "frozen" else None
        ),
        "matched_frozen_acquisition_mismatches": matched_trace_mismatches,
        "aggregates": aggregates,
        "objective_fidelity": objective_fidelity,
        "posterior_projection": posterior_projection,
        "paused_survival_included": args.include_paused_survival,
        "reference_path_diagnostics_included": (args.include_reference_path_diagnostics),
        "runs": runs,
        "interpretation_guardrails": [
            *(
                [
                    "this output authorizes timing and memory interpretation only",
                    "P1.5 support is irrelevant; discovery metrics are not results",
                ]
                if args.compute_relevance_only
                else []
            ),
            (
                "GP parameters were loaded from the recorded disjoint calibration freeze"
                if calibration_freeze_payload is not None
                else "engineering hyperparameters were not loaded from a calibration freeze"
            ),
            "WBM oracle energies were reconstructed from the frozen parity environment",
            "prototype and WBM-MP overlap clustering remain pending",
            "no paper-level GO can be inferred from this run",
        ],
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if {"fifo", "free_same_fifo"}.issubset(strategy_names):
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
