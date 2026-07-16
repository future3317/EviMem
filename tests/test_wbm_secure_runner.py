from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from evimem.matmem import (
    AppendOnlyWBMEventLog,
    CalibrationUtilityBuilder,
    CompositionHullState,
    CorrectedPhaseEntry,
    FacilityLocationCoresetPlanner,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    HullSnapshot,
    MaterialIdentity,
    MaterialQuery,
    PersistentFIFOEvidence,
    PolicyState,
    PolicySubprocess,
    ProtocolCompatibilityResolver,
    ReconstructedFIFOEvidence,
    SecureWBMRunner,
    StreamingCalibrationCoreset,
    StreamingCoresetEvidence,
    WBMOracleRecord,
    WBMOracleVault,
    assert_exact_emulation,
    replay_wbm_event_log,
)
from evimem.matmem.protocols import ProtocolCertificate

NOW = datetime(2023, 2, 7, tzinfo=UTC)
OBSERVED = NOW + timedelta(days=1)
SYSTEM = ("Fe", "Zr")


def _initial_diagram():
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    return PhaseDiagram(
        [
            ComputedEntry("Fe", -8.0, entry_id="mp-fe"),
            ComputedEntry("Zr", -9.0, entry_id="mp-zr"),
        ]
    )


def _entry(query_id: str, composition: str, total_energy: float):
    from pymatgen.entries.computed_entries import ComputedEntry

    return ComputedEntry(composition, total_energy, entry_id=query_id)


def _protocol() -> ProtocolCertificate:
    return ProtocolCertificate(
        functional="PBE",
        pseudopotential_set="PAW",
        correction_scheme="MP2020",
        relaxation_protocol="WBM-relaxed",
        calculation_code="VASP",
    )


def _query(
    query_id: str,
    composition: str,
    prediction: float,
    embedding: tuple[float, float],
) -> MaterialQuery:
    return MaterialQuery(
        query_id=query_id,
        structure_hash=f"structure-{query_id}",
        identity=MaterialIdentity(
            exact_calculation_id=query_id,
            canonical_structure_id=f"canonical-{query_id}",
            composition_family="Fe-Zr",
            prototype_family=f"prototype-{query_id}",
        ),
        composition=composition,
        embedding=embedding,
        protocol=_protocol(),
        hull_snapshot=HullSnapshot(
            snapshot_id=f"initial:{query_id}",
            chemical_system=SYSTEM,
            reference_hull_energy_ev_per_atom=0.0,
            phase_set_checksum="sha256:" + "1" * 64,
            known_through=NOW,
            built_at=NOW,
            source_version="MP-2022.10.28",
        ),
        base_predicted_formation_energy_ev_per_atom=prediction,
        as_of=NOW,
    )


def _fixture(
    *,
    second_formation_energy: float = -0.10,
) -> tuple[
    tuple[MaterialQuery, ...],
    tuple[WBMOracleRecord, ...],
    dict[str, object],
    tuple[CorrectedPhaseEntry, ...],
]:
    queries = (
        _query("q1", "FeZr", -0.22, (1.0, 0.0)),
        _query("q2", "FeZr2", -0.11, (0.0, 1.0)),
        _query("q3", "Fe2Zr", -0.05, (0.7, 0.7)),
    )
    totals = {
        "q1": -17.4,
        "q2": -26.0 + 3 * second_formation_energy,
        "q3": -25.1,
    }
    compositions = {"q1": "FeZr", "q2": "FeZr2", "q3": "Fe2Zr"}
    formations = {
        "q1": -0.20,
        "q2": second_formation_energy,
        "q3": -0.0333333333333333,
    }
    entries = {
        query_id: _entry(query_id, compositions[query_id], total)
        for query_id, total in totals.items()
    }
    records = tuple(
        WBMOracleRecord(
            query_id=query.query_id,
            structure_hash=query.structure_hash,
            corrected_total_energy_ev=totals[query.query_id],
            corrected_formation_energy_ev_per_atom=formations[query.query_id],
            source_record_locator=query.query_id,
            observed_at=OBSERVED,
        )
        for query in queries
    )
    universe = tuple(
        CorrectedPhaseEntry(
            query_id=query_id,
            corrected_total_energy_ev=totals[query_id],
            entry=entries[query_id],
        )
        for query_id in ("q1", "q2", "q3")
    )
    return queries, records, entries, universe


def _run(
    tmp_path: Path,
    *,
    log_name: str,
    evidence: object,
    policy: PolicySubprocess | None = None,
    budget: float = 2,
    second_formation_energy: float = -0.10,
):
    queries, records, entries, universe = _fixture(
        second_formation_energy=second_formation_energy
    )
    log = AppendOnlyWBMEventLog(tmp_path / log_name)
    runner = SecureWBMRunner(
        queries=queries,
        vault=WBMOracleVault(records, entries, source_version="WBM-2021.68"),
        hull_state=CompositionHullState(
            _initial_diagram(),
            chemical_system=SYSTEM,
            source_version="MP-2022.10.28",
        ),
        policy=policy or PolicySubprocess("frozen"),
        evidence_access=evidence,
        oracle_universe=universe,
        event_log=log,
    )
    result = runner.run(oracle_budget=budget)
    log.close()
    return result, entries, universe


class _HullAwareEvidenceSpy:
    capacity = 1

    def __init__(self) -> None:
        self._active = ()
        self.seen_snapshot_ids: list[tuple[str, ...]] = []

    def active(self, archive):
        del archive
        return self._active

    def admit(self, card, query_pool):
        self.seen_snapshot_ids.append(
            tuple(query.hull_snapshot.snapshot_id for query in query_pool)
        )
        self._active = (card,)


def test_policy_serialization_is_allow_listed_and_order_invariant() -> None:
    queries, _, _, _ = _fixture()
    policy = PolicySubprocess("frozen")
    state = PolicyState.create(
        round_index=1,
        remaining_budget=2,
        queries=reversed(queries),
        witnesses=(),
        history_query_ids=(),
        active_witness_capacity=0,
        policy_identity_checksum=policy.identity_checksum,
    )
    payload = state.serialized_for_policy()
    lowered = payload.lower()
    assert "corrected_total_energy" not in lowered
    assert "corrected_formation_energy" not in lowered
    assert "stable_label" not in lowered
    assert "phase_diagram" not in lowered
    assert [item["query_id"] for item in json.loads(payload)["queries"]] == [
        "q1",
        "q2",
        "q3",
    ]
    forward = policy.select(state)
    reversed_state = PolicyState.create(
        round_index=1,
        remaining_budget=2,
        queries=queries,
        witnesses=(),
        history_query_ids=(),
        active_witness_capacity=0,
        policy_identity_checksum=policy.identity_checksum,
    )
    assert forward == policy.select(reversed_state)


def test_unqueried_oracle_counterfactual_cannot_change_actions(tmp_path: Path) -> None:
    baseline, _, _ = _run(
        tmp_path,
        log_name="baseline.jsonl",
        evidence=PersistentFIFOEvidence(1),
    )
    changed, _, _ = _run(
        tmp_path,
        log_name="counterfactual.jsonl",
        evidence=PersistentFIFOEvidence(1),
        second_formation_energy=0.40,
    )
    assert baseline.selected_query_ids == changed.selected_query_ids


def test_secure_runner_updates_composition_hull_before_evidence_admission(
    tmp_path: Path,
) -> None:
    evidence = _HullAwareEvidenceSpy()
    _run(
        tmp_path,
        log_name="hull-before-admission.jsonl",
        evidence=evidence,
        budget=1,
    )
    assert len(evidence.seen_snapshot_ids) == 1
    assert all(
        snapshot_id.startswith("causal:MP-2022.10.28:2:")
        for snapshot_id in evidence.seen_snapshot_ids[0]
    )


def test_secure_runner_supports_streaming_calibration_coreset(tmp_path: Path) -> None:
    resolver = ProtocolCompatibilityResolver()
    coreset = StreamingCalibrationCoreset(
        FacilityLocationCoresetPlanner(
            1,
            CalibrationUtilityBuilder(
                FixedKernelResidualGP(
                    resolver,
                    config=FixedKernelGPConfig(length_scale=0.2),
                )
            ),
        )
    )
    result, _, _ = _run(
        tmp_path,
        log_name="streaming-coreset.jsonl",
        evidence=StreamingCoresetEvidence(coreset),
    )
    assert result.selected_query_ids
    assert len(coreset.cards()) <= 1


def test_zero_weight_survival_worker_matches_gp_uncertainty_actions(
    tmp_path: Path,
) -> None:
    config = FixedKernelGPConfig(length_scale=0.2)
    uncertainty, _, _ = _run(
        tmp_path,
        log_name="gp-uncertainty.jsonl",
        evidence=PersistentFIFOEvidence(1),
        policy=PolicySubprocess("gp_uncertainty", gp_config=config),
    )
    zero_survival, _, _ = _run(
        tmp_path,
        log_name="zero-survival.jsonl",
        evidence=PersistentFIFOEvidence(1),
        policy=PolicySubprocess(
            "survival_coreset",
            gp_config=config,
            survival_weight=0,
        ),
    )
    assert zero_survival.selected_query_ids == uncertainty.selected_query_ids


def test_vault_requires_persisted_action_and_rejects_duplicate(tmp_path: Path) -> None:
    queries, records, entries, _ = _fixture()
    vault = WBMOracleVault(records, entries, source_version="WBM-2021.68")
    log = AppendOnlyWBMEventLog(tmp_path / "authorization.jsonl")
    with pytest.raises(RuntimeError, match="persisted action"):
        vault.reveal(queries[0], authorization=object(), event_log=log)  # type: ignore[arg-type]
    authorization = log.append_action(
        round_index=1,
        selected_query_id="q1",
        pre_reveal_state_checksum="sha256:" + "2" * 64,
    )
    assert (tmp_path / "authorization.jsonl").stat().st_size > 0
    vault.reveal(queries[0], authorization=authorization, event_log=log)
    log.append_reveal(
        round_index=1,
        selected_query_id="q1",
        action_checksum=authorization.action_checksum,
        causal_discovery=True,
        active_witness_ids=("wbm-card:q1",),
        post_reveal_hull_checksum="sha256:" + "3" * 64,
        archive_checksum="sha256:" + "4" * 64,
    )
    duplicate = log.append_action(
        round_index=2,
        selected_query_id="q1",
        pre_reveal_state_checksum="sha256:" + "5" * 64,
    )
    with pytest.raises(ValueError, match="already been revealed"):
        vault.reveal(queries[0], authorization=duplicate, event_log=log)
    log.close()


def test_total_energy_and_per_atom_energy_cannot_be_mixed() -> None:
    queries, records, entries, _ = _fixture()
    wrong = records[0].model_copy(update={"corrected_total_energy_ev": -8.7})
    with pytest.raises(ValueError, match="total energy"):
        WBMOracleVault(
            (wrong, *records[1:]),
            entries,
            source_version="WBM-2021.68",
        )
    assert queries[0].base_predicted_formation_energy_ev_per_atom == -0.22


def test_hypothetical_hull_and_cross_system_are_isolated() -> None:
    _, _, _, universe = _fixture()
    causal = CompositionHullState(
        _initial_diagram(),
        chemical_system=SYSTEM,
        source_version="MP-2022.10.28",
    )
    before = causal.phase_set_checksum
    hypothetical = causal.hypothetical(universe[0])
    assert causal.phase_set_checksum == before
    assert hypothetical.phase_set_checksum != before

    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    other = CompositionHullState(
        PhaseDiagram(
            [
                ComputedEntry("Li", -1.0, entry_id="mp-li"),
                ComputedEntry("O", -2.0, entry_id="mp-o"),
            ]
        ),
        chemical_system=("Li", "O"),
        source_version="MP-2022.10.28",
    )
    other_before = other.phase_set_checksum
    causal.add_revealed(universe[0])
    assert other.phase_set_checksum == other_before
    with pytest.raises(ValueError, match="another chemical system"):
        other.add_revealed(universe[0])


def test_three_hulls_distinguish_selected_and_oracle_final() -> None:
    from pymatgen.entries.computed_entries import ComputedEntry

    causal = CompositionHullState(
        _initial_diagram(),
        chemical_system=SYSTEM,
        source_version="MP-2022.10.28",
    )
    selected = CorrectedPhaseEntry(
        query_id="selected",
        corrected_total_energy_ev=-17.2,
        entry=ComputedEntry("FeZr", -17.2, entry_id="selected"),
    )
    hidden = CorrectedPhaseEntry(
        query_id="hidden",
        corrected_total_energy_ev=-17.6,
        entry=ComputedEntry("FeZr", -17.6, entry_id="hidden"),
    )
    causal.add_revealed(selected)
    assert causal.selected_final_stability() == {"selected": True}
    assert causal.oracle_final_stability((selected, hidden)) == {"selected": False}


def test_event_replay_and_free_same_fifo_are_exact(tmp_path: Path) -> None:
    persistent, entries, universe = _run(
        tmp_path,
        log_name="persistent.jsonl",
        evidence=PersistentFIFOEvidence(1),
    )
    reconstructed, _, _ = _run(
        tmp_path,
        log_name="reconstructed.jsonl",
        evidence=ReconstructedFIFOEvidence(1),
    )
    audit = assert_exact_emulation(persistent, reconstructed)
    assert audit.passed
    phase_registry = {item.query_id: item for item in universe}
    replay = replay_wbm_event_log(
        tmp_path / "persistent.jsonl",
        initial_phase_diagram=_initial_diagram(),
        chemical_system=SYSTEM,
        source_version="MP-2022.10.28",
        phase_entries=phase_registry,
    )
    assert replay.passed
    assert replay.trace_checksum == persistent.trace_checksum
    assert set(entries) == {"q1", "q2", "q3"}
