from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from matmem import (
    CalibrationUtilityBuilder,
    CanonicalGroupSplit,
    CompatibilityKind,
    FacilityLocationCoresetPlanner,
    FIFOBoundedMemory,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    HullSnapshot,
    MatchedResidualPair,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    ProtocolCertificate,
    ProtocolCompatibilityResolver,
    ProtocolRiskController,
    ProtocolTransportMap,
    ResidualCorrector,
    ScreeningDecision,
    SourceProvenance,
    StreamingCalibrationCoreset,
    StructureArtifactIdentity,
)


def _coreset(
    capacity: int,
    resolver: ProtocolCompatibilityResolver,
    *,
    false_stable_cost: float = 5.0,
) -> StreamingCalibrationCoreset:
    posterior = FixedKernelResidualGP(
        resolver,
        config=FixedKernelGPConfig(length_scale=0.2),
    )
    builder = CalibrationUtilityBuilder(
        posterior,
        false_stable_cost=false_stable_cost,
    )
    return StreamingCalibrationCoreset(
        FacilityLocationCoresetPlanner(capacity, builder)
    )


def _protocol(functional: str = "PBE") -> ProtocolCertificate:
    return ProtocolCertificate(
        functional=functional,
        hubbard_u_ev={"Fe": 5.3} if functional == "PBE+U" else {},
        pseudopotential_set="PAW-2024",
        correction_scheme="MP2020",
        relaxation_protocol="static-after-relax",
        calculation_code="VASP-6.4",
    )


def _snapshot(reference: float = -1.0, version: str = "v1") -> HullSnapshot:
    return HullSnapshot(
        snapshot_id=f"hull-{version}",
        chemical_system=("Fe", "O"),
        reference_hull_energy_ev_per_atom=reference,
        phase_set_checksum="sha256:" + ("a" if version == "v1" else "b") * 64,
        known_through=datetime(2025, 12, 31, tzinfo=UTC),
        built_at=datetime(2026, 1, 1, tzinfo=UTC),
        source_version=version,
    )


def _query(
    query_id: str = "q1",
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
    base_energy: float = -1.03,
    protocol: ProtocolCertificate | None = None,
    snapshot: HullSnapshot | None = None,
) -> MaterialQuery:
    return MaterialQuery(
        query_id=query_id,
        structure_hash=f"structure-{query_id}",
        structure_identity=StructureArtifactIdentity.initial(
            query_id, f"structure-{query_id}"
        ),
        identity=MaterialIdentity(
            exact_calculation_id=f"calculation-{query_id}",
            canonical_structure_id=f"canonical-{query_id}",
            composition_family="Fe-O",
            prototype_family="corundum",
        ),
        composition="Fe2O3",
        embedding=embedding,
        protocol=protocol or _protocol(),
        hull_snapshot=snapshot or _snapshot(),
        base_predicted_formation_energy_ev_per_atom=base_energy,
    )


def _card(
    card_id: str = "c1",
    *,
    embedding: tuple[float, ...] = (1.0, 0.0),
    formation_energy: float = -0.94,
    base_energy: float = -1.03,
    protocol: ProtocolCertificate | None = None,
) -> MaterialMemoryCard:
    return MaterialMemoryCard(
        card_id=card_id,
        material_id=f"mp-{card_id}",
        structure_hash=f"structure-{card_id}",
        structure_identity=StructureArtifactIdentity.initial(
            f"mp-{card_id}", f"structure-{card_id}"
        ),
        identity=MaterialIdentity(
            exact_calculation_id=f"calculation-{card_id}",
            canonical_structure_id=f"canonical-{card_id}",
            composition_family="Fe-O",
            prototype_family="corundum",
        ),
        composition="Fe2O3",
        embedding=embedding,
        protocol=protocol or _protocol(),
        provenance=SourceProvenance(
            source_name="Materials Project",
            source_version="2026.01",
            record_locator=f"mp-{card_id}",
        ),
        formation_energy_ev_per_atom=formation_energy,
        base_predicted_formation_energy_ev_per_atom=base_energy,
        oracle_residual_ev_per_atom=formation_energy - base_energy,
        hull_snapshot=_snapshot(),
        recorded_hull_distance_ev_per_atom=formation_energy - _snapshot().reference_hull_energy_ev_per_atom,
    )


def test_card_preserves_raw_energy_and_recomputes_hull_on_new_snapshot() -> None:
    card = _card(formation_energy=-0.94)
    assert card.hull_distance() == pytest.approx(0.06)
    assert card.hull_distance(_snapshot(reference=-0.90, version="v2")) == pytest.approx(-0.04)
    invalid = card.model_copy(update={"oracle_residual_ev_per_atom": 0.0})
    with pytest.raises(ValidationError, match="oracle residual"):
        MaterialMemoryCard.model_validate(invalid.model_dump())


def test_hull_snapshot_and_query_time_are_causal() -> None:
    query = _query()
    future_query = query.model_copy(update={"as_of": datetime(2025, 1, 1, tzinfo=UTC)})
    with pytest.raises(ValidationError, match="built in its future"):
        MaterialQuery.model_validate(future_query.model_dump())


def test_protocol_resolver_is_fail_closed_without_same_structure_transport() -> None:
    source = _protocol("PBE")
    target = _protocol("PBE+U")
    resolver = ProtocolCompatibilityResolver()
    assert resolver.resolve(source, source).kind == CompatibilityKind.DIRECT
    assert resolver.resolve(source, target).kind == CompatibilityKind.REJECT
    correction = ResidualCorrector(resolver).correct(_query(protocol=target), [_card(protocol=source)])
    assert correction.status == "abstain_no_certificate_compatible_neighbor"


def test_explicit_transport_is_the_only_cross_protocol_path() -> None:
    source = _protocol("PBE")
    target = _protocol("PBE+U")
    transport = ProtocolTransportMap(
        source_protocol=source,
        target_protocol=target,
        slope=0.5,
        intercept_ev_per_atom=0.01,
        error_radius_ev_per_atom=0.02,
        matched_structure_count=12,
        calibration_group_checksum="sha256:" + "e" * 64,
        calibration_id="same-structure-match-v1",
    )
    resolver = ProtocolCompatibilityResolver([transport])
    correction = ResidualCorrector(resolver).correct(_query(protocol=target), [_card(protocol=source)])
    assert correction.status == "corrected"
    assert correction.residual_shift_ev_per_atom == pytest.approx(0.055)
    assert correction.uncertainty_radius_ev_per_atom == pytest.approx(0.02)
    with pytest.raises(ValidationError, match="identical protocols"):
        ProtocolTransportMap(
            source_protocol=source,
            target_protocol=source,
            slope=1.0,
            intercept_ev_per_atom=0.0,
            error_radius_ev_per_atom=0.0,
            matched_structure_count=3,
            calibration_group_checksum="sha256:" + "d" * 64,
            calibration_id="invalid-direct-transport",
        )


def test_transport_fit_requires_same_structure_calibration_and_has_no_identity_fallback() -> None:
    source = _protocol("PBE")
    target = _protocol("PBE+U")
    fitted = ProtocolTransportMap.fit_same_structure(
        source,
        target,
        [
                MatchedResidualPair(
                exact_calculation_id=f"calculation-{index}",
                canonical_structure_id=f"canonical-{index}",
                source_residual_ev_per_atom=float(index),
                target_residual_ev_per_atom=2.0 * index + 0.1,
            )
            for index in range(3)
        ],
        calibration_id="training-only-matched-structures",
    )
    assert fitted.slope == pytest.approx(2.0)
    assert fitted.intercept_ev_per_atom == pytest.approx(0.1)
    with pytest.raises(ValueError, match="three unique"):
        ProtocolTransportMap.fit_same_structure(
            source,
            target,
            [
                MatchedResidualPair(
                    exact_calculation_id="only-one-calculation",
                    canonical_structure_id="only-one-canonical",
                    source_residual_ev_per_atom=0.1,
                    target_residual_ev_per_atom=0.2,
                )
            ],
            calibration_id="underidentified",
        )


def test_canonical_group_split_blocks_equivalent_structure_leakage() -> None:
    split = CanonicalGroupSplit(
        calibration_groups=("canonical-calibration",),
        memory_groups=("canonical-memory",),
        evaluation_groups=("canonical-evaluation",),
    )
    split.assert_partition(
        MaterialIdentity(
            exact_calculation_id="database-mirror-a",
            canonical_structure_id="canonical-evaluation",
            composition_family="Fe-O",
        ),
        "evaluation",
    )
    with pytest.raises(ValueError, match="cross partitions"):
        CanonicalGroupSplit(
            calibration_groups=("canonical-shared",),
            memory_groups=("canonical-shared",),
            evaluation_groups=("canonical-evaluation",),
        )


def test_transport_fit_rejects_a_held_out_canonical_structure() -> None:
    source = _protocol("PBE")
    target = _protocol("PBE+U")
    pairs = [
        MatchedResidualPair(
            exact_calculation_id=f"calculation-{index}",
            canonical_structure_id=f"canonical-{index}",
            source_residual_ev_per_atom=float(index),
            target_residual_ev_per_atom=float(index),
        )
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="leaks held-out"):
        ProtocolTransportMap.fit_same_structure(
            source,
            target,
            pairs,
            calibration_id="leaky-transport",
            held_out_canonical_structure_ids=("canonical-1",),
        )


def test_coreset_selects_a_costly_false_stable_near_miss_not_a_redundant_card() -> None:
    resolver = ProtocolCompatibilityResolver()
    policy = _coreset(1, resolver, false_stable_cost=10.0)
    query = _query(base_energy=-1.03)
    near_miss = _card("near-miss", embedding=(1.0, 0.0), formation_energy=-0.94)
    harmless = _card("harmless", embedding=(0.9, 0.1), formation_energy=-1.02)
    selection = policy.planner.select_from_archive_greedy(
        [near_miss, harmless], [query]
    )
    assert selection.selected_card_ids == ("near-miss",)
    assert selection.objective_value > 0
    assert selection.facility_proxy_risk < selection.baseline_decision_risk


def test_redundant_failure_kill_test_preserves_distinct_risk_coverage() -> None:
    resolver = ProtocolCompatibilityResolver()
    policy = _coreset(2, resolver, false_stable_cost=10.0)
    queries = [_query("cluster-a", embedding=(1.0, 0.0)), _query("cluster-b", embedding=(0.0, 1.0))]
    cards = [
        _card("redundant-a1", embedding=(1.0, 0.0), formation_energy=-0.94),
        _card("redundant-a2", embedding=(1.0, 0.0), formation_energy=-0.94),
        _card("distinct-b", embedding=(0.0, 1.0), formation_energy=-0.94),
    ]
    selection = policy.planner.select_from_archive_greedy(cards, queries)
    assert "distinct-b" in selection.selected_card_ids
    assert len({"redundant-a1", "redundant-a2"} & set(selection.selected_card_ids)) == 1


def test_concept_recurrence_kill_test_retains_old_risk_card_beyond_fifo() -> None:
    resolver = ProtocolCompatibilityResolver()
    old = _card("old", embedding=(1.0, 0.0), formation_energy=-0.94)
    recent = _card("recent", embedding=(0.0, 1.0), formation_energy=-1.02)
    recurring = _query("recurring", embedding=(1.0, 0.0))
    coreset = _coreset(1, resolver)
    coreset.admit(old, [recurring])
    coreset.admit(recent, [recurring])
    fifo = FIFOBoundedMemory(capacity=1)
    fifo.admit(old)
    fifo.admit(recent)
    assert [card.card_id for card in coreset.cards()] == ["old"]
    assert [card.card_id for card in fifo.cards()] == ["recent"]


def test_adversarial_candidate_order_does_not_change_coreset_selection() -> None:
    resolver = ProtocolCompatibilityResolver()
    policy = _coreset(1, resolver)
    query = _query("order")
    cards = [
        _card("a", formation_energy=-0.94),
        _card("b", formation_energy=-0.94),
    ]
    first = policy.planner.select_from_archive_greedy(cards, [query])
    second = policy.planner.select_from_archive_greedy(list(reversed(cards)), [query])
    assert first.selected_card_ids == second.selected_card_ids
    assert first.objective_value == second.objective_value


def test_risk_controller_rejects_uncalibrated_stable_screen_and_controls_upper_bound() -> None:
    query = _query(base_energy=-1.03)
    resolver = ProtocolCompatibilityResolver()
    correction = ResidualCorrector(resolver).correct(query, [_card(formation_energy=-1.06)])
    controller = ProtocolRiskController(minimum_calibration_size=3)
    assert controller.screen(query, correction).decision == ScreeningDecision.ABSTAIN
    controller.fit(
        query,
        [0.01, 0.02, 0.03, 0.04],
        alpha=0.2,
        calibration_id="iid-protocol-calibration-v1",
        exchangeability_assumed=True,
    )
    decision = controller.screen(query, correction)
    assert decision.decision == ScreeningDecision.STABLE
    assert decision.upper_hull_distance_ev_per_atom == pytest.approx(-0.02)
