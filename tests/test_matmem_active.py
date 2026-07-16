from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from evimem.matmem import (
    ActiveDiscoveryEvaluator,
    BaseBoundaryAcquisition,
    BoundaryRiskPotential,
    BoundaryRiskRetention,
    CardOracleVault,
    DecisionAwareOnlineCoreset,
    FIFOBoundedMemory,
    HullSnapshot,
    MatchedAccessCostModel,
    MatchedAccessOperationLedger,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    ProtocolAwareBoundaryAcquisition,
    ProtocolCertificate,
    ProtocolCompatibilityResolver,
    RetentionAwareBoundaryAcquisition,
    SourceProvenance,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class _Case:
    query: MaterialQuery
    oracle_card: MaterialMemoryCard


def _vault(cases: list[_Case] | tuple[_Case, ...]) -> CardOracleVault:
    return CardOracleVault({case.query.query_id: case.oracle_card for case in cases})


def _evaluate(evaluator: ActiveDiscoveryEvaluator, cases: list[_Case] | tuple[_Case, ...]):
    return evaluator.evaluate((case.query for case in cases), _vault(cases))


class _SyntheticReviser:
    """Test double only; production causal updates require a phase diagram."""

    def revise(self, observed: MaterialMemoryCard, remaining_queries, *, call_index: int):
        result = {}
        for query in remaining_queries:
            reference = min(query.hull_snapshot.reference_hull_energy_ev_per_atom, observed.formation_energy_ev_per_atom)
            result[query.query_id] = query.hull_snapshot.model_copy(update={
                "snapshot_id": f"{query.hull_snapshot.snapshot_id}:test:{call_index}",
                "reference_hull_energy_ev_per_atom": reference,
                "built_at": query.as_of,
                "known_through": query.as_of,
            })
        return result

    def final_stability(self, selected_cards):
        cards = tuple(selected_cards)
        reference = min(card.formation_energy_ev_per_atom for card in cards)
        return {card.material_id: card.formation_energy_ev_per_atom - reference <= 0 for card in cards}


def _protocol(functional: str = "PBE") -> ProtocolCertificate:
    return ProtocolCertificate(
        functional=functional,
        pseudopotential_set="PAW-v1",
        correction_scheme="none",
        relaxation_protocol="full-relax",
        calculation_code="VASP-6",
    )


def _snapshot() -> HullSnapshot:
    return HullSnapshot(
        snapshot_id="hull-v1",
        chemical_system=("A", "B"),
        reference_hull_energy_ev_per_atom=-1.0,
        known_through=START,
        built_at=START,
        phase_set_checksum="sha256:" + "1" * 64,
        source_version="synthetic-v1",
    )


def _item(
    query_id: str,
    *,
    embedding: tuple[float, float],
    base_energy: float,
    oracle_energy: float,
    protocol: ProtocolCertificate | None = None,
) -> _Case:
    selected_protocol = protocol or _protocol()
    identity = MaterialIdentity(
        exact_calculation_id=f"calculation-{query_id}",
        canonical_structure_id=f"canonical-{query_id}",
        composition_family="A-B",
        prototype_family=f"prototype-{query_id}",
    )
    query = MaterialQuery(
        query_id=query_id,
        structure_hash=f"structure-{query_id}",
        identity=identity,
        composition="AB",
        embedding=embedding,
        base_predicted_formation_energy_ev_per_atom=base_energy,
        protocol=selected_protocol,
        hull_snapshot=_snapshot(),
        stability_threshold_ev_per_atom=0.0,
        as_of=START + timedelta(days=1),
    )
    card = MaterialMemoryCard(
        card_id=f"card-{query_id}",
        material_id=f"material-{query_id}",
        structure_hash=query.structure_hash,
        identity=identity,
        composition=query.composition,
        embedding=embedding,
        formation_energy_ev_per_atom=oracle_energy,
        base_predicted_formation_energy_ev_per_atom=base_energy,
        oracle_residual_ev_per_atom=oracle_energy - base_energy,
        protocol=selected_protocol,
        provenance=SourceProvenance(
            source_name="synthetic-mechanism-test",
            source_version="v1",
            record_locator=query_id,
        ),
        hull_snapshot=_snapshot(),
        recorded_hull_distance_ev_per_atom=oracle_energy + 1.0,
        observed_at=START + timedelta(days=2),
    )
    return _Case(query=query, oracle_card=card)


def test_synthetic_vault_rejects_oracle_from_another_protocol() -> None:
    item = _item(
        "protocol",
        embedding=(1.0, 0.0),
        base_energy=-1.02,
        oracle_energy=-0.95,
    )
    with pytest.raises(ValueError, match="query scientific protocol"):
        CardOracleVault(
            {
                item.query.query_id: item.oracle_card.model_copy(
                    update={"protocol": _protocol("SCAN")}
                )
            }
        ).benchmark_card(item.query)


def test_boundary_acquisition_uses_only_compatible_past_witnesses() -> None:
    resolver = ProtocolCompatibilityResolver()
    acquisition = ProtocolAwareBoundaryAcquisition(resolver, prior_strength=0.1)
    target = _item(
        "target",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
    )
    same_protocol = _item(
        "past",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
    ).oracle_card
    unsupported = _item(
        "unsupported",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.14,
        protocol=_protocol("SCAN"),
    ).oracle_card
    base_score = acquisition.score(target.query, [])
    supported_score = acquisition.score(target.query, [same_protocol])
    unsupported_score = acquisition.score(target.query, [unsupported])
    assert supported_score.stable_score < base_score.stable_score
    assert unsupported_score.stable_score == pytest.approx(base_score.stable_score)
    assert unsupported_score.compatible_witness_count == 0


def test_active_evaluator_reveals_only_selected_oracle_and_respects_budgets() -> None:
    candidates = [
        _item("a", embedding=(1.0, 0.0), base_energy=-1.04, oracle_energy=-0.94),
        _item("b", embedding=(0.0, 1.0), base_energy=-1.03, oracle_energy=-1.02),
        _item("c", embedding=(0.7, 0.7), base_energy=-1.01, oracle_energy=-1.01),
    ]
    metrics = _evaluate(ActiveDiscoveryEvaluator(
        BaseBoundaryAcquisition(),
        FIFOBoundedMemory(capacity=1),
        oracle_budget=2,
    ), candidates)
    assert metrics.oracle_calls == 2
    assert len(metrics.selected_query_ids) == 2
    assert metrics.average_memory_size == 1.0
    assert metrics.active_witness_budget == 1
    assert metrics.selected_query_ids[0] == "a"


def test_joint_acquisition_retention_avoids_recurring_false_stable_cluster() -> None:
    candidates = [
        *[
            _item(
                f"false-{index}",
                embedding=(1.0, 0.0),
                base_energy=-1.05 + index * 0.001,
                oracle_energy=-0.94,
            )
            for index in range(5)
        ],
        *[
            _item(
                f"stable-{index}",
                embedding=(0.0, 1.0),
                base_energy=-1.03 + index * 0.001,
                oracle_energy=-1.02,
            )
            for index in range(5)
        ],
    ]
    resolver = ProtocolCompatibilityResolver()
    base = _evaluate(ActiveDiscoveryEvaluator(
        BaseBoundaryAcquisition(),
        FIFOBoundedMemory(capacity=1),
        oracle_budget=5,
    ), candidates)
    joint = _evaluate(ActiveDiscoveryEvaluator(
        ProtocolAwareBoundaryAcquisition(resolver, prior_strength=0.1),
        DecisionAwareOnlineCoreset(capacity=1, resolver=resolver, false_stable_cost=5.0),
        oracle_budget=5,
    ), candidates)
    assert joint.cumulative_true_discoveries > base.cumulative_true_discoveries
    assert joint.unstable_oracle_calls < base.unstable_oracle_calls
    assert joint.cumulative_discovery_regret < base.cumulative_discovery_regret


def _with_cost(item: _Case, cost: float) -> _Case:
    return _Case(
        query=item.query.model_copy(update={"oracle_cost": cost}),
        oracle_card=item.oracle_card,
    )


def test_variable_cost_budget_never_overspends_and_stops_when_unaffordable() -> None:
    candidates = [
        _with_cost(
            _item("costly", embedding=(1.0, 0.0), base_energy=-1.20, oracle_energy=-1.05),
            2.0,
        ),
        _with_cost(
            _item("also-costly", embedding=(0.0, 1.0), base_energy=-1.05, oracle_energy=-1.02),
            1.0,
        ),
    ]
    result = _evaluate(ActiveDiscoveryEvaluator(
        BaseBoundaryAcquisition(),
        FIFOBoundedMemory(capacity=0),
        oracle_budget=2.5,
    ), candidates)
    assert result.selected_query_ids == ("costly",)
    assert result.oracle_cost_spent == 2.0
    assert result.oracle_cost_spent <= result.oracle_budget
    assert result.active_witness_budget == 0
    assert result.archive_size == 1
    assert result.steps[0].memory_size_after_observation == 0


def test_archive_grows_while_active_witness_set_respects_capacity() -> None:
    candidates = [
        _item(f"archive-{index}", embedding=(1.0, float(index)), base_energy=-1.04, oracle_energy=-1.03)
        for index in range(3)
    ]
    result = _evaluate(ActiveDiscoveryEvaluator(
        BaseBoundaryAcquisition(),
        FIFOBoundedMemory(capacity=1),
        oracle_budget=3,
    ), candidates)
    assert [step.archive_size_after_observation for step in result.steps] == [1, 2, 3]
    assert all(step.memory_size_after_observation <= 1 for step in result.steps)
    assert result.archive_size == 3


def test_pool_permutation_does_not_change_policy_actions() -> None:
    candidates = [
        _item("perm-c", embedding=(0.7, 0.7), base_energy=-1.02, oracle_energy=-1.01),
        _item("perm-a", embedding=(1.0, 0.0), base_energy=-1.04, oracle_energy=-0.94),
        _item("perm-b", embedding=(0.0, 1.0), base_energy=-1.03, oracle_energy=-1.05),
    ]

    def run(pool: list[_Case]) -> tuple[str, ...]:
        potential = BoundaryRiskPotential(ProtocolCompatibilityResolver())
        return _evaluate(ActiveDiscoveryEvaluator(
            RetentionAwareBoundaryAcquisition(potential, active_witness_budget=1),
            BoundaryRiskRetention(1, potential),
            oracle_budget=3,
        ), pool).selected_query_ids

    assert run(candidates) == run(list(reversed(candidates)))


def test_unrevealed_oracles_do_not_interfere_at_later_rounds() -> None:
    candidates = [
        _item("blind-1", embedding=(1.0, 0.0), base_energy=-1.05, oracle_energy=-0.95),
        _item("blind-2", embedding=(0.0, 1.0), base_energy=-1.04, oracle_energy=-1.08),
        _item("blind-3", embedding=(0.7, 0.7), base_energy=-1.03, oracle_energy=-0.96),
        _item("blind-4", embedding=(-0.7, 0.7), base_energy=-1.02, oracle_energy=-1.10),
    ]

    def run(pool: list[_Case]) -> tuple[str, ...]:
        potential = BoundaryRiskPotential(ProtocolCompatibilityResolver())
        return _evaluate(ActiveDiscoveryEvaluator(
            RetentionAwareBoundaryAcquisition(potential, active_witness_budget=1),
            BoundaryRiskRetention(1, potential),
            oracle_budget=2,
        ), pool).selected_query_ids

    selected = run(candidates)
    altered = []
    for item in candidates:
        if item.query.query_id in selected:
            altered.append(item)
            continue
        new_energy = -1.30 if item.oracle_card.formation_energy_ev_per_atom > -1.0 else -0.80
        altered_card = item.oracle_card.model_copy(
            update={
                "formation_energy_ev_per_atom": new_energy,
                "oracle_residual_ev_per_atom": new_energy
                - item.query.base_predicted_formation_energy_ev_per_atom,
                "recorded_hull_distance_ev_per_atom": new_energy + 1.0,
            }
        )
        altered.append(_Case(query=item.query, oracle_card=altered_card))
    assert run(altered) == selected


def test_causal_and_final_hull_discoveries_are_reported_separately() -> None:
    candidates = [
        _item(
            "provisional",
            embedding=(1.0, 0.0),
            base_energy=-1.05,
            oracle_energy=-1.02,
        ),
        _item(
            "deep-later",
            embedding=(0.0, 1.0),
            base_energy=-1.04,
            oracle_energy=-1.20,
        ),
    ]
    result = _evaluate(ActiveDiscoveryEvaluator(
        BaseBoundaryAcquisition(),
        FIFOBoundedMemory(capacity=0),
        oracle_budget=2,
        causal_hull_updates=True,
        causal_hull_reviser=_SyntheticReviser(),
    ), candidates)
    assert result.query_time_causal_discoveries == 2
    assert result.final_hull_confirmed_discoveries == 1
    assert result.invalidated_provisional_discoveries == 1
    assert result.steps[0].actual_stable
    assert not result.steps[0].final_hull_stable


def test_matched_access_ledger_exposes_hull_churn_break_even() -> None:
    candidates = [
        _item(
            "provisional",
            embedding=(1.0, 0.0),
            base_energy=-1.05,
            oracle_energy=-1.02,
        ),
        _item(
            "deep-later",
            embedding=(0.0, 1.0),
            base_energy=-1.04,
            oracle_energy=-1.20,
        ),
    ]
    metrics = _evaluate(ActiveDiscoveryEvaluator(
        BaseBoundaryAcquisition(),
        FIFOBoundedMemory(capacity=2),
        oracle_budget=2,
        causal_hull_updates=True,
        causal_hull_reviser=_SyntheticReviser(),
    ), candidates)
    ledger = MatchedAccessOperationLedger.from_metrics(metrics)
    assert metrics.steps[0].causal_hull_transition_after_observation
    assert ledger.oracle_admission_certifications == 2
    assert ledger.common_witness_scans == 1
    assert ledger.persistent_hull_recertifications == 1
    assert ledger.on_demand_archive_retrievals == 1
    assert ledger.on_demand_recertifications == 1

    free_retrieval = MatchedAccessCostModel(
        archive_retrieval_cost=0.0,
        persistent_recertification_cost=2.0,
        on_demand_recertification_cost=1.0,
    ).evaluate(ledger)
    assert free_retrieval.persistent_net_savings == pytest.approx(-1.0)

    costly_retrieval = MatchedAccessCostModel(
        archive_retrieval_cost=2.0,
        persistent_recertification_cost=2.0,
        on_demand_recertification_cost=1.0,
    ).evaluate(ledger)
    assert costly_retrieval.persistent_net_savings == pytest.approx(1.0)


def test_matched_access_ledger_has_no_persistent_recertification_without_hull_churn() -> None:
    candidates = [
        _item("one", embedding=(1.0, 0.0), base_energy=-1.04, oracle_energy=-1.02),
        _item("two", embedding=(0.0, 1.0), base_energy=-1.03, oracle_energy=-1.01),
    ]
    metrics = _evaluate(ActiveDiscoveryEvaluator(
        BaseBoundaryAcquisition(),
        FIFOBoundedMemory(capacity=2),
        oracle_budget=2,
        causal_hull_updates=False,
    ), candidates)
    ledger = MatchedAccessOperationLedger.from_metrics(metrics)
    assert ledger.persistent_hull_recertifications == 0
    assert ledger.on_demand_archive_retrievals == 1


def test_hypothetical_witnesses_never_enter_the_real_archive() -> None:
    candidate = _item(
        "real-only",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.02,
    )
    potential = BoundaryRiskPotential(ProtocolCompatibilityResolver())
    result = _evaluate(ActiveDiscoveryEvaluator(
        RetentionAwareBoundaryAcquisition(potential, active_witness_budget=1),
        BoundaryRiskRetention(1, potential),
        oracle_budget=1,
    ), [candidate])
    assert result.archive_card_ids == (candidate.oracle_card.card_id,)
    assert all(not card_id.startswith("hypothetical:") for card_id in result.archive_card_ids)
