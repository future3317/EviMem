from __future__ import annotations

import itertools
import random

from evimem.matmem import (
    ActiveDiscoveryEvaluator,
    BoundaryRiskConfig,
    BoundaryRiskPotential,
    BoundaryRiskRetention,
    CandidatePoolItem,
    ProtocolCompatibilityResolver,
    RetentionAwareBoundaryAcquisition,
)

from .test_matmem_active import _item, _protocol, _SyntheticReviser


def test_boundary_potential_links_coverage_to_hull_margin_ambiguity() -> None:
    resolver = ProtocolCompatibilityResolver()
    potential = BoundaryRiskPotential(
        resolver,
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.05,
            prior_radius_ev_per_atom=0.15,
            calibration_radius_ev_per_atom=0.01,
        ),
    )
    target = _item(
        "target",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
    )
    witness = _item(
        "witness",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
    ).oracle_card
    without_memory = potential.estimate(target.query, [])
    with_memory = potential.estimate(target.query, [witness])
    assert without_memory.boundary_ambiguous
    assert without_memory.weighted_risk_upper_bound > 0
    assert not with_memory.boundary_ambiguous
    assert with_memory.weighted_risk_upper_bound == 0


def test_boundary_potential_rejects_unsupported_protocol_witness() -> None:
    potential = BoundaryRiskPotential(ProtocolCompatibilityResolver())
    target = _item(
        "target-protocol",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
    )
    unsupported = _item(
        "other-protocol",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
        protocol=_protocol("SCAN"),
    ).oracle_card
    base = potential.estimate(target.query, [])
    rejected = potential.estimate(target.query, [unsupported])
    assert rejected == base


def test_boundary_retention_optimizes_remaining_pool_without_future_labels() -> None:
    resolver = ProtocolCompatibilityResolver()
    potential = BoundaryRiskPotential(
        resolver,
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.0,
            calibration_radius_ev_per_atom=0.0,
        ),
    )
    false_witness = _item(
        "false-witness",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
    ).oracle_card
    stable_witness = _item(
        "stable-witness",
        embedding=(0.0, 1.0),
        base_energy=-1.04,
        oracle_energy=-1.20,
    ).oracle_card
    future = [
        _item(
            f"future-a-{index}",
            embedding=(1.0, 0.0),
            base_energy=-1.04,
            oracle_energy=-0.94,
        ).query
        for index in range(2)
    ]
    future.append(
        _item(
            "future-b",
            embedding=(0.0, 1.0),
            base_energy=-1.04,
            oracle_energy=-1.20,
        ).query
    )
    selection = BoundaryRiskRetention(1, potential).select(
        [false_witness, stable_witness],
        future,
    )
    assert selection.selected_card_ids == (false_witness.card_id,)


def test_retention_aware_information_value_depends_on_active_witness_budget() -> None:
    resolver = ProtocolCompatibilityResolver()
    potential = BoundaryRiskPotential(
        resolver,
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.0,
            prior_radius_ev_per_atom=0.15,
            calibration_radius_ev_per_atom=0.0,
        ),
    )
    current = _item(
        "current-a",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.94,
    ).oracle_card
    candidate = _item(
        "candidate-b",
        embedding=(0.0, 1.0),
        base_energy=-1.04,
        oracle_energy=-1.20,
    ).query
    future = (
        _item(
            "future-a",
            embedding=(1.0, 0.0),
            base_energy=-1.04,
            oracle_energy=-0.94,
        ).query,
        _item(
            "future-b",
            embedding=(0.0, 1.0),
            base_energy=-1.04,
            oracle_energy=-1.20,
        ).query,
    )
    capacity_one = RetentionAwareBoundaryAcquisition(
        potential,
        active_witness_budget=1,
        outcome_margin_ev_per_atom=0.03,
    ).score(candidate, future, (current,))
    capacity_two = RetentionAwareBoundaryAcquisition(
        potential,
        active_witness_budget=2,
        outcome_margin_ev_per_atom=0.03,
    ).score(candidate, future, (current,))
    assert capacity_one.downstream_risk_reduction == 0
    assert capacity_two.downstream_risk_reduction > capacity_one.downstream_risk_reduction


def test_first_acquisition_is_invariant_to_hidden_oracle_outcomes() -> None:
    resolver = ProtocolCompatibilityResolver()
    potential = BoundaryRiskPotential(resolver)
    queries = [
        _item(
            "blind-a",
            embedding=(1.0, 0.0),
            base_energy=-1.04,
            oracle_energy=-0.94,
        ),
        _item(
            "blind-b",
            embedding=(0.0, 1.0),
            base_energy=-1.03,
            oracle_energy=-1.20,
        ),
    ]
    swapped = [
        CandidatePoolItem(
            query=queries[0].query,
            oracle_card=queries[0].oracle_card.model_copy(
                update={
                    "formation_energy_ev_per_atom": -1.20,
                    "oracle_residual_ev_per_atom": -1.20
                    - queries[0].query.base_predicted_formation_energy_ev_per_atom,
                    "recorded_hull_distance_ev_per_atom": -0.20,
                }
            ),
        ),
        CandidatePoolItem(
            query=queries[1].query,
            oracle_card=queries[1].oracle_card.model_copy(
                update={
                    "formation_energy_ev_per_atom": -0.94,
                    "oracle_residual_ev_per_atom": -0.94
                    - queries[1].query.base_predicted_formation_energy_ev_per_atom,
                    "recorded_hull_distance_ev_per_atom": 0.06,
                }
            ),
        ),
    ]

    def first_selected(pool: list[CandidatePoolItem]) -> str:
        result = ActiveDiscoveryEvaluator(
            RetentionAwareBoundaryAcquisition(potential, active_witness_budget=1),
            BoundaryRiskRetention(1, potential),
            oracle_budget=1,
        ).evaluate(pool)
        return result.selected_query_ids[0]

    assert first_selected(queries) == first_selected(swapped)


def test_causal_hull_revision_uses_only_an_already_observed_phase() -> None:
    resolver = ProtocolCompatibilityResolver()
    potential = BoundaryRiskPotential(resolver)
    candidates = [
        _item(
            "deep-phase",
            embedding=(1.0, 0.0),
            base_energy=-1.20,
            oracle_energy=-1.20,
        ),
        _item(
            "later-candidate",
            embedding=(0.0, 1.0),
            base_energy=-1.03,
            oracle_energy=-1.02,
        ),
    ]
    result = ActiveDiscoveryEvaluator(
        RetentionAwareBoundaryAcquisition(potential, active_witness_budget=1),
        BoundaryRiskRetention(1, potential),
        oracle_budget=2,
        causal_hull_updates=True,
        causal_hull_reviser=_SyntheticReviser(),
    ).evaluate(candidates)
    assert result.selected_query_ids[0] == "deep-phase"
    assert result.hull_revision_count == 1
    assert not result.steps[1].actual_stable


def test_information_value_excludes_queried_item_removal() -> None:
    potential = BoundaryRiskPotential(
        ProtocolCompatibilityResolver(),
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.0,
            calibration_radius_ev_per_atom=0.0,
        ),
    )
    candidate = _item(
        "orthogonal-query",
        embedding=(1.0, 0.0),
        base_energy=-0.90,
        oracle_energy=-0.90,
    ).query
    different_system = candidate.hull_snapshot.model_copy(
        update={"snapshot_id": "other-system", "chemical_system": ("C", "D")}
    )
    candidate = candidate.model_copy(update={"hull_snapshot": different_system})
    remaining = (
        _item(
            "unaffected-future",
            embedding=(0.0, 1.0),
            base_energy=-1.04,
            oracle_energy=-1.04,
        ).query,
    )
    acquisition = RetentionAwareBoundaryAcquisition(potential, active_witness_budget=1)
    assert acquisition.information_value(candidate, remaining, ()) == 0.0


def test_boundary_weight_is_fixed_across_working_sets() -> None:
    potential = BoundaryRiskPotential(
        ProtocolCompatibilityResolver(),
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.0,
            calibration_radius_ev_per_atom=0.0,
            minimum_boundary_weight=0.1,
        ),
    )
    target = _item(
        "fixed-weight-target",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.04,
    )
    shifting_witness = _item(
        "fixed-weight-witness",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.96,
    ).oracle_card
    without = potential.estimate(target.query, ())
    with_witness = potential.estimate(target.query, (shifting_witness,))
    assert with_witness.center_hull_distance_ev_per_atom != without.center_hull_distance_ev_per_atom
    assert with_witness.boundary_weight == without.boundary_weight


def test_compatible_intervals_intersect_and_conflicts_fail_closed() -> None:
    potential = BoundaryRiskPotential(
        ProtocolCompatibilityResolver(),
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.0,
            calibration_radius_ev_per_atom=0.02,
        ),
    )
    target = _item(
        "intersection-target",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.04,
    )
    first = _item(
        "intersection-first",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.02,
    ).oracle_card
    second = _item(
        "intersection-second",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.00,
    ).oracle_card
    one = potential.estimate(target.query, (first,))
    both = potential.estimate(target.query, (first, second))
    assert both.radius_ev_per_atom < one.radius_ev_per_atom
    assert both.source_witness_ids == (first.card_id, second.card_id)
    assert not both.interval_conflict

    conflicting = BoundaryRiskPotential(
        ProtocolCompatibilityResolver(),
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.0,
            calibration_radius_ev_per_atom=0.001,
        ),
    ).estimate(target.query, (first, second))
    assert conflicting.interval_conflict
    assert conflicting.abstained
    assert conflicting.weighted_risk_upper_bound > 0


def test_retention_matches_manual_exhaustive_search_on_random_small_instances() -> None:
    rng = random.Random(17)
    for trial in range(8):
        potential = BoundaryRiskPotential(
            ProtocolCompatibilityResolver(),
            BoundaryRiskConfig(
                residual_lipschitz_ev_per_atom=0.02,
                calibration_radius_ev_per_atom=0.01,
            ),
        )
        cards = [
            _item(
                f"exact-card-{trial}-{index}",
                embedding=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
                base_energy=-1.04,
                oracle_energy=-1.04 + rng.uniform(-0.08, 0.08),
            ).oracle_card
            for index in range(4)
        ]
        queries = tuple(
            _item(
                f"exact-query-{trial}-{index}",
                embedding=(rng.uniform(-1, 1), rng.uniform(-1, 1)),
                base_energy=-1.04 + rng.uniform(-0.03, 0.03),
                oracle_energy=-1.04,
            ).query
            for index in range(3)
        )
        manual = []
        for size in range(3):
            for retained in itertools.combinations(cards, size):
                ids = tuple(sorted(card.card_id for card in retained))
                manual.append((potential.evaluate(queries, retained).total, -size, ids))
        expected = min(manual, key=lambda item: (item[0], item[1], item[2]))[2]
        selected = BoundaryRiskRetention(2, potential).select(cards, queries)
        assert selected.selected_card_ids == expected


def test_exact_retention_can_remove_two_old_witnesses_after_conflicting_admission() -> None:
    """The best legal set may have fewer than K members when intervals conflict."""

    potential = BoundaryRiskPotential(
        ProtocolCompatibilityResolver(),
        BoundaryRiskConfig(
            residual_lipschitz_ev_per_atom=0.0,
            calibration_radius_ev_per_atom=0.02,
        ),
    )
    target = _item(
        "multi-eviction-target",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.04,
    ).query
    old_one = _item(
        "old-ambiguous-one",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.00,
    ).oracle_card
    old_two = _item(
        "old-ambiguous-two",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-0.99,
    ).oracle_card
    new_witness = _item(
        "new-certifying-witness",
        embedding=(1.0, 0.0),
        base_energy=-1.04,
        oracle_energy=-1.10,
    ).oracle_card

    assert potential.estimate(target, (new_witness, old_one)).interval_conflict
    assert potential.estimate(target, (new_witness, old_two)).interval_conflict
    assert potential.evaluate((target,), (new_witness,)).total == 0.0
    assert potential.evaluate((target,), (old_one,)).total > 0.0
    assert potential.evaluate((target,), (old_two,)).total > 0.0

    selected = BoundaryRiskRetention(2, potential).select(
        (old_one, old_two, new_witness),
        (target,),
    )
    assert selected.selected_card_ids == (new_witness.card_id,)
    assert selected.evicted_card_ids == (old_one.card_id, old_two.card_id)
