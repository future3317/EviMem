from __future__ import annotations

import itertools

import pytest

from matmem.hull_certificate import (
    ActionValueInterval,
    PhaseEnergyInterval,
    RobustHullDecisionCertifier,
    RobustHullDecisionKind,
    certify_epsilon_optimal_actions,
    clustered_conformal_quantile,
)


def _phase(
    phase_id: str,
    composition: dict[str, float],
    lower: float,
    upper: float | None = None,
) -> PhaseEnergyInterval:
    return PhaseEnergyInterval(
        phase_id=phase_id,
        element_fractions=composition,
        lower_energy_ev_per_atom=lower,
        upper_energy_ev_per_atom=lower if upper is None else upper,
    )


def test_robust_hull_certifies_stable_unstable_and_abstain() -> None:
    references = (
        _phase("A", {"A": 1.0}, 0.0),
        _phase("B", {"B": 1.0}, 0.0),
    )
    certifier = RobustHullDecisionCertifier()
    stable = certifier.certify(
        _phase("stable", {"A": 0.5, "B": 0.5}, -0.2, -0.1), references
    )
    unstable = certifier.certify(
        _phase("unstable", {"A": 0.5, "B": 0.5}, 0.1, 0.2), references
    )
    abstain = certifier.certify(
        _phase("uncertain", {"A": 0.5, "B": 0.5}, -0.1, 0.1), references
    )
    assert stable.kind is RobustHullDecisionKind.STABLE
    assert unstable.kind is RobustHullDecisionKind.UNSTABLE
    assert abstain.kind is RobustHullDecisionKind.ABSTAIN


def test_candidate_is_excluded_from_its_own_competing_hull() -> None:
    candidate = _phase("candidate", {"A": 0.5, "B": 0.5}, -1.0)
    references = (
        candidate,
        _phase("A", {"A": 1.0}, 0.0),
        _phase("B", {"B": 1.0}, 0.0),
    )
    decision = RobustHullDecisionCertifier().certify(candidate, references)
    assert decision.lower_competing_hull_ev_per_atom == pytest.approx(0.0)
    assert decision.kind is RobustHullDecisionKind.STABLE


def test_certified_decisions_are_sound_for_every_interval_endpoint() -> None:
    references = (
        _phase("A", {"A": 1.0}, -0.02, 0.02),
        _phase("B", {"B": 1.0}, -0.03, 0.01),
    )
    candidates = (
        _phase("stable", {"A": 0.5, "B": 0.5}, -0.25, -0.15),
        _phase("unstable", {"A": 0.5, "B": 0.5}, 0.15, 0.25),
    )
    certifier = RobustHullDecisionCertifier()
    for candidate in candidates:
        decision = certifier.certify(candidate, references)
        assert decision.kind is not RobustHullDecisionKind.ABSTAIN
        for a_energy, b_energy, candidate_energy in itertools.product(
            (-0.02, 0.02), (-0.03, 0.01),
            (candidate.lower_energy_ev_per_atom, candidate.upper_energy_ev_per_atom),
        ):
            actual_hull = 0.5 * a_energy + 0.5 * b_energy
            actually_stable = candidate_energy <= actual_hull
            assert actually_stable is (decision.kind is RobustHullDecisionKind.STABLE)


def test_infeasible_competing_composition_abstains() -> None:
    decision = RobustHullDecisionCertifier().certify(
        _phase("AB", {"A": 0.5, "B": 0.5}, -0.1, 0.1),
        (_phase("A", {"A": 1.0}, 0.0),),
    )
    assert decision.kind is RobustHullDecisionKind.ABSTAIN
    assert decision.reason == "competing_hull_infeasible"


def test_epsilon_optimal_action_sets_have_guaranteed_and_possible_semantics() -> None:
    result = certify_epsilon_optimal_actions(
        (
            ActionValueInterval(action_id="a", lower_value=0.9, upper_value=1.0),
            ActionValueInterval(action_id="b", lower_value=0.5, upper_value=0.6),
            ActionValueInterval(action_id="c", lower_value=0.7, upper_value=1.1),
        ),
        epsilon=0.2,
    )
    assert result.guaranteed_epsilon_optimal_action_ids == ("a",)
    assert result.possible_epsilon_optimal_action_ids == ("a", "c")


def test_clustered_conformal_quantile_uses_finite_sample_order_statistic() -> None:
    calibration = clustered_conformal_quantile(
        tuple(float(index) for index in range(1, 11)), alpha=0.1
    )
    assert calibration.order_statistic_one_based == 10
    assert calibration.radius == 10.0
    with pytest.raises(ValueError, match="too few"):
        clustered_conformal_quantile(tuple(float(index) for index in range(1, 9)), alpha=0.1)
