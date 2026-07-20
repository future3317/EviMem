from __future__ import annotations

import itertools

import numpy as np
import pytest

from matmem import (
    CalibrationUtilityBuilder,
    CalibrationUtilityMatrix,
    ExactArchivePosteriorProjectionPlanner,
    FacilityLocationCoresetPlanner,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    FrozenHullDistanceAcquisition,
    GPVarianceOneSwapMemory,
    JointPosteriorRiskOneSwapPlanner,
    PosteriorProjectionOneSwapPlanner,
    PosteriorProjectionScorer,
    ProperPosteriorDivergence,
    ProtocolCompatibilityResolver,
    ResidualPrediction,
    SurvivalConditionedAcquisition,
    bernoulli_brier_divergence,
    bernoulli_log_divergence,
    compare_facility_and_joint_objectives,
    reference_decision_regret,
    threshold_weighted_crps_divergence,
)

from .test_matmem import _card, _protocol, _query


class _FixedUtilityBuilder:
    def __init__(
        self,
        gain_by_card: dict[str, tuple[float, ...]],
        baseline: float = 10.0,
    ):
        self.gain_by_card = gain_by_card
        self.baseline = baseline

    def build(self, queries, witnesses):
        query_items = tuple(queries)
        witness_items = tuple(witnesses)
        columns = []
        for card in witness_items:
            gain = next(
                (
                    values
                    for prefix, values in self.gain_by_card.items()
                    if card.card_id.startswith(prefix)
                ),
                (0.0,) * len(query_items),
            )
            if len(gain) != len(query_items):
                raise ValueError("fixed test gain has the wrong query dimension")
            columns.append(gain)
        gains = (
            np.asarray(columns, dtype=float).T
            if columns
            else np.zeros((len(query_items), 0), dtype=float)
        )
        return (
            CalibrationUtilityMatrix(
                tuple(item.query_id for item in query_items),
                tuple(item.card_id for item in witness_items),
                gains,
            ),
            self.baseline,
        )


class _ExplodingPosterior:
    def fit(self, cards):
        raise AssertionError("zero-weight survival acquisition must not fit a posterior")


class _ZeroPosterior:
    def fit(self, cards):
        self.cards = tuple(cards)
        return self

    def predict(self, queries):
        items = tuple(queries)
        return ResidualPrediction(
            query_ids=tuple(item.query_id for item in items),
            mean_ev_per_atom=(0.0,) * len(items),
            std_ev_per_atom=(0.1,) * len(items),
            stable_probability=(0.5,) * len(items),
            compatible_witness_count=(len(getattr(self, "cards", ())),) * len(items),
        )

    def sample_residuals(self, query, *, num_samples: int, seed: int):
        del query, seed
        return np.zeros(num_samples)


def test_facility_location_matrix_is_monotone_and_submodular() -> None:
    matrix = CalibrationUtilityMatrix(
        query_ids=("u1", "u2", "u3"),
        witness_ids=("a", "b", "c"),
        gains=np.asarray(
            [
                [4.0, 1.0, 2.0],
                [0.0, 5.0, 2.0],
                [1.0, 1.0, 3.0],
            ]
        ),
    )
    witnesses = set(matrix.witness_ids)
    for size in range(4):
        for subset in itertools.combinations(matrix.witness_ids, size):
            for candidate in witnesses - set(subset):
                assert matrix.value((*subset, candidate)) >= matrix.value(subset)
    for a_size in range(4):
        for a_tuple in itertools.combinations(matrix.witness_ids, a_size):
            a = set(a_tuple)
            for b_size in range(a_size, 4):
                for b_tuple in itertools.combinations(matrix.witness_ids, b_size):
                    b = set(b_tuple)
                    if not a.issubset(b):
                        continue
                    for candidate in witnesses - b:
                        assert matrix.marginal_gain(a, candidate) >= matrix.marginal_gain(
                            b, candidate
                        )


def test_streaming_one_swap_matches_exhaustive_union_optimum() -> None:
    queries = (_query("u1"), _query("u2"))
    current = (_card("a"), _card("b"))
    new = _card("new")
    planner = FacilityLocationCoresetPlanner(
        2,
        _FixedUtilityBuilder(
            {
                "a": (4.0, 0.0),
                "b": (0.0, 4.0),
                "new": (5.0, 5.0),
            }
        ),
    )
    preview = planner.preview_admit(current, new, queries)
    matrix, _ = planner.build_utility_matrix(queries, (*current, new))
    exhaustive_value = max(
        matrix.value(subset)
        for size in range(3)
        for subset in itertools.combinations(matrix.witness_ids, size)
    )
    assert preview.objective_value == exhaustive_value
    assert preview.admitted_new_card
    assert len(preview.evicted_card_ids) == 1


def test_streaming_rejects_redundant_or_too_small_gain() -> None:
    query = (_query("u"),)
    current = (_card("a"),)
    planner = FacilityLocationCoresetPlanner(
        1,
        _FixedUtilityBuilder({"a": (3.0,), "new": (3.1,)}),
        min_admission_gain=0.2,
    )
    preview = planner.preview_admit(current, _card("new"), query)
    assert preview.selected_card_ids == ("a",)
    assert not preview.admitted_new_card
    assert preview.objective_improvement == 0


def test_fixed_kernel_posterior_is_protocol_safe_and_deterministic() -> None:
    resolver = ProtocolCompatibilityResolver()
    posterior = FixedKernelResidualGP(
        resolver,
        config=FixedKernelGPConfig(length_scale=0.2),
    ).fit((_card("positive", formation_energy=-0.90),))
    compatible = posterior.predict((_query("compatible"),))
    incompatible = posterior.predict((_query("incompatible", protocol=_protocol("PBE+U")),))
    assert compatible.compatible_witness_count == (1,)
    assert compatible.mean_ev_per_atom[0] > 0
    assert incompatible.compatible_witness_count == (0,)
    assert incompatible.mean_ev_per_atom == (0.0,)
    left = posterior.sample_residuals(_query("sample"), num_samples=5, seed=7)
    right = posterior.sample_residuals(_query("sample"), num_samples=5, seed=7)
    assert np.array_equal(left, right)


def test_fixed_gp_predictive_discrepancy_and_transport_uncertainty_are_distinct() -> None:
    config = FixedKernelGPConfig(
        length_scale=0.35,
        signal_std_ev_per_atom=0.08,
        noise_std_ev_per_atom=0.02,
        jitter=1e-12,
    )
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=config,
    )
    query = _query("variance-semantics", embedding=(1.0, 0.0))

    prior = posterior.fit(()).predict((query,))
    assert prior.std_ev_per_atom[0] ** 2 == pytest.approx(
        config.signal_std_ev_per_atom**2 + config.noise_std_ev_per_atom**2,
        abs=1e-12,
    )

    exact_card = _card(
        "variance-semantics-card",
        embedding=(1.0, 0.0),
        formation_energy=-1.04,
    )
    conditioned = posterior.fit((exact_card,)).predict((query,))
    # The independent predictive discrepancy remains after conditioning. An
    # exact-protocol card has no transport radius, so it is not double-counted.
    assert (
        conditioned.std_ev_per_atom[0] ** 2
        >= config.noise_std_ev_per_atom**2 - 1e-10
    )
    assert conditioned.std_ev_per_atom[0] < prior.std_ev_per_atom[0]


def test_baseline_risk_uses_the_same_boundary_weights_as_gains() -> None:
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(
            ProtocolCompatibilityResolver(),
            config=FixedKernelGPConfig(length_scale=0.2),
        ),
        boundary_scale_ev_per_atom=0.05,
    )
    queries = (
        _query("near", base_energy=-0.99),
        _query("far", base_energy=-0.70),
    )
    _, baseline = builder.build(queries, ())
    assert baseline == pytest.approx(builder.weighted_decision_risk(queries, ()))
    unweighted = sum(
        builder.posterior_template.clone_unfit()
        .fit(())
        .decision_risks(
            queries,
            false_stable_cost=builder.false_stable_cost,
            false_unstable_cost=builder.false_unstable_cost,
        )
        .values()
    )
    assert baseline != pytest.approx(unweighted)


@pytest.mark.parametrize("seed", range(10))
def test_single_witness_helper_equals_general_one_card_gp(seed: int) -> None:
    generator = np.random.default_rng(seed)
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(
            ProtocolCompatibilityResolver(),
            config=FixedKernelGPConfig(length_scale=0.35),
        )
    )
    queries = tuple(
        _query(
            f"single-q-{index}",
            embedding=tuple(generator.normal(size=4)),
            base_energy=float(-1 + generator.normal(scale=0.08)),
        )
        for index in range(5)
    )
    witness = _card(
        "single-card",
        embedding=tuple(generator.normal(size=4)),
        formation_energy=float(-1 + generator.normal(scale=0.08)),
        base_energy=-1.0,
    )
    fast = builder.posterior_template.single_witness_decision_risks(
        queries,
        witness,
        false_stable_cost=builder.false_stable_cost,
        false_unstable_cost=builder.false_unstable_cost,
    )
    general = (
        builder.posterior_template.clone_unfit()
        .fit((witness,))
        .decision_risks(
            queries,
            false_stable_cost=builder.false_stable_cost,
            false_unstable_cost=builder.false_unstable_cost,
        )
    )
    assert fast == pytest.approx(general, abs=1e-12)


def test_proper_projection_rejects_wrong_extreme_self_risk_preference() -> None:
    reference = 0.99
    wrong_extreme = 0.01
    correct_extreme = 0.99

    def self_risk(probability: float) -> float:
        return min(5.0 * (1 - probability), probability)

    assert self_risk(wrong_extreme) < self_risk(correct_extreme)
    assert bernoulli_brier_divergence(reference, correct_extreme) == 0
    assert bernoulli_brier_divergence(reference, wrong_extreme) > bernoulli_brier_divergence(
        reference, correct_extreme
    )
    assert bernoulli_log_divergence(reference, wrong_extreme) > 0
    assert (
        reference_decision_regret(
            reference,
            wrong_extreme,
            false_stable_cost=5.0,
            false_unstable_cost=1.0,
        )
        > 0
    )


def test_threshold_weighted_crps_is_zero_only_for_matching_gaussian() -> None:
    assert (
        threshold_weighted_crps_divergence(0.0, 0.1, 0.0, 0.1, threshold=0.02, bandwidth=0.05) == 0
    )
    assert (
        threshold_weighted_crps_divergence(0.0, 0.1, 0.08, 0.1, threshold=0.02, bandwidth=0.05) > 0
    )


@pytest.mark.parametrize("kind", ("brier", "log", "gaussian_kl", "threshold_weighted_crps"))
def test_posterior_projection_one_swap_is_exact_in_union(kind: str) -> None:
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    planner = PosteriorProjectionOneSwapPlanner(
        2,
        posterior,
        ProperPosteriorDivergence(kind=kind),  # type: ignore[arg-type]
    )
    queries = (
        _query("p3c-q-a", embedding=(1.0, 0.0), base_energy=-1.01),
        _query("p3c-q-b", embedding=(0.0, 1.0), base_energy=-0.97),
    )
    current = (
        _card("p3c-old-a", embedding=(1.0, 0.0), formation_energy=-1.07),
        _card("p3c-old-b", embedding=(0.0, 1.0), formation_energy=-0.96),
    )
    new = _card("p3c-new", embedding=(0.7, 0.7), formation_energy=-1.03)
    selection = planner.preview_admit(current, new, queries)
    expected = min(
        selection.candidates,
        key=lambda item: (
            item.proper_divergence + item.reactivation_cost,
            tuple(sorted(item.selected_card_ids)),
        ),
    )
    assert len(selection.candidates) == 3
    assert selection.selected_card_ids == expected.selected_card_ids
    assert selection.reference_card_ids == ("p3c-old-a", "p3c-old-b", "p3c-new")


def test_posterior_projection_candidates_are_nondegenerate_real_gp_posteriors() -> None:
    generator = np.random.default_rng(0)
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    queries = tuple(
        _query(
            f"nondegenerate-q-{index}",
            embedding=tuple(generator.normal(size=3)),
            base_energy=float(-1.0 + generator.normal(scale=0.05)),
        )
        for index in range(5)
    )
    current = tuple(
        _card(
            f"nondegenerate-card-{index}",
            embedding=tuple(generator.normal(size=3)),
            formation_energy=float(-1.0 + generator.normal(scale=0.1)),
            base_energy=-1.0,
        )
        for index in range(2)
    )
    new_card = _card(
        "nondegenerate-new",
        embedding=tuple(generator.normal(size=3)),
        formation_energy=float(-1.0 + generator.normal(scale=0.1)),
        base_energy=-1.0,
    )
    planner = PosteriorProjectionOneSwapPlanner(
        2,
        posterior,
        ProperPosteriorDivergence(kind="log"),
    )
    selection = planner.preview_admit(current, new_card, queries)
    cards_by_id = {card.card_id: card for card in (*current, new_card)}
    candidate_posteriors = {
        candidate.selected_card_ids: posterior.clone_unfit()
        .fit(tuple(cards_by_id[card_id] for card_id in candidate.selected_card_ids))
        .predict(queries)
        for candidate in selection.candidates
    }

    objective_values = [candidate.proper_divergence for candidate in selection.candidates]
    assert max(objective_values) - min(objective_values) > 1e-6
    assert len(
        {
            (
                tuple(np.round(prediction.mean_ev_per_atom, 12)),
                tuple(np.round(prediction.std_ev_per_atom, 12)),
            )
            for prediction in candidate_posteriors.values()
        }
    ) == len(selection.candidates)
    expected = min(
        selection.candidates,
        key=lambda item: (
            item.proper_divergence + item.reactivation_cost,
            tuple(sorted(item.selected_card_ids)),
        ),
    )
    assert selection.selected_card_ids == expected.selected_card_ids
    assert selection.selected_card_ids != tuple(card.card_id for card in current)


def test_p3c_and_gp_variance_can_select_different_real_gp_subsets() -> None:
    generator = np.random.default_rng(0)
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    queries = tuple(
        _query(
            f"different-objective-q-{index}",
            embedding=tuple(generator.normal(size=3)),
            base_energy=float(-1.0 + generator.normal(scale=0.05)),
        )
        for index in range(5)
    )
    current = tuple(
        _card(
            f"different-objective-card-{index}",
            embedding=tuple(generator.normal(size=3)),
            formation_energy=float(-1.0 + generator.normal(scale=0.1)),
            base_energy=-1.0,
        )
        for index in range(2)
    )
    new_card = _card(
        "different-objective-new",
        embedding=tuple(generator.normal(size=3)),
        formation_energy=float(-1.0 + generator.normal(scale=0.1)),
        base_energy=-1.0,
    )
    p3c = PosteriorProjectionOneSwapPlanner(
        2,
        posterior,
        ProperPosteriorDivergence(kind="log"),
    ).preview_admit(current, new_card, queries)
    gp_variance = GPVarianceOneSwapMemory(2, posterior)
    for card in current:
        gp_variance.admit(card, queries)
    assert tuple(card.card_id for card in gp_variance.cards()) == tuple(
        card.card_id for card in current
    )
    gp_variance.admit(new_card, queries)

    gp_variance_ids = tuple(card.card_id for card in gp_variance.cards())
    assert set(p3c.selected_card_ids) != set(gp_variance_ids)


def test_posterior_projection_constraint_fallback_is_deterministic() -> None:
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    planner = PosteriorProjectionOneSwapPlanner(
        2,
        posterior,
        ProperPosteriorDivergence(kind="threshold_weighted_crps"),
        max_decision_regret=0.0,
        max_log_divergence=0.0,
    )
    queries = (
        _query("constraint-a", embedding=(1.0, 0.0), base_energy=-1.01),
        _query("constraint-b", embedding=(0.0, 1.0), base_energy=-0.97),
    )
    current = (
        _card("constraint-old-a", embedding=(1.0, 0.0), formation_energy=-1.07),
        _card("constraint-old-b", embedding=(0.0, 1.0), formation_energy=-0.96),
    )
    selection = planner.preview_admit(
        current,
        _card("constraint-new", embedding=(0.7, 0.7), formation_energy=-1.03),
        queries,
    )
    assert selection.used_constraint_fallback
    expected = min(
        selection.candidates,
        key=lambda item: (
            item.normalized_constraint_violation,
            item.proper_divergence + item.reactivation_cost,
            tuple(sorted(item.selected_card_ids)),
        ),
    )
    assert selection.selected_card_ids == expected.selected_card_ids


def test_archive_projection_enumerates_every_subset_up_to_capacity() -> None:
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    planner = ExactArchivePosteriorProjectionPlanner(
        2, posterior, ProperPosteriorDivergence(kind="gaussian_kl")
    )
    archive = tuple(
        _card(
            f"archive-{index}",
            embedding=tuple(np.eye(4)[index]),
            formation_energy=-1 + 0.02 * index,
        )
        for index in range(4)
    )
    selection = planner.select(
        archive,
        (_query("archive-q", embedding=(0.5, 0.5, 0.5, 0.5)),),
    )
    assert len(selection.candidates) == 1 + 4 + 6
    assert len(selection.selected_card_ids) <= 2


def test_explicit_projection_scorer_factorizes_reference_and_search_space() -> None:
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    scorer = PosteriorProjectionScorer(
        posterior,
        ProperPosteriorDivergence(kind="log"),
        false_stable_cost=5.0,
        false_unstable_cost=1.0,
        max_decision_regret=None,
        max_log_divergence=None,
        reactivation_weight=0.0,
    )
    near_a = _card(
        "factor-a",
        embedding=(1.0, 0.0),
        formation_energy=-1.03,
        base_energy=-1.03,
    )
    influential_b = _card(
        "factor-b",
        embedding=(0.0, 1.0),
        formation_energy=-1.25,
        base_energy=-1.03,
    )
    queries = (_query("factor-q", embedding=(0.0, 1.0), base_energy=-1.0),)
    cards_by_id = {item.card_id: item for item in (near_a, influential_b)}
    online_candidates = ((near_a.card_id,),)
    archive_candidates = ((near_a.card_id,), (influential_b.card_id,))

    union_online = scorer.score(
        reference_cards=(near_a,),
        candidate_sets=online_candidates,
        cards_by_id=cards_by_id,
        queries=queries,
        current_ids=(),
    )
    union_archive = scorer.score(
        reference_cards=(near_a,),
        candidate_sets=archive_candidates,
        cards_by_id=cards_by_id,
        queries=queries,
        current_ids=(),
    )
    archive_online = scorer.score(
        reference_cards=(near_a, influential_b),
        candidate_sets=online_candidates,
        cards_by_id=cards_by_id,
        queries=queries,
        current_ids=(),
    )
    archive_archive = scorer.score(
        reference_cards=(near_a, influential_b),
        candidate_sets=archive_candidates,
        cards_by_id=cards_by_id,
        queries=queries,
        current_ids=(),
    )

    assert union_online.selected_card_ids == (near_a.card_id,)
    assert union_archive.selected_card_ids == (near_a.card_id,)
    assert archive_online.selected_card_ids == (near_a.card_id,)
    assert archive_archive.selected_card_ids == (influential_b.card_id,)
    assert archive_archive.selected_proper_divergence < (archive_online.selected_proper_divergence)
    assert union_archive.reference_card_ids != archive_archive.reference_card_ids


@pytest.mark.parametrize("seed", range(5))
def test_joint_posterior_risk_one_swap_matches_manual_neighborhood(seed: int) -> None:
    generator = np.random.default_rng(seed)
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(
            ProtocolCompatibilityResolver(),
            config=FixedKernelGPConfig(length_scale=0.35),
        )
    )
    planner = JointPosteriorRiskOneSwapPlanner(2, builder)
    queries = tuple(
        _query(
            f"joint-q-{index}",
            embedding=tuple(generator.normal(size=3)),
            base_energy=float(-1 + generator.normal(scale=0.05)),
        )
        for index in range(4)
    )
    current = tuple(
        _card(
            f"joint-c-{index}",
            embedding=tuple(generator.normal(size=3)),
            formation_energy=float(-1 + generator.normal(scale=0.08)),
            base_energy=-1.0,
        )
        for index in range(2)
    )
    new_card = _card(
        "joint-new",
        embedding=tuple(generator.normal(size=3)),
        formation_energy=float(-1 + generator.normal(scale=0.08)),
        base_energy=-1.0,
    )
    selection = planner.preview_admit(current, new_card, queries)
    candidates = (
        current,
        (new_card, current[1]),
        (current[0], new_card),
    )
    manual = {
        tuple(card.card_id for card in cards): builder.weighted_decision_risk(queries, cards)
        for cards in candidates
    }
    expected = min(manual, key=lambda ids: (manual[ids], ids))
    current_ids = tuple(card.card_id for card in current)
    if manual[expected] >= manual[current_ids]:
        expected = current_ids
    assert selection.selected_card_ids == expected
    assert selection.weighted_joint_risk == pytest.approx(manual[expected])
    assert len(selection.candidate_weighted_joint_risks) == 3


def test_objective_fidelity_scores_identical_one_swap_candidates() -> None:
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(
            ProtocolCompatibilityResolver(),
            config=FixedKernelGPConfig(length_scale=0.3),
        )
    )
    facility = FacilityLocationCoresetPlanner(2, builder)
    joint = JointPosteriorRiskOneSwapPlanner(2, builder)
    queries = (
        _query("fidelity-a", embedding=(1.0, 0.0), base_energy=-0.98),
        _query("fidelity-b", embedding=(0.0, 1.0), base_energy=-1.02),
    )
    current = (
        _card("fidelity-old-a", embedding=(1.0, 0.0), formation_energy=-1.04),
        _card("fidelity-old-b", embedding=(0.0, 1.0), formation_energy=-0.94),
    )
    new_card = _card(
        "fidelity-new",
        embedding=(0.7, 0.7),
        formation_energy=-1.01,
    )
    diagnostic = compare_facility_and_joint_objectives(current, new_card, queries, facility, joint)
    facility_preview = facility.preview_admit(current, new_card, queries)
    joint_preview = joint.preview_admit(current, new_card, queries)
    assert len(diagnostic.candidates) == 3
    assert {row.candidate_key for row in diagnostic.candidates} == {
        "fidelity-old-a|fidelity-old-b",
        "fidelity-new|fidelity-old-a",
        "fidelity-new|fidelity-old-b",
    }
    assert diagnostic.facility_selected_card_ids == facility_preview.selected_card_ids
    assert diagnostic.joint_risk_selected_card_ids == joint_preview.selected_card_ids
    assert diagnostic.facility_joint_risk_regret >= 0


@pytest.mark.parametrize("seed", range(5))
def test_gp_variance_one_swap_matches_manual_neighborhood(seed: int) -> None:
    generator = np.random.default_rng(seed)
    posterior = FixedKernelResidualGP(
        ProtocolCompatibilityResolver(),
        config=FixedKernelGPConfig(length_scale=0.35),
    )
    memory = GPVarianceOneSwapMemory(2, posterior)
    queries = tuple(
        _query(f"variance-q-{index}", embedding=tuple(generator.normal(size=3)))
        for index in range(4)
    )
    current = tuple(
        _card(f"variance-c-{index}", embedding=tuple(generator.normal(size=3)))
        for index in range(2)
    )
    for card in current:
        memory.admit(card, queries)
    before = memory.cards()
    new_card = _card("variance-new", embedding=tuple(generator.normal(size=3)))
    memory.admit(new_card, queries)
    cards_by_id = {card.card_id: card for card in (*before, new_card)}
    before_ids = tuple(card.card_id for card in before)
    candidates = (
        before_ids,
        (new_card.card_id, before_ids[1]),
        (before_ids[0], new_card.card_id),
    )

    def objective(ids: tuple[str, ...]) -> float:
        prediction = (
            posterior.clone_unfit()
            .fit(tuple(cards_by_id[card_id] for card_id in ids))
            .predict(queries)
        )
        return sum(value * value for value in prediction.std_ev_per_atom)

    objectives = {ids: objective(ids) for ids in candidates}
    improving = [ids for ids in candidates[1:] if objectives[ids] < objectives[before_ids]]
    expected = (
        min(improving, key=lambda ids: (objectives[ids], tuple(sorted(ids))))
        if improving
        else before_ids
    )
    assert tuple(card.card_id for card in memory.cards()) == expected


def test_zero_survival_weight_returns_base_ranking_verbatim() -> None:
    queries = (_query("a", base_energy=-1.05), _query("b", base_energy=-1.01))
    proposal = FrozenHullDistanceAcquisition()
    base = proposal.rank(queries, ())
    acquisition = SurvivalConditionedAcquisition(
        proposal,
        _ExplodingPosterior(),  # type: ignore[arg-type]
        FacilityLocationCoresetPlanner(1, _FixedUtilityBuilder({})),
        survival_weight=0,
    )
    assert acquisition.rank(queries, ()) == base


def test_redundant_fantasy_has_zero_survival_bonus_and_is_not_admitted() -> None:
    current = (_card("current"),)
    queries = (_query("a", base_energy=-1.05), _query("b", base_energy=-1.01))
    planner = FacilityLocationCoresetPlanner(
        1,
        _FixedUtilityBuilder(
            {
                "current": (1.0,),
                "fantasy:": (1.0,),
            }
        ),
    )
    acquisition = SurvivalConditionedAcquisition(
        FrozenHullDistanceAcquisition(),
        _ZeroPosterior(),  # type: ignore[arg-type]
        planner,
        proposal_size=2,
        num_fantasies=3,
        survival_weight=1.0,
    )
    scores = acquisition.rank(queries, current)
    assert all(item.downstream_risk_reduction == 0 for item in scores)
    assert [card.card_id for card in current] == ["current"]
