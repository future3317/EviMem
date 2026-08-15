from __future__ import annotations

import numpy as np
import pytest

import matmem.hull_geometry as hull_geometry
from matmem.hull_geometry import (
    FixedCompositionHullTemplate,
    _CausalHullEnvelope,
)
from matmem.posterior import (
    ProtocolTargetEnergyPosterior,
    _sample_gaussian,
    _sample_gaussian_blocks,
    protocol_target_energy_posterior,
)
from matmem.protocol_acquisition import (
    _cal_hull_values,
    _CalHullRuntimePlan,
    _condition_gaussian_on_scalar,
    _gaussian_hull_entropy,
    _simultaneous_paired_lower_bounds,
    _source_rollout_rewards,
    _unique_query_composition_grid,
    complete_pool_posterior_mean_hull_margin,
    conformal_one_deviation_source_rollout,
    constrained_dual_horizon_source_rollout,
    delta_hull_active_search,
    delta_hull_anchored_rollout,
    diagonal_independent_confirmation_source_rollout,
    fit_conformal_source_rollout_calibration,
    hull_ens,
    independent_confirmation_source_rollout,
    independent_world_confirmation_source_rollout,
    matched_local_hull_probability,
    posterior_rank_diagnostics,
    protocol_hull_entropy,
    protocol_hull_knowledge_gradient,
    protocol_hull_risk_reduction,
    safe_hull_ens,
    selective_delta_hull,
    source_rollout_delta_hull,
    source_rollout_system_score,
    ungated_source_rollout_delta_hull,
)
from matmem.protocol_knowledge_gradient import (
    fit_protocol_kernel_transport,
    fit_protocol_ridge_transport,
)
from matmem.transport import FrozenProtocolRidgeTransport


def test_complete_pool_posterior_mean_margin_uses_leave_one_out_pool() -> None:
    result = complete_pool_posterior_mean_hull_margin(
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
        ),
        posterior_mean=np.asarray((0.1, 0.2)),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.asarray((0.0, 0.0)),
    )

    assert result == pytest.approx((0.1, 0.2))


def test_complete_pool_mean_margin_matches_explicit_leave_one_out_envelopes() -> None:
    query_compositions = (
        {"A": 1.0, "B": 0.0, "C": 0.0},
        {"A": 0.0, "B": 1.0, "C": 0.0},
        {"A": 0.0, "B": 0.0, "C": 1.0},
        {"A": 0.5, "B": 0.25, "C": 0.25},
        {"A": 0.25, "B": 0.5, "C": 0.25},
    )
    references = (
        {"A": 1.0},
        {"B": 1.0},
        {"C": 1.0},
    )
    means = np.asarray((-0.1, -0.2, -0.3, -0.25, -0.15))
    result = complete_pool_posterior_mean_hull_margin(
        query_compositions=query_compositions,
        posterior_mean=means,
        reference_compositions=references,
        reference_energies=np.zeros(3),
    )

    expected = []
    for candidate_index in range(len(means)):
        remaining = tuple(
            index for index in range(len(means)) if index != candidate_index
        )
        envelope = _CausalHullEnvelope.build(
            query_compositions=query_compositions,
            reference_compositions=references,
            selected_query_indices=remaining,
        )
        active_energies = np.concatenate(
            (np.zeros(len(references)), means[list(remaining)])
        )
        expected.append(float(envelope.competing_hull_energies(active_energies)[0, candidate_index]))

    np.testing.assert_allclose(
        np.asarray(result), means - np.asarray(expected), rtol=0.0, atol=1e-12
    )


def test_cal_grid_deduplicates_normalized_query_compositions() -> None:
    grid = _unique_query_composition_grid(
        (
            {"A": 1.0, "B": 1.0},
            {"B": 2.0, "A": 2.0},
            {"A": 3.0, "B": 1.0},
        )
    )
    assert grid == (
        {"A": 0.5, "B": 0.5},
        {"A": 0.75, "B": 0.25},
    )


def test_cal_joint_hull_entropy_uses_ridge_for_rank_deficient_vectors() -> None:
    hull_values = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (2.0, 0.0),
        )
    )
    entropy = _gaussian_hull_entropy(hull_values, relative_ridge=1e-10)
    assert np.isfinite(entropy)
    assert entropy == _gaussian_hull_entropy(hull_values, relative_ridge=1e-10)


def test_cal_gaussian_conditioning_matches_scalar_normal_formula() -> None:
    conditional_mean, conditional_covariance = _condition_gaussian_on_scalar(
        np.asarray((1.0, 2.0)),
        np.asarray(((4.0, 2.0), (2.0, 3.0))),
        index=0,
        outcome=3.0,
    )
    np.testing.assert_allclose(conditional_mean, (3.0, 3.0))
    np.testing.assert_allclose(conditional_covariance, ((0.0, 0.0), (0.0, 2.0)))


def _cal_test_posterior() -> ProtocolTargetEnergyPosterior:
    return ProtocolTargetEnergyPosterior(
        mean=(-0.40, -0.25),
        covariance=((0.04, 0.012), (0.012, 0.03)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )


def test_cal_entropy_uses_query_grid_and_is_deterministic() -> None:
    result = protocol_hull_entropy(
        _cal_test_posterior(),
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        fantasy_count=2,
        seed=19,
    )
    repeat = protocol_hull_entropy(
        _cal_test_posterior(),
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        fantasy_count=2,
        seed=19,
    )
    assert result.evaluation_composition_count == 2
    assert result == repeat
    assert all(np.isfinite(value) for value in result.scores)


def test_cal_entropy_accepts_the_frozen_fixed_composition_backend() -> None:
    query_compositions = (
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
    )
    fixed_template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
    )
    result = protocol_hull_entropy(
        _cal_test_posterior(),
        query_compositions=query_compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        fantasy_count=2,
        seed=19,
        fixed_template=fixed_template,
    )
    assert result.evaluation_composition_count == 2


def test_cal_runtime_plan_matches_rebuilt_fixed_backend_hull_values() -> None:
    query_compositions = (
        {"A": 1.0, "B": 1.0},
        {"A": 2.0, "B": 2.0},
        {"A": 3.0, "B": 1.0},
    )
    references = ({"A": 1.0}, {"B": 1.0})
    evaluation_compositions = _unique_query_composition_grid(query_compositions)
    template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=references,
    )
    samples = np.asarray(
        ((-0.35, -0.40, -0.28), (-0.31, -0.38, -0.27), (-0.33, -0.36, -0.30))
    )

    runtime_plan = _CalHullRuntimePlan.from_inputs(
        query_compositions=query_compositions,
        reference_compositions=references,
        evaluation_compositions=evaluation_compositions,
        fixed_template=template,
    )
    baseline = _cal_hull_values(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(2),
        evaluation_compositions=evaluation_compositions,
        fixed_template=template,
    )
    planned = _cal_hull_values(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(2),
        evaluation_compositions=evaluation_compositions,
        fixed_template=template,
        runtime_plan=runtime_plan,
    )

    np.testing.assert_allclose(planned, baseline, rtol=0.0, atol=0.0)


def test_cal_entropy_candidate_parallelism_preserves_fixed_backend_scores() -> None:
    """Parallel CAL candidates must retain the serial deterministic objective."""

    query_compositions = (
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
        {"A": 0.75, "B": 0.25},
    )
    fixed_template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
    )
    kwargs = dict(
        query_compositions=query_compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(3),
        posterior_sample_count=8,
        fantasy_count=2,
        seed=19,
        fixed_template=fixed_template,
    )
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.40, -0.25, -0.32),
        covariance=((0.04, 0.012, 0.008), (0.012, 0.03, 0.006), (0.008, 0.006, 0.02)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )

    serial_timing: dict[str, float] = {}
    serial = protocol_hull_entropy(
        posterior,
        candidate_workers=1,
        timing_output=serial_timing,
        **kwargs,
    )
    parallel = protocol_hull_entropy(posterior, candidate_workers=2, **kwargs)

    np.testing.assert_allclose(parallel.scores, serial.scores, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        parallel.expected_conditional_entropies,
        serial.expected_conditional_entropies,
        rtol=0.0,
        atol=0.0,
    )
    assert int(np.argmax(parallel.scores)) == int(np.argmax(serial.scores))
    required_timing_keys = {
        "runtime_plan_build_seconds",
        "conditional_gaussian_seconds",
        "sparse_hull_kernel_seconds",
        "entropy_seconds",
        "other_seconds",
        "total_wall_seconds",
    }
    assert required_timing_keys <= serial_timing.keys()
    assert all(
        np.isfinite(serial_timing[key]) and serial_timing[key] >= 0.0
        for key in required_timing_keys
    )


def test_cal_entropy_matches_unbatched_reference_sampling() -> None:
    query_compositions = (
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
        {"A": 0.75, "B": 0.25},
    )
    reference_compositions = ({"A": 1.0}, {"B": 1.0})
    reference_energies = np.zeros(2)
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.40, -0.25, -0.32),
        covariance=((0.04, 0.012, 0.008), (0.012, 0.03, 0.006), (0.008, 0.006, 0.02)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=reference_compositions,
    )
    sample_count = 8
    fantasy_count = 2
    seed = 19
    grid = _unique_query_composition_grid(query_compositions)
    runtime_plan = _CalHullRuntimePlan.from_inputs(
        query_compositions=query_compositions,
        reference_compositions=reference_compositions,
        evaluation_compositions=grid,
        fixed_template=template,
    )
    mean = np.asarray(posterior.mean)
    covariance = np.asarray(posterior.covariance)
    current = _sample_gaussian(mean, covariance, sample_count=sample_count, seed=seed)
    current_entropy = _gaussian_hull_entropy(
        _cal_hull_values(
            query_compositions=query_compositions,
            sampled_query_energies=current,
            reference_compositions=reference_compositions,
            reference_energies=reference_energies,
            evaluation_compositions=grid,
            fixed_template=template,
            runtime_plan=runtime_plan,
        )
    )
    expected = []
    for query_index in range(len(mean)):
        variance = float(covariance[query_index, query_index])
        fantasies = _sample_gaussian(
            np.asarray((mean[query_index],)),
            np.asarray(((variance,),)),
            sample_count=fantasy_count,
            seed=seed + 9001,
        )[:, 0]
        entropy_sum = 0.0
        for fantasy_index, outcome in enumerate(fantasies):
            conditional_mean, conditional_covariance = _condition_gaussian_on_scalar(
                mean,
                covariance,
                index=query_index,
                outcome=float(outcome),
            )
            conditional_samples = _sample_gaussian(
                conditional_mean,
                conditional_covariance,
                sample_count=sample_count,
                seed=seed + 104729 * (fantasy_index + 1),
            )
            entropy_sum += _gaussian_hull_entropy(
                _cal_hull_values(
                    query_compositions=query_compositions,
                    sampled_query_energies=conditional_samples,
                    reference_compositions=reference_compositions,
                    reference_energies=reference_energies,
                    evaluation_compositions=grid,
                    fixed_template=template,
                    runtime_plan=runtime_plan,
                )
            )
        expected.append(entropy_sum / fantasy_count)

    optimized = protocol_hull_entropy(
        posterior,
        query_compositions=query_compositions,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        costs=np.ones(3),
        posterior_sample_count=sample_count,
        fantasy_count=fantasy_count,
        seed=seed,
        fixed_template=template,
    )
    np.testing.assert_allclose(optimized.current_entropy, current_entropy, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        optimized.expected_conditional_entropies,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_cal_fixed_backend_is_invariant_to_query_order() -> None:
    first = (
        {"A": 0.25, "B": 0.75},
        {"A": 0.5, "B": 0.5},
    )
    second = tuple(reversed(first))
    references = ({"A": 1.0}, {"B": 1.0})
    samples = np.asarray(((-0.25, -0.40), (-0.20, -0.45), (-0.30, -0.35)))
    first_values = _cal_hull_values(
        query_compositions=first,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(2),
        evaluation_compositions=first,
        fixed_template=FixedCompositionHullTemplate.from_compositions(
            query_compositions=first,
            reference_compositions=references,
        ),
    )
    second_values = _cal_hull_values(
        query_compositions=second,
        sampled_query_energies=samples[:, ::-1],
        reference_compositions=references,
        reference_energies=np.zeros(2),
        evaluation_compositions=second,
        fixed_template=FixedCompositionHullTemplate.from_compositions(
            query_compositions=second,
            reference_compositions=references,
        ),
    )
    np.testing.assert_allclose(first_values, second_values[:, ::-1])


def test_cal_entropy_zero_variance_candidate_has_zero_information_gain() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.4, -0.25),
        covariance=((0.0, 0.0), (0.0, 0.03)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    result = protocol_hull_entropy(
        posterior,
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        fantasy_count=2,
        seed=19,
    )
    assert result.expected_entropy_reductions[0] == pytest.approx(0.0, abs=1e-12)


def test_cal_entropy_rejects_nonuniform_costs() -> None:
    with pytest.raises(ValueError, match="equal query costs"):
        protocol_hull_entropy(
            _cal_test_posterior(),
            query_compositions=(
                {"A": 0.5, "B": 0.5},
                {"A": 0.25, "B": 0.75},
            ),
            reference_compositions=({"A": 1.0}, {"B": 1.0}),
            reference_energies=np.zeros(2),
            costs=np.asarray((1.0, 2.0)),
            posterior_sample_count=8,
            fantasy_count=2,
            seed=19,
        )
def _ic_rollout(
    *,
    selected: int,
    advantages: tuple[float, ...],
    lower_bounds: tuple[float, ...],
) -> object:
    """Small immutable SARR result for IC-SARR gate failure tests."""

    from matmem.protocol_acquisition import SourceRolloutDeltaHullResult

    return SourceRolloutDeltaHullResult(
        scores=(0.0, 0.2, 0.1),
        block_scores=tuple((0.0, 0.2, 0.1) for _ in range(16)),
        final_stability_probabilities=(0.0, 0.0, 0.0),
        paired_advantages_over_source=advantages,
        paired_advantage_lower_bounds=lower_bounds,
        source_action_index=0,
        selected_action_index=selected,
        posterior_sample_count=32,
        sobol_scramble_count=16,
        simultaneous_comparison_count=2,
        horizon=2,
        fallback_reason=None,
    )


def _ic_kwargs() -> dict[str, object]:
    return {
        "posterior": ProtocolTargetEnergyPosterior(
            mean=(-0.2, -0.5, -0.5),
            covariance=((1e-12, 0.0, 0.0), (0.0, 1e-12, 0.0), (0.0, 0.0, 1e-12)),
            system_offset_mean=0.0,
            system_offset_variance=0.0,
            history_count=0,
        ),
        "query_compositions": (
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        "query_source_energies": np.asarray([-0.45, -0.4, -0.4]),
        "query_ids": ("source", "left", "right"),
        "reference_compositions": ({"A": 1.0}, {"B": 1.0}),
        "reference_energies": np.zeros(2),
        "current_competing_hull_energies": np.zeros(3),
        "costs": np.ones(3),
        "remaining_budget": 2.0,
        "stage_one_posterior_sample_count": 32,
        "stage_two_posterior_sample_count": 64,
        "seed": 11,
    }


def test_ic_sarr_preserves_accepted_sarr_action_without_stage_two() -> None:
    kwargs = _ic_kwargs()
    stage_one = source_rollout_delta_hull(
        kwargs["posterior"],
        **{
            key: value
            for key, value in kwargs.items()
            if key
            not in {
                "posterior",
                "stage_one_posterior_sample_count",
                "stage_two_posterior_sample_count",
            }
        },
        posterior_sample_count=32,
    )
    assert stage_one.selected_action_index != stage_one.source_action_index
    result = independent_confirmation_source_rollout(**kwargs)
    assert result.selected_action_index == stage_one.selected_action_index
    assert not result.stage_two_used
    assert result.fallback_reason == "stage_one_accepted"


def test_p0_rollout_ablations_are_legal_and_do_not_mutate_ic_sarr() -> None:
    kwargs = _ic_kwargs()
    rollout_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"posterior", "stage_one_posterior_sample_count", "stage_two_posterior_sample_count"}
    }
    ungated = ungated_source_rollout_delta_hull(
        kwargs["posterior"], **rollout_kwargs, posterior_sample_count=32
    )
    diagonal = diagonal_independent_confirmation_source_rollout(**kwargs)
    independent_worlds = independent_world_confirmation_source_rollout(**kwargs)

    assert 0 <= ungated.selected_action_index < 3
    assert 0 <= diagonal.selected_action_index < 3
    assert 0 <= independent_worlds.selected_action_index < 3
    assert diagonal.stage_one_seed == kwargs["seed"]
    assert independent_worlds.stage_one_seed == kwargs["seed"]


def test_ic_sarr_falls_back_without_positive_stage_one_advantage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import matmem.protocol_acquisition as acquisition

    monkeypatch.setattr(
        acquisition,
        "source_rollout_delta_hull",
        lambda *_args, **_kwargs: _ic_rollout(
            selected=0, advantages=(0.0, 0.0, -0.1), lower_bounds=(0.0, -0.2, -0.3)
        ),
    )
    result = independent_confirmation_source_rollout(**_ic_kwargs())
    assert result.selected_action_index == result.source_action_index == 0
    assert not result.stage_two_used
    assert result.fallback_reason == "no_positive_stage_one_advantage"


@pytest.mark.parametrize(("candidate_reward", "expected_selected"), ((0.0, 0), (1.0, 1)))
def test_ic_sarr_uses_independent_single_comparison_gate(
    monkeypatch: pytest.MonkeyPatch,
    candidate_reward: float,
    expected_selected: int,
) -> None:
    import matmem.protocol_acquisition as acquisition

    monkeypatch.setattr(
        acquisition,
        "source_rollout_delta_hull",
        lambda *_args, **_kwargs: _ic_rollout(
            selected=0, advantages=(0.0, 0.2, 0.1), lower_bounds=(0.0, -0.2, -0.3)
        ),
    )
    observed_seeds: list[int] = []
    monkeypatch.setattr(
        acquisition,
        "_sample_gaussian_blocks",
        lambda mean, covariance, *, sample_count, seeds: (
            observed_seeds.extend(seeds)
            or tuple(np.zeros((sample_count, len(mean))) for _ in seeds)
        ),
    )
    monkeypatch.setattr(
        acquisition,
        "_final_hull_membership",
        lambda **kwargs: np.zeros_like(kwargs["sampled_query_energies"], dtype=bool),
    )
    monkeypatch.setattr(
        acquisition,
        "_source_rollout_rewards",
        lambda *, sampled_query_energies, first_action_indices, **_kwargs: np.column_stack(
            (
                np.zeros(len(sampled_query_energies)),
                np.full(len(sampled_query_energies), candidate_reward),
            )
        ),
    )
    result = independent_confirmation_source_rollout(**_ic_kwargs())
    assert result.stage_two_used
    assert result.screened_action_index == 1
    assert result.selected_action_index == expected_selected
    assert result.stage_two_seed is not None
    assert result.stage_two_seed != result.stage_one_seed
    assert len(observed_seeds) == 16
    assert min(observed_seeds) == result.stage_two_seed
    assert result.stage_two_paired_lower_bound is not None
    if expected_selected:
        assert result.stage_two_paired_lower_bound > 0
    else:
        assert result.stage_two_paired_lower_bound <= 0


def test_scrambled_sobol_gaussian_samples_are_deterministic_and_nested() -> None:

    mean = np.asarray([0.2, -0.1, 0.4])
    covariance = np.asarray(
        [
            [0.5, 0.1, 0.0],
            [0.1, 0.3, -0.05],
            [0.0, -0.05, 0.2],
        ]
    )
    small = _sample_gaussian(mean, covariance, sample_count=8, seed=17)
    repeated = _sample_gaussian(mean, covariance, sample_count=8, seed=17)
    large = _sample_gaussian(mean, covariance, sample_count=32, seed=17)
    other_seed = _sample_gaussian(mean, covariance, sample_count=8, seed=18)

    assert np.array_equal(small, repeated)
    assert np.array_equal(small, large[: len(small)])
    assert not np.array_equal(small, other_seed)
    assert np.isfinite(large).all()


def test_hoisted_gaussian_factor_preserves_each_registered_sobol_block() -> None:
    mean = np.asarray([0.2, -0.1, 0.4])
    covariance = np.asarray(
        [[0.5, 0.1, 0.0], [0.1, 0.3, -0.05], [0.0, -0.05, 0.2]]
    )
    seeds = (17, 104746, 209475)
    expected = tuple(
        _sample_gaussian(mean, covariance, sample_count=8, seed=seed) for seed in seeds
    )
    actual = _sample_gaussian_blocks(mean, covariance, sample_count=8, seeds=seeds)
    for left, right in zip(expected, actual, strict=True):
        np.testing.assert_array_equal(left, right)


def test_simultaneous_paired_bounds_apply_familywise_bonferroni_correction() -> None:

    block_differences = np.asarray(
        [
            [0.10, 0.30, -0.02],
            [0.06, 0.28, 0.01],
            [0.14, 0.35, -0.01],
            [0.08, 0.25, 0.03],
            [0.12, 0.32, -0.03],
            [0.09, 0.29, 0.00],
            [0.11, 0.31, 0.02],
            [0.07, 0.27, -0.01],
        ]
    )
    marginal = _simultaneous_paired_lower_bounds(
        block_differences,
        confidence=0.95,
        comparison_count=1,
    )
    simultaneous = _simultaneous_paired_lower_bounds(
        block_differences,
        confidence=0.95,
        comparison_count=3,
    )

    assert np.all(simultaneous <= marginal)
    assert simultaneous[0] < marginal[0]
    assert simultaneous[1] < marginal[1]

    near_threshold = np.asarray([[-0.05, 0.0], [0.35, 0.0], [-0.05, 0.0], [0.35, 0.0]] * 2)
    marginal_near_threshold = _simultaneous_paired_lower_bounds(
        near_threshold,
        confidence=0.95,
        comparison_count=1,
    )
    simultaneous_near_threshold = _simultaneous_paired_lower_bounds(
        near_threshold,
        confidence=0.95,
        comparison_count=3,
    )
    assert marginal_near_threshold[0] > 0
    assert simultaneous_near_threshold[0] < 0


def test_source_rollout_reports_simultaneous_candidate_count() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.2, -0.5, -0.5),
        covariance=(
            (1e-12, 0.0, 0.0),
            (0.0, 1e-12, 0.0),
            (0.0, 0.0, 1e-12),
        ),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    result = source_rollout_delta_hull(
        posterior,
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        query_source_energies=np.asarray([-0.45, -0.4, -0.4]),
        query_ids=("source", "left", "right"),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        current_competing_hull_energies=np.zeros(3),
        costs=np.ones(3),
        remaining_budget=2.0,
        posterior_sample_count=32,
        seed=11,
    )

    assert result.sobol_scramble_count == 16
    assert result.simultaneous_comparison_count == 2
    assert len(result.block_scores) == 16
    assert all(len(block) == 3 for block in result.block_scores)
    assert result.fallback_reason in {None, "no_positive_simultaneous_lower_bound"}


def _dual_kwargs() -> dict[str, object]:
    return {
        "posterior": ProtocolTargetEnergyPosterior(
            mean=(-0.2, -0.3, -0.4),
            covariance=((1e-12, 0.0, 0.0), (0.0, 1e-12, 0.0), (0.0, 0.0, 1e-12)),
            system_offset_mean=0.0,
            system_offset_variance=0.0,
            history_count=0,
        ),
        "query_compositions": (
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        "query_source_energies": np.asarray([-0.5, -0.4, -0.4]),
        "query_ids": ("source", "candidate-a", "candidate-b"),
        "reference_compositions": ({"A": 1.0}, {"B": 1.0}),
        "reference_energies": np.zeros(2),
        "current_competing_hull_energies": np.zeros(3),
        "costs": np.ones(3),
        "remaining_budget": 2.0,
        "posterior_sample_count": 4,
        "sobol_scramble_count": 2,
        "seed": 23,
    }


@pytest.mark.parametrize(
    ("causal_values", "expected"),
    [
        ((0.0, -0.1, -0.2), 0),  # terminal-only improvement is rejected
        ((0.0, 1.0, 1.5), 2),  # both gates pass; terminal mean wins
        ((0.0, -0.01, -0.01), 0),  # no feasible deviation falls back to source
    ],
)
def test_dual_horizon_requires_terminal_and_causal_gates(
    monkeypatch: pytest.MonkeyPatch,
    causal_values: tuple[float, ...],
    expected: int,
) -> None:
    import matmem.protocol_acquisition as acquisition

    monkeypatch.setattr(
        acquisition,
        "_sample_gaussian",
        lambda mean, covariance, *, sample_count, seed: np.zeros((sample_count, len(mean))),
    )
    monkeypatch.setattr(
        acquisition,
        "_final_hull_membership",
        lambda **kwargs: np.zeros_like(kwargs["sampled_query_energies"], dtype=bool),
    )

    def fake_rewards(*, sampled_query_energies, causal_rewards_output, **_kwargs):
        # Two identical Sobol blocks make the t lower bound exact and avoid a
        # stochastic test that can intermittently cross a gate.
        terminal = np.tile(np.asarray([[0.0, 1.0, 2.0]]), (len(sampled_query_energies), 1))
        causal_rewards_output[:] = np.tile(
            np.asarray(causal_values), (len(sampled_query_energies), 1)
        )
        return terminal

    monkeypatch.setattr(acquisition, "_source_rollout_rewards", fake_rewards)
    result = constrained_dual_horizon_source_rollout(**_dual_kwargs())
    assert result.source_action_index == 0
    assert result.selected_action_index == expected
    if expected == 0:
        assert result.fallback_reason == "no_dual_horizon_feasible_deviation"
    else:
        assert result.feasible_mask[expected]


def test_dual_horizon_causal_reward_uses_selected_history_only() -> None:

    samples = np.asarray([[-0.1, -0.2, 0.3]])
    terminal_labels = np.asarray([[True, True, False]])
    causal = np.empty((1, 3), dtype=float)
    rewards = _source_rollout_rewards(
        sampled_query_energies=samples,
        final_hull_membership=terminal_labels,
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        query_source_energies=np.asarray([-0.1, -0.1, -0.1]),
        query_ids=("a", "b", "c"),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        horizon=2,
        causal_rewards_output=causal,
    )
    assert rewards.shape == (1, 3)
    # Every simulated continuation contains exactly two selected outcomes;
    # causal rewards are computed from those two outcomes, never from the
    # unselected third candidate.
    assert np.all(causal <= 2.0)
    assert np.all(causal >= 0.0)


def test_dual_horizon_causal_reward_matches_manual_phase_diagram() -> None:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    compositions = (
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
        {"A": 0.75, "B": 0.25},
    )
    energies = np.asarray([[-0.2, -0.1, -0.3]])
    references = ({"A": 1.0}, {"B": 1.0})
    reference_energies = np.zeros(2)
    causal = np.empty((1, 3), dtype=float)
    rewards = _source_rollout_rewards(
        sampled_query_energies=energies,
        final_hull_membership=np.zeros((1, 3), dtype=bool),
        query_compositions=compositions,
        query_source_energies=np.zeros(3),
        query_ids=("a", "b", "c"),
        reference_compositions=references,
        reference_energies=reference_energies,
        horizon=1,
        causal_rewards_output=causal,
    )
    expected = []
    for index in range(3):
        entries = [
            ComputedEntry(Composition(comp), energy * Composition(comp).num_atoms)
            for comp, energy in zip(references, reference_energies, strict=True)
        ]
        comp = Composition(compositions[index])
        entries.append(ComputedEntry(comp, energies[0, index] * comp.num_atoms))
        stable = PhaseDiagram(entries).stable_entries
        expected.append(
            float(
                any(entry.composition.reduced_formula == comp.reduced_formula for entry in stable)
            )
        )
    np.testing.assert_array_equal(causal[0], np.asarray(expected))
    assert rewards.shape == (1, 3)


def test_conformal_source_rollout_calibration_is_system_clustered() -> None:
    score = source_rollout_system_score(
        np.asarray([[0.2, 0.1], [0.0, -0.3]]),
        np.asarray([[0.1, 0.2], [0.0, -0.4]]),
    )
    assert score == pytest.approx(0.1)
    calibration = fit_conformal_source_rollout_calibration(
        (0.1, 0.2, 0.3, 0.4),
        system_ids=("s1", "s2", "s3", "s4"),
        alpha=0.2,
    )
    assert calibration.order_statistic_one_based == 4
    assert calibration.radius == pytest.approx(0.4)
    assert calibration.identity_checksum.startswith("sha256:")
    with pytest.raises(ValueError, match="too few exact systems"):
        fit_conformal_source_rollout_calibration((0.1, 0.2), system_ids=("s1", "s2"), alpha=0.1)


def test_conformal_source_rollout_allows_only_one_deviation() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.2, -0.5, -0.5),
        covariance=(
            (1e-12, 0.0, 0.0),
            (0.0, 1e-12, 0.0),
            (0.0, 0.0, 1e-12),
        ),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    kwargs = dict(
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        query_source_energies=np.asarray([-0.45, -0.4, -0.4]),
        query_ids=("source", "left", "right"),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        current_competing_hull_energies=np.zeros(3),
        costs=np.ones(3),
        remaining_budget=2.0,
        conformal_radius=0.0,
        posterior_sample_count=32,
        seed=11,
    )
    first = conformal_one_deviation_source_rollout(posterior, **kwargs)
    assert first.deviation_selected
    assert first.selected_action_index == 1
    second = conformal_one_deviation_source_rollout(posterior, deviation_used=True, **kwargs)
    assert not second.deviation_selected
    assert second.selected_action_index == second.source_action_index == 0
    assert second.fallback_reason == "deviation_already_used"


def _transport_model():
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.8, 1.0],
            [1.0, 1.0],
        ]
    )
    source = np.asarray([-0.4, -0.2, -0.3, -0.1])
    target = source + np.asarray([0.04, 0.05, -0.03, -0.02])
    return fit_protocol_ridge_transport(
        features=features,
        source_energies=source,
        target_energies=target,
        system_ids=("A-B", "A-B", "C-D", "C-D"),
    )


def test_protocol_transport_is_frozen_and_system_disjoint() -> None:
    model = _transport_model()
    assert model.fit_system_ids == ("A-B", "C-D")
    assert model.fit_element_ids == ("A", "B", "C", "D")
    assert model.fit_row_count == 4
    assert model.identity_checksum.startswith("sha256:")
    assert model.within_system_variance > 0
    assert model.between_system_variance > 0


def test_hierarchical_kernel_transport_fits_only_registered_systems() -> None:
    features = np.asarray([[x, float(system)] for system in range(3) for x in (0.0, 0.3, 0.7, 1.0)])
    source = np.zeros(len(features))
    target = np.asarray(
        [
            0.15 * np.sin(4.0 * x) + 0.02 * system
            for system in range(3)
            for x in (0.0, 0.3, 0.7, 1.0)
        ]
    )
    systems = tuple(system for system in ("A-B", "C-D", "E-F") for _ in range(4))
    model = fit_protocol_kernel_transport(
        features=features,
        kernel_features=features[:, :1],
        source_energies=source,
        target_energies=target,
        system_ids=systems,
        kernel_feature_encoder="fixture-structure-encoder",
        kernel_feature_encoder_checksum="sha256:fixture",
    )
    assert model.local_kernel == "matern52"
    assert model.local_kernel_fit_system_count == 3
    assert model.local_kernel_signal_variance > 0
    assert model.local_kernel_noise_variance > 0
    assert model.local_kernel_nll_per_row is not None
    assert model.local_kernel_optimizer_success is True
    assert model.local_kernel_optimizer_status is not None
    assert model.local_kernel_optimizer_message
    assert model.local_kernel_optimizer_gradient_norm is not None


def test_local_discrepancy_reveal_updates_nearby_candidate_more() -> None:
    model = FrozenProtocolRidgeTransport(
        feature_mean=(0.0, 0.0),
        feature_scale=(1.0, 1.0),
        coefficients=(0.0, 0.0, 0.0),
        precision=((1e8, 0.0, 0.0), (0.0, 1e8, 0.0), (0.0, 0.0, 1e8)),
        within_system_variance=0.0901,
        between_system_variance=1e-8,
        ridge_penalty=1.0,
        fit_system_ids=("A-B", "C-D"),
        fit_element_ids=("A", "B", "C", "D"),
        fit_row_count=8,
        local_kernel="matern52",
        local_kernel_signal_variance=0.09,
        local_kernel_noise_variance=0.0001,
        local_kernel_length_scale=0.3,
        local_kernel_fit_system_count=2,
        local_kernel_nll_per_row=0.0,
        kernel_feature_mean=(0.0,),
        kernel_feature_scale=(1.0,),
        kernel_feature_encoder="fixture-structure-encoder",
        kernel_feature_encoder_checksum="sha256:fixture",
    )
    posterior = protocol_target_energy_posterior(
        model,
        # The global-mean features are deliberately reversed relative to the
        # frozen structure embedding.  The local update must follow the latter.
        query_features=np.asarray([[2.0], [0.05]]),
        query_source_energies=np.zeros(2),
        history_features=np.asarray([[0.0]]),
        history_source_energies=np.zeros(1),
        history_target_energies=np.ones(1),
        query_kernel_features=np.asarray([[0.05], [2.0]]),
        history_kernel_features=np.asarray([[0.0]]),
    )
    assert posterior.mean[0] > posterior.mean[1] + 0.8
    assert posterior.covariance[0][0] < posterior.covariance[1][1]

    with pytest.raises(ValueError, match="local-kernel embeddings"):
        protocol_target_energy_posterior(
            model,
            query_features=np.asarray([[2.0], [0.05]]),
            query_source_energies=np.zeros(2),
            history_features=np.asarray([[0.0]]),
            history_source_energies=np.zeros(1),
            history_target_energies=np.ones(1),
        )


def test_every_revealed_outcome_updates_order_invariant_system_state() -> None:
    model = _transport_model()
    query_features = np.asarray([[0.4, 0.2], [0.6, 0.8]])
    query_source = np.asarray([-0.25, -0.15])
    history_features = np.asarray([[0.3, 0.1], [0.7, 0.9]])
    history_source = np.asarray([-0.30, -0.12])
    unconditioned = protocol_target_energy_posterior(
        model,
        query_features=query_features,
        query_source_energies=query_source,
        history_features=np.empty((0, 2)),
        history_source_energies=np.empty(0),
        history_target_energies=np.empty(0),
    )
    conditioned = protocol_target_energy_posterior(
        model,
        query_features=query_features,
        query_source_energies=query_source,
        history_features=history_features,
        history_source_energies=history_source,
        history_target_energies=history_source + 0.20,
    )
    reversed_conditioned = protocol_target_energy_posterior(
        model,
        query_features=query_features,
        query_source_energies=query_source,
        history_features=history_features[::-1],
        history_source_energies=history_source[::-1],
        history_target_energies=(history_source + 0.20)[::-1],
    )
    assert conditioned.history_count == 2
    assert conditioned.system_offset_mean > unconditioned.system_offset_mean
    assert np.asarray(conditioned.mean) == pytest.approx(reversed_conditioned.mean)
    assert not np.allclose(conditioned.mean, unconditioned.mean)
    np.testing.assert_allclose(
        np.asarray(conditioned.covariance),
        np.asarray(reversed_conditioned.covariance),
    )


def test_delta_hull_active_search_prefers_final_support_phase() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.5, 0.2),
        covariance=((1e-8, 0.0), (0.0, 1e-8)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    result = delta_hull_active_search(
        posterior,
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "B": 0.5},
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        seed=7,
    )
    assert result.final_stability_probabilities == pytest.approx((1.0, 0.0))
    assert np.argmax(result.scores) == 0


def test_matched_local_hull_probability_equals_manual_sample_frequency() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.12, -0.08),
        covariance=((0.01, 0.004), (0.004, 0.02)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=1,
    )
    current_hull = np.asarray([-0.10, -0.15])
    sample_count = 32
    seed = 91

    result = matched_local_hull_probability(
        posterior,
        current_competing_hull_energies=current_hull,
        costs=np.ones(2),
        posterior_sample_count=sample_count,
        seed=seed,
    )
    samples = _sample_gaussian(
        np.asarray(posterior.mean),
        np.asarray(posterior.covariance),
        sample_count=sample_count,
        seed=seed,
    )
    expected = np.mean(samples <= current_hull[None, :], axis=0)

    assert result.scores == pytest.approx(expected)
    assert result.final_stability_probabilities == pytest.approx(expected)


def test_matched_local_hull_probability_is_seed_deterministic() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(0.0, 0.0),
        covariance=((1.0, 0.25), (0.25, 1.0)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    kwargs = {
        "current_competing_hull_energies": np.asarray([0.1, -0.2]),
        "costs": np.ones(2),
        "posterior_sample_count": 16,
        "seed": 4,
    }
    first = matched_local_hull_probability(posterior, **kwargs)
    second = matched_local_hull_probability(posterior, **kwargs)
    assert first == second


def test_matched_local_hull_probability_rejects_unequal_costs() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(0.0, 0.0),
        covariance=((1.0, 0.0), (0.0, 1.0)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    with pytest.raises(ValueError, match="equal query costs"):
        matched_local_hull_probability(
            posterior,
            current_competing_hull_energies=np.zeros(2),
            costs=np.asarray([1.0, 2.0]),
            posterior_sample_count=8,
        )


def test_hull_ens_horizon_one_has_delta_hull_action_parity() -> None:
    kwargs = _ic_kwargs()
    delta = delta_hull_active_search(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        costs=kwargs["costs"],
        posterior_sample_count=16,
        seed=kwargs["seed"],
    )
    result = hull_ens(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        query_ids=kwargs["query_ids"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        costs=kwargs["costs"],
        remaining_budget=1.0,
        posterior_sample_count=16,
        fantasy_sample_count=4,
        seed=kwargs["seed"],
    )
    assert result.horizon == 1
    assert result.selected_action_index == min(
        range(3), key=lambda index: (-delta.scores[index], kwargs["query_ids"][index])
    )
    assert result.scores == pytest.approx(delta.scores)
    assert result.expected_future_values == pytest.approx((0.0, 0.0, 0.0))


def test_hull_ens_candidate_parallelism_preserves_scores() -> None:
    kwargs = _ic_kwargs()
    serial = hull_ens(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        query_ids=kwargs["query_ids"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        costs=kwargs["costs"],
        remaining_budget=2.0,
        posterior_sample_count=16,
        fantasy_sample_count=4,
        seed=kwargs["seed"],
        candidate_workers=1,
    )
    parallel = hull_ens(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        query_ids=kwargs["query_ids"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        costs=kwargs["costs"],
        remaining_budget=2.0,
        posterior_sample_count=16,
        fantasy_sample_count=4,
        seed=kwargs["seed"],
        candidate_workers=2,
    )
    assert parallel.selected_action_index == serial.selected_action_index
    assert parallel.delta_hull_action_index == serial.delta_hull_action_index
    assert parallel.scores == pytest.approx(serial.scores)
    assert parallel.expected_future_values == pytest.approx(serial.expected_future_values)
    assert parallel.information_values == pytest.approx(serial.information_values)


def test_safe_hull_ens_falls_back_to_delta_hull_when_horizon_is_one() -> None:
    kwargs = _ic_kwargs()
    result = safe_hull_ens(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        query_ids=kwargs["query_ids"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        costs=kwargs["costs"],
        remaining_budget=1.0,
        posterior_sample_count=16,
        fantasy_sample_count=4,
        certificate_sample_count=16,
        seed=kwargs["seed"],
    )
    assert result.selected_action_index == result.delta_hull_action_index
    assert result.gate_used is False
    assert result.fallback_reason == "no_positive_delta_relative_certificate"
    assert all(value <= 0.0 for value in result.certificate_lower_bounds)


def test_delta_hull_anchored_rollout_records_rank_and_coupling_diagnostics() -> None:
    kwargs = _ic_kwargs()
    result = delta_hull_anchored_rollout(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        query_ids=kwargs["query_ids"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        costs=kwargs["costs"],
        remaining_budget=kwargs["remaining_budget"],
        posterior_sample_count=16,
        continuation_sample_count=8,
        diagnostic_sample_count=8,
        seed=kwargs["seed"],
    )
    assert result.horizon == 2
    assert len(result.scores) == 3
    assert len(result.cross_candidate_influence) == 3
    assert 0.0 <= result.rank_switch_probability <= 1.0
    assert result.coupling_score >= result.coupling_score_normalized


def test_posterior_rank_diagnostics_are_posterior_only_and_bounded() -> None:
    kwargs = _ic_kwargs()
    diagnostics = posterior_rank_diagnostics(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        query_ids=kwargs["query_ids"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        posterior_sample_count=8,
        conditional_sample_count=4,
        observation_count=1,
        seed=kwargs["seed"],
    )
    assert 0.0 <= diagnostics["posterior_rank_switch_probability"] <= 1.0
    assert diagnostics["posterior_rank_margin"] >= 0.0
    assert diagnostics["posterior_coupling_score"] >= 0.0


def test_selective_delta_hull_falls_back_without_positive_net_headroom() -> None:
    kwargs = _ic_kwargs()
    result = selective_delta_hull(
        kwargs["posterior"],
        query_compositions=kwargs["query_compositions"],
        query_ids=kwargs["query_ids"],
        reference_compositions=kwargs["reference_compositions"],
        reference_energies=kwargs["reference_energies"],
        costs=kwargs["costs"],
        remaining_budget=kwargs["remaining_budget"],
        two_step_posterior_sample_count=16,
        two_step_inner_sample_count=4,
        rollout_posterior_sample_count=16,
        rollout_continuation_sample_count=4,
        seed=kwargs["seed"],
        model_penalty=10.0,
    )
    assert result.selected_action_index == result.delta_hull_action_index
    assert result.gate_used is False
    assert result.rollout_action_index is None
    assert len(result.conditional_p_final) == len(kwargs["query_ids"])


def test_delta_hull_active_search_rejects_ratio_heuristic_costs() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.5, 0.2),
        covariance=((0.01, 0.0), (0.0, 0.01)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    with pytest.raises(ValueError, match="equal query costs"):
        delta_hull_active_search(
            posterior,
            query_compositions=(
                {"A": 0.5, "B": 0.5},
                {"A": 0.5, "B": 0.5},
            ),
            reference_compositions=({"A": 1.0}, {"B": 1.0}),
            reference_energies=np.zeros(2),
            costs=np.asarray([1.0, 2.0]),
        )


def test_delta_hull_scores_equal_manual_joint_final_membership_probability() -> None:
    from matmem.protocol_knowledge_gradient import (
        _final_hull_membership,
        _sample_gaussian,
    )

    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.25, -0.18, -0.08),
        covariance=(
            (0.02, 0.01, 0.0),
            (0.01, 0.03, 0.005),
            (0.0, 0.005, 0.01),
        ),
        system_offset_mean=0.0,
        system_offset_variance=0.01,
        history_count=2,
    )
    compositions = (
        {"A": 0.25, "B": 0.75},
        {"A": 0.5, "B": 0.5},
        {"A": 0.75, "B": 0.25},
    )
    result = delta_hull_active_search(
        posterior,
        query_compositions=compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(3),
        posterior_sample_count=32,
        seed=29,
    )
    manual = _final_hull_membership(
        query_compositions=compositions,
        sampled_query_energies=_sample_gaussian(
            np.asarray(posterior.mean),
            np.asarray(posterior.covariance),
            sample_count=32,
            seed=29,
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
    ).mean(axis=0)
    assert result.scores == pytest.approx(manual)
    assert result.final_stability_probabilities == pytest.approx(manual)


@pytest.mark.parametrize(
    ("query_compositions", "reference_compositions", "selected"),
    (
        (
            (
                {"A": 0.25, "B": 0.75},
                {"A": 0.5, "B": 0.5},
                {"A": 0.75, "B": 0.25},
            ),
            ({"A": 1.0}, {"B": 1.0}),
            (0, 2),
        ),
        (
            (
                {"A": 0.5, "B": 0.5},
                {"A": 0.5, "C": 0.5},
                {"B": 0.5, "C": 0.5},
                {"A": 0.25, "B": 0.25, "C": 0.5},
            ),
            ({"A": 1.0}, {"B": 1.0}, {"C": 1.0}),
            (0, 1, 2),
        ),
        (
            (
                {"A": 0.5, "B": 0.5},
                {"C": 0.5, "D": 0.5},
                {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25},
                {"A": 0.1, "B": 0.2, "C": 0.3, "D": 0.4},
                {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1},
            ),
            ({"A": 1.0}, {"B": 1.0}, {"C": 1.0}, {"D": 1.0}),
            (0, 1, 2),
        ),
        (
            (
                {"A": 0.2, "B": 0.2, "C": 0.2, "D": 0.2, "E": 0.2},
                {"A": 0.4, "B": 0.1, "C": 0.1, "D": 0.1, "E": 0.3},
                {"A": 0.1, "B": 0.4, "C": 0.2, "D": 0.2, "E": 0.1},
                {"A": 0.3, "B": 0.2, "C": 0.1, "D": 0.3, "E": 0.1},
                {"A": 0.1, "B": 0.1, "C": 0.4, "D": 0.1, "E": 0.3},
                {"A": 0.2, "B": 0.1, "C": 0.1, "D": 0.5, "E": 0.1},
            ),
            (
                {"A": 1.0},
                {"B": 1.0},
                {"C": 1.0},
                {"D": 1.0},
                {"E": 1.0},
            ),
            (0, 2, 4),
        ),
    ),
)
def test_cached_causal_hull_envelope_matches_pymatgen_competing_hull(
    query_compositions: tuple[dict[str, float], ...],
    reference_compositions: tuple[dict[str, float], ...],
    selected: tuple[int, ...],
) -> None:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    rng = np.random.default_rng(72)
    reference_energies = np.zeros(len(reference_compositions))
    sampled = rng.normal(-0.2, 0.15, size=(7, len(query_compositions)))
    envelope = _CausalHullEnvelope.build(
        query_compositions=query_compositions,
        reference_compositions=reference_compositions,
        selected_query_indices=selected,
    )
    active = np.column_stack((reference_energies[None, :].repeat(7, axis=0), sampled[:, selected]))
    actual = envelope.competing_hull_energies(active)
    expected = np.empty_like(actual)
    for sample_index in range(len(sampled)):
        entries = [
            ComputedEntry(value, 0.0, entry_id=f"reference:{index}")
            for index, value in enumerate(reference_compositions)
        ]
        entries.extend(
            ComputedEntry(
                query_compositions[index],
                sampled[sample_index, index] * Composition(query_compositions[index]).num_atoms,
                entry_id=f"selected:{index}",
            )
            for index in selected
        )
        diagram = PhaseDiagram(entries)
        expected[sample_index] = [
            diagram.get_hull_energy_per_atom(Composition(value)) for value in query_compositions
        ]
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_causal_hull_tiling_matches_dense_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(812)
    positions = rng.integers(0, 6, size=(7, 4), dtype=np.int64)
    weights = rng.random((7, 4))
    weights /= weights.sum(axis=1, keepdims=True)
    second_positions = rng.integers(0, 6, size=(5, 4), dtype=np.int64)
    second_weights = rng.random((5, 4))
    second_weights /= second_weights.sum(axis=1, keepdims=True)
    envelope = _CausalHullEnvelope(
        query_active_positions=(positions, second_positions),
        query_weights=(weights, second_weights),
        active_count=6,
        query_count=2,
        retained_simplex_count=2,
        feasible_nnz=12,
    )
    values = rng.normal(size=(11, 6))
    monkeypatch.setattr(hull_geometry, "_CAUSAL_HULL_DECOMPOSITION_CHUNK_SIZE", 3)
    monkeypatch.setattr(hull_geometry, "_CAUSAL_HULL_SCRATCH_BUDGET_BYTES", 128)

    actual = envelope.competing_hull_energies(values)
    expected = np.column_stack(
        (
            np.min(np.einsum("mdk,dk->md", values[:, positions], weights), axis=1),
            np.min(
                np.einsum(
                    "mdk,dk->md", values[:, second_positions], second_weights
                ),
                axis=1,
            ),
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_source_rollout_evaluator_matches_manual_pymatgen_continuation() -> None:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    from matmem.protocol_knowledge_gradient import (
        _final_hull_membership,
        _source_rollout_rewards,
    )

    compositions = (
        {"A": 0.25, "B": 0.75},
        {"A": 0.5, "B": 0.5},
        {"A": 0.75, "B": 0.25},
        {"A": 0.4, "B": 0.6},
    )
    references = ({"A": 1.0}, {"B": 1.0})
    reference_energies = np.zeros(2)
    source = np.asarray([-0.31, -0.28, -0.26, -0.20])
    identifiers = ("q3", "q1", "q4", "q2")
    samples = np.asarray(
        [
            [-0.42, -0.20, -0.38, -0.15],
            [-0.15, -0.48, -0.41, -0.30],
            [-0.36, -0.34, -0.12, -0.43],
        ]
    )
    labels = _final_hull_membership(
        query_compositions=compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=reference_energies,
    )
    actual = _source_rollout_rewards(
        sampled_query_energies=samples,
        final_hull_membership=labels,
        query_compositions=compositions,
        query_source_energies=source,
        query_ids=identifiers,
        reference_compositions=references,
        reference_energies=reference_energies,
        horizon=3,
    )
    expected = np.empty_like(actual)
    for sample_index, sample in enumerate(samples):
        for first_action in range(len(compositions)):
            selected = [first_action]
            for _ in range(1, 3):
                entries = [
                    ComputedEntry(value, 0.0, entry_id=f"reference:{index}")
                    for index, value in enumerate(references)
                ]
                entries.extend(
                    ComputedEntry(
                        compositions[index],
                        sample[index] * Composition(compositions[index]).num_atoms,
                        entry_id=f"selected:{index}",
                    )
                    for index in selected
                )
                diagram = PhaseDiagram(entries)
                remaining = set(range(len(compositions))) - set(selected)
                action = min(
                    remaining,
                    key=lambda index: (
                        source[index]
                        - diagram.get_hull_energy_per_atom(Composition(compositions[index])),
                        identifiers[index],
                    ),
                )
                selected.append(action)
            expected[sample_index, first_action] = labels[sample_index, selected].sum()
    np.testing.assert_array_equal(actual, expected)


def test_source_rollout_finds_full_budget_improvement_over_myopic_source() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.2, -0.5, -0.5),
        covariance=(
            (1e-12, 0.0, 0.0),
            (0.0, 1e-12, 0.0),
            (0.0, 0.0, 1e-12),
        ),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    result = source_rollout_delta_hull(
        posterior,
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        query_source_energies=np.asarray([-0.45, -0.4, -0.4]),
        query_ids=("source", "left", "right"),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        current_competing_hull_energies=np.zeros(3),
        costs=np.ones(3),
        remaining_budget=2.0,
        posterior_sample_count=32,
        seed=11,
    )
    assert result.horizon == 2
    assert result.source_action_index == 0
    assert result.selected_action_index == 1
    assert result.scores == pytest.approx((1.0, 2.0, 2.0))
    assert result.paired_advantage_lower_bounds[1] > 0


def test_fixed_template_preserves_all_source_rollout_values() -> None:
    """The cached final-hull backend is an implementation optimization only."""

    query_compositions = (
        {"A": 0.25, "B": 0.75},
        {"A": 0.5, "B": 0.5},
        {"A": 0.75, "B": 0.25},
        {"A": 0.4, "B": 0.6},
    )
    reference_compositions = ({"A": 1.0}, {"B": 1.0})
    kwargs = dict(
        query_compositions=query_compositions,
        query_source_energies=np.asarray([-0.31, -0.28, -0.26, -0.20]),
        query_ids=("q3", "q1", "q4", "q2"),
        reference_compositions=reference_compositions,
        reference_energies=np.zeros(2),
        current_competing_hull_energies=np.zeros(4),
        costs=np.ones(4),
        remaining_budget=3.0,
        posterior_sample_count=32,
        seed=91,
    )
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.42, -0.20, -0.38, -0.15),
        covariance=(
            (0.02, 0.005, 0.0, 0.0),
            (0.005, 0.03, 0.004, 0.0),
            (0.0, 0.004, 0.02, 0.003),
            (0.0, 0.0, 0.003, 0.01),
        ),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    reference = source_rollout_delta_hull(posterior, **kwargs)
    cached = source_rollout_delta_hull(
        posterior,
        **kwargs,
        fixed_template=FixedCompositionHullTemplate.from_compositions(
            query_compositions=query_compositions,
            reference_compositions=reference_compositions,
        ),
    )
    assert cached.model_dump() == reference.model_dump()


def test_fixed_binary_chain_matches_pymatgen_for_random_and_collinear_samples() -> None:
    """The binary fast path keeps pymatgen's non-vertex tie semantics."""

    from matmem.protocol_knowledge_gradient import (
        FixedCompositionHullTemplate,
        _final_hull_membership,
    )

    query_compositions = (
        {"A": 0.25, "B": 0.75},
        {"A": 0.5, "B": 0.5},
        {"A": 0.75, "B": 0.25},
        {"A": 0.6, "B": 0.4},
    )
    references = ({"A": 1.0}, {"B": 1.0})
    random_samples = np.random.default_rng(20260722).normal(
        -0.2, 0.2, size=(256, len(query_compositions))
    )
    # The first three values are exactly collinear with the elemental line in
    # reduced-composition order. Qhull excludes the redundant middle vertex;
    # this is the edge case that a mere energy-above-hull test would get wrong.
    samples = np.vstack((random_samples, np.asarray([[-0.30, -0.20, -0.10, -0.16]])))
    template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=references,
    )
    expected = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(2),
    )
    actual = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(2),
        fixed_template=template,
    )
    np.testing.assert_array_equal(actual, expected)


def test_fixed_ternary_oriented_facets_match_pymatgen_random_samples() -> None:
    """The oriented-facet fast path preserves pymatgen stable membership."""

    from matmem.protocol_knowledge_gradient import (
        FixedCompositionHullTemplate,
        _final_hull_membership,
    )

    query_compositions = (
        {"A": 0.5, "B": 0.5},
        {"A": 0.5, "C": 0.5},
        {"B": 0.5, "C": 0.5},
        {"A": 0.333333333333, "B": 0.333333333333, "C": 0.333333333334},
        {"A": 0.25, "B": 0.25, "C": 0.5},
        {"A": 0.5, "B": 0.5},
    )
    references = ({"A": 1.0}, {"B": 1.0}, {"C": 1.0})
    samples = np.random.default_rng(20260729).normal(
        -0.15, 0.25, size=(256, len(query_compositions))
    )
    template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=references,
    )
    expected = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(3),
    )
    actual = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(3),
        fixed_template=template,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", (31, 47, 73, 101))
def test_fixed_ternary_oriented_facets_match_pymatgen_random_compositions(
    seed: int,
) -> None:
    """Random ternary geometries preserve lower-facet vertex semantics."""

    from matmem.protocol_knowledge_gradient import (
        FixedCompositionHullTemplate,
        _final_hull_membership,
    )

    rng = np.random.default_rng(seed)
    elements = ("A", "B", "C")
    integer_compositions = rng.integers(1, 9, size=(12, 3))
    query_compositions = tuple(
        dict(zip(elements, row, strict=True)) for row in integer_compositions
    )
    references = ({"A": 1.0}, {"B": 1.0}, {"C": 1.0})
    samples = rng.normal(-0.1, 0.3, size=(64, len(query_compositions)))
    template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=references,
    )
    expected = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(3),
    )
    actual = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=references,
        reference_energies=np.zeros(3),
        fixed_template=template,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    ("query_compositions", "reference_compositions"),
    (
        (
            (
                {"A": 0.5, "B": 0.5},
                {"A": 0.25, "B": 0.75},
                {"A": 0.75, "B": 0.25},
                {"A": 0.5, "B": 0.5},
            ),
            ({"A": 1.0}, {"B": 1.0}),
        ),
        (
            (
                {"A": 0.5, "B": 0.5},
                {"A": 0.5, "C": 0.5},
                {"B": 0.5, "C": 0.5},
                {"A": 0.333333333333, "B": 0.333333333333, "C": 0.333333333334},
                {"A": 0.25, "B": 0.25, "C": 0.5},
            ),
            ({"A": 1.0}, {"B": 1.0}, {"C": 1.0}),
        ),
    ),
)
def test_fixed_composition_hull_backend_matches_pymatgen(
    query_compositions: tuple[dict[str, float], ...],
    reference_compositions: tuple[dict[str, float], ...],
) -> None:
    from matmem.protocol_knowledge_gradient import (
        FixedCompositionHullTemplate,
        _final_hull_membership,
    )

    rng = np.random.default_rng(20260721)
    sampled = rng.normal(-0.05, 0.25, size=(24, len(query_compositions)))
    references = np.zeros(len(reference_compositions), dtype=float)
    template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=reference_compositions,
    )
    pymatgen_labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=sampled,
        reference_compositions=reference_compositions,
        reference_energies=references,
    )
    fixed_labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=sampled,
        reference_compositions=reference_compositions,
        reference_energies=references,
        fixed_template=template,
    )
    np.testing.assert_array_equal(fixed_labels, pymatgen_labels)


def test_two_step_protocol_hull_policy_rejects_nonuniform_costs() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.5, 0.2),
        covariance=((0.01, 0.0), (0.0, 0.01)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    with pytest.raises(ValueError, match="unit costs"):
        protocol_hull_knowledge_gradient(
            posterior,
            query_compositions=(
                {"A": 0.5, "B": 0.5},
                {"A": 0.5, "B": 0.5},
            ),
            reference_compositions=({"A": 1.0}, {"B": 1.0}),
            reference_energies=np.zeros(2),
            costs=np.asarray([1.0, 2.0]),
            remaining_budget=2.0,
        )


def test_hull_risk_reduction_remains_informative_when_membership_saturates() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-1.0, -1.0),
        covariance=((0.04, 0.0), (0.0, 0.0025)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    result = protocol_hull_risk_reduction(
        posterior,
        query_compositions=(
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=32,
        fantasy_count=3,
        seed=13,
    )
    assert result.current_hull_risk > 0
    assert result.risk_reductions[0] > result.risk_reductions[1]
    assert np.argmax(result.scores) == 0


def test_hull_risk_grid_is_invariant_when_query_becomes_reference() -> None:
    compositions = (
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
    )
    before = protocol_hull_risk_reduction(
        ProtocolTargetEnergyPosterior(
            mean=(-0.4, -0.3),
            covariance=((0.01, 0.0), (0.0, 0.01)),
            system_offset_mean=0.0,
            system_offset_variance=0.0,
            history_count=0,
        ),
        query_compositions=compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        fantasy_count=1,
        seed=5,
    )
    after = protocol_hull_risk_reduction(
        ProtocolTargetEnergyPosterior(
            mean=(-0.3,),
            covariance=((0.01,),),
            system_offset_mean=0.0,
            system_offset_variance=0.0,
            history_count=1,
        ),
        query_compositions=(compositions[1],),
        reference_compositions=({"A": 1.0}, {"B": 1.0}, compositions[0]),
        reference_energies=np.asarray([0.0, 0.0, -0.4]),
        costs=np.ones(1),
        posterior_sample_count=8,
        fantasy_count=1,
        seed=5,
    )
    assert before.evaluation_composition_count == 4
    assert after.evaluation_composition_count == before.evaluation_composition_count
