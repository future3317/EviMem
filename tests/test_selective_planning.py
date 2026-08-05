from __future__ import annotations

import numpy as np

from matmem.hull_ens_audit import (
    evaluate_hull_ens_policy,
    exact_branch_value,
    exact_hull_ens,
    sampled_hull_ens,
)
from matmem.random_delayed_label_benchmark import generate_random_instances
from matmem.selective_planning import (
    EvaluatorTrace,
    StateTrace,
    planning_value_decomposition,
    selective_gate,
    top_two_exchange_gap,
)


def test_top_two_exchange_cancels_direct_reward() -> None:
    probabilities = (0.8, 0.7, 0.2)
    information = (0.01, 0.04, 0.2)
    assert np.isclose(top_two_exchange_gap(probabilities, information, 0, 1), 0.03)


def test_state_and_evaluator_traces_are_separate_and_aligned() -> None:
    trace = StateTrace(
        system_id="s1",
        fold_id="fold0",
        round_id=0,
        remaining_budget=2,
        candidate_ids=("a", "b"),
        p_final=(0.8, 0.7),
        greedy_order=("a", "b"),
        rank_gaps=(0.1,),
        fantasy_candidate="a",
        fantasy_energy=-0.2,
        conditional_p_final=((0.8, 0.75), (0.8, 0.7)),
        signed_delta_p=((0.0, 0.05), (0.0, 0.0)),
        information_value=(0.01, 0.04),
        q_two_step=(1.5, 1.54),
        q_rollout=(1.5, 1.54),
        q_standard_error=(0.01, 0.01),
        greedy_action="a",
        planner_action="b",
        posterior_stream_id="outer-1",
        inner_stream_id="inner-1",
    )
    evaluator = EvaluatorTrace(
        realized_final_labels=(True, False),
        realized_greedy_return=1.0,
        realized_planner_return=0.0,
    )
    assert trace.system_id == "s1"
    assert evaluator.realized_planner_return < evaluator.realized_greedy_return


def test_planning_value_decomposition_is_an_identity() -> None:
    decomposition = planning_value_decomposition(
        true_planner_value=1.3,
        true_greedy_value=1.0,
        posterior_optimal_value=1.2,
        posterior_greedy_value=0.9,
        posterior_planner_value=1.1,
        planner_cost=4.0,
        greedy_cost=1.0,
        cost_weight=0.05,
    )
    np.testing.assert_allclose(decomposition.planning_gain, 0.3, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(decomposition.cost_adjusted_gain, 0.15, rtol=0.0, atol=1e-12)


def test_selective_gate_falls_back_until_net_headroom_is_positive() -> None:
    fallback = selective_gate(
        greedy_action_index=1,
        rollout_action_index=2,
        headroom_mean=0.03,
        headroom_standard_error=0.01,
        model_penalty=0.01,
        cost_penalty=0.01,
    )
    assert fallback.action_index == 1
    assert not fallback.gate_used
    accepted = selective_gate(
        greedy_action_index=1,
        rollout_action_index=2,
        headroom_mean=0.20,
        headroom_standard_error=0.01,
        model_penalty=0.01,
        cost_penalty=0.01,
    )
    assert accepted.action_index == 2
    assert accepted.gate_used


def test_exact_hull_ens_is_budget_one_delta_hull_rule() -> None:
    instance = generate_random_instances(count=1, seed=20270804)[0]
    exact = exact_hull_ens(instance, remaining_budget=1)
    probabilities = np.asarray(exact.scores)
    assert exact.selected_action == int(np.argmax(probabilities))


def test_hull_ens_continuation_is_observation_measurable() -> None:
    instance = generate_random_instances(count=1, seed=20270811)[0]
    action = 0
    matching = [
        world
        for world in range(4)
        if instance.world_energies[world][action] == instance.world_energies[0][action]
    ]
    # The registered generator normally has distinct observations; use a
    # synthetic duplicate branch when it does not provide one.
    if len(matching) < 2:
        worlds = [list(row) for row in instance.world_energies]
        worlds[1][action] = worlds[0][action]
        instance = instance.__class__(
            instance_id=instance.instance_id,
            budget=instance.budget,
            source_signal=instance.source_signal,
            energy_correlation=instance.energy_correlation,
            delayed_label_coupling=instance.delayed_label_coupling,
            posterior_noise=instance.posterior_noise,
            competing_facet_count=instance.competing_facet_count,
            source_energies=instance.source_energies,
            world_energies=tuple(tuple(row) for row in worlds),
            world_probabilities=instance.world_probabilities,
        )
        matching = [0, 1]
    assert exact_branch_value(instance, action=action, observed_world=matching[0]) == exact_branch_value(
        instance, action=action, observed_world=matching[1]
    )


def test_double_sampling_is_explicit_and_reproducible() -> None:
    instance = generate_random_instances(count=1, seed=20270812)[0]
    first = sampled_hull_ens(
        instance,
        posterior_sample_count=64,
        inner_sample_count=8,
        seed=7,
        remaining_budget=2,
        independent_inner_stream=True,
    )
    second = sampled_hull_ens(
        instance,
        posterior_sample_count=64,
        inner_sample_count=8,
        seed=7,
        remaining_budget=2,
        independent_inner_stream=True,
    )
    assert first.independent_inner_stream
    assert first.selected_action == second.selected_action
    assert first.scores == second.scores


def test_repeated_exact_hull_ens_has_a_finite_world_value() -> None:
    instance = generate_random_instances(count=1, seed=20270813)[0]
    value = evaluate_hull_ens_policy(instance, mode="exact")
    assert 0.0 <= value <= float(instance.budget) + 1e-12
