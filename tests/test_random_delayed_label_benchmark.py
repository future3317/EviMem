from __future__ import annotations

import pytest

from matmem.random_delayed_label_benchmark import (
    evaluate_random_instance,
    evaluate_random_suite,
    generate_random_instances,
)


def test_registered_generator_is_deterministic_and_respects_bounds() -> None:
    first = generate_random_instances(count=12)
    second = generate_random_instances(count=12)

    assert first == second
    assert {instance.pool_size for instance in first} <= set(range(5, 11))
    assert {instance.budget for instance in first} <= set(range(1, 5))
    assert {len(instance.world_energies) for instance in first} == {4}


def test_exact_dp_dominates_fixed_rollout_policies() -> None:
    for instance in generate_random_instances(count=24):
        dp = evaluate_random_instance(instance, "optimal_dp").value
        rollout = evaluate_random_instance(instance, "source_rollout").value
        source = evaluate_random_instance(instance, "source_margin").value
        greedy = evaluate_random_instance(instance, "greedy_final").value
        assert dp + 1e-12 >= rollout
        assert dp + 1e-12 >= source
        assert dp + 1e-12 >= greedy


def test_suite_contains_all_registered_policy_rows() -> None:
    rows = evaluate_random_suite(count=5)

    assert len(rows) == 25
    assert {row.policy for row in rows} == {
        "source_margin",
        "greedy_final",
        "source_rollout",
        "optimal_dp",
        "ic_sarr",
    }
    assert all(row.value >= 0.0 for row in rows)


def test_budget_one_rollout_is_greedy_final() -> None:
    for instance in generate_random_instances(count=30):
        if instance.budget == 1:
            assert evaluate_random_instance(instance, "source_rollout").value == pytest.approx(
                evaluate_random_instance(instance, "greedy_final").value
            )
