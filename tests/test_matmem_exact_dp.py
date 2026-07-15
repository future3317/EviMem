from __future__ import annotations

import pytest

from evimem.matmem.exact_dp import (
    BinaryWitnessDP,
    BinaryWitnessState,
    exact_policy_comparison,
)


@pytest.mark.parametrize("capacity", [0, 1, 2])
@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_exact_joint_upper_bounds_bounded_heuristics(capacity: int, budget: int) -> None:
    result = exact_policy_comparison(
        (3, 2, 2),
        oracle_budget=budget,
        active_witness_budget=capacity,
    )
    assert result["exact_joint"] + 1e-12 >= result["one_step_joint"]
    assert result["exact_joint"] + 1e-12 >= result["decoupled"]


def test_exact_retention_can_condition_on_the_observed_outcome() -> None:
    model = BinaryWitnessDP(active_witness_budget=1)
    state = BinaryWitnessState((2, 2))
    observations = (*state.active_observations, (0, 1), (1, 0))
    options = model.retention_options(observations)
    assert ((0, 1),) in options
    assert ((1, 0),) in options
