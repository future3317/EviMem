from __future__ import annotations

import pytest

from matmem.exact_joint_separation import (
    ExactSeparationInstance,
    enumerate_nonadaptive_sequence_values,
    evaluate_exact_joint_separation,
    evaluate_joint_policy,
)


def test_exact_joint_policy_has_a_strict_state_feedback_separation() -> None:
    report = evaluate_exact_joint_separation()

    assert report.joint.terminal_confirmations == pytest.approx(2.0)
    assert report.nonadaptive.terminal_confirmations == pytest.approx(5.0 / 3.0)
    assert report.myopic.terminal_confirmations == pytest.approx(5.0 / 3.0)
    assert report.joint_nonadaptive_gap == pytest.approx(1.0 / 3.0)
    assert report.joint.final_causal_confirmations == pytest.approx(2.0)
    assert report.nonadaptive.final_causal_confirmations == pytest.approx(3.0)
    assert report.joint.false_stable == pytest.approx(0.0)
    assert report.nonadaptive.false_stable == pytest.approx(4.0 / 3.0)
    assert report.joint.false_unstable == pytest.approx(0.0)


def test_nonadaptive_upper_bound_covers_every_deterministic_sequence() -> None:
    values = enumerate_nonadaptive_sequence_values(ExactSeparationInstance())

    assert len(values) == 336
    assert max(value for _, value in values) == pytest.approx(5.0 / 3.0)
    assert all(value <= 5.0 / 3.0 for _, value in values)


def test_joint_policy_retains_the_protocol_compatible_witness() -> None:
    report = evaluate_exact_joint_separation()

    for path in report.joint.paths:
        assert path.actions == (
            "probe",
            f"group-{path.world}-a",
            f"group-{path.world}-b",
        )
        assert path.retained_states == (path.world, path.world, path.world)
    assert report.joint.memory_turnovers == pytest.approx(1.0)


def test_registered_memory_and_information_order_nulls_are_exact() -> None:
    report = evaluate_exact_joint_separation()

    assert report.zero_memory.terminal_confirmations == pytest.approx(
        report.nonadaptive.terminal_confirmations
    )
    assert report.full_history.terminal_confirmations == pytest.approx(
        report.joint.terminal_confirmations
    )
    assert report.zero_access_cost.terminal_confirmations == pytest.approx(
        report.full_history.terminal_confirmations
    )
    assert evaluate_joint_policy(
        ExactSeparationInstance(memory_capacity=3)
    ).terminal_confirmations == pytest.approx(report.full_history.terminal_confirmations)


def test_uninformative_or_unsupported_witness_fails_closed_to_the_baseline() -> None:
    report = evaluate_exact_joint_separation()

    assert report.uninformative_witness.terminal_confirmations == pytest.approx(
        report.nonadaptive.terminal_confirmations
    )
    assert report.unsupported_witness.terminal_confirmations == pytest.approx(
        report.nonadaptive.terminal_confirmations
    )
    assert all("probe" not in path.actions for path in report.unsupported_witness.paths)
