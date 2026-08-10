from __future__ import annotations

import numpy as np
import pytest

from matmem.paired_randomization import paired_sign_randomization


def test_exact_sign_randomization_detects_uniform_positive_effect() -> None:
    result = paired_sign_randomization(
        np.ones(8),
        confidence=0.95,
        draws=10_000,
        seed=17,
        exact_max_pairs=12,
    )
    assert result.method == "exact_sign_randomization"
    assert result.mean_effect == 1.0
    assert result.p_value == pytest.approx(2.0 / 256.0)
    assert result.wins_ties_losses == (8, 0, 0)
    assert result.ci_low <= 1.0 <= result.ci_high
    assert not (result.ci_low <= 0.0 <= result.ci_high)


def test_randomization_is_symmetric_under_effect_sign_reversal() -> None:
    values = np.asarray([0.8, 0.4, 0.2, -0.1, 0.5, 0.0, 0.3, 0.7])
    positive = paired_sign_randomization(values, draws=20_000, seed=9)
    negative = paired_sign_randomization(-values, draws=20_000, seed=9)
    assert negative.mean_effect == pytest.approx(-positive.mean_effect)
    assert negative.p_value == pytest.approx(positive.p_value)
    assert negative.ci_low == pytest.approx(-positive.ci_high, abs=1e-6)
    assert negative.ci_high == pytest.approx(-positive.ci_low, abs=1e-6)


def test_monte_carlo_randomization_is_deterministic_and_interval_matches_test() -> None:
    values = np.asarray([1.0, 0.0, 1.0, -1.0, 1.0] * 8)
    first = paired_sign_randomization(
        values,
        confidence=0.95,
        draws=20_000,
        seed=20260810,
        exact_max_pairs=10,
    )
    second = paired_sign_randomization(
        values,
        confidence=0.95,
        draws=20_000,
        seed=20260810,
        exact_max_pairs=10,
    )
    assert first == second
    assert first.method == "monte_carlo_sign_randomization"
    assert (first.p_value >= 0.05) == (first.ci_low <= 0.0 <= first.ci_high)
    assert first.randomization_resolution == pytest.approx(1.0 / 20_001.0)


def test_all_ties_return_zero_effect_and_unit_p_value() -> None:
    result = paired_sign_randomization(np.zeros(20), draws=1_000, seed=3)
    assert result.mean_effect == 0.0
    assert result.p_value == 1.0
    assert result.ci_low == 0.0
    assert result.ci_high == 0.0
    assert result.wins_ties_losses == (0, 20, 0)


def test_randomization_rejects_nonfinite_or_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        paired_sign_randomization(np.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        paired_sign_randomization(np.asarray([0.0, np.nan]))
