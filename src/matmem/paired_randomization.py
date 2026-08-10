"""Paired sign-randomization inference with test-inverted intervals."""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class PairedRandomizationResult(BaseModel):
    """One internally consistent paired randomization summary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean_effect: float
    ci_low: float
    ci_high: float
    confidence: float = Field(gt=0.0, lt=1.0)
    p_value: float = Field(ge=0.0, le=1.0)
    wins_ties_losses: tuple[int, int, int]
    system_count: int = Field(gt=0)
    method: str
    randomization_draws: int = Field(gt=0)
    randomization_seed: int
    randomization_resolution: float = Field(gt=0.0)


def _sign_matrix(*, count: int, draws: int, seed: int, exact: bool) -> np.ndarray:
    if exact:
        states = np.arange(1 << count, dtype=np.uint64)[:, None]
        bits = (states >> np.arange(count, dtype=np.uint64)[None, :]) & 1
        return np.where(bits == 0, -1.0, 1.0)
    rng = np.random.default_rng(seed)
    return rng.choice(np.asarray([-1.0, 1.0]), size=(draws, count), replace=True)


def paired_sign_randomization(
    values: np.ndarray,
    *,
    confidence: float = 0.95,
    draws: int = 100_000,
    seed: int = 20260810,
    exact_max_pairs: int = 18,
) -> PairedRandomizationResult:
    """Test a zero paired mean and invert the same test for its interval.

    The randomization distribution uses sign flips of the centered paired
    differences. Exact enumeration is used for small samples; larger samples
    use one frozen Rademacher matrix for both the p-value and every point in
    the interval inversion.
    """

    differences = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(differences):
        raise ValueError("paired randomization requires a non-empty input")
    if not np.isfinite(differences).all():
        raise ValueError("paired randomization values must be finite")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if draws < 100:
        raise ValueError("paired randomization needs at least 100 draws")
    if exact_max_pairs < 1:
        raise ValueError("exact_max_pairs must be positive")

    count = len(differences)
    wins = int(np.sum(differences > 0.0))
    ties = int(np.sum(differences == 0.0))
    losses = int(np.sum(differences < 0.0))
    mean_effect = float(np.mean(differences))
    exact = count <= exact_max_pairs
    effective_draws = (1 << count) if exact else draws
    method = "exact_sign_randomization" if exact else "monte_carlo_sign_randomization"
    resolution = 1.0 / effective_draws if exact else 1.0 / (effective_draws + 1.0)

    if ties == count:
        return PairedRandomizationResult(
            mean_effect=0.0,
            ci_low=0.0,
            ci_high=0.0,
            confidence=confidence,
            p_value=1.0,
            wins_ties_losses=(wins, ties, losses),
            system_count=count,
            method=method,
            randomization_draws=effective_draws,
            randomization_seed=seed,
            randomization_resolution=resolution,
        )

    signs = _sign_matrix(count=count, draws=draws, seed=seed, exact=exact)
    randomized_data_mean = signs @ differences / count
    randomized_sign_mean = np.mean(signs, axis=1)

    def p_value(theta: float) -> float:
        observed = abs(mean_effect - theta)
        randomized = np.abs(randomized_data_mean - theta * randomized_sign_mean)
        exceedances = int(np.count_nonzero(randomized >= observed - 1e-15))
        if exact:
            return exceedances / effective_draws
        return (exceedances + 1.0) / (effective_draws + 1.0)

    null_p = float(p_value(0.0))
    alpha = 1.0 - confidence

    def accepted(theta: float) -> bool:
        return p_value(theta) >= alpha

    scale = max(
        float(np.std(differences, ddof=0)),
        float(np.ptp(differences)),
        abs(mean_effect),
        1.0,
    )

    lower_rejected = mean_effect - scale
    for _ in range(32):
        if not accepted(lower_rejected):
            break
        scale *= 2.0
        lower_rejected = mean_effect - scale
    else:
        lower_rejected = -math.inf

    upper_scale = max(
        float(np.std(differences, ddof=0)),
        float(np.ptp(differences)),
        abs(mean_effect),
        1.0,
    )
    upper_rejected = mean_effect + upper_scale
    for _ in range(32):
        if not accepted(upper_rejected):
            break
        upper_scale *= 2.0
        upper_rejected = mean_effect + upper_scale
    else:
        upper_rejected = math.inf

    if math.isfinite(lower_rejected):
        rejected = lower_rejected
        retained = mean_effect
        for _ in range(60):
            midpoint = 0.5 * (rejected + retained)
            if accepted(midpoint):
                retained = midpoint
            else:
                rejected = midpoint
        ci_low = retained
    else:
        ci_low = lower_rejected

    if math.isfinite(upper_rejected):
        retained = mean_effect
        rejected = upper_rejected
        for _ in range(60):
            midpoint = 0.5 * (retained + rejected)
            if accepted(midpoint):
                retained = midpoint
            else:
                rejected = midpoint
        ci_high = retained
    else:
        ci_high = upper_rejected

    return PairedRandomizationResult(
        mean_effect=mean_effect,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=confidence,
        p_value=null_p,
        wins_ties_losses=(wins, ties, losses),
        system_count=count,
        method=method,
        randomization_draws=effective_draws,
        randomization_seed=seed,
        randomization_resolution=resolution,
    )
