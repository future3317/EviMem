"""Gaussian working posterior over target-protocol pool energies.

The posterior conditions a frozen transport model on every revealed
 current-system outcome.  It also supplies deterministic scrambled-Sobol
posterior sampling used by all acquisition and campaign functions.
"""

from __future__ import annotations

import math

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.special import ndtri
from scipy.stats import qmc

from .constants import (
    COVARIANCE_CLIP_EPSILON,
    HULL_PSD_TOLERANCE,
    HULL_SYMMETRY_TOLERANCE,
    KERNEL_JITTER,
    PSD_REGULARIZATION,
)
from .transport import FrozenProtocolRidgeTransport, _matern52_covariance


class ProtocolTargetEnergyPosterior(BaseModel):
    """Joint Gaussian working posterior over target-protocol pool energies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    system_offset_mean: float
    system_offset_variance: float = Field(ge=0)
    history_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _dimensions(self) -> ProtocolTargetEnergyPosterior:
        size = len(self.mean)
        if (
            size == 0
            or len(self.covariance) != size
            or any(len(row) != size for row in self.covariance)
        ):
            raise ValueError("target posterior dimensions are inconsistent")
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("target posterior must be finite")
        if not np.allclose(covariance, covariance.T, atol=HULL_SYMMETRY_TOLERANCE):
            raise ValueError("target posterior covariance must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -HULL_PSD_TOLERANCE:
            raise ValueError("target posterior covariance must be positive semidefinite")
        return self


def _sample_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Draw a deterministic nested scrambled-Sobol Gaussian design.

    Hull membership is a discontinuous functional, so ordinary pseudo-random
    Monte Carlo can change an action merely because the requested sample count
    changed.  A scrambled Sobol design preserves randomized-QMC error control
    while making every power-of-two run a prefix of the next run for the same
    seed.  That gives the policy an observable numerical-convergence check
    without changing its posterior or acquisition objective.
    """

    if sample_count < 1:
        raise ValueError("Gaussian posterior sampling requires a positive count")
    dimension = len(mean)
    if dimension < 1:
        raise ValueError("Gaussian posterior sampling requires a nonempty mean")
    factor = _gaussian_factor(covariance)
    return _sample_gaussian_from_factor(mean, factor, sample_count=sample_count, seed=seed)


def _gaussian_factor(covariance: np.ndarray) -> np.ndarray:
    """Return the frozen symmetric-eigendecomposition Gaussian factor."""

    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))


def _sample_gaussian_from_factor(
    mean: np.ndarray,
    factor: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Draw one original Sobol block from an already computed exact factor."""

    if sample_count < 1:
        raise ValueError("Gaussian posterior sampling requires a positive count")
    dimension = len(mean)
    if factor.shape != (dimension, dimension):
        raise ValueError("Gaussian factor dimensions disagree with mean")
    normal = _standard_normal_block(
        dimension=dimension,
        sample_count=sample_count,
        seed=seed,
    )
    return mean + normal @ factor.T


def _standard_normal_block(
    *,
    dimension: int,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Return one registered scrambled-Sobol standard-normal block."""

    if dimension < 1:
        raise ValueError("Gaussian sampling requires a positive dimension")
    if sample_count < 1:
        raise ValueError("Gaussian sampling requires a positive sample count")
    exponent = math.ceil(math.log2(sample_count))
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)
    unit = unit[:sample_count]
    epsilon = COVARIANCE_CLIP_EPSILON
    return ndtri(np.clip(unit, epsilon, 1.0 - epsilon))


def _sample_gaussian_blocks(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    sample_count: int,
    seeds: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    """Draw independent registered Sobol blocks with one exact factorization.

    The block seed sequence and each block's Sobol construction remain exactly
    those of independent :func:`_sample_gaussian` calls.  Only the repeated
    covariance symmetrization/eigendecomposition is removed.
    """

    if not seeds:
        raise ValueError("Gaussian block sampling requires at least one seed")
    factor = _gaussian_factor(covariance)
    return tuple(
        _sample_gaussian_from_factor(mean, factor, sample_count=sample_count, seed=seed)
        for seed in seeds
    )


def protocol_target_energy_posterior(
    model: FrozenProtocolRidgeTransport,
    *,
    query_features: np.ndarray,
    query_source_energies: np.ndarray,
    history_features: np.ndarray,
    history_source_energies: np.ndarray,
    history_target_energies: np.ndarray,
    query_kernel_features: np.ndarray | None = None,
    history_kernel_features: np.ndarray | None = None,
) -> ProtocolTargetEnergyPosterior:
    """Condition the frozen transport on every revealed current-system outcome."""

    from .transport import _raw_features

    query_raw = _raw_features(query_features, query_source_energies)
    history_x = np.asarray(history_features, dtype=np.float64)
    history_source = np.asarray(history_source_energies, dtype=np.float64).reshape(-1)
    history_target = np.asarray(history_target_energies, dtype=np.float64).reshape(-1)
    if history_x.ndim != 2 or history_x.shape[1] != query_raw.shape[1] - 1:
        raise ValueError("protocol posterior history feature dimension disagrees")
    if len(history_x) != len(history_source) or len(history_x) != len(history_target):
        raise ValueError("protocol posterior history arrays disagree")
    if any(not np.isfinite(values).all() for values in (history_x, history_source, history_target)):
        raise ValueError("protocol posterior history must be finite")

    feature_mean = np.asarray(model.feature_mean)
    feature_scale = np.asarray(model.feature_scale)
    coefficients = np.asarray(model.coefficients)
    precision = np.asarray(model.precision)

    def design(features: np.ndarray, source: np.ndarray) -> np.ndarray:
        raw = np.column_stack((features, source))
        return np.column_stack((np.ones(len(raw)), (raw - feature_mean) / feature_scale))

    query_design = design(
        np.asarray(query_features, dtype=np.float64),
        np.asarray(query_source_energies, dtype=np.float64),
    )
    within = model.within_system_variance
    between = model.between_system_variance
    if model.local_kernel == "matern52":
        query_kernel = np.asarray(query_kernel_features, dtype=np.float64)
        history_kernel = np.asarray(history_kernel_features, dtype=np.float64)
        kernel_mean = np.asarray(model.kernel_feature_mean, dtype=np.float64)
        kernel_scale = np.asarray(model.kernel_feature_scale, dtype=np.float64)
        if (
            query_kernel.ndim != 2
            or query_kernel.shape != (len(query_raw), len(kernel_mean))
            or history_kernel.ndim != 2
            or history_kernel.shape != (len(history_x), len(kernel_mean))
            or not np.isfinite(query_kernel).all()
            or not np.isfinite(history_kernel).all()
        ):
            raise ValueError("frozen local-kernel embeddings are missing or inconsistent")
        coefficient_covariance = np.linalg.inv(precision) * (within + between)
        query_source = np.asarray(query_source_energies, dtype=np.float64)
        predicted_mean = query_source + query_design @ coefficients
        query_standardized = (query_kernel - kernel_mean) / kernel_scale
        covariance = query_design @ coefficient_covariance @ query_design.T
        covariance += between * np.ones((len(query_raw), len(query_raw)))
        covariance += model.local_kernel_signal_variance * _matern52_covariance(
            query_standardized,
            query_standardized,
            length_scale=model.local_kernel_length_scale,
        )
        covariance += model.local_kernel_noise_variance * np.eye(len(query_raw))
        system_offset_mean = 0.0
        system_offset_variance = between
        if len(history_x):
            history_design = design(history_x, history_source)
            history_standardized = (history_kernel - kernel_mean) / kernel_scale
            history_mean = history_source + history_design @ coefficients
            history_covariance = history_design @ coefficient_covariance @ history_design.T
            history_covariance += between * np.ones((len(history_x), len(history_x)))
            history_covariance += model.local_kernel_signal_variance * _matern52_covariance(
                history_standardized,
                history_standardized,
                length_scale=model.local_kernel_length_scale,
            )
            history_covariance += model.local_kernel_noise_variance * np.eye(len(history_x))
            history_covariance += KERNEL_JITTER * np.eye(len(history_x))
            cross_covariance = query_design @ coefficient_covariance @ history_design.T
            cross_covariance += between * np.ones((len(query_raw), len(history_x)))
            cross_covariance += model.local_kernel_signal_variance * _matern52_covariance(
                query_standardized,
                history_standardized,
                length_scale=model.local_kernel_length_scale,
            )
            factor = np.linalg.cholesky(history_covariance)
            innovation = history_target - history_mean
            solved_innovation = np.linalg.solve(factor.T, np.linalg.solve(factor, innovation))
            predicted_mean = predicted_mean + cross_covariance @ solved_innovation
            solved_cross = np.linalg.solve(factor, cross_covariance.T)
            covariance = covariance - solved_cross.T @ solved_cross
            ones = np.ones(len(history_x), dtype=np.float64)
            solved_ones = np.linalg.solve(factor.T, np.linalg.solve(factor, ones))
            system_offset_mean = float(between * ones @ solved_innovation)
            system_offset_variance = max(
                float(between - between**2 * ones @ solved_ones),
                0.0,
            )
        covariance = 0.5 * (covariance + covariance.T)
        covariance += PSD_REGULARIZATION * np.eye(len(query_raw))
        return ProtocolTargetEnergyPosterior(
            mean=tuple(float(value) for value in predicted_mean),
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            system_offset_mean=system_offset_mean,
            system_offset_variance=system_offset_variance,
            history_count=len(history_x),
        )

    parameter_count = len(coefficients) + 1
    prior_precision = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    prior_precision[:-1, :-1] = precision / (within + between)
    prior_precision[-1, -1] = 1.0 / between
    prior_mean = np.concatenate((coefficients, [0.0]))
    posterior_precision = prior_precision.copy()
    posterior_natural = prior_precision @ prior_mean
    if len(history_x):
        history_design = design(history_x, history_source)
        history_joint_design = np.column_stack((history_design, np.ones(len(history_design))))
        history_discrepancy = history_target - history_source
        posterior_precision += history_joint_design.T @ history_joint_design / within
        posterior_natural += history_joint_design.T @ history_discrepancy / within
    parameter_covariance = np.linalg.inv(posterior_precision)
    parameter_mean = parameter_covariance @ posterior_natural
    query_joint_design = np.column_stack((query_design, np.ones(len(query_design))))
    predicted_mean = np.asarray(query_source_energies, dtype=np.float64) + (
        query_joint_design @ parameter_mean
    )
    covariance = query_joint_design @ parameter_covariance @ query_joint_design.T
    covariance += within * np.eye(len(query_raw))
    covariance = 0.5 * (covariance + covariance.T)
    covariance += PSD_REGULARIZATION * np.eye(len(query_raw))
    return ProtocolTargetEnergyPosterior(
        mean=tuple(float(value) for value in predicted_mean),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        system_offset_mean=float(parameter_mean[-1]),
        system_offset_variance=float(parameter_covariance[-1, -1]),
        history_count=len(history_x),
    )
