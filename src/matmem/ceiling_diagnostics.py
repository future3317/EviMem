"""Model-ceiling diagnostics for fixed-kernel residual posteriors.

These functions are evaluator-only.  They never participate in acquisition or
retention and therefore may consume revealed residuals after a trace is frozen.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.special import ndtri
from sklearn.gaussian_process.kernels import RBF, Matern

from .residual_posterior import FixedKernelGPConfig


def normalized_rows(values: np.ndarray) -> np.ndarray:
    """Return finite unit-norm rows or fail closed."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("kernel diagnostics require a nonempty matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or np.any(norms <= 0):
        raise ValueError("kernel diagnostic embeddings must be finite and nonzero")
    return matrix / norms[:, None]


def signal_kernel_matrix(
    embeddings: np.ndarray,
    config: FixedKernelGPConfig,
) -> np.ndarray:
    """Evaluate the exact frozen signal kernel, excluding predictive white noise."""

    matrix = normalized_rows(embeddings)
    base = (
        Matern(length_scale=config.length_scale, nu=2.5)
        if config.kernel == "matern52"
        else RBF(length_scale=config.length_scale)
    )
    return config.signal_std_ev_per_atom**2 * np.asarray(base(matrix), dtype=float)


def regularized_effective_dimension(
    embeddings: np.ndarray,
    config: FixedKernelGPConfig,
) -> float:
    """Compute ``tr[K(K + lambda I)^-1]`` for frozen GP noise ``lambda``."""

    kernel = signal_kernel_matrix(embeddings, config)
    regularizer = config.noise_std_ev_per_atom**2 + config.jitter
    eigenvalues = np.maximum(np.linalg.eigvalsh(kernel), 0.0)
    return float(np.sum(eigenvalues / (eigenvalues + regularizer)))


def centered_kernel_target_alignment(kernel: np.ndarray, residuals: np.ndarray) -> float:
    """Centered alignment between a kernel and the residual outer product."""

    matrix = np.asarray(kernel, dtype=float)
    target = np.asarray(residuals, dtype=float)
    if matrix.shape != (len(target), len(target)) or len(target) < 2:
        raise ValueError("kernel alignment requires matching square data")
    centering = np.eye(len(target)) - np.ones((len(target), len(target))) / len(target)
    centered_kernel = centering @ matrix @ centering
    centered_target = np.outer(target - np.mean(target), target - np.mean(target))
    denominator = np.linalg.norm(centered_kernel) * np.linalg.norm(centered_target)
    return float(np.sum(centered_kernel * centered_target) / denominator) if denominator else 0.0


def kernel_moran_autocorrelation(kernel: np.ndarray, residuals: np.ndarray) -> float:
    """Moran-style residual autocorrelation using off-diagonal kernel weights."""

    weights = np.asarray(kernel, dtype=float).copy()
    target = np.asarray(residuals, dtype=float)
    if weights.shape != (len(target), len(target)) or len(target) < 2:
        raise ValueError("kernel autocorrelation requires matching square data")
    np.fill_diagonal(weights, 0.0)
    centered = target - np.mean(target)
    denominator = float(np.sum(centered**2))
    weight_sum = float(np.sum(weights))
    if denominator <= 0 or weight_sum <= 0:
        return 0.0
    return float(len(target) * centered @ weights @ centered / (weight_sum * denominator))


def nearest_neighbor_sign_agreement(kernel: np.ndarray, residuals: np.ndarray) -> float:
    """Fraction whose strongest non-self kernel neighbor has the same residual sign."""

    matrix = np.asarray(kernel, dtype=float).copy()
    target = np.asarray(residuals, dtype=float)
    if matrix.shape != (len(target), len(target)) or len(target) < 2:
        raise ValueError("nearest-neighbor agreement requires matching square data")
    np.fill_diagonal(matrix, -math.inf)
    neighbors = np.argmax(matrix, axis=1)
    return float(np.mean(np.sign(target) == np.sign(target[neighbors])))


def gaussian_dispersion_diagnostics(
    truth: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float | int]:
    """Measure standardized dispersion and central Gaussian interval coverage."""

    targets = np.asarray(truth, dtype=float)
    means = np.asarray(mean, dtype=float)
    scales = np.asarray(std, dtype=float)
    if targets.ndim != 1 or means.shape != targets.shape or scales.shape != targets.shape:
        raise ValueError("dispersion diagnostics require matching one-dimensional arrays")
    if not len(targets) or not np.all(np.isfinite((targets, means, scales))):
        raise ValueError("dispersion diagnostics require nonempty finite arrays")
    if np.any(scales <= 0):
        raise ValueError("dispersion diagnostics require positive standard deviations")
    absolute_z = np.abs((targets - means) / scales)
    result: dict[str, float | int] = {
        "observation_count": len(targets),
        "mean_squared_standardized_residual": float(np.mean(absolute_z**2)),
        "median_squared_standardized_residual": float(np.median(absolute_z**2)),
    }
    for nominal in (0.5, 0.8, 0.9):
        cutoff = float(ndtri((1.0 + nominal) / 2.0))
        result[f"central_{int(100 * nominal)}_coverage"] = float(
            np.mean(absolute_z <= cutoff)
        )
    return result


def deterministic_bootstrap_mean(
    values: Sequence[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, float | int]:
    """Exact-system bootstrap summary with deterministic sampling."""

    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector) or iterations < 1:
        raise ValueError("bootstrap requires nonempty values and positive iterations")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(vector), size=(iterations, len(vector)))
    means = np.mean(vector[indices], axis=1)
    return {
        "system_count": len(vector),
        "mean": float(np.mean(vector)),
        "median": float(np.median(vector)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
    }
