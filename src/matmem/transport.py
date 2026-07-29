"""Frozen source-to-target protocol transport models.

A transport model is fitted on disjoint exact chemical systems and then used
to form a working posterior over target-protocol energies for a new query
system.  No query trajectory or acquisition trace participates in the fit.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize

from .constants import (
    BETWEEN_SYSTEM_VARIANCE_FLOOR,
    DEFAULT_LOCAL_KERNEL_LENGTH_SCALE,
    LOCAL_KERNEL_BOUND_ACTIVE_TOLERANCE,
    LOCAL_KERNEL_LENGTH_FLOOR,
    LOCAL_KERNEL_OPTIMIZER_MESSAGE,
    LOCAL_KERNEL_QUANTILE_LOWER,
    LOCAL_KERNEL_QUANTILE_UPPER,
    LOCAL_KERNEL_SIGNAL_VARIANCE_FRACTION,
    LOCAL_KERNEL_VARIANCE_FLOOR,
    LOCAL_KERNEL_VARIANCE_MULTIPLIER,
    MIN_FEATURE_SCALE,
    MIN_POSITIVE_DISTANCE,
    TRANSPORT_RIDGE_INTERCEPT_FLOOR,
    WITHIN_SYSTEM_VARIANCE_FLOOR,
)


class FrozenProtocolRidgeTransport(BaseModel):
    """System-balanced ridge transport fitted on disjoint exact systems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    within_system_variance: float = Field(gt=0)
    between_system_variance: float = Field(gt=0)
    ridge_penalty: float = Field(gt=0)
    fit_system_ids: tuple[str, ...]
    fit_element_ids: tuple[str, ...]
    fit_row_count: int = Field(gt=0)
    local_kernel: Literal["independent", "matern52"] = "independent"
    local_kernel_signal_variance: float = Field(default=0.0, ge=0)
    local_kernel_noise_variance: float = Field(default=0.0, ge=0)
    local_kernel_length_scale: float = Field(default=DEFAULT_LOCAL_KERNEL_LENGTH_SCALE, gt=0)
    local_kernel_fit_system_count: int = Field(default=0, ge=0)
    local_kernel_nll_per_row: float | None = None
    local_kernel_optimizer_success: bool | None = None
    local_kernel_optimizer_status: int | None = None
    local_kernel_optimizer_message: str | None = LOCAL_KERNEL_OPTIMIZER_MESSAGE
    local_kernel_optimizer_gradient_norm: float | None = Field(default=None, ge=0)
    local_kernel_optimizer_bounds_active: tuple[str, ...] = ()
    kernel_feature_mean: tuple[float, ...] = ()
    kernel_feature_scale: tuple[float, ...] = ()
    kernel_feature_encoder: str | None = None
    kernel_feature_encoder_checksum: str | None = None

    @model_validator(mode="after")
    def _dimensions(self) -> FrozenProtocolRidgeTransport:
        feature_count = len(self.feature_mean)
        if feature_count == 0 or len(self.feature_scale) != feature_count:
            raise ValueError("transport feature normalization is inconsistent")
        if any(value <= 0 or not math.isfinite(value) for value in self.feature_scale):
            raise ValueError("transport feature scales must be finite and positive")
        if len(self.coefficients) != feature_count + 1:
            raise ValueError("transport coefficient dimension is inconsistent")
        if len(self.precision) != len(self.coefficients) or any(
            len(row) != len(self.coefficients) for row in self.precision
        ):
            raise ValueError("transport precision dimension is inconsistent")
        if not self.fit_system_ids or len(set(self.fit_system_ids)) != len(self.fit_system_ids):
            raise ValueError("transport fit systems must be unique and nonempty")
        if not self.fit_element_ids or len(set(self.fit_element_ids)) != len(self.fit_element_ids):
            raise ValueError("transport fit elements must be unique and nonempty")
        arrays = (
            np.asarray(self.feature_mean),
            np.asarray(self.feature_scale),
            np.asarray(self.coefficients),
            np.asarray(self.precision),
        )
        if any(not np.isfinite(values).all() for values in arrays):
            raise ValueError("transport parameters must be finite")
        if self.local_kernel == "independent":
            if (
                self.local_kernel_signal_variance != 0
                or self.kernel_feature_mean
                or self.kernel_feature_scale
                or self.kernel_feature_encoder is not None
                or self.kernel_feature_encoder_checksum is not None
            ):
                raise ValueError("independent transport cannot carry local-kernel state")
        elif (
            self.local_kernel_signal_variance <= 0
            or self.local_kernel_noise_variance <= 0
            or self.local_kernel_fit_system_count < 2
            or self.local_kernel_nll_per_row is None
            or not math.isfinite(self.local_kernel_nll_per_row)
            or not self.kernel_feature_mean
            or len(self.kernel_feature_scale) != len(self.kernel_feature_mean)
            or any(value <= 0 or not math.isfinite(value) for value in self.kernel_feature_scale)
            or not self.kernel_feature_encoder
            or not self.kernel_feature_encoder_checksum
        ):
            raise ValueError("Matérn transport requires a frozen observable kernel representation")
        return self

    @property
    def identity_checksum(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


def _raw_features(features: np.ndarray, source_energies: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    source = np.asarray(source_energies, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or not len(values) or len(source) != len(values):
        raise ValueError("protocol transport features and source energies disagree")
    if not np.isfinite(values).all() or not np.isfinite(source).all():
        raise ValueError("protocol transport inputs must be finite")
    return np.column_stack((values, source))


def _matern52_covariance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    length_scale: float,
) -> np.ndarray:
    """Unit-variance Matérn-5/2 covariance on standardized observables."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if (
        left_values.ndim != 2
        or right_values.ndim != 2
        or left_values.shape[1] != right_values.shape[1]
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
        or not math.isfinite(length_scale)
        or length_scale <= 0
    ):
        raise ValueError("Matérn covariance inputs are inconsistent")
    distances = np.linalg.norm(
        left_values[:, None, :] - right_values[None, :, :],
        axis=2,
    )
    scaled = math.sqrt(5.0) * distances / length_scale
    return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def fit_protocol_ridge_transport(
    *,
    features: np.ndarray,
    source_energies: np.ndarray,
    target_energies: np.ndarray,
    system_ids: Sequence[str],
    ridge_penalty: float = 1.0,
) -> FrozenProtocolRidgeTransport:
    """Fit a system-balanced source-to-target discrepancy model.

    Each exact system has equal total regression weight.  Variance components
    are then estimated from within-system residuals and held-out-system random
    intercepts instead of being tuned on a query trajectory.
    """

    raw = _raw_features(features, source_energies)
    target = np.asarray(target_energies, dtype=np.float64).reshape(-1)
    systems = tuple(str(value) for value in system_ids)
    if len(target) != len(raw) or len(systems) != len(raw):
        raise ValueError("protocol transport targets and systems disagree")
    if not np.isfinite(target).all():
        raise ValueError("protocol transport targets must be finite")
    if not math.isfinite(ridge_penalty) or ridge_penalty <= 0:
        raise ValueError("protocol transport ridge penalty must be positive")
    unique_systems = tuple(sorted(set(systems)))
    if len(unique_systems) < 2:
        raise ValueError("protocol transport requires at least two fit systems")

    feature_mean = raw.mean(axis=0)
    feature_scale = raw.std(axis=0)
    feature_scale[feature_scale < MIN_FEATURE_SCALE] = 1.0
    standardized = (raw - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(raw)), standardized))
    counts = Counter(systems)
    weights = np.asarray(
        [len(raw) / (len(unique_systems) * counts[system]) for system in systems],
        dtype=np.float64,
    )
    penalty = ridge_penalty * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = TRANSPORT_RIDGE_INTERCEPT_FLOOR
    precision = design.T @ (weights[:, None] * design) + penalty
    discrepancy = target - np.asarray(source_energies, dtype=np.float64)
    coefficients = np.linalg.solve(
        precision,
        design.T @ (weights * discrepancy),
    )
    residuals = discrepancy - design @ coefficients
    system_means = {
        system: float(np.mean(residuals[np.asarray(systems) == system]))
        for system in unique_systems
    }
    centered = np.asarray(
        [residual - system_means[system] for residual, system in zip(residuals, systems)],
        dtype=np.float64,
    )
    within_variance = max(float(np.mean(centered**2)), WITHIN_SYSTEM_VARIANCE_FLOOR)
    means = np.asarray([system_means[system] for system in unique_systems])
    sampling_variance = float(
        np.mean([within_variance / counts[system] for system in unique_systems])
    )
    between_variance = max(
        float(np.var(means, ddof=1)) - sampling_variance, BETWEEN_SYSTEM_VARIANCE_FLOOR
    )
    return FrozenProtocolRidgeTransport(
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        coefficients=tuple(float(value) for value in coefficients),
        precision=tuple(tuple(float(value) for value in row) for row in precision),
        within_system_variance=within_variance,
        between_system_variance=between_variance,
        ridge_penalty=ridge_penalty,
        fit_system_ids=unique_systems,
        fit_element_ids=tuple(
            sorted({element for system in unique_systems for element in system.split("-")})
        ),
        fit_row_count=len(raw),
    )


def fit_protocol_kernel_transport(
    *,
    features: np.ndarray,
    kernel_features: np.ndarray,
    source_energies: np.ndarray,
    target_energies: np.ndarray,
    system_ids: Sequence[str],
    kernel_feature_encoder: str,
    kernel_feature_encoder_checksum: str,
    ridge_penalty: float = 1.0,
) -> FrozenProtocolRidgeTransport:
    """Fit a hierarchical autoregressive discrepancy posterior.

    The cross-system mean remains the system-balanced ridge transport over
    composition and source-protocol observables.  A separate frozen structure
    representation defines a shared Matérn-5/2 covariance for the residual
    discrepancy inside each exact chemical system.  Kernel scales are
    empirical-Bayes estimates from the disjoint fit systems; no query-system
    outcome or acquisition trace participates in the fit.
    """

    model = fit_protocol_ridge_transport(
        features=features,
        source_energies=source_energies,
        target_energies=target_energies,
        system_ids=system_ids,
        ridge_penalty=ridge_penalty,
    )
    raw = _raw_features(features, source_energies)
    kernel_raw = np.asarray(kernel_features, dtype=np.float64)
    if (
        kernel_raw.ndim != 2
        or len(kernel_raw) != len(raw)
        or kernel_raw.shape[1] == 0
        or not np.isfinite(kernel_raw).all()
        or not kernel_feature_encoder.strip()
        or not kernel_feature_encoder_checksum.strip()
    ):
        raise ValueError("local protocol kernel features or provenance are inconsistent")
    target = np.asarray(target_energies, dtype=np.float64).reshape(-1)
    systems = np.asarray([str(value) for value in system_ids], dtype=object)
    standardized = (raw - np.asarray(model.feature_mean, dtype=np.float64)) / np.asarray(
        model.feature_scale, dtype=np.float64
    )
    kernel_feature_mean = kernel_raw.mean(axis=0)
    kernel_feature_scale = kernel_raw.std(axis=0)
    kernel_feature_scale[kernel_feature_scale < MIN_FEATURE_SCALE] = 1.0
    kernel_standardized = (kernel_raw - kernel_feature_mean) / kernel_feature_scale
    design = np.column_stack((np.ones(len(raw)), standardized))
    discrepancy = target - np.asarray(source_energies, dtype=np.float64)
    residuals = discrepancy - design @ np.asarray(model.coefficients)

    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    positive_distances: list[np.ndarray] = []
    for system in model.fit_system_ids:
        mask = systems == system
        system_x = kernel_standardized[mask]
        system_y = residuals[mask]
        if len(system_y) < 2:
            continue
        system_y = system_y - float(np.mean(system_y))
        blocks.append((system_x, system_y))
        distances = np.linalg.norm(system_x[:, None, :] - system_x[None, :, :], axis=2)
        values = distances[np.triu_indices(len(system_x), k=1)]
        positive = values[values > MIN_POSITIVE_DISTANCE]
        if len(positive):
            positive_distances.append(positive)
    if len(blocks) < 2 or not positive_distances:
        raise ValueError("local protocol kernel requires two non-degenerate fit systems")

    distances = np.concatenate(positive_distances)
    lower_length = max(
        float(np.quantile(distances, LOCAL_KERNEL_QUANTILE_LOWER)), LOCAL_KERNEL_LENGTH_FLOOR
    )
    upper_length = max(
        float(np.quantile(distances, LOCAL_KERNEL_QUANTILE_UPPER)), 2.0 * lower_length
    )
    initial_length = float(np.clip(np.median(distances), lower_length, upper_length))
    total_variance = max(model.within_system_variance, WITHIN_SYSTEM_VARIANCE_FLOOR)
    lower_variance = max(
        total_variance * LOCAL_KERNEL_SIGNAL_VARIANCE_FRACTION, LOCAL_KERNEL_VARIANCE_FLOOR
    )
    upper_variance = max(
        total_variance * LOCAL_KERNEL_VARIANCE_MULTIPLIER,
        lower_variance * LOCAL_KERNEL_VARIANCE_MULTIPLIER,
    )

    def objective(log_parameters: np.ndarray) -> float:
        length_scale, signal_variance, noise_variance = np.exp(log_parameters)
        values = []
        for system_x, system_y in blocks:
            covariance = signal_variance * _matern52_covariance(
                system_x,
                system_x,
                length_scale=length_scale,
            )
            covariance += noise_variance * np.eye(len(system_y))
            covariance += 1e-10 * np.eye(len(system_y))
            try:
                factor = np.linalg.cholesky(covariance)
                solved = np.linalg.solve(factor, system_y)
            except np.linalg.LinAlgError:
                return float("inf")
            nll = 0.5 * float(solved @ solved)
            nll += float(np.log(np.diag(factor)).sum())
            nll += 0.5 * len(system_y) * math.log(2.0 * math.pi)
            values.append(nll / len(system_y))
        return float(np.mean(values))

    initial = np.log(np.asarray([initial_length, total_variance / 2.0, total_variance / 2.0]))
    bounds = (
        (math.log(lower_length), math.log(upper_length)),
        (math.log(lower_variance), math.log(upper_variance)),
        (math.log(lower_variance), math.log(upper_variance)),
    )
    optimized = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    if not np.isfinite(optimized.fun):
        raise RuntimeError("local protocol kernel marginal likelihood is non-finite")
    if not bool(optimized.success):
        raise RuntimeError(
            "local protocol kernel marginal likelihood optimizer failed: "
            f"status={optimized.status} message={optimized.message}"
        )
    length_scale, signal_variance, noise_variance = np.exp(optimized.x)
    gradient = np.asarray(getattr(optimized, "jac", ()), dtype=float)
    gradient_norm = (
        float(np.linalg.norm(gradient)) if gradient.size and np.isfinite(gradient).all() else None
    )
    bound_names = ("length_scale", "signal_variance", "noise_variance")
    bounds_active = tuple(
        name
        for name, value, (lower, upper) in zip(bound_names, optimized.x, bounds, strict=True)
        if abs(float(value) - lower) <= LOCAL_KERNEL_BOUND_ACTIVE_TOLERANCE
        or abs(float(value) - upper) <= LOCAL_KERNEL_BOUND_ACTIVE_TOLERANCE
    )
    payload = model.model_dump()
    payload.update(
        {
            "local_kernel": "matern52",
            "local_kernel_signal_variance": float(signal_variance),
            "local_kernel_noise_variance": float(noise_variance),
            "local_kernel_length_scale": float(length_scale),
            "local_kernel_fit_system_count": len(blocks),
            "local_kernel_nll_per_row": float(optimized.fun),
            "local_kernel_optimizer_success": bool(optimized.success),
            "local_kernel_optimizer_status": int(optimized.status),
            "local_kernel_optimizer_message": str(optimized.message),
            "local_kernel_optimizer_gradient_norm": gradient_norm,
            "local_kernel_optimizer_bounds_active": bounds_active,
            "kernel_feature_mean": tuple(float(value) for value in kernel_feature_mean),
            "kernel_feature_scale": tuple(float(value) for value in kernel_feature_scale),
            "kernel_feature_encoder": kernel_feature_encoder,
            "kernel_feature_encoder_checksum": kernel_feature_encoder_checksum,
        }
    )
    return FrozenProtocolRidgeTransport.model_validate(payload)
