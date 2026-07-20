"""Environment-conditional, fail-closed directed protocol transport."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

from .protocols import ProtocolCertificate


class MatchedEnvironmentEnergyPair(BaseModel):
    """Same-candidate source/target energy pair with source-only observables."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_calculation_id: str
    canonical_structure_id: str
    chemical_system: str
    element_fractions: dict[str, float]
    source_descriptor: tuple[float, ...]
    source_energy_ev_per_atom: float
    target_energy_ev_per_atom: float

    @field_validator("exact_calculation_id", "canonical_structure_id", "chemical_system")
    @classmethod
    def _identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("environment transport pair requires non-empty identity")
        return value

    @field_validator("element_fractions")
    @classmethod
    def _fractions(cls, values: dict[str, float]) -> dict[str, float]:
        normalized = {key.strip(): float(value) for key, value in values.items()}
        if (
            not normalized
            or any(
                not key or not math.isfinite(value) or value < 0
                for key, value in normalized.items()
            )
            or not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("environment pair fractions must be finite and sum to one")
        return dict(sorted(normalized.items()))

    @field_validator("source_descriptor")
    @classmethod
    def _descriptor(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) < 2 or any(not math.isfinite(value) for value in values):
            raise ValueError("source environment descriptor must contain finite values")
        return values

    @field_validator("source_energy_ev_per_atom", "target_energy_ev_per_atom")
    @classmethod
    def _energy(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("environment pair energies must be finite")
        return value


class EnvironmentTransportStatus(StrEnum):
    CERTIFIED = "certified"
    REJECT_UNSEEN_ELEMENT = "reject_unseen_element"
    REJECT_DESCRIPTOR_DIMENSION = "reject_descriptor_dimension"


class EnvironmentTransportPrediction(BaseModel):
    """Point transport plus a simultaneous interval when support is certified."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EnvironmentTransportStatus
    target_energy_ev_per_atom: float | None = None
    lower_energy_ev_per_atom: float | None = None
    upper_energy_ev_per_atom: float | None = None
    leverage_scale: float | None = None
    interval_half_width_ev_per_atom: float | None = None
    reason: str


class EnvironmentConditionalProtocolTransportMap(BaseModel):
    """Ridge delta transport with observable support and clustered intervals.

    The same-candidate source energy remains an explicit base.  Ridge predicts
    only the target-minus-source correction from source energy and the frozen
    source-structure descriptor.  The error radius is the split-conformal
    quantile of maximum within-exact-system errors after the input-only support
    gate has been frozen on fit observables.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_protocol: ProtocolCertificate
    target_protocol: ProtocolCertificate
    descriptor_dimension: int = Field(ge=2)
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    delta_coefficient: tuple[float, ...]
    delta_intercept_ev_per_atom: float
    posterior_precision_inverse: tuple[tuple[float, ...], ...]
    supported_elements: tuple[str, ...]
    normalized_error_quantile_ev_per_atom: float = Field(ge=0)
    fit_residual_variance_ev2_per_atom2: float = Field(gt=0)
    ridge_penalty: float = Field(gt=0)
    ridge_selection_rule: str
    radius_alpha: float = Field(gt=0, lt=1)
    fit_structure_count: int = Field(ge=3)
    fit_chemical_system_count: int = Field(ge=2)
    radius_calibration_structure_count: int = Field(ge=3)
    radius_calibration_chemical_system_count: int = Field(ge=2)
    supported_radius_chemical_system_count: int = Field(ge=1)
    calibration_group_checksum: str
    calibration_id: str

    @field_validator(
        "delta_intercept_ev_per_atom",
        "normalized_error_quantile_ev_per_atom",
        "fit_residual_variance_ev2_per_atom2",
        "ridge_penalty",
    )
    @classmethod
    def _finite_scalar(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("environment transport parameters must be finite")
        return value

    @field_validator("feature_mean", "feature_scale", "delta_coefficient")
    @classmethod
    def _finite_vector(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(not math.isfinite(value) for value in values):
            raise ValueError("environment transport vectors must be finite")
        return values

    @field_validator("supported_elements")
    @classmethod
    def _elements(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({value.strip() for value in values}))
        if not normalized or any(not value for value in normalized):
            raise ValueError("environment transport requires a supported element vocabulary")
        return normalized

    @field_validator("calibration_id")
    @classmethod
    def _calibration_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("environment transport requires immutable calibration identity")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> EnvironmentConditionalProtocolTransportMap:
        expected = self.descriptor_dimension + 1
        if not (
            len(self.feature_mean)
            == len(self.feature_scale)
            == len(self.delta_coefficient)
            == expected
        ):
            raise ValueError("environment transport vector dimensions disagree")
        augmented_dimension = expected + 1
        if (
            len(self.posterior_precision_inverse) != augmented_dimension
            or any(
                len(row) != augmented_dimension
                for row in self.posterior_precision_inverse
            )
        ):
            raise ValueError("environment transport leverage matrix dimension disagrees")
        if any(value <= 0 for value in self.feature_scale):
            raise ValueError("environment transport feature scales must be positive")
        if (
            self.source_protocol.scientific_fingerprint
            == self.target_protocol.scientific_fingerprint
        ):
            raise ValueError("identical protocols must use direct outcomes, not transport")
        return self

    def _raw_feature(
        self, source_energy_ev_per_atom: float, source_descriptor: tuple[float, ...]
    ) -> np.ndarray | None:
        if len(source_descriptor) != self.descriptor_dimension:
            return None
        values = np.asarray((source_energy_ev_per_atom, *source_descriptor), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("environment transport input must be finite")
        return values

    def predict(
        self,
        source_energy_ev_per_atom: float,
        source_descriptor: tuple[float, ...],
        element_fractions: dict[str, float],
    ) -> EnvironmentTransportPrediction:
        if not set(element_fractions) <= set(self.supported_elements):
            return EnvironmentTransportPrediction(
                status=EnvironmentTransportStatus.REJECT_UNSEEN_ELEMENT,
                reason="candidate_contains_element_absent_from_transport_fit",
            )
        raw = self._raw_feature(source_energy_ev_per_atom, source_descriptor)
        if raw is None:
            return EnvironmentTransportPrediction(
                status=EnvironmentTransportStatus.REJECT_DESCRIPTOR_DIMENSION,
                reason="candidate_descriptor_dimension_differs_from_frozen_transport",
            )
        standardized = (raw - np.asarray(self.feature_mean)) / np.asarray(
            self.feature_scale
        )
        augmented = np.concatenate(([1.0], standardized))
        precision_inverse = np.asarray(self.posterior_precision_inverse)
        leverage = math.sqrt(
            max(1.0 + float(augmented @ precision_inverse @ augmented), 1.0)
        )
        correction = float(
            standardized @ np.asarray(self.delta_coefficient)
            + self.delta_intercept_ev_per_atom
        )
        target = source_energy_ev_per_atom + correction
        radius = self.normalized_error_quantile_ev_per_atom * leverage
        return EnvironmentTransportPrediction(
            status=EnvironmentTransportStatus.CERTIFIED,
            target_energy_ev_per_atom=target,
            lower_energy_ev_per_atom=target - radius,
            upper_energy_ev_per_atom=target + radius,
            leverage_scale=leverage,
            interval_half_width_ev_per_atom=radius,
            reason="environment_supported_leverage_scaled_cluster_calibrated_transport",
        )

    @classmethod
    def fit_same_candidate_system_split(
        cls,
        source_protocol: ProtocolCertificate,
        target_protocol: ProtocolCertificate,
        fit_pairs: Iterable[MatchedEnvironmentEnergyPair],
        radius_calibration_pairs: Iterable[MatchedEnvironmentEnergyPair],
        *,
        calibration_id: str,
        alpha: float = 0.1,
        ridge_penalty: float | None = None,
        held_out_canonical_structure_ids: Iterable[str] = (),
    ) -> EnvironmentConditionalProtocolTransportMap:
        if not 0 < alpha < 1:
            raise ValueError("environment transport alpha must be in (0, 1)")
        if ridge_penalty is not None and (
            not math.isfinite(ridge_penalty) or ridge_penalty <= 0
        ):
            raise ValueError("environment transport ridge penalty must be positive")
        fit_items = tuple(fit_pairs)
        radius_items = tuple(radius_calibration_pairs)
        fit_groups = {item.canonical_structure_id for item in fit_items}
        radius_groups = {item.canonical_structure_id for item in radius_items}
        fit_systems = {item.chemical_system for item in fit_items}
        radius_systems = {item.chemical_system for item in radius_items}
        held_out = {item.strip() for item in held_out_canonical_structure_ids}
        if len(fit_groups) < 3 or len(radius_groups) < 3:
            raise ValueError("environment transport needs three structures per partition")
        if len(fit_systems) < 2 or len(radius_systems) < 2:
            raise ValueError("environment transport needs two exact systems per partition")
        if fit_groups & radius_groups or fit_systems & radius_systems:
            raise ValueError("environment transport fit and radius partitions overlap")
        if not all(held_out) or (fit_groups | radius_groups) & held_out:
            raise ValueError("environment transport leaks held-out structures")
        dimensions = {len(item.source_descriptor) for item in (*fit_items, *radius_items)}
        if len(dimensions) != 1:
            raise ValueError("environment transport descriptor dimensions disagree")
        descriptor_dimension = dimensions.pop()
        supported_elements = tuple(
            sorted({element for item in fit_items for element in item.element_fractions})
        )
        raw_fit = np.asarray(
            [
                (item.source_energy_ev_per_atom, *item.source_descriptor)
                for item in fit_items
            ],
            dtype=float,
        )
        mean = raw_fit.mean(axis=0)
        scale = raw_fit.std(axis=0)
        scale[scale <= np.finfo(float).eps] = 1.0
        standardized_fit = (raw_fit - mean) / scale
        target_delta = np.asarray(
            [
                item.target_energy_ev_per_atom - item.source_energy_ev_per_atom
                for item in fit_items
            ],
            dtype=float,
        )
        if ridge_penalty is None:
            groups = np.asarray([item.chemical_system for item in fit_items])
            logo = LeaveOneGroupOut()
            ridge_cv = RidgeCV(
                alphas=np.logspace(-6, 6, 25),
                fit_intercept=True,
                cv=logo.split(standardized_fit, target_delta, groups),
                scoring="neg_mean_squared_error",
            ).fit(standardized_fit, target_delta)
            fitted_ridge_penalty = float(ridge_cv.alpha_)
            ridge_selection_rule = "leave_one_exact_system_out_log_grid_1e-6_to_1e6"
        else:
            fitted_ridge_penalty = ridge_penalty
            ridge_selection_rule = "externally_frozen"
        ridge = Ridge(alpha=fitted_ridge_penalty, fit_intercept=True).fit(
            standardized_fit, target_delta
        )
        fit_error = target_delta - ridge.predict(standardized_fit)
        fit_residual_variance = max(float(np.mean(fit_error**2)), np.finfo(float).eps)
        augmented_fit = np.column_stack(
            (np.ones(len(standardized_fit), dtype=float), standardized_fit)
        )
        penalty = np.eye(augmented_fit.shape[1], dtype=float) * fitted_ridge_penalty
        penalty[0, 0] = 0.0
        precision_inverse = np.linalg.pinv(augmented_fit.T @ augmented_fit + penalty)
        provisional = cls(
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            descriptor_dimension=descriptor_dimension,
            feature_mean=tuple(float(value) for value in mean),
            feature_scale=tuple(float(value) for value in scale),
            delta_coefficient=tuple(float(value) for value in np.asarray(ridge.coef_)),
            delta_intercept_ev_per_atom=float(ridge.intercept_),
            posterior_precision_inverse=tuple(
                tuple(float(value) for value in row) for row in precision_inverse
            ),
            supported_elements=supported_elements,
            normalized_error_quantile_ev_per_atom=0.0,
            fit_residual_variance_ev2_per_atom2=fit_residual_variance,
            ridge_penalty=fitted_ridge_penalty,
            ridge_selection_rule=ridge_selection_rule,
            radius_alpha=alpha,
            fit_structure_count=len(fit_groups),
            fit_chemical_system_count=len(fit_systems),
            radius_calibration_structure_count=len(radius_groups),
            radius_calibration_chemical_system_count=len(radius_systems),
            supported_radius_chemical_system_count=1,
            calibration_group_checksum="sha256:" + "0" * 64,
            calibration_id=calibration_id,
        )
        errors_by_system: dict[str, list[float]] = defaultdict(list)
        for item in radius_items:
            prediction = provisional.predict(
                item.source_energy_ev_per_atom,
                item.source_descriptor,
                item.element_fractions,
            )
            if prediction.status is not EnvironmentTransportStatus.CERTIFIED:
                continue
            assert prediction.target_energy_ev_per_atom is not None
            assert prediction.leverage_scale is not None
            errors_by_system[item.chemical_system].append(
                abs(item.target_energy_ev_per_atom - prediction.target_energy_ev_per_atom)
                / prediction.leverage_scale
            )
        cluster_scores = sorted(max(values) for values in errors_by_system.values())
        index = math.ceil((len(cluster_scores) + 1) * (1 - alpha)) - 1
        if not cluster_scores or index >= len(cluster_scores):
            raise ValueError(
                "too few supported radius systems for finite clustered conformal radius"
            )
        all_groups = fit_groups | radius_groups
        return provisional.model_copy(
            update={
                "normalized_error_quantile_ev_per_atom": cluster_scores[index],
                "supported_radius_chemical_system_count": len(cluster_scores),
                "calibration_group_checksum": "sha256:"
                + hashlib.sha256("\n".join(sorted(all_groups)).encode()).hexdigest(),
            }
        )
