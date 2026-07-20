"""Scientific-protocol identity and explicit residual transport.

Source names and release versions are provenance.  They are intentionally not
used as a substitute for a compatible DFT calculation protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from enum import StrEnum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CompatibilityKind(StrEnum):
    DIRECT = "direct"
    TRANSPORTED = "transported"
    REJECT = "reject"


class ProtocolCertificate(BaseModel):
    """Fields that determine whether formation-energy residuals can transfer."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    functional: str
    hubbard_u_ev: dict[str, float] = Field(default_factory=dict)
    pseudopotential_set: str
    correction_scheme: str
    relaxation_protocol: str
    calculation_code: str

    @field_validator(
        "functional",
        "pseudopotential_set",
        "correction_scheme",
        "relaxation_protocol",
        "calculation_code",
    )
    @classmethod
    def _require_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("protocol certificate fields must be non-empty")
        return value

    @field_validator("hubbard_u_ev")
    @classmethod
    def _validate_u(cls, values: dict[str, float]) -> dict[str, float]:
        if any(not element.strip() or not math.isfinite(value) for element, value in values.items()):
            raise ValueError("Hubbard-U entries require element keys and finite values")
        return dict(sorted(values.items()))

    @property
    def scientific_fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class ProtocolTransportMap(BaseModel):
    """A directed transport calibrated only on same-structure matched records.

    ``slope * residual + intercept`` maps a residual from ``source_protocol``
    to the target protocol.  It is not inferred from shared metadata.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_protocol: ProtocolCertificate
    target_protocol: ProtocolCertificate
    slope: float
    intercept_ev_per_atom: float
    error_radius_ev_per_atom: float = Field(ge=0)
    matched_structure_count: int = Field(ge=3)
    fit_structure_count: int | None = Field(default=None, ge=3)
    radius_calibration_structure_count: int | None = Field(default=None, ge=3)
    radius_alpha: float | None = Field(default=None, gt=0, lt=1)
    calibration_group_checksum: str
    calibration_id: str

    @field_validator("slope", "intercept_ev_per_atom", "error_radius_ev_per_atom")
    @classmethod
    def _require_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("transport parameters must be finite")
        return value

    @field_validator("calibration_id")
    @classmethod
    def _require_calibration_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transport map requires immutable calibration identity")
        return value.strip()

    def transport(self, residual_ev_per_atom: float) -> float:
        return self.slope * residual_ev_per_atom + self.intercept_ev_per_atom

    @classmethod
    def fit_same_structure(
        cls,
        source_protocol: ProtocolCertificate,
        target_protocol: ProtocolCertificate,
        pairs: Iterable[MatchedResidualPair],
        *,
        calibration_id: str,
        alpha: float = 0.1,
        held_out_canonical_structure_ids: Iterable[str] = (),
    ) -> ProtocolTransportMap:
        """Fit a directed affine map using only matched structure residuals.

        Callers must construct pairs from a calibration partition disjoint from
        the online evaluation stream. This method rejects an underidentified
        fit instead of substituting an identity map.
        """

        if not 0 < alpha < 1:
            raise ValueError("transport calibration alpha must be in (0, 1)")
        items = list(pairs)
        unique_structures = {item.canonical_structure_id for item in items}
        if len(unique_structures) < 3:
            raise ValueError("transport requires at least three unique matched structures")
        held_out = {item.strip() for item in held_out_canonical_structure_ids}
        if not all(held_out):
            raise ValueError("held-out canonical structure IDs must be non-empty")
        overlap = unique_structures & held_out
        if overlap:
            raise ValueError("transport calibration leaks held-out canonical structures")
        source = [item.source_residual_ev_per_atom for item in items]
        target = [item.target_residual_ev_per_atom for item in items]
        mean_source = sum(source) / len(source)
        mean_target = sum(target) / len(target)
        variance = sum((value - mean_source) ** 2 for value in source)
        if variance == 0:
            raise ValueError("transport source residuals must have non-zero variance")
        slope = sum(
            (left - mean_source) * (right - mean_target)
            for left, right in zip(source, target, strict=True)
        ) / variance
        intercept = mean_target - slope * mean_source
        errors = sorted(
            abs(right - (slope * left + intercept))
            for left, right in zip(source, target, strict=True)
        )
        index = min(len(errors) - 1, math.ceil((len(errors) + 1) * (1 - alpha)) - 1)
        return cls(
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            slope=slope,
            intercept_ev_per_atom=intercept,
            error_radius_ev_per_atom=errors[index],
            matched_structure_count=len(unique_structures),
            calibration_group_checksum="sha256:"
            + hashlib.sha256("\n".join(sorted(unique_structures)).encode("utf-8")).hexdigest(),
            calibration_id=calibration_id,
        )

    @classmethod
    def fit_same_structure_split(
        cls,
        source_protocol: ProtocolCertificate,
        target_protocol: ProtocolCertificate,
        fit_pairs: Iterable[MatchedResidualPair],
        radius_calibration_pairs: Iterable[MatchedResidualPair],
        *,
        calibration_id: str,
        alpha: float = 0.1,
        held_out_canonical_structure_ids: Iterable[str] = (),
    ) -> ProtocolTransportMap:
        """Fit an affine map and calibrate its radius on disjoint structures."""

        if not 0 < alpha < 1:
            raise ValueError("transport calibration alpha must be in (0, 1)")
        fit_items = tuple(fit_pairs)
        radius_items = tuple(radius_calibration_pairs)
        fit_groups = {item.canonical_structure_id for item in fit_items}
        radius_groups = {item.canonical_structure_id for item in radius_items}
        held_out = {item.strip() for item in held_out_canonical_structure_ids}
        if len(fit_groups) < 3 or len(radius_groups) < 3:
            raise ValueError("split transport requires three unique structures in each partition")
        if not all(held_out):
            raise ValueError("held-out canonical structure IDs must be non-empty")
        if fit_groups & radius_groups:
            raise ValueError("transport fit and radius-calibration structures overlap")
        if (fit_groups | radius_groups) & held_out:
            raise ValueError("transport calibration leaks held-out canonical structures")

        source = [item.source_residual_ev_per_atom for item in fit_items]
        target = [item.target_residual_ev_per_atom for item in fit_items]
        mean_source = sum(source) / len(source)
        mean_target = sum(target) / len(target)
        variance = sum((value - mean_source) ** 2 for value in source)
        if variance == 0:
            raise ValueError("transport source residuals must have non-zero variance")
        slope = sum(
            (left - mean_source) * (right - mean_target)
            for left, right in zip(source, target, strict=True)
        ) / variance
        intercept = mean_target - slope * mean_source
        errors = sorted(
            abs(
                item.target_residual_ev_per_atom
                - (slope * item.source_residual_ev_per_atom + intercept)
            )
            for item in radius_items
        )
        index = math.ceil((len(errors) + 1) * (1 - alpha)) - 1
        if index >= len(errors):
            raise ValueError("radius-calibration partition is too small for finite conformal radius")
        all_groups = fit_groups | radius_groups
        return cls(
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            slope=slope,
            intercept_ev_per_atom=intercept,
            error_radius_ev_per_atom=errors[index],
            matched_structure_count=len(all_groups),
            fit_structure_count=len(fit_groups),
            radius_calibration_structure_count=len(radius_groups),
            radius_alpha=alpha,
            calibration_group_checksum="sha256:"
            + hashlib.sha256("\n".join(sorted(all_groups)).encode()).hexdigest(),
            calibration_id=calibration_id,
        )

    @model_validator(mode="after")
    def _require_actual_protocol_change(self) -> ProtocolTransportMap:
        if (
            self.source_protocol.scientific_fingerprint
            == self.target_protocol.scientific_fingerprint
        ):
            raise ValueError("identical protocols must use direct residual reuse, not transport")
        return self


class MatchedResidualPair(BaseModel):
    """Residuals for exactly one structure calculated under two protocols."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_calculation_id: str
    canonical_structure_id: str
    source_residual_ev_per_atom: float
    target_residual_ev_per_atom: float

    @field_validator("exact_calculation_id", "canonical_structure_id")
    @classmethod
    def _require_structure(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("matched residual pair requires identity")
        return value.strip()

    @field_validator("source_residual_ev_per_atom", "target_residual_ev_per_atom")
    @classmethod
    def _finite_residual(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("matched residuals must be finite")
        return value


class MatchedEnergyPair(BaseModel):
    """Same-material energies with an observable elemental composition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_calculation_id: str
    canonical_structure_id: str
    chemical_system: str
    element_fractions: dict[str, float]
    source_energy_ev_per_atom: float
    target_energy_ev_per_atom: float

    @field_validator("exact_calculation_id", "canonical_structure_id", "chemical_system")
    @classmethod
    def _identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("matched energy pair requires non-empty identity")
        return value.strip()

    @field_validator("element_fractions")
    @classmethod
    def _fractions(cls, values: dict[str, float]) -> dict[str, float]:
        normalized = {key.strip(): float(value) for key, value in values.items()}
        if (
            not normalized
            or any(not key or not math.isfinite(value) or value < 0 for key, value in normalized.items())
            or not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-8)
        ):
            raise ValueError("element fractions must be finite, non-negative, and sum to one")
        return dict(sorted(normalized.items()))

    @field_validator("source_energy_ev_per_atom", "target_energy_ev_per_atom")
    @classmethod
    def _energy(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("matched energies must be finite")
        return value


class CompositionAwareProtocolTransportMap(BaseModel):
    """Directed energy transport with elemental reference-energy offsets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_protocol: ProtocolCertificate
    target_protocol: ProtocolCertificate
    source_slope: float
    element_offset_ev_per_atom: dict[str, float]
    error_radius_ev_per_atom: float = Field(ge=0)
    fit_structure_count: int = Field(ge=3)
    radius_calibration_structure_count: int = Field(ge=3)
    fit_chemical_system_count: int = Field(ge=2)
    radius_calibration_chemical_system_count: int = Field(ge=2)
    radius_alpha: float = Field(gt=0, lt=1)
    ridge_penalty: float = Field(gt=0)
    calibration_group_checksum: str
    calibration_id: str

    @field_validator("source_slope", "error_radius_ev_per_atom", "ridge_penalty")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("composition-aware transport parameters must be finite")
        return value

    @field_validator("element_offset_ev_per_atom")
    @classmethod
    def _offsets(cls, values: dict[str, float]) -> dict[str, float]:
        normalized = {key.strip(): float(value) for key, value in values.items()}
        if not normalized or any(
            not key or not math.isfinite(value) for key, value in normalized.items()
        ):
            raise ValueError("composition-aware transport requires finite element offsets")
        return dict(sorted(normalized.items()))

    @model_validator(mode="after")
    def _directed(self) -> CompositionAwareProtocolTransportMap:
        if (
            self.source_protocol.scientific_fingerprint
            == self.target_protocol.scientific_fingerprint
        ):
            raise ValueError("identical protocols must not use composition-aware transport")
        return self

    def transport(
        self, source_energy_ev_per_atom: float, element_fractions: dict[str, float]
    ) -> float | None:
        if not set(element_fractions) <= set(self.element_offset_ev_per_atom):
            return None
        return self.source_slope * source_energy_ev_per_atom + sum(
            float(fraction) * self.element_offset_ev_per_atom[element]
            for element, fraction in element_fractions.items()
        )

    @classmethod
    def fit_same_structure_system_split(
        cls,
        source_protocol: ProtocolCertificate,
        target_protocol: ProtocolCertificate,
        fit_pairs: Iterable[MatchedEnergyPair],
        radius_calibration_pairs: Iterable[MatchedEnergyPair],
        *,
        calibration_id: str,
        alpha: float = 0.2,
        ridge_penalty: float = 1e-3,
        held_out_canonical_structure_ids: Iterable[str] = (),
    ) -> CompositionAwareProtocolTransportMap:
        """Fit on systems and conformalize maximum error on disjoint systems."""

        if not 0 < alpha < 1 or not math.isfinite(ridge_penalty) or ridge_penalty <= 0:
            raise ValueError("composition-aware transport requires valid alpha and ridge")
        fit_items = tuple(fit_pairs)
        radius_items = tuple(radius_calibration_pairs)
        fit_groups = {item.canonical_structure_id for item in fit_items}
        radius_groups = {item.canonical_structure_id for item in radius_items}
        fit_systems = {item.chemical_system for item in fit_items}
        radius_systems = {item.chemical_system for item in radius_items}
        held_out = {item.strip() for item in held_out_canonical_structure_ids}
        if len(fit_groups) < 3 or len(radius_groups) < 3:
            raise ValueError("composition-aware transport needs three structures per partition")
        if len(fit_systems) < 2 or len(radius_systems) < 2:
            raise ValueError("composition-aware transport needs two exact systems per partition")
        if fit_groups & radius_groups or fit_systems & radius_systems:
            raise ValueError("transport fit and radius partitions overlap")
        if not all(held_out) or (fit_groups | radius_groups) & held_out:
            raise ValueError("composition-aware transport leaks held-out structures")
        vocabulary = sorted(
            {element for item in fit_items for element in item.element_fractions}
        )
        if any(not set(item.element_fractions) <= set(vocabulary) for item in radius_items):
            raise ValueError("radius calibration contains an element absent from transport fit")
        design = np.asarray(
            [
                [item.source_energy_ev_per_atom]
                + [item.element_fractions.get(element, 0.0) for element in vocabulary]
                for item in fit_items
            ],
            dtype=float,
        )
        target = np.asarray(
            [item.target_energy_ev_per_atom for item in fit_items], dtype=float
        )
        penalty = np.eye(design.shape[1]) * ridge_penalty
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        slope = float(coefficients[0])
        offsets = {
            element: float(value)
            for element, value in zip(vocabulary, coefficients[1:], strict=True)
        }
        errors_by_system: dict[str, list[float]] = {}
        provisional = cls(
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            source_slope=slope,
            element_offset_ev_per_atom=offsets,
            error_radius_ev_per_atom=0.0,
            fit_structure_count=len(fit_groups),
            radius_calibration_structure_count=len(radius_groups),
            fit_chemical_system_count=len(fit_systems),
            radius_calibration_chemical_system_count=len(radius_systems),
            radius_alpha=alpha,
            ridge_penalty=ridge_penalty,
            calibration_group_checksum="sha256:" + "0" * 64,
            calibration_id=calibration_id,
        )
        for item in radius_items:
            prediction = provisional.transport(
                item.source_energy_ev_per_atom, item.element_fractions
            )
            if prediction is None:
                raise ValueError("radius calibration is outside the fitted element vocabulary")
            errors_by_system.setdefault(item.chemical_system, []).append(
                abs(item.target_energy_ev_per_atom - prediction)
            )
        cluster_scores = sorted(max(values) for values in errors_by_system.values())
        index = math.ceil((len(cluster_scores) + 1) * (1 - alpha)) - 1
        if index >= len(cluster_scores):
            raise ValueError("too few radius systems for a finite clustered conformal radius")
        all_groups = fit_groups | radius_groups
        return cls(
            source_protocol=source_protocol,
            target_protocol=target_protocol,
            source_slope=slope,
            element_offset_ev_per_atom=offsets,
            error_radius_ev_per_atom=cluster_scores[index],
            fit_structure_count=len(fit_groups),
            radius_calibration_structure_count=len(radius_groups),
            fit_chemical_system_count=len(fit_systems),
            radius_calibration_chemical_system_count=len(radius_systems),
            radius_alpha=alpha,
            ridge_penalty=ridge_penalty,
            calibration_group_checksum="sha256:"
            + hashlib.sha256("\n".join(sorted(all_groups)).encode()).hexdigest(),
            calibration_id=calibration_id,
        )


class ProtocolCompatibility(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: CompatibilityKind
    uncertainty_radius_ev_per_atom: float = Field(ge=0)
    transport: ProtocolTransportMap | None = None
    reason: str

    def transfer_residual(self, residual_ev_per_atom: float) -> float | None:
        if self.kind == CompatibilityKind.REJECT:
            return None
        if self.transport is None:
            return residual_ev_per_atom
        return self.transport.transport(residual_ev_per_atom)


class ProtocolCompatibilityResolver:
    """Fail-closed protocol resolver with no metadata-only compatibility path."""

    def __init__(self, transports: Iterable[ProtocolTransportMap] = ()) -> None:
        self._transports = {
            (item.source_protocol.scientific_fingerprint, item.target_protocol.scientific_fingerprint): item
            for item in transports
        }

    def resolve(
        self,
        source: ProtocolCertificate,
        target: ProtocolCertificate,
    ) -> ProtocolCompatibility:
        source_id = source.scientific_fingerprint
        target_id = target.scientific_fingerprint
        if source_id == target_id:
            return ProtocolCompatibility(
                kind=CompatibilityKind.DIRECT,
                uncertainty_radius_ev_per_atom=0.0,
                reason="identical_scientific_protocol",
            )
        transport = self._transports.get((source_id, target_id))
        if transport is not None:
            return ProtocolCompatibility(
                kind=CompatibilityKind.TRANSPORTED,
                uncertainty_radius_ev_per_atom=transport.error_radius_ev_per_atom,
                transport=transport,
                reason="same_structure_calibrated_transport",
            )
        return ProtocolCompatibility(
            kind=CompatibilityKind.REJECT,
            uncertainty_radius_ev_per_atom=0.0,
            reason="no_calibrated_protocol_transport",
        )
