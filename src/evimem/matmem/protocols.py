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
