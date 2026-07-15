"""Protocol-stratified calibration and conservative screening decisions."""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .cards import MaterialQuery
from .residual import ResidualCorrection


class ScreeningDecision(StrEnum):
    STABLE = "stable"
    NOT_STABLE = "not_stable"
    ABSTAIN = "abstain"


class ConformalCalibration(BaseModel):
    """Finite-sample split-conformal radius for one target protocol stratum.

    The coverage interpretation applies only when the documented calibration
    and evaluation residuals are approximately exchangeable.  The model stores
    the assumption explicitly instead of claiming it under protocol shift.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol_fingerprint: str
    alpha: float = Field(gt=0, lt=1)
    sample_count: int = Field(ge=1)
    radius_ev_per_atom: float = Field(ge=0)
    exchangeability_assumed: bool
    calibration_id: str

    @field_validator("protocol_fingerprint", "calibration_id")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("calibration requires a non-empty identity")
        return value.strip()


class RiskDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    decision: ScreeningDecision
    upper_hull_distance_ev_per_atom: float | None = None
    reason: str
    calibration_id: str | None = None


class ProtocolRiskController:
    """Only certifies a stable screen when its calibrated upper bound is stable."""

    def __init__(self, minimum_calibration_size: int = 30) -> None:
        if minimum_calibration_size < 1:
            raise ValueError("minimum calibration size must be positive")
        self.minimum_calibration_size = minimum_calibration_size
        self._calibrations: dict[str, ConformalCalibration] = {}

    def fit(
        self,
        query: MaterialQuery,
        absolute_errors_ev_per_atom: Iterable[float],
        *,
        alpha: float,
        calibration_id: str,
        exchangeability_assumed: bool,
    ) -> ConformalCalibration:
        errors = sorted(float(error) for error in absolute_errors_ev_per_atom)
        if len(errors) < self.minimum_calibration_size:
            raise ValueError("insufficient protocol-stratified calibration residuals")
        if any(error < 0 or not math.isfinite(error) for error in errors):
            raise ValueError("calibration errors must be finite non-negative values")
        index = min(len(errors) - 1, math.ceil((len(errors) + 1) * (1 - alpha)) - 1)
        calibration = ConformalCalibration(
            protocol_fingerprint=query.protocol.scientific_fingerprint,
            alpha=alpha,
            sample_count=len(errors),
            radius_ev_per_atom=errors[index],
            exchangeability_assumed=exchangeability_assumed,
            calibration_id=calibration_id,
        )
        self._calibrations[calibration.protocol_fingerprint] = calibration
        return calibration

    def screen(self, query: MaterialQuery, correction: ResidualCorrection) -> RiskDecision:
        if correction.status != "corrected" or correction.corrected_hull_distance_ev_per_atom is None:
            return RiskDecision(decision=ScreeningDecision.ABSTAIN, reason=correction.status)
        calibration = self._calibrations.get(query.protocol.scientific_fingerprint)
        if calibration is None:
            return RiskDecision(
                decision=ScreeningDecision.ABSTAIN,
                reason="no_protocol_stratified_calibration",
            )
        radius = calibration.radius_ev_per_atom + (correction.uncertainty_radius_ev_per_atom or 0.0)
        upper = correction.corrected_hull_distance_ev_per_atom + radius
        if upper <= query.stability_threshold_ev_per_atom:
            return RiskDecision(
                decision=ScreeningDecision.STABLE,
                upper_hull_distance_ev_per_atom=upper,
                reason="calibrated_upper_bound_is_stable",
                calibration_id=calibration.calibration_id,
            )
        return RiskDecision(
            decision=ScreeningDecision.NOT_STABLE,
            upper_hull_distance_ev_per_atom=upper,
            reason="calibrated_upper_bound_exceeds_stability_threshold",
            calibration_id=calibration.calibration_id,
        )
