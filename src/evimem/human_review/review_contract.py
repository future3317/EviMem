"""Typed, evidence-preserving human review exchange."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evimem.contracts import EvidenceRef


class HumanReviewRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    run_id: str
    candidate_id: str
    slot_name: str
    evidence_release_id: str
    evidence_bundle: tuple[EvidenceRef, ...]
    ambiguity_reason_codes: tuple[str, ...]
    policy_version: str
    policy_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _fixed_release(self) -> HumanReviewRequest:
        if not self.evidence_bundle:
            raise ValueError("human review requires an evidence bundle")
        if any(ref.release_id != self.evidence_release_id for ref in self.evidence_bundle):
            raise ValueError("review bundle cannot mix evidence releases")
        return self


class ReviewCorrection(BaseModel):
    model_config = ConfigDict(frozen=True)

    field_name: str
    previous_value: Any = None
    corrected_value: Any
    reason: str


class HumanReviewResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    reviewer_id: str
    decision: Literal["confirm", "correct", "reject", "defer"]
    corrections: tuple[ReviewCorrection, ...] = ()
    supporting_evidence: tuple[EvidenceRef, ...]
    completed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _correction_has_edits(self) -> HumanReviewResponse:
        if self.decision == "correct" and not self.corrections:
            raise ValueError("a correction response must describe its edits")
        return self
