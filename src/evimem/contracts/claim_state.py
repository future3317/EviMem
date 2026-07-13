"""Verifier-owned slot values used in deterministic certificates."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, model_validator

from .evidence import EvidenceRef


class SlotStatus(StrEnum):
    MISSING = "missing"
    CANDIDATE = "candidate"
    BOUND = "bound"
    VERIFIED = "verified"
    AMBIGUOUS = "ambiguous"
    CONFLICTING = "conflicting"
    INVALID = "invalid"


class VerificationSlot(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: SlotStatus = SlotStatus.MISSING
    evidence_refs: tuple[EvidenceRef, ...] = ()
    reason_codes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _verified_slots_require_evidence(self) -> VerificationSlot:
        if self.status in {SlotStatus.BOUND, SlotStatus.VERIFIED} and not self.evidence_refs:
            raise ValueError(f"{self.status.value} slots require immutable evidence refs")
        return self
