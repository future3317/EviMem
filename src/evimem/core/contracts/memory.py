"""Canonical evidence-warranted memory contract."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .evidence import EvidenceRef


class MemoryType(StrEnum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    CONFLICT = "conflict"
    CORRECTION = "correction"
    POLICY = "policy"


class ClaimSignature(BaseModel):
    model_config = ConfigDict(frozen=True)

    domain: str
    property_key: str
    material_family: str | None = None
    material_identity: str | None = None
    composition: str | None = None
    condition_signature: str | None = None


class MemoryDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["published", "rejected", "deferred", "superseded"]
    reason: str


class MemoryAuthority(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["deterministic_harness", "human_curator"]
    level: int = Field(ge=1, le=4)


class WarrantedMemoryItem(BaseModel):
    """Reusable history whose evidence and decision authority are explicit."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    memory_id: str
    memory_type: MemoryType
    claim_signature: ClaimSignature
    normalized_content: dict[str, Any]
    evidence_refs: tuple[EvidenceRef, ...]
    certificate_id: str
    decision: MemoryDecision
    authority: MemoryAuthority
    policy_version: str
    policy_hash: str
    evidence_release_id: str
    valid_from: datetime = Field(default_factory=lambda: datetime.now(UTC))
    valid_until: datetime | None = None
    status: Literal["active", "superseded"] = "active"
    supersedes: tuple[str, ...] = ()
    contradicted_by: tuple[str, ...] = ()
    supersession_reason: str | None = None

    @field_validator(
        "memory_id", "certificate_id", "policy_version", "policy_hash", "evidence_release_id"
    )
    @classmethod
    def _require_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("memory identity fields must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_warrant(self) -> WarrantedMemoryItem:
        if not self.evidence_refs:
            raise ValueError("warranted memory requires immutable evidence")
        if any(ref.release_id != self.evidence_release_id for ref in self.evidence_refs):
            raise ValueError("all evidence refs must belong to the memory evidence release")
        if self.valid_until is not None and self.valid_until <= self.valid_from:
            raise ValueError("valid_until must be later than valid_from")
        if self.status == "superseded" and not self.supersession_reason:
            raise ValueError("superseded memory requires a reason")
        return self
