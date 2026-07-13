"""Verifier-owned state for one candidate observation."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


class CurationBudget(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    human_queries: int = Field(default=0, ge=0)
    wall_clock_seconds: float = Field(default=0.0, ge=0.0)

    def consume(
        self,
        *,
        token: int = 0,
        tool_calls: int = 0,
        human_queries: int = 0,
        wall_clock_seconds: float = 0.0,
    ) -> CurationBudget:
        costs = (token, tool_calls, human_queries, wall_clock_seconds)
        remaining = (self.token, self.tool_calls, self.human_queries, self.wall_clock_seconds)
        if any(cost < 0 for cost in costs):
            raise ValueError("budget costs cannot be negative")
        if any(cost > available for cost, available in zip(costs, remaining)):
            raise ValueError("action cost exceeds remaining budget")
        return CurationBudget(
            token=self.token - token,
            tool_calls=self.tool_calls - tool_calls,
            human_queries=self.human_queries - human_queries,
            wall_clock_seconds=self.wall_clock_seconds - wall_clock_seconds,
        )


class ClaimState(BaseModel):
    """Current deterministic verification state for a candidate."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    candidate_id: str
    evidence_release_id: str
    domain_pack_id: str
    domain_pack_version: str
    domain_pack_hash: str
    slots: dict[str, VerificationSlot]
    unresolved_slots: tuple[str, ...] = ()
    conflict_status: Literal[
        "unknown", "clear", "distinct_context", "resolvable", "unresolved"
    ] = "unknown"
    remaining_budget: CurationBudget
    terminal_status: Literal[
        "active", "request_publication", "deferred", "rejected", "stopped"
    ] = "active"

    @field_validator(
        "candidate_id",
        "evidence_release_id",
        "domain_pack_id",
        "domain_pack_version",
        "domain_pack_hash",
    )
    @classmethod
    def _require_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("claim-state identity fields must be non-empty")
        return value

    @model_validator(mode="after")
    def _derive_unresolved_slots(self) -> ClaimState:
        unresolved = tuple(
            sorted(
                name
                for name, slot in self.slots.items()
                if slot.status != SlotStatus.VERIFIED
            )
        )
        if self.unresolved_slots and tuple(sorted(self.unresolved_slots)) != unresolved:
            raise ValueError("unresolved_slots must match slot verification statuses")
        object.__setattr__(self, "unresolved_slots", unresolved)
        return self
