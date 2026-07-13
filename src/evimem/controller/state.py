"""Memory-conditioned controller state."""

from __future__ import annotations

import hashlib
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evimem.contracts import (
    ActionCost,
    CandidateObservation,
    ClaimState,
    EvidenceRef,
    VerifierDelta,
)
from evimem.memory.retriever import MemoryHints

from .actions import CurationAction


class EvidenceIndexEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    evidence_ref: EvidenceRef
    kind: Literal["passage", "table", "caption", "figure", "metadata"]
    labels: tuple[str, ...] = ()


class ActionRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: CurationAction
    result_digest: str
    cost: ActionCost
    verifier_delta: VerifierDelta


class ControllerState(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.controller_state.v1"

    candidate: CandidateObservation
    claim_state: ClaimState
    evidence_index: tuple[EvidenceIndexEntry, ...] = ()
    memory_hints: MemoryHints = Field(default_factory=MemoryHints)
    domain_requirements: dict[str, Any] = Field(default_factory=dict)
    gathered_evidence: tuple[EvidenceRef, ...] = ()
    action_history: tuple[ActionRecord, ...] = ()

    @model_validator(mode="after")
    def _validate_fixed_episode_identity(self) -> ControllerState:
        if self.candidate.candidate_id != self.claim_state.candidate_id:
            raise ValueError("candidate and claim state identities differ")
        refs = [entry.evidence_ref for entry in self.evidence_index]
        refs.extend(self.gathered_evidence)
        if any(ref.release_id != self.claim_state.evidence_release_id for ref in refs):
            raise ValueError("one episode cannot mix evidence releases")
        return self

    def state_hash(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
