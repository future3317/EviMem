"""Leakage-safe continual episodes and separately held oracle annotations."""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evimem.contracts import (
    AdmissionAction,
    EvidenceRef,
    ScientificClaimRecord,
    ScientificMemoryRecord,
    UpdateOperation,
)


class ScientificDocument(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    text: str
    timestamp: datetime | None = None
    dataset_name: str
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str
    candidate_claim: ScientificClaimRecord | None = None


class BenchmarkEpisode(BaseModel):
    """Everything visible during inference; no gold field is permitted."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: ClassVar[str] = "scimem_curate.episode.v1"

    episode_id: str
    stream_position: int = Field(ge=0)
    history: tuple[ScientificMemoryRecord, ...] = ()
    current_document: ScientificDocument
    query: MemoryQuery

    @model_validator(mode="after")
    def _reject_future_memory(self) -> BenchmarkEpisode:
        if self.current_document.timestamp is None:
            if self.history:
                raise ValueError("undated documents cannot receive timestamped memory history")
            return self
        future = [
            record.memory_id
            for record in self.history
            if record.observed_at > self.current_document.timestamp
        ]
        if future:
            raise ValueError(f"history contains future memories: {sorted(future)}")
        return self


class OracleAnnotation(BaseModel):
    """Gold data stored outside policy-facing episodes and used only for scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    relevant_memory_ids: tuple[str, ...] = ()
    evidence_refs: tuple[EvidenceRef, ...] = ()
    final_record: ScientificClaimRecord | None = None
    admission: AdmissionAction | None = None
    memory_operation: UpdateOperation | None = None
    target_memory_ids: tuple[str, ...] = ()


class EpisodePrediction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    episode_id: str
    retrieved_memory_ids: tuple[str, ...] = ()
    predicted_record: ScientificClaimRecord | None = None
    evidence_refs: tuple[EvidenceRef, ...] = ()
    admission: AdmissionAction
    memory_operation: UpdateOperation
    target_memory_ids: tuple[str, ...] = ()
    publication_requested: bool = False
    publication_authorized: bool = False
    certificate_id: str | None = None
    memory_size: int = Field(default=0, ge=0)
    retrieval_tokens: int = Field(default=0, ge=0)
