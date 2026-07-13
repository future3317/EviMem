"""Task-specific, provenance-preserving dataset views.

These records are audit artifacts, not generic instruction samples.  Retrieval,
admission, and update labels are intentionally disjoint, and a missing gold label
is represented as ``None`` rather than fabricated as IGNORE or ADD.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evimem.contracts import AdmissionAction, ScientificClaimRecord, UpdateOperation

from .datasets import DataView


class ConversionOrigin(StrEnum):
    NATIVE = "native"
    DETERMINISTIC_DERIVED = "deterministic_derived"
    CONTROLLED_CORRUPTION = "controlled_corruption"


class AlignedEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_document_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    text: str
    source_field: str

    @model_validator(mode="after")
    def _valid_interval(self) -> AlignedEvidence:
        if self.end <= self.start:
            raise ValueError("evidence end must exceed start")
        return self


class ViewSample(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    dataset_name: str
    source_document_id: str
    split: str
    view: DataView | None
    native_task: str
    query_text: str
    source_text: str
    target: dict[str, Any] = Field(default_factory=dict)
    claim: ScientificClaimRecord | None = None
    evidence: tuple[AlignedEvidence, ...] = ()
    admission: AdmissionAction | None = None
    memory_operation: UpdateOperation | None = None
    origin: ConversionOrigin
    label_source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _enforce_view_separation_and_alignment(self) -> ViewSample:
        for span in self.evidence:
            if span.source_document_id != self.source_document_id:
                raise ValueError("evidence points to a different source_document_id")
            if span.end > len(self.source_text):
                raise ValueError("evidence locator exceeds source text")
            if self.source_text[span.start : span.end] != span.text:
                raise ValueError("transformed evidence does not round-trip to source text")
        if self.view == DataView.RETRIEVAL and (
            self.admission is not None or self.memory_operation is not None
        ):
            raise ValueError("retrieval_view cannot carry admission or update gold")
        if self.view == DataView.ADMISSION and self.memory_operation is not None:
            raise ValueError("admission_view cannot carry update gold")
        if self.view == DataView.UPDATE and self.admission is not None:
            raise ValueError("update_view cannot carry admission gold")
        if (
            self.memory_operation == UpdateOperation.SUPERSEDE
            and self.label_source != "human_reviewed_update"
        ):
            raise ValueError("SUPERSEDE requires a human-reviewed update annotation")
        return self

    def model_input(self) -> InferenceViewInput:
        """Return the policy-facing object; oracle targets/evidence cannot appear here."""

        if self.view is None:
            raise ValueError("non-EviMem native tasks do not have a policy-facing data view")
        return InferenceViewInput(
            sample_id=self.sample_id,
            dataset_name=self.dataset_name,
            source_document_id=self.source_document_id,
            split=self.split,
            view=self.view,
            query_text=self.query_text,
            source_text=self.source_text,
            metadata={"native_task": self.native_task, "origin": self.origin.value},
        )

    def oracle_payload(self) -> OracleViewTarget:
        """Return the separately persisted scorer-only payload."""

        return OracleViewTarget(
            sample_id=self.sample_id,
            target=self.target,
            claim=self.claim,
            evidence=self.evidence,
            admission=self.admission,
            memory_operation=self.memory_operation,
            label_source=self.label_source,
        )


class InferenceViewInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    dataset_name: str
    source_document_id: str
    split: str
    view: DataView
    query_text: str
    source_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class OracleViewTarget(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    target: dict[str, Any] = Field(default_factory=dict)
    claim: ScientificClaimRecord | None = None
    evidence: tuple[AlignedEvidence, ...] = ()
    admission: AdmissionAction | None = None
    memory_operation: UpdateOperation | None = None
    label_source: str


class RejectedConversion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_name: str
    source_row_id: str
    split: str
    reason: str
    source_checksum: str


def exact_evidence(
    source_text: str,
    evidence_text: str,
    *,
    source_document_id: str,
    source_field: str,
    start_hint: int | None = None,
) -> AlignedEvidence:
    """Create an evidence span only when the text is an exact source substring."""

    if not evidence_text:
        raise ValueError("empty evidence cannot be aligned")
    start = start_hint if start_hint is not None else source_text.find(evidence_text)
    if start < 0 or source_text[start : start + len(evidence_text)] != evidence_text:
        raise ValueError("evidence text is not an exact substring of source text")
    return AlignedEvidence(
        source_document_id=source_document_id,
        start=start,
        end=start + len(evidence_text),
        text=evidence_text,
        source_field=source_field,
    )
