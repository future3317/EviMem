"""V2 candidate observation contract.

A ``CandidateObservation`` is the output of the proposer stage: a claim plus
proposed evidence and proposer provenance.  It is intentionally not yet
verified; verification produces a ``VerificationCertificate``.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from .claim import ScientificClaim
from .evidence import EvidenceRef
from .provenance import ProposerProvenance


class CandidateObservation(BaseModel):
    """A proposed observation awaiting harness verification."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    candidate_id: str
    run_id: str
    doi: str
    candidate_version: int = 1
    parent_candidate_id: str | None = None
    claim: ScientificClaim
    proposed_evidence: list[EvidenceRef] = Field(default_factory=list)
    evidence_hints: tuple[str, ...] = ()
    proposer_provenance: ProposerProvenance
    publication_status: Literal["unpublished"] = "unpublished"
