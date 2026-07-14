"""Canonical Pydantic contracts for EviMem.

All models are frozen, serializable and versioned. Publication and governed
memory code must consume these contracts directly rather than inferred legacy
anchors or compatibility dictionaries.
"""

from __future__ import annotations

from .candidate import CandidateObservation
from .certificate import CheckResult, VerificationCertificate
from .claim import ScientificClaim
from .claim_state import SlotStatus, VerificationSlot
from .evidence import EvidenceRef, evidence_ref_from_block
from .identity import (
    make_candidate_fingerprint,
    make_candidate_id,
    make_certificate_id,
    make_observation_id,
    make_observation_key,
    make_publication_commit_id,
)
from .locators import CaptionLocator, TableCellLocator, TextSpanLocator
from .memory import (
    AdmissionAction,
    AuthorityRelation,
    ClaimSignature,
    CorrectionBasis,
    CorrectionEvidenceScope,
    EvidenceSufficiency,
    MemoryAuthority,
    MemoryDecision,
    MemoryManagerAction,
    MemoryOrigin,
    MemoryStatus,
    MemoryType,
    ScientificClaimRecord,
    ScientificMemoryRecord,
    ScopeRelation,
    SemanticRelation,
    UpdateOperation,
    VerifiedCorrectionEvidence,
)
from .provenance import ProposerProvenance, VerifierProvenance
from .published import PublishedObservation

__all__ = [
    "AdmissionAction",
    "AuthorityRelation",
    "CandidateObservation",
    "CaptionLocator",
    "ClaimSignature",
    "CorrectionBasis",
    "CorrectionEvidenceScope",
    "CheckResult",
    "EvidenceRef",
    "EvidenceSufficiency",
    "evidence_ref_from_block",
    "MemoryAuthority",
    "MemoryDecision",
    "MemoryManagerAction",
    "MemoryOrigin",
    "MemoryStatus",
    "MemoryType",
    "ProposerProvenance",
    "PublishedObservation",
    "ScientificClaim",
    "ScientificClaimRecord",
    "ScientificMemoryRecord",
    "ScopeRelation",
    "SemanticRelation",
    "SlotStatus",
    "TableCellLocator",
    "TextSpanLocator",
    "VerificationCertificate",
    "VerificationSlot",
    "VerifierProvenance",
    "UpdateOperation",
    "VerifiedCorrectionEvidence",
    "make_candidate_fingerprint",
    "make_candidate_id",
    "make_certificate_id",
    "make_observation_id",
    "make_observation_key",
    "make_publication_commit_id",
]
