"""Canonical Pydantic contracts for EviMem-RL.

All models are frozen, serializable and versioned. Publication and governed
memory code must consume these contracts directly rather than inferred legacy
anchors or compatibility dictionaries.
"""

from __future__ import annotations

from .candidate import CandidateObservation
from .certificate import CheckResult, VerificationCertificate
from .claim import ScientificClaim
from .claim_state import ClaimState, CurationBudget, SlotStatus, VerificationSlot
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
    ClaimSignature,
    MemoryAuthority,
    MemoryDecision,
    MemoryType,
    WarrantedMemoryItem,
)
from .provenance import ProposerProvenance, VerifierProvenance
from .published import PublishedObservation
from .trajectory import ActionCost, CurationStep, CurationTrajectory, VerifierDelta

__all__ = [
    "CandidateObservation",
    "CaptionLocator",
    "ClaimSignature",
    "ClaimState",
    "CheckResult",
    "CurationBudget",
    "CurationStep",
    "CurationTrajectory",
    "EvidenceRef",
    "evidence_ref_from_block",
    "MemoryAuthority",
    "MemoryDecision",
    "MemoryType",
    "ProposerProvenance",
    "PublishedObservation",
    "ScientificClaim",
    "SlotStatus",
    "TableCellLocator",
    "TextSpanLocator",
    "VerificationCertificate",
    "VerificationSlot",
    "VerifierProvenance",
    "VerifierDelta",
    "WarrantedMemoryItem",
    "ActionCost",
    "make_candidate_fingerprint",
    "make_candidate_id",
    "make_certificate_id",
    "make_observation_id",
    "make_observation_key",
    "make_publication_commit_id",
]
