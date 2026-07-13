"""Deterministic verification certificate contract.

A ``VerificationCertificate`` is the immutable output of the verifier stage.
It records the normalized claim, resolved evidence, check results, and the
publication decision.  Certificates are referenced by published observations
but are themselves versioned audit artifacts.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .claim import ScientificClaim
from .claim_state import SlotStatus
from .evidence import EvidenceRef


class CheckResult(BaseModel):
    """Result of a single harness check."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    check_id: str
    check_type: str
    passed: bool
    severity: Literal["error", "warning", "info"]
    reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationCertificate(BaseModel):
    """Immutable certification of a candidate observation."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    certificate_id: str
    run_id: str
    candidate_id: str
    normalized_claim: ScientificClaim
    certified_claim_hash: str
    resolved_evidence: list[EvidenceRef]
    checks: list[CheckResult]
    slot_verification: dict[str, SlotStatus]
    binding_method: str
    support_tier: Literal[
        "verified_strong",
        "verified_weak",
        "structured_prompt_support",
        "ambiguous",
        "unbound",
    ]
    constraint_result: Literal["pass", "fail", "not_run"]
    conflict_result: Literal[
        "pass",
        "distinct_context",
        "exact_duplicate",
        "resolvable_conflict",
        "unresolved_conflict",
        "not_run",
    ]
    final_decision: Literal["publish", "review", "reject", "defer"]
    exclusion_reasons: list[str] = Field(default_factory=list)
    evidence_release_id: str
    domain_pack_id: str
    domain_pack_version: str
    domain_pack_hash: str
    verifier_version: str

    @field_validator("certified_claim_hash")
    @classmethod
    def _validate_claim_hash(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("certified_claim_hash must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"

    @model_validator(mode="after")
    def _validate_evidence_identity(self) -> VerificationCertificate:
        if any(ref.release_id != self.evidence_release_id for ref in self.resolved_evidence):
            raise ValueError("certificate evidence must belong to evidence_release_id")
        if self.final_decision == "publish" and not self.resolved_evidence:
            raise ValueError("publish certificate requires resolved immutable evidence")
        if self.final_decision == "publish" and self.exclusion_reasons:
            raise ValueError("publish certificate cannot carry exclusion reasons")
        if self.certified_claim_hash != self.normalized_claim.memory_fingerprint():
            raise ValueError("certified_claim_hash does not match normalized_claim")
        return self
