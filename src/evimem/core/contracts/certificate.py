"""Deterministic verification certificate contract.

A ``VerificationCertificate`` is the immutable output of the verifier stage.
It records the normalized claim, resolved evidence, check results, and the
publication decision.  Certificates are referenced by published observations
but are themselves versioned audit artifacts.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

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
