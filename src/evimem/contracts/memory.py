"""Evidence-certified memory contracts for continual scientific curation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .certificate import VerificationCertificate
from .claim import ScientificClaim
from .evidence import EvidenceRef


class MemoryType(StrEnum):
    VERIFIED = "verified"
    REJECTED = "rejected"
    CONFLICT = "conflict"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class AdmissionAction(StrEnum):
    WRITE_VERIFIED = "WRITE_VERIFIED"
    WRITE_REJECTED = "WRITE_REJECTED"
    WRITE_CONFLICT = "WRITE_CONFLICT"
    EPHEMERAL_ONLY = "EPHEMERAL_ONLY"
    IGNORE = "IGNORE"


class UpdateOperation(StrEnum):
    ADD = "ADD"
    MERGE = "MERGE"
    LINK = "LINK"
    CONFLICT = "CONFLICT"
    SUPERSEDE = "SUPERSEDE"
    IGNORE = "IGNORE"


class SemanticRelation(StrEnum):
    EQUIVALENT = "EQUIVALENT"
    COMPATIBLE_DISTINCT = "COMPATIBLE_DISTINCT"
    CONTRADICTORY = "CONTRADICTORY"
    UNRELATED = "UNRELATED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"


class ScopeRelation(StrEnum):
    SAME_SCOPE = "SAME_SCOPE"
    NARROWER_SCOPE = "NARROWER_SCOPE"
    BROADER_SCOPE = "BROADER_SCOPE"
    DIFFERENT_SCOPE = "DIFFERENT_SCOPE"
    UNKNOWN_SCOPE = "UNKNOWN_SCOPE"


class AuthorityRelation(StrEnum):
    NEWER_MORE_AUTHORITATIVE = "NEWER_MORE_AUTHORITATIVE"
    OLDER_MORE_AUTHORITATIVE = "OLDER_MORE_AUTHORITATIVE"
    EQUAL_AUTHORITY = "EQUAL_AUTHORITY"
    UNRESOLVED = "UNRESOLVED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class EvidenceSufficiency(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PARTIAL = "PARTIAL"
    INSUFFICIENT = "INSUFFICIENT"


class CorrectionBasis(StrEnum):
    NONE = "NONE"
    CORRECTION = "CORRECTION"
    RETRACTION = "RETRACTION"
    CURATOR_CORRECTION = "CURATOR_CORRECTION"


class CorrectionEvidenceScope(StrEnum):
    SOURCE_LEVEL = "SOURCE_LEVEL"
    CLAIM_LEVEL = "CLAIM_LEVEL"


class VerifiedCorrectionEvidence(BaseModel):
    """Non-model evidence authorizing a destructive claim-level update."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    basis: CorrectionBasis
    scope: CorrectionEvidenceScope
    source_id: str
    source_checksum: str
    verified_by: Literal["deterministic_verifier", "human_curator"]

    @field_validator("source_id")
    @classmethod
    def _require_source_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("correction evidence requires a source identity")
        return normalized

    @field_validator("source_checksum")
    @classmethod
    def _require_sha256(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("source_checksum must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"

    @model_validator(mode="after")
    def _reject_empty_basis(self) -> VerifiedCorrectionEvidence:
        if self.basis == CorrectionBasis.NONE:
            raise ValueError("verified correction evidence cannot use NONE")
        return self


class ScientificClaimRecord(BaseModel):
    """Minimal cross-dataset claim schema; absent fields remain null."""

    model_config = ConfigDict(frozen=True)

    subject: str
    relation: str
    object: str | None = None
    value: float | str | None = None
    unit: str | None = None
    condition: dict[str, Any] | None = None
    qualifiers: dict[str, Any] | None = None

    @field_validator("subject", "relation")
    @classmethod
    def _require_core_field(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("claim subject and relation must be non-empty")
        return normalized

    @classmethod
    def from_material_claim(cls, claim: ScientificClaim) -> ScientificClaimRecord:
        return cls.model_validate(claim.memory_claim_payload())

    def canonical_key(self, *, include_value: bool = True) -> str:
        payload = self.model_dump(mode="json")
        if not include_value:
            payload.pop("value", None)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return "sha256:" + hashlib.sha256(self.canonical_key().encode("utf-8")).hexdigest()


class ClaimSignature(BaseModel):
    """Fields used for structure-aware retrieval and typed updates."""

    model_config = ConfigDict(frozen=True)

    domain: str
    subject: str
    relation: str
    object: str | None = None
    unit: str | None = None
    condition_signature: str | None = None
    measurement_setting: str | None = None

    @classmethod
    def from_claim(
        cls,
        claim: ScientificClaimRecord,
        *,
        domain: str,
        measurement_setting: str | None = None,
    ) -> ClaimSignature:
        condition = (
            json.dumps(claim.condition, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if claim.condition
            else None
        )
        return cls(
            domain=domain,
            subject=claim.subject,
            relation=claim.relation,
            object=claim.object,
            unit=claim.unit,
            condition_signature=condition,
            measurement_setting=measurement_setting,
        )


class MemoryDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: Literal["published", "rejected", "conflict", "ambiguous", "superseded"]
    reason: str


class MemoryAuthority(BaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["deterministic_verifier", "dataset_annotation", "human_curator"]
    level: int = Field(ge=1, le=4)


class MemoryOrigin(BaseModel):
    """Provenance needed to distinguish natural labels from stress-test corruption."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str
    split: Literal["train", "validation", "test", "ood", "scale", "case_study"]
    annotation_kind: Literal[
        "natural_annotation", "controlled_corruption", "deterministic_verifier"
    ]
    license_id: str


class ScientificMemoryRecord(BaseModel):
    """A claim whose evidence, certificate, decision, time and policy are explicit."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.scientific_memory.v1"

    memory_id: str
    memory_type: MemoryType
    claim: ScientificClaimRecord
    claim_signature: ClaimSignature
    evidence_refs: tuple[EvidenceRef, ...]
    certificate: VerificationCertificate
    decision: MemoryDecision
    source_document: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    policy_version: str
    policy_hash: str
    evidence_release_id: str
    authority: MemoryAuthority
    origin: MemoryOrigin
    status: MemoryStatus = MemoryStatus.ACTIVE
    valid_until: datetime | None = None
    superseded_by: tuple[str, ...] = ()
    supersession_reason: str | None = None

    @property
    def certificate_id(self) -> str:
        return self.certificate.certificate_id

    @field_validator(
        "memory_id", "source_document", "policy_version", "policy_hash", "evidence_release_id"
    )
    @classmethod
    def _require_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("memory identity fields must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_warrant(self) -> ScientificMemoryRecord:
        if not self.evidence_refs:
            raise ValueError("long-term memory requires immutable evidence")
        if self.certificate.evidence_release_id != self.evidence_release_id:
            raise ValueError("certificate and memory evidence releases differ")
        if self.certificate.domain_pack_version != self.policy_version:
            raise ValueError("certificate and memory policy versions differ")
        if self.certificate.domain_pack_hash != self.policy_hash:
            raise ValueError("certificate and memory policy hashes differ")
        if any(ref.release_id != self.evidence_release_id for ref in self.evidence_refs):
            raise ValueError("all evidence refs must belong to the memory evidence release")
        certified = {ref.model_dump_json() for ref in self.certificate.resolved_evidence}
        if {ref.model_dump_json() for ref in self.evidence_refs} - certified:
            raise ValueError("memory evidence must be present in its certificate")
        if self.claim.fingerprint() != self.certificate.certified_claim_hash:
            raise ValueError("memory claim does not match its verification certificate")
        if self.valid_until is not None and self.valid_until <= self.observed_at:
            raise ValueError("valid_until must be later than observed_at")
        if self.status == MemoryStatus.SUPERSEDED and not self.supersession_reason:
            raise ValueError("superseded memory requires a reason")
        return self


class MemoryManagerAction(BaseModel):
    """Hierarchical relation assessment accepted from a learned manager.

    The contract deliberately has no compiled operation field. Only the
    deterministic ``UpdateCompiler`` may derive an ``UpdateOperation``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    admission: AdmissionAction
    semantic_relation: SemanticRelation
    scope_relation: ScopeRelation
    authority_relation: AuthorityRelation
    evidence_sufficiency: EvidenceSufficiency
    target_memory_ids: tuple[str, ...] = ()
    reason_code: str

    @model_validator(mode="after")
    def _validate_targets(self) -> MemoryManagerAction:
        if self.admission in {AdmissionAction.EPHEMERAL_ONLY, AdmissionAction.IGNORE}:
            if self.evidence_sufficiency == EvidenceSufficiency.SUFFICIENT:
                raise ValueError("non-writing admission cannot claim sufficient evidence")
        if not self.target_memory_ids:
            if self.semantic_relation not in {
                SemanticRelation.UNRELATED,
                SemanticRelation.INSUFFICIENT_CONTEXT,
            }:
                raise ValueError("relational labels require at least one target memory")
            if self.scope_relation not in {
                ScopeRelation.DIFFERENT_SCOPE,
                ScopeRelation.UNKNOWN_SCOPE,
            }:
                raise ValueError("target-free assessment cannot assert matching scope")
            if self.authority_relation != AuthorityRelation.NOT_APPLICABLE:
                raise ValueError("target-free assessment requires NOT_APPLICABLE authority")
        if not self.reason_code.strip():
            raise ValueError("memory manager action requires a reason code")
        return self
