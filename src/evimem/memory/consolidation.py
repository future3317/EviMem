"""Certificate-driven conversion from a curation outcome to scientific memory."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.contracts import (
    CandidateObservation,
    ClaimSignature,
    MemoryAuthority,
    MemoryDecision,
    MemoryOrigin,
    MemoryType,
    ScientificClaimRecord,
    ScientificMemoryRecord,
    VerificationCertificate,
)
from evimem.contracts.ids import deterministic_id


@dataclass(frozen=True)
class ConsolidationResult:
    eligible_for_admission: bool
    record: ScientificMemoryRecord | None
    reason_codes: tuple[str, ...]


class MemoryConsolidator:
    """Build memory only from a verifier certificate, never free-form reflection."""

    @staticmethod
    def consolidate(
        *,
        candidate: CandidateObservation,
        certificate: VerificationCertificate,
        domain: str,
        source_document: str,
        origin: MemoryOrigin,
    ) -> ConsolidationResult:
        if certificate.candidate_id != candidate.candidate_id:
            return ConsolidationResult(False, None, ("candidate_certificate_mismatch",))
        if certificate.evidence_release_id not in {
            ref.release_id for ref in candidate.proposed_evidence
        }:
            return ConsolidationResult(False, None, ("candidate_release_mismatch",))
        if not certificate.resolved_evidence or certificate.support_tier == "unbound":
            return ConsolidationResult(False, None, ("certificate_lacks_bound_evidence",))

        if certificate.final_decision == "publish" and certificate.support_tier == "verified_strong":
            memory_type = MemoryType.VERIFIED
            status = "published"
            reason = "verified_strong"
            authority_level = 3
        elif certificate.final_decision == "reject" and certificate.exclusion_reasons:
            memory_type = MemoryType.REJECTED
            status = "rejected"
            reason = certificate.exclusion_reasons[0]
            authority_level = 3
        elif (
            certificate.conflict_result == "unresolved_conflict"
            and certificate.final_decision in {"review", "defer"}
        ):
            memory_type = MemoryType.CONFLICT
            status = "conflict"
            reason = "unresolved_conflict"
            authority_level = 2
        else:
            return ConsolidationResult(False, None, ("outcome_not_long_term_admissible",))

        claim = ScientificClaimRecord.from_material_claim(certificate.normalized_claim)
        signature = ClaimSignature.from_claim(claim, domain=domain)
        memory_id = deterministic_id(
            certificate.certificate_id,
            memory_type.value,
            claim.canonical_key(),
            namespace="mem",
            length=32,
        )
        record = ScientificMemoryRecord(
            memory_id=memory_id,
            memory_type=memory_type,
            claim=claim,
            claim_signature=signature,
            evidence_refs=tuple(certificate.resolved_evidence),
            certificate=certificate,
            decision=MemoryDecision(status=status, reason=reason),
            source_document=source_document,
            observed_at=candidate.proposer_provenance.extraction_timestamp,
            policy_version=certificate.domain_pack_version,
            policy_hash=certificate.domain_pack_hash,
            evidence_release_id=certificate.evidence_release_id,
            authority=MemoryAuthority(source="deterministic_verifier", level=authority_level),
            origin=origin,
        )
        return ConsolidationResult(True, record, ())
