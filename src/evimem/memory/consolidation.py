"""Certificate-driven conversion from an episode outcome to warranted memory."""

from __future__ import annotations

import json
from dataclasses import dataclass

from evimem.contracts import (
    CandidateObservation,
    ClaimSignature,
    CurationTrajectory,
    MemoryAuthority,
    MemoryDecision,
    MemoryType,
    VerificationCertificate,
    WarrantedMemoryItem,
)
from evimem.contracts.ids import deterministic_id


@dataclass(frozen=True)
class ConsolidationResult:
    admitted_to_long_term: bool
    item: WarrantedMemoryItem | None
    reason_codes: tuple[str, ...]


class MemoryConsolidator:
    """Build memory from audited outcomes, never from free-form reflection."""

    @staticmethod
    def consolidate(
        *,
        candidate: CandidateObservation,
        certificate: VerificationCertificate,
        trajectory: CurationTrajectory,
        domain: str,
        material_family: str | None = None,
    ) -> ConsolidationResult:
        if certificate.candidate_id != candidate.candidate_id:
            return ConsolidationResult(False, None, ("candidate_certificate_mismatch",))
        if trajectory.candidate_id != candidate.candidate_id:
            return ConsolidationResult(False, None, ("candidate_trajectory_mismatch",))
        if trajectory.evidence_release_id != certificate.evidence_release_id:
            return ConsolidationResult(False, None, ("trajectory_release_mismatch",))
        if trajectory.domain_pack_id != certificate.domain_pack_id:
            return ConsolidationResult(False, None, ("trajectory_domain_pack_mismatch",))
        if trajectory.domain_pack_version != certificate.domain_pack_version:
            return ConsolidationResult(False, None, ("trajectory_policy_version_mismatch",))
        if trajectory.domain_pack_hash != certificate.domain_pack_hash:
            return ConsolidationResult(False, None, ("trajectory_policy_hash_mismatch",))
        if not certificate.resolved_evidence:
            return ConsolidationResult(False, None, ("certificate_has_no_resolved_evidence",))
        if certificate.support_tier == "unbound":
            return ConsolidationResult(False, None, ("unbound_certificate_not_admissible",))

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
            status = "deferred"
            reason = "unresolved_conflict"
            authority_level = 2
        else:
            return ConsolidationResult(False, None, ("outcome_not_long_term_admissible",))

        claim = certificate.normalized_claim
        conditions = json.dumps(claim.conditions, sort_keys=True, separators=(",", ":"))
        signature = ClaimSignature(
            domain=domain,
            property_key=claim.property_key,
            material_family=material_family,
            material_identity=claim.material_normalized or claim.material_raw,
            composition=claim.composition_normalized or claim.composition_raw,
            condition_signature=conditions if claim.conditions else claim.conditions_raw,
        )
        memory_id = deterministic_id(
            certificate.certificate_id,
            memory_type.value,
            signature.model_dump_json(),
            length=32,
            namespace="mem",
        )
        item = WarrantedMemoryItem(
            memory_id=memory_id,
            memory_type=memory_type,
            claim_signature=signature,
            normalized_content={
                **claim.model_dump(mode="json"),
                "successful_actions": [step.action for step in trajectory.steps],
            },
            evidence_refs=tuple(certificate.resolved_evidence),
            certificate_id=certificate.certificate_id,
            decision=MemoryDecision(status=status, reason=reason),
            authority=MemoryAuthority(
                source="deterministic_harness",
                level=authority_level,
            ),
            policy_version=certificate.domain_pack_version,
            policy_hash=certificate.domain_pack_hash,
            evidence_release_id=certificate.evidence_release_id,
            valid_from=candidate.proposer_provenance.extraction_timestamp,
        )
        return ConsolidationResult(True, item, ())
