"""Evidence-to-certificate deterministic Phase 0 verifier."""

from __future__ import annotations

from collections.abc import Callable

from evimem.contracts import (
    CheckResult,
    ScientificClaim,
    VerificationCertificate,
    make_certificate_id,
)
from evimem.contracts.ids import deterministic_id
from evimem.controller import EpisodeOutcome
from evimem.domains import DomainPack, DomainValidator
from evimem.evidence import EvidenceBinder, EvidenceBlockStore

from .conflicts import ConflictResolver
from .gate import PublicationGate


class TupleVerifier:
    verifier_version = "evimem-phase0-verifier.v1"

    def __init__(
        self,
        *,
        domain_pack: DomainPack,
        evidence_store: EvidenceBlockStore,
        existing_claims: Callable[[ScientificClaim], tuple[ScientificClaim, ...]] | None = None,
    ) -> None:
        self.domain_pack = domain_pack
        self.evidence_store = evidence_store
        self.binder = EvidenceBinder(evidence_store, domain_pack)
        self.domain_validator = DomainValidator(domain_pack)
        self.gate = PublicationGate(domain_pack)
        self.existing_claims = existing_claims or (lambda claim: ())

    def certify(self, *, outcome: EpisodeOutcome) -> VerificationCertificate:
        candidate = outcome.final_state.candidate
        refs = outcome.final_state.gathered_evidence or tuple(candidate.proposed_evidence)
        binding = self.binder.bind(candidate, evidence_refs=refs)
        domain_result = self.domain_validator.validate(candidate.claim)
        normalized_claim = candidate.claim.model_copy(
            update={
                "property_key": domain_result.canonical_property or candidate.claim.property_key,
                "unit_canonical": domain_result.canonical_unit or candidate.claim.unit_canonical,
            }
        )
        conflict = ConflictResolver.assess(
            normalized_claim,
            self.existing_claims(normalized_claim),
        )
        evidence_text = "\n".join(ref.quote or "" for ref in binding.resolved_evidence)
        gate = self.gate.evaluate(
            evidence_text=evidence_text,
            publication_requested=outcome.publication_requested,
            domain_validation=domain_result,
            binding=binding,
            conflict_result=conflict.result,
        )
        checks = [
            self._check(
                candidate.candidate_id,
                "evidence_binding",
                binding.support_tier == "verified_strong",
                None if binding.support_tier == "verified_strong" else ",".join(binding.reason_codes),
            ),
            self._check(
                candidate.candidate_id,
                "domain_validation",
                domain_result.passed,
                None if domain_result.passed else ",".join(domain_result.reason_codes),
            ),
            self._check(
                candidate.candidate_id,
                "predictive_language",
                not gate.predictive_or_hypothetical,
                "predictive_or_hypothetical_claim" if gate.predictive_or_hypothetical else None,
            ),
            self._check(
                candidate.candidate_id,
                "conflict_resolution",
                conflict.result in {"pass", "distinct_context", "exact_duplicate"},
                ",".join(conflict.reason_codes) or None,
            ),
            self._check(
                candidate.candidate_id,
                "publication_gate",
                gate.final_decision == "publish",
                ",".join(gate.reason_codes) or None,
            ),
        ]
        resolved = list(binding.resolved_evidence)
        certificate_id = make_certificate_id(
            candidate.candidate_id,
            resolved,
            self.verifier_version,
        )
        return VerificationCertificate(
            certificate_id=certificate_id,
            run_id=candidate.run_id,
            candidate_id=candidate.candidate_id,
            normalized_claim=normalized_claim,
            resolved_evidence=resolved,
            checks=checks,
            slot_verification=binding.slot_status,
            binding_method=binding.binding_method,
            support_tier=binding.support_tier,
            constraint_result="pass" if domain_result.passed else "fail",
            conflict_result=conflict.result,
            final_decision=gate.final_decision,
            exclusion_reasons=list(gate.reason_codes),
            evidence_release_id=outcome.final_state.claim_state.evidence_release_id,
            domain_pack_id=self.domain_pack.domain_id,
            domain_pack_version=self.domain_pack.version,
            domain_pack_hash=self.domain_pack.pack_hash,
            verifier_version=self.verifier_version,
        )

    @staticmethod
    def _check(candidate_id: str, check_type: str, passed: bool, reason: str | None) -> CheckResult:
        return CheckResult(
            check_id=deterministic_id(candidate_id, check_type, passed, reason, namespace="check"),
            check_type=check_type,
            passed=passed,
            severity="info" if passed else "error",
            reason=reason,
        )

