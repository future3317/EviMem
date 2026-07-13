from __future__ import annotations

from datetime import UTC, datetime

from evimem.contracts import (
    CandidateObservation,
    CheckResult,
    EvidenceRef,
    ProposerProvenance,
    ScientificClaim,
    SlotStatus,
    TextSpanLocator,
    VerificationCertificate,
)


def evidence_ref(
    block_id: str = "ev-1",
    *,
    release_id: str = "release-1",
    document_id: str = "doi:10.1000/example",
) -> EvidenceRef:
    return EvidenceRef(
        release_id=release_id,
        document_id=document_id,
        block_id=block_id,
        checksum="sha256:" + "a" * 64,
        quote="PZT has d33 = 350 pC/N",
        locator=TextSpanLocator(block_id=block_id, start=0, end=24),
    )


def claim() -> ScientificClaim:
    return ScientificClaim(
        property_key="d33",
        value_raw="350",
        value_num=350.0,
        unit_raw="pC/N",
        unit_canonical="pC/N",
        material_raw="PZT",
        material_normalized="PZT",
        composition_raw="Pb(Zr,Ti)O3",
        conditions_raw="room temperature",
        conditions={"temperature": "room_temperature"},
    )


def candidate() -> CandidateObservation:
    return CandidateObservation(
        candidate_id="candidate-1",
        run_id="run-1",
        doi="10.1000/example",
        claim=claim(),
        proposed_evidence=[evidence_ref()],
        proposer_provenance=ProposerProvenance(
            provider="test",
            model="test-proposer",
            extraction_schema_version="evimem.v1",
            prompt_hash="prompt-hash",
            extraction_timestamp=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )


def certificate(
    *,
    certificate_id: str = "certificate-1",
    final_decision: str = "publish",
    support_tier: str = "verified_strong",
    conflict_result: str = "pass",
    exclusion_reasons: list[str] | None = None,
) -> VerificationCertificate:
    return VerificationCertificate(
        certificate_id=certificate_id,
        run_id="run-1",
        candidate_id="candidate-1",
        normalized_claim=claim(),
        resolved_evidence=[evidence_ref()],
        checks=[
            CheckResult(
                check_id="tuple",
                check_type="tuple_verification",
                passed=final_decision == "publish",
                severity="info" if final_decision == "publish" else "error",
            )
        ],
        slot_verification={
            "property": SlotStatus.VERIFIED,
            "value": SlotStatus.VERIFIED,
            "unit": SlotStatus.VERIFIED,
            "material": SlotStatus.VERIFIED,
        },
        binding_method="exact_quote_match",
        support_tier=support_tier,
        constraint_result="pass" if final_decision == "publish" else "fail",
        conflict_result=conflict_result,
        final_decision=final_decision,
        exclusion_reasons=exclusion_reasons or [],
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        verifier_version="harness-1",
    )
