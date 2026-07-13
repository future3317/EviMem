from __future__ import annotations

from evimem.contracts import MemoryType, ScientificClaim
from evimem.runtime import DeterministicPhase0Runtime


def test_predictive_claim_reaches_gate_then_is_rejected_and_remembered(tmp_path) -> None:
    runtime = DeterministicPhase0Runtime(tmp_path / "phase0")
    result = runtime.run_document(
        run_id="reject-run",
        doi="10.1000/phase0.reject",
        document_text=(
            "The predicted d33 may reach 190 pC/N in future optimized BaTiO3 samples."
        ),
        proposed_claim=ScientificClaim(
            property_key="d33",
            value_raw="190",
            value_num=190.0,
            unit_raw="pC/N",
            unit_canonical="pC/N",
            material_raw="BaTiO3",
            material_normalized="BaTiO3",
            conditions_raw="future optimized",
            conditions={"optimization": "future optimized"},
        ),
        release_id="phase0-reject-release",
    )

    assert result.outcome.publication_requested is True
    assert result.trajectory.terminal_action == "REQUEST_PUBLICATION"
    assert result.certificate.support_tier == "verified_strong"
    assert result.certificate.final_decision == "reject"
    assert "predictive_or_hypothetical_claim" in result.certificate.exclusion_reasons
    assert result.publication is None
    assert runtime.publication_store.count_observations() == 0
    assert runtime.publication_store.count_certificates() == 0
    assert result.rejection_audit is not None
    assert runtime.audit_store.count() == 1
    assert runtime.audit_store.get(result.certificate.certificate_id) == result.certificate
    memory = runtime.memory_store.get(result.memory_id or "")
    assert memory is not None
    assert memory.memory_type == MemoryType.REJECTED
    assert memory.decision.reason == "predictive_or_hypothetical_claim"

