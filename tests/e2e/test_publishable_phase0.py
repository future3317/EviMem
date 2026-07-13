from __future__ import annotations

from evimem.contracts import MemoryType, ScientificClaim
from evimem.runtime import DeterministicPhase0Runtime


def test_publishable_document_runs_real_phase0_and_is_idempotent(tmp_path) -> None:
    runtime = DeterministicPhase0Runtime(tmp_path / "phase0")
    claim = ScientificClaim(
        property_key="d33",
        value_raw="190",
        value_num=190.0,
        unit_raw="pC/N",
        unit_canonical="pC/N",
        material_raw="BaTiO3",
        material_normalized="BaTiO3",
        conditions_raw="room temperature",
        conditions={"temperature": "room temperature"},
    )
    arguments = {
        "run_id": "publish-run",
        "doi": "10.1000/phase0.publish",
        "document_text": "BaTiO3 exhibits a d33 value of 190 pC/N at room temperature.",
        "proposed_claim": claim,
        "release_id": "phase0-publish-release",
    }

    first = runtime.run_document(**arguments)
    second = runtime.run_document(**arguments)

    assert first.outcome.publication_requested is True
    assert first.trajectory.terminal_action == "REQUEST_PUBLICATION"
    assert first.certificate.final_decision == "publish"
    assert first.certificate.binding_method == "exact_quote_match"
    assert first.certificate.resolved_evidence[0].release_id == "phase0-publish-release"
    assert first.publication is not None and not first.publication.idempotent
    assert second.publication is not None and second.publication.idempotent
    assert first.publication.observation.certificate_id == first.certificate.certificate_id
    assert runtime.publication_store.count_observations() == 1
    assert runtime.publication_store.count_certificates() == 1
    assert runtime.publication_store.count_commits() == 1
    assert runtime.audit_store.count() == 0
    assert first.memory_id == second.memory_id
    memory = runtime.memory_store.get(first.memory_id or "")
    assert memory is not None
    assert memory.memory_type == MemoryType.VERIFIED
    assert memory.certificate_id == first.certificate.certificate_id
    assert memory.evidence_refs == tuple(first.certificate.resolved_evidence)

