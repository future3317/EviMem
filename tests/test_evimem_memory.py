from __future__ import annotations

from datetime import UTC, datetime

import pytest

from evimem.core.contracts import (
    ClaimSignature,
    CurationTrajectory,
    MemoryAuthority,
    MemoryDecision,
    MemoryType,
    WarrantedMemoryItem,
)
from evimem.memory import (
    GovernedMemoryStore,
    MemoryAdmissionError,
    MemoryConsolidator,
    MemoryRetriever,
    MemorySupersessionService,
    RetrievalQuery,
)

from .evimem_helpers import candidate, certificate, evidence_ref


def _trajectory() -> CurationTrajectory:
    return CurationTrajectory(
        trajectory_id="trajectory-1",
        run_id="run-1",
        candidate_id="candidate-1",
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        initial_state_hash="state-hash",
        terminal_action="REQUEST_PUBLICATION",
        final_certificate_id="certificate-1",
    )


def _verified_item(memory_id: str = "memory-1") -> WarrantedMemoryItem:
    return WarrantedMemoryItem(
        memory_id=memory_id,
        memory_type=MemoryType.VERIFIED,
        claim_signature=ClaimSignature(
            domain="piezoelectric",
            property_key="d33",
            material_family="PZT",
            material_identity="PZT",
            condition_signature="room_temperature",
        ),
        normalized_content={"value": 350.0, "unit": "pC/N"},
        evidence_refs=(evidence_ref(),),
        certificate_id="certificate-1",
        decision=MemoryDecision(status="published", reason="verified_strong"),
        authority=MemoryAuthority(source="deterministic_harness", level=3),
        policy_version="1.3.0",
        policy_hash="policy-hash",
        evidence_release_id="release-1",
        valid_from=datetime(2026, 7, 13, tzinfo=UTC),
    )


def test_governed_store_admits_only_certificate_backed_memory(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    item = _verified_item()
    first = store.admit(item, certificate())
    second = store.admit(item, certificate())
    assert first.admitted and not first.idempotent
    assert second.admitted and second.idempotent
    assert store.get(item.memory_id) == item
    assert store.verify_integrity()["ok"] is True


def test_governed_store_rejects_policy_mismatch(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    item = _verified_item().model_copy(update={"policy_hash": "wrong"})
    with pytest.raises(MemoryAdmissionError, match="policy_hash_mismatch"):
        store.admit(item, certificate())


def test_governed_store_does_not_overwrite_same_id(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    item = _verified_item()
    store.admit(item, certificate())
    changed = item.model_copy(update={"normalized_content": {"value": 999.0}})
    with pytest.raises(MemoryAdmissionError, match="memory_id_collision"):
        store.admit(changed, certificate())


def test_supersession_preserves_old_item_and_audit_chain(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    old = _verified_item("old")
    new = _verified_item("new")
    store.admit(old, certificate())
    store.admit(new, certificate())
    service = MemorySupersessionService(store)
    assert service.supersede(
        previous_memory_id="old",
        successor_memory_id="new",
        reason="new_stronger_evidence",
    )
    effective_old = store.get("old")
    assert effective_old is not None
    assert effective_old.status == "superseded"
    assert effective_old.contradicted_by == ("new",)
    assert [item.memory_id for item in store.query(domain="piezoelectric")] == ["new"]


def test_consolidator_uses_certificate_not_model_reflection() -> None:
    result = MemoryConsolidator.consolidate(
        candidate=candidate(),
        certificate=certificate(),
        trajectory=_trajectory(),
        domain="piezoelectric",
        material_family="PZT",
    )
    assert result.admitted_to_long_term
    assert result.item is not None
    assert result.item.memory_type == MemoryType.VERIFIED
    assert result.item.certificate_id == "certificate-1"


def test_ambiguous_deferral_is_not_long_term_memory() -> None:
    cert = certificate(
        final_decision="defer",
        support_tier="ambiguous",
        conflict_result="not_run",
    )
    trajectory = _trajectory().model_copy(update={"terminal_action": "DEFER_FOR_REVIEW"})
    result = MemoryConsolidator.consolidate(
        candidate=candidate(),
        certificate=cert,
        trajectory=trajectory,
        domain="piezoelectric",
    )
    assert not result.admitted_to_long_term
    assert result.reason_codes == ("outcome_not_long_term_admissible",)


def test_retriever_prefers_policy_compatible_structural_match(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    compatible = _verified_item("compatible")
    stale_policy = _verified_item("stale").model_copy(
        update={"policy_version": "1.2.0", "policy_hash": "old-policy"}
    )
    store.admit(compatible, certificate())
    old_certificate = certificate().model_copy(
        update={"domain_pack_version": "1.2.0", "domain_pack_hash": "old-policy"}
    )
    store.admit(stale_policy, old_certificate)
    results = MemoryRetriever(store).retrieve(
        RetrievalQuery(
            signature=compatible.claim_signature,
            policy_version="1.3.0",
            policy_hash="policy-hash",
        )
    )
    assert [result.item.memory_id for result in results][:2] == ["compatible", "stale"]
    assert results[0].policy_compatibility == 1.0
