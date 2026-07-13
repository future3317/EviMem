from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evimem.contracts import (
    AdmissionAction,
    MemoryManagerAction,
    MemoryType,
    UpdateOperation,
)
from evimem.memory import (
    FullHistoryBaseline,
    GovernedMemoryStore,
    MemoryAdmissionError,
    MemoryConsolidator,
    MemoryRetriever,
    NoMemoryBaseline,
    RetrievalQuery,
    TypedMemoryUpdateService,
)

from .evimem_helpers import candidate, certificate, memory_record


def test_governed_store_requires_matching_structured_admission(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    record = memory_record()
    first = store.admit(record, AdmissionAction.WRITE_VERIFIED)
    second = store.admit(record, AdmissionAction.WRITE_VERIFIED)
    assert first.admitted and not first.idempotent
    assert second.admitted and second.idempotent
    assert store.get(record.memory_id) == record
    with pytest.raises(MemoryAdmissionError, match="admission_memory_type_mismatch"):
        store.admit(memory_record("wrong"), AdmissionAction.WRITE_REJECTED)


def test_ephemeral_prediction_never_crosses_long_term_boundary(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    decision = store.admit(memory_record(), AdmissionAction.EPHEMERAL_ONLY)
    assert not decision.admitted
    assert store.get("memory-1") is None


def test_memory_cannot_replace_its_certified_evidence() -> None:
    record = memory_record()
    changed_ref = record.evidence_refs[0].model_copy(update={"quote": "fabricated quote"})
    changed = record.model_copy(update={"evidence_refs": (changed_ref,)})
    with pytest.raises(ValidationError, match="memory evidence must be present"):
        type(record).model_validate(changed.model_dump())


def test_memory_cannot_relabel_a_certificate_as_a_different_claim() -> None:
    record = memory_record()
    changed_claim = record.claim.model_copy(update={"value": 999.0})
    changed = record.model_copy(update={"claim": changed_claim})
    with pytest.raises(ValidationError, match="claim does not match"):
        type(record).model_validate(changed.model_dump())


def test_certificate_claim_hash_is_self_verifying() -> None:
    cert = certificate()
    changed = cert.model_copy(update={"certified_claim_hash": "sha256:" + "0" * 64})
    with pytest.raises(ValidationError, match="does not match normalized_claim"):
        type(cert).model_validate(changed.model_dump())


def test_typed_conflict_update_creates_edge_without_overwrite(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    old = memory_record("old", value=350.0)
    store.admit(old, AdmissionAction.WRITE_VERIFIED)
    conflict = memory_record("conflict", value=190.0, memory_type=MemoryType.CONFLICT)
    action = MemoryManagerAction(
        admission=AdmissionAction.WRITE_CONFLICT,
        update_operation=UpdateOperation.CONFLICT,
        target_memory_ids=("old",),
        reason_code="same_context_incompatible_value",
    )
    result = TypedMemoryUpdateService(store).apply(new_record=conflict, action=action)
    assert result.applied
    assert store.get("old") == old
    assert store.get("conflict") == conflict


def test_supersession_preserves_lineage_and_requires_newer_evidence(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    old = memory_record("old", observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    store.admit(old, AdmissionAction.WRITE_VERIFIED)
    new = memory_record("new", value=360.0, observed_at=datetime(2026, 1, 1, tzinfo=UTC))
    action = MemoryManagerAction(
        admission=AdmissionAction.WRITE_VERIFIED,
        update_operation=UpdateOperation.SUPERSEDE,
        target_memory_ids=("old",),
        reason_code="newer_stronger_evidence",
    )
    TypedMemoryUpdateService(store).apply(new_record=new, action=action)
    superseded = store.get("old")
    assert superseded is not None
    assert superseded.status.value == "superseded"
    assert superseded.superseded_by == ("new",)
    assert [record.memory_id for record in store.query(domain="piezoelectric")] == ["new"]


def test_retrieval_enforces_as_of_time_and_returns_full_certificate(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    past = memory_record("past", observed_at=datetime(2024, 1, 1, tzinfo=UTC))
    future = memory_record("future", observed_at=datetime(2027, 1, 1, tzinfo=UTC))
    store.admit(past, AdmissionAction.WRITE_VERIFIED)
    store.admit(future, AdmissionAction.WRITE_VERIFIED)
    results = MemoryRetriever(store).retrieve(
        RetrievalQuery(
            signature=past.claim_signature,
            policy_version="1.3.0",
            policy_hash="policy-hash",
            as_of=datetime(2026, 1, 1, tzinfo=UTC),
        )
    )
    assert [result.record.memory_id for result in results] == ["past"]
    assert results[0].record.certificate.certificate_id == "certificate-1"
    assert NoMemoryBaseline().retrieve(
        RetrievalQuery(
            signature=past.claim_signature,
            policy_version="1.3.0",
            policy_hash="policy-hash",
        )
    ) == []
    assert [
        result.record.memory_id
        for result in FullHistoryBaseline(store).retrieve(
            RetrievalQuery(
                signature=past.claim_signature,
                policy_version="1.3.0",
                policy_hash="policy-hash",
                as_of=datetime(2026, 1, 1, tzinfo=UTC),
            )
        )
    ] == ["past"]


def test_consolidation_no_longer_depends_on_controller_trajectory() -> None:
    from evimem.contracts import MemoryOrigin

    result = MemoryConsolidator.consolidate(
        candidate=candidate(),
        certificate=certificate(),
        domain="piezoelectric",
        source_document="doi:10.1000/example",
        origin=MemoryOrigin(
            dataset_name="fixture",
            split="train",
            annotation_kind="deterministic_verifier",
            license_id="test-only",
        ),
    )
    assert result.eligible_for_admission
    assert result.record is not None
    assert result.record.memory_type == MemoryType.VERIFIED


def test_invalid_supersession_is_rejected(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    old = memory_record("old", observed_at=datetime(2026, 1, 1, tzinfo=UTC))
    store.admit(old, AdmissionAction.WRITE_VERIFIED)
    older = memory_record("older", observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    action = MemoryManagerAction(
        admission=AdmissionAction.WRITE_VERIFIED,
        update_operation=UpdateOperation.SUPERSEDE,
        target_memory_ids=("old",),
        reason_code="invalid_time",
    )
    with pytest.raises(MemoryAdmissionError, match="supersede_requires_newer_evidence"):
        TypedMemoryUpdateService(store).apply(new_record=older, action=action)
