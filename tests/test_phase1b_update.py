from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from evimem.contracts import (
    AdmissionAction,
    AuthorityRelation,
    CorrectionBasis,
    CorrectionEvidenceScope,
    EvidenceSufficiency,
    MemoryManagerAction,
    ScopeRelation,
    SemanticRelation,
    UpdateOperation,
    VerifiedCorrectionEvidence,
)
from evimem.memory import GovernedMemoryStore, UpdateCompiler

from .evimem_helpers import memory_record


def assessment(
    semantic: SemanticRelation,
    scope: ScopeRelation,
    authority: AuthorityRelation,
    evidence: EvidenceSufficiency = EvidenceSufficiency.SUFFICIENT,
    *,
    targets: tuple[str, ...] = ("old",),
) -> MemoryManagerAction:
    return MemoryManagerAction(
        admission=AdmissionAction.WRITE_VERIFIED,
        semantic_relation=semantic,
        scope_relation=scope,
        authority_relation=authority,
        evidence_sufficiency=evidence,
        target_memory_ids=targets,
        reason_code="phase1b_test",
    )


def correction(scope: CorrectionEvidenceScope) -> VerifiedCorrectionEvidence:
    return VerifiedCorrectionEvidence(
        basis=CorrectionBasis.CORRECTION,
        scope=scope,
        source_id="curator:verified-correction",
        source_checksum="sha256:" + "e" * 64,
        verified_by="human_curator",
    )


def test_hierarchical_label_contract_rejects_flat_operation() -> None:
    with pytest.raises(ValidationError, match="update_operation"):
        MemoryManagerAction.model_validate(
            {
                "admission": "WRITE_VERIFIED",
                "semantic_relation": "UNRELATED",
                "scope_relation": "DIFFERENT_SCOPE",
                "authority_relation": "NOT_APPLICABLE",
                "evidence_sufficiency": "SUFFICIENT",
                "update_operation": "SUPERSEDE",
                "reason_code": "illegal_direct_operation",
            }
        )


@pytest.mark.parametrize(
    ("semantic", "scope", "authority", "evidence", "expected"),
    [
        (
            SemanticRelation.EQUIVALENT,
            ScopeRelation.SAME_SCOPE,
            AuthorityRelation.EQUAL_AUTHORITY,
            EvidenceSufficiency.SUFFICIENT,
            UpdateOperation.MERGE,
        ),
        (
            SemanticRelation.COMPATIBLE_DISTINCT,
            ScopeRelation.NARROWER_SCOPE,
            AuthorityRelation.NOT_APPLICABLE,
            EvidenceSufficiency.PARTIAL,
            UpdateOperation.LINK,
        ),
        (
            SemanticRelation.CONTRADICTORY,
            ScopeRelation.SAME_SCOPE,
            AuthorityRelation.UNRESOLVED,
            EvidenceSufficiency.SUFFICIENT,
            UpdateOperation.CONFLICT,
        ),
        (
            SemanticRelation.CONTRADICTORY,
            ScopeRelation.UNKNOWN_SCOPE,
            AuthorityRelation.UNRESOLVED,
            EvidenceSufficiency.SUFFICIENT,
            UpdateOperation.IGNORE,
        ),
        (
            SemanticRelation.INSUFFICIENT_CONTEXT,
            ScopeRelation.UNKNOWN_SCOPE,
            AuthorityRelation.UNRESOLVED,
            EvidenceSufficiency.INSUFFICIENT,
            UpdateOperation.IGNORE,
        ),
    ],
)
def test_update_compiler_truth_table(
    tmp_path,
    semantic: SemanticRelation,
    scope: ScopeRelation,
    authority: AuthorityRelation,
    evidence: EvidenceSufficiency,
    expected: UpdateOperation,
) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    old = memory_record("old", observed_at=datetime(2025, 1, 1, tzinfo=UTC))
    store.admit(old, AdmissionAction.WRITE_VERIFIED)
    new = memory_record(
        "new",
        value=350.0 if semantic == SemanticRelation.EQUIVALENT else 360.0,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    compiled = UpdateCompiler(store).compile(
        new_record=new,
        assessment=assessment(semantic, scope, authority, evidence),
    )
    assert compiled.operation == expected


def test_target_free_eligible_claim_compiles_to_add(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    compiled = UpdateCompiler(store).compile(
        new_record=memory_record("new"),
        assessment=assessment(
            SemanticRelation.UNRELATED,
            ScopeRelation.DIFFERENT_SCOPE,
            AuthorityRelation.NOT_APPLICABLE,
            targets=(),
        ),
    )
    assert compiled.operation == UpdateOperation.ADD
    assert not compiled.target_memory_ids


def test_source_level_retraction_never_compiles_claim_supersede(tmp_path) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    store.admit(
        memory_record("old", observed_at=datetime(2025, 1, 1, tzinfo=UTC), authority_level=2),
        AdmissionAction.WRITE_VERIFIED,
    )
    new = memory_record(
        "new",
        value=360.0,
        observed_at=datetime(2026, 1, 1, tzinfo=UTC),
        authority_level=4,
    )
    compiled = UpdateCompiler(store).compile(
        new_record=new,
        assessment=assessment(
            SemanticRelation.CONTRADICTORY,
            ScopeRelation.SAME_SCOPE,
            AuthorityRelation.NEWER_MORE_AUTHORITATIVE,
        ),
        correction_evidence=correction(CorrectionEvidenceScope.SOURCE_LEVEL),
    )
    assert compiled.operation == UpdateOperation.CONFLICT


@pytest.mark.parametrize(
    ("new_authority", "new_date", "expected"),
    [
        (3, datetime(2026, 1, 1, tzinfo=UTC), UpdateOperation.CONFLICT),
        (4, datetime(2024, 1, 1, tzinfo=UTC), UpdateOperation.CONFLICT),
        (4, datetime(2026, 1, 1, tzinfo=UTC), UpdateOperation.SUPERSEDE),
    ],
)
def test_supersede_requires_higher_authority_and_newer_evidence(
    tmp_path,
    new_authority: int,
    new_date: datetime,
    expected: UpdateOperation,
) -> None:
    store = GovernedMemoryStore(tmp_path / "memory.sqlite")
    store.admit(
        memory_record("old", observed_at=datetime(2025, 1, 1, tzinfo=UTC), authority_level=3),
        AdmissionAction.WRITE_VERIFIED,
    )
    compiled = UpdateCompiler(store).compile(
        new_record=memory_record(
            "new",
            value=360.0,
            observed_at=new_date,
            authority_level=new_authority,
        ),
        assessment=assessment(
            SemanticRelation.CONTRADICTORY,
            ScopeRelation.SAME_SCOPE,
            AuthorityRelation.NEWER_MORE_AUTHORITATIVE,
        ),
        correction_evidence=correction(CorrectionEvidenceScope.CLAIM_LEVEL),
    )
    assert compiled.operation == expected
