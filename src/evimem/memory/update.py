"""Deterministic compilation of hierarchical relation labels into memory updates."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.contracts import (
    AdmissionAction,
    AuthorityRelation,
    CorrectionEvidenceScope,
    EvidenceSufficiency,
    MemoryManagerAction,
    ScientificMemoryRecord,
    ScopeRelation,
    SemanticRelation,
    UpdateOperation,
    VerifiedCorrectionEvidence,
)

from .governed_store import GovernedMemoryStore, MemoryAdmissionGate


@dataclass(frozen=True)
class CompiledUpdate:
    """Operation produced by policy code, never by a model or annotator."""

    admission: AdmissionAction
    operation: UpdateOperation
    target_memory_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class UpdateResult:
    applied: bool
    admitted: bool
    operation: UpdateOperation
    reason_codes: tuple[str, ...] = ()


class UpdateCompiler:
    """Fail-closed truth table from relation assessment and store state to an operation."""

    def __init__(self, store: GovernedMemoryStore):
        self.store = store

    @staticmethod
    def _same_claim_scope(left: ScientificMemoryRecord, right: ScientificMemoryRecord) -> bool:
        return left.claim.canonical_key(include_value=False) == right.claim.canonical_key(
            include_value=False
        )

    @staticmethod
    def _has_claim_level_correction(
        correction_evidence: VerifiedCorrectionEvidence | None,
    ) -> bool:
        return bool(
            correction_evidence
            and correction_evidence.scope == CorrectionEvidenceScope.CLAIM_LEVEL
        )

    def compile(
        self,
        *,
        new_record: ScientificMemoryRecord,
        assessment: MemoryManagerAction,
        correction_evidence: VerifiedCorrectionEvidence | None = None,
    ) -> CompiledUpdate:
        admission = MemoryAdmissionGate.assess(new_record, assessment.admission)
        if not admission.admitted:
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=admission.reason_codes or ("non_writing_admission",),
            )

        if assessment.evidence_sufficiency == EvidenceSufficiency.INSUFFICIENT:
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=("insufficient_pair_evidence",),
            )

        targets = tuple(
            record
            for memory_id in assessment.target_memory_ids
            if (record := self.store.get(memory_id)) is not None
        )
        if len(targets) != len(assessment.target_memory_ids):
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=("unknown_target_memory",),
            )

        relation = assessment.semantic_relation
        scope = assessment.scope_relation
        sufficient = assessment.evidence_sufficiency == EvidenceSufficiency.SUFFICIENT

        if relation == SemanticRelation.INSUFFICIENT_CONTEXT:
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=("insufficient_relation_context",),
            )

        if relation == SemanticRelation.UNRELATED:
            if sufficient:
                return CompiledUpdate(
                    admission=assessment.admission,
                    operation=UpdateOperation.ADD,
                    reason_codes=("independent_eligible_claim",),
                )
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=("partial_evidence_for_independent_claim",),
            )

        if not targets:
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=("relation_requires_target",),
            )

        if relation == SemanticRelation.EQUIVALENT:
            identical_claims = all(
                new_record.claim.canonical_key() == target.claim.canonical_key()
                for target in targets
            )
            if scope == ScopeRelation.SAME_SCOPE and sufficient and identical_claims:
                return CompiledUpdate(
                    admission=assessment.admission,
                    operation=UpdateOperation.MERGE,
                    target_memory_ids=assessment.target_memory_ids,
                    reason_codes=("equivalent_same_scope",),
                )
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=("equivalence_not_established_at_same_scope",),
            )

        if relation == SemanticRelation.COMPATIBLE_DISTINCT:
            if scope != ScopeRelation.UNKNOWN_SCOPE:
                return CompiledUpdate(
                    admission=assessment.admission,
                    operation=UpdateOperation.LINK,
                    target_memory_ids=assessment.target_memory_ids,
                    reason_codes=("compatible_distinct_claims",),
                )
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.IGNORE,
                reason_codes=("compatible_relation_has_unknown_scope",),
            )

        if relation == SemanticRelation.CONTRADICTORY:
            if scope != ScopeRelation.SAME_SCOPE:
                if scope == ScopeRelation.UNKNOWN_SCOPE:
                    return CompiledUpdate(
                        admission=assessment.admission,
                        operation=UpdateOperation.IGNORE,
                        reason_codes=("contradiction_scope_unresolved",),
                    )
                return CompiledUpdate(
                    admission=assessment.admission,
                    operation=UpdateOperation.LINK,
                    target_memory_ids=assessment.target_memory_ids,
                    reason_codes=("apparent_contradiction_differs_in_scope",),
                )
            if not sufficient:
                return CompiledUpdate(
                    admission=assessment.admission,
                    operation=UpdateOperation.IGNORE,
                    reason_codes=("contradiction_evidence_partial",),
                )
            if not all(self._same_claim_scope(new_record, target) for target in targets):
                return CompiledUpdate(
                    admission=assessment.admission,
                    operation=UpdateOperation.IGNORE,
                    reason_codes=("structured_scope_mismatch",),
                )
            destructive_ok = (
                assessment.authority_relation == AuthorityRelation.NEWER_MORE_AUTHORITATIVE
                and self._has_claim_level_correction(correction_evidence)
                and all(new_record.authority.level > target.authority.level for target in targets)
                and all(new_record.observed_at > target.observed_at for target in targets)
            )
            if destructive_ok:
                return CompiledUpdate(
                    admission=assessment.admission,
                    operation=UpdateOperation.SUPERSEDE,
                    target_memory_ids=assessment.target_memory_ids,
                    reason_codes=("verified_claim_level_correction",),
                )
            return CompiledUpdate(
                admission=assessment.admission,
                operation=UpdateOperation.CONFLICT,
                target_memory_ids=assessment.target_memory_ids,
                reason_codes=("same_scope_contradiction_without_supersession_authority",),
            )

        return CompiledUpdate(
            admission=assessment.admission,
            operation=UpdateOperation.IGNORE,
            reason_codes=("unhandled_relation_state",),
        )


class TypedMemoryUpdateService:
    """Apply only operations compiled from hierarchical relation assessments."""

    def __init__(self, store: GovernedMemoryStore):
        self.store = store
        self.compiler = UpdateCompiler(store)

    def apply(
        self,
        *,
        new_record: ScientificMemoryRecord,
        assessment: MemoryManagerAction,
        correction_evidence: VerifiedCorrectionEvidence | None = None,
    ) -> UpdateResult:
        compiled = self.compiler.compile(
            new_record=new_record,
            assessment=assessment,
            correction_evidence=correction_evidence,
        )
        if compiled.operation == UpdateOperation.IGNORE:
            return UpdateResult(False, False, compiled.operation, compiled.reason_codes)

        admission = self.store.admit(new_record, compiled.admission)
        for target_memory_id in compiled.target_memory_ids:
            self.store.register_relation(
                source_memory_id=new_record.memory_id,
                target_memory_id=target_memory_id,
                operation=compiled.operation,
                reason=",".join(compiled.reason_codes),
            )
        return UpdateResult(
            applied=True,
            admitted=admission.admitted,
            operation=compiled.operation,
            reason_codes=compiled.reason_codes,
        )
