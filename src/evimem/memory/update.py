"""Typed memory update validation and append-only relation registration."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.contracts import (
    MemoryManagerAction,
    ScientificMemoryRecord,
    UpdateOperation,
)

from .governed_store import GovernedMemoryStore, MemoryAdmissionError


@dataclass(frozen=True)
class UpdateResult:
    applied: bool
    admitted: bool
    operation: UpdateOperation
    reason_codes: tuple[str, ...] = ()


class TypedMemoryUpdateGate:
    """Validate semantic preconditions before a predicted update mutates memory."""

    @staticmethod
    def _same_context(left: ScientificMemoryRecord, right: ScientificMemoryRecord) -> bool:
        return left.claim.canonical_key(include_value=False) == right.claim.canonical_key(
            include_value=False
        )

    @classmethod
    def validate(
        cls,
        new_record: ScientificMemoryRecord,
        action: MemoryManagerAction,
        targets: tuple[ScientificMemoryRecord, ...],
    ) -> tuple[str, ...]:
        operation = action.update_operation
        if operation == UpdateOperation.ADD:
            return ()
        if operation == UpdateOperation.IGNORE:
            return ()
        if len(targets) != len(action.target_memory_ids):
            return ("unknown_target_memory",)
        reasons: list[str] = []
        for target in targets:
            same_context = cls._same_context(new_record, target)
            same_claim = new_record.claim.canonical_key() == target.claim.canonical_key()
            if operation == UpdateOperation.MERGE and not same_claim:
                reasons.append("merge_requires_identical_claim")
            elif operation == UpdateOperation.LINK:
                same_subject_relation = (
                    new_record.claim.subject == target.claim.subject
                    and new_record.claim.relation == target.claim.relation
                )
                if not same_subject_relation or same_context:
                    reasons.append("link_requires_related_distinct_context")
            elif operation == UpdateOperation.CONFLICT:
                if not same_context or new_record.claim.value == target.claim.value:
                    reasons.append("conflict_requires_same_context_incompatible_value")
            elif operation == UpdateOperation.SUPERSEDE:
                if not same_context:
                    reasons.append("supersede_requires_same_context")
                if new_record.observed_at <= target.observed_at:
                    reasons.append("supersede_requires_newer_evidence")
                if new_record.authority.level < target.authority.level:
                    reasons.append("supersede_cannot_lower_authority")
        return tuple(dict.fromkeys(reasons))


class TypedMemoryUpdateService:
    def __init__(self, store: GovernedMemoryStore):
        self.store = store

    def apply(
        self,
        *,
        new_record: ScientificMemoryRecord,
        action: MemoryManagerAction,
    ) -> UpdateResult:
        if action.update_operation == UpdateOperation.IGNORE:
            return UpdateResult(False, False, UpdateOperation.IGNORE, (action.reason_code,))
        targets = tuple(
            record
            for memory_id in action.target_memory_ids
            if (record := self.store.get(memory_id)) is not None
        )
        reasons = TypedMemoryUpdateGate.validate(new_record, action, targets)
        if reasons:
            raise MemoryAdmissionError(",".join(reasons))
        admission = self.store.admit(new_record, action.admission)
        for target in targets:
            self.store.register_relation(
                source_memory_id=new_record.memory_id,
                target_memory_id=target.memory_id,
                operation=action.update_operation,
                reason=action.reason_code,
            )
        return UpdateResult(
            applied=True,
            admitted=admission.admitted,
            operation=action.update_operation,
        )
