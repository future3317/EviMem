"""Auditable memory supersession without destructive overwrite."""

from __future__ import annotations

from .governed_store import GovernedMemoryStore, MemoryAdmissionError


class MemorySupersessionService:
    def __init__(self, store: GovernedMemoryStore):
        self.store = store

    def supersede(
        self,
        *,
        previous_memory_id: str,
        successor_memory_id: str,
        reason: str,
    ) -> bool:
        previous = self.store.get(previous_memory_id)
        successor = self.store.get(successor_memory_id)
        if previous is None or successor is None:
            raise MemoryAdmissionError("both memories must exist")
        if previous.claim_signature.domain != successor.claim_signature.domain:
            raise MemoryAdmissionError("cross-domain supersession is forbidden")
        return self.store.register_supersession(
            superseded_memory_id=previous_memory_id,
            successor_memory_id=successor_memory_id,
            reason=reason,
        )
