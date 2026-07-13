"""Construct verifier-owned initial state from a candidate and fixed episode identity."""

from __future__ import annotations

from evimem.core.contracts import (
    CandidateObservation,
    ClaimState,
    CurationBudget,
    SlotStatus,
    VerificationSlot,
)
from evimem.memory.retriever import MemoryHints

from .state import ControllerState, EvidenceIndexEntry

_CLAIM_FIELDS = {
    "property": "property_key",
    "value": "value_raw",
    "unit": "unit_raw",
    "material": "material_raw",
    "composition": "composition_raw",
}


class StateBuilder:
    @staticmethod
    def build(
        *,
        candidate: CandidateObservation,
        required_slots: tuple[str, ...],
        evidence_release_id: str,
        domain_pack_id: str,
        domain_pack_version: str,
        domain_pack_hash: str,
        budget: CurationBudget,
        evidence_index: tuple[EvidenceIndexEntry, ...] = (),
        memory_hints: MemoryHints | None = None,
        domain_requirements: dict[str, object] | None = None,
    ) -> ControllerState:
        slots: dict[str, VerificationSlot] = {}
        claim = candidate.claim
        for slot_name in required_slots:
            if slot_name.startswith("condition."):
                key = slot_name.removeprefix("condition.")
                present = key in claim.conditions and claim.conditions[key] not in (None, "")
            else:
                field_name = _CLAIM_FIELDS.get(slot_name)
                present = bool(field_name and getattr(claim, field_name, None) not in (None, ""))
            slots[slot_name] = VerificationSlot(
                status=SlotStatus.CANDIDATE if present else SlotStatus.MISSING
            )
        claim_state = ClaimState(
            candidate_id=candidate.candidate_id,
            evidence_release_id=evidence_release_id,
            domain_pack_id=domain_pack_id,
            domain_pack_version=domain_pack_version,
            domain_pack_hash=domain_pack_hash,
            slots=slots,
            remaining_budget=budget,
        )
        return ControllerState(
            candidate=candidate,
            claim_state=claim_state,
            evidence_index=evidence_index,
            memory_hints=memory_hints or MemoryHints(),
            domain_requirements=dict(domain_requirements or {}),
        )
