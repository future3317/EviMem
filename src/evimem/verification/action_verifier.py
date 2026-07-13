"""Verifier-owned ClaimState updates after evidence actions."""

from __future__ import annotations

from evimem.contracts import VerificationSlot, VerifierDelta
from evimem.controller import (
    ActionToolResult,
    ControllerState,
    CurationAction,
    VerificationUpdate,
)
from evimem.evidence import EvidenceBinder


class DeterministicActionVerifier:
    def __init__(self, binder: EvidenceBinder):
        self.binder = binder

    def verify(
        self,
        *,
        action: CurationAction,
        state: ControllerState,
        tool_result: ActionToolResult,
    ) -> VerificationUpdate:
        refs = tuple(
            dict.fromkeys((*state.gathered_evidence, *tool_result.evidence_refs))
        )
        binding = self.binder.bind(state.candidate, evidence_refs=refs)
        updates: dict[str, VerificationSlot] = {}
        newly_verified: list[str] = []
        newly_bound: list[str] = []
        for slot_name in state.claim_state.slots:
            status = binding.slot_status.get(slot_name)
            if status is None:
                continue
            evidence = binding.slot_evidence.get(slot_name, ())
            updates[slot_name] = VerificationSlot(
                status=status,
                evidence_refs=evidence,
                reason_codes=() if evidence else (f"unbound_slot:{slot_name}",),
            )
            previous = state.claim_state.slots[slot_name].status
            if status.value == "verified" and previous.value != "verified":
                newly_verified.append(slot_name)
            elif status.value == "bound" and previous.value not in {"bound", "verified"}:
                newly_bound.append(slot_name)
        return VerificationUpdate(
            slot_updates=updates,
            delta=VerifierDelta(
                newly_bound_slots=tuple(newly_bound),
                newly_verified_slots=tuple(newly_verified),
            ),
        )

