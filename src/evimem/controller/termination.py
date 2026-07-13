"""Deterministic episode termination checks."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.contracts import SlotStatus

from .state import ControllerState


@dataclass(frozen=True)
class TerminationDecision:
    terminal: bool
    reason: str


def evaluate_termination(state: ControllerState, *, max_steps: int) -> TerminationDecision:
    if state.claim_state.terminal_status != "active":
        return TerminationDecision(True, state.claim_state.terminal_status)
    if len(state.action_history) >= max_steps:
        return TerminationDecision(True, "max_steps")
    budget = state.claim_state.remaining_budget
    if budget.tool_calls == 0 and state.claim_state.unresolved_slots:
        return TerminationDecision(True, "tool_budget_exhausted")
    if any(slot.status == SlotStatus.INVALID for slot in state.claim_state.slots.values()):
        return TerminationDecision(True, "invalid_candidate")
    if state.claim_state.conflict_status == "unresolved":
        return TerminationDecision(True, "unresolved_conflict")
    return TerminationDecision(False, "active")
