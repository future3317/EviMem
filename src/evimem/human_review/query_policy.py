"""Expected-value gate for the scarce ASK_HUMAN action."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.controller.state import ControllerState
from evimem.core.contracts import SlotStatus


@dataclass(frozen=True)
class ReviewPolicyDecision:
    ask_human: bool
    expected_value: float
    reason: str


class ExpectedValueReviewPolicy:
    def __init__(self, *, minimum_expected_value: float = 0.0):
        self.minimum_expected_value = minimum_expected_value

    def evaluate(
        self,
        state: ControllerState,
        *,
        recovery_probability: float,
        verified_record_value: float,
        query_cost: float,
    ) -> ReviewPolicyDecision:
        if state.claim_state.remaining_budget.human_queries < 1:
            return ReviewPolicyDecision(False, float("-inf"), "human_budget_exhausted")
        difficult = [
            name
            for name, slot in state.claim_state.slots.items()
            if slot.status in {SlotStatus.AMBIGUOUS, SlotStatus.CONFLICTING}
        ]
        if not difficult and state.claim_state.conflict_status != "unresolved":
            return ReviewPolicyDecision(False, -query_cost, "no_human_review_trigger")
        expected_value = recovery_probability * verified_record_value - query_cost
        return ReviewPolicyDecision(
            ask_human=expected_value > self.minimum_expected_value,
            expected_value=expected_value,
            reason="positive_expected_value" if expected_value > self.minimum_expected_value else "cost_exceeds_value",
        )
