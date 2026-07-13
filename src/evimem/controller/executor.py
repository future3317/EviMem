"""Single deterministic execution boundary for every controller action."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from evimem.core.contracts import (
    ActionCost,
    ClaimState,
    EvidenceRef,
    VerificationSlot,
    VerifierDelta,
)
from evimem.memory.retriever import MemoryHints

from .actions import TERMINAL_ACTIONS, ActionType, CurationAction
from .state import ActionRecord, ControllerState


class ActionToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    payload: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: tuple[EvidenceRef, ...] = ()
    memory_hints: MemoryHints | None = None


class VerificationUpdate(BaseModel):
    model_config = ConfigDict(frozen=True)

    slot_updates: dict[str, VerificationSlot] = Field(default_factory=dict)
    conflict_status: str | None = None
    delta: VerifierDelta = Field(default_factory=VerifierDelta)


class DeterministicVerifier(Protocol):
    def verify(
        self,
        *,
        action: CurationAction,
        state: ControllerState,
        tool_result: ActionToolResult,
    ) -> VerificationUpdate: ...


@dataclass(frozen=True)
class RegisteredAction:
    handler: Callable[[CurationAction, ControllerState], ActionToolResult]
    cost: ActionCost


@dataclass(frozen=True)
class ExecutionOutcome:
    state: ControllerState
    action_record: ActionRecord
    tool_result: ActionToolResult


class ActionExecutor:
    """Executes registered tools; it has no publication-store dependency."""

    def __init__(
        self,
        *,
        actions: dict[ActionType, RegisteredAction],
        verifier: DeterministicVerifier,
    ) -> None:
        if any(action in TERMINAL_ACTIONS for action in actions):
            raise ValueError("terminal actions are executor-owned and cannot be registered")
        self._actions = dict(actions)
        self._verifier = verifier

    def legal_actions(self, state: ControllerState) -> frozenset[ActionType]:
        if state.claim_state.terminal_status != "active":
            return frozenset()
        budget = state.claim_state.remaining_budget
        legal = set(TERMINAL_ACTIONS)
        for action_type, registered in self._actions.items():
            cost = registered.cost
            if (
                cost.tokens <= budget.token
                and cost.tool_calls <= budget.tool_calls
                and cost.human_queries <= budget.human_queries
                and cost.wall_clock_seconds <= budget.wall_clock_seconds
            ):
                legal.add(action_type)
        return frozenset(legal)

    @staticmethod
    def _terminal_status(action_type: ActionType) -> str:
        return {
            ActionType.REQUEST_PUBLICATION: "request_publication",
            ActionType.DEFER_FOR_REVIEW: "deferred",
            ActionType.REJECT_CANDIDATE: "rejected",
            ActionType.STOP_NO_RECORD: "stopped",
        }[action_type]

    @staticmethod
    def _digest(tool_result: ActionToolResult) -> str:
        payload = json.dumps(
            tool_result.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def execute(self, action: CurationAction, state: ControllerState) -> ExecutionOutcome:
        if action.type not in self.legal_actions(state):
            raise ValueError(f"illegal or unaffordable action: {action.type.value}")

        if action.type in TERMINAL_ACTIONS:
            cost = ActionCost()
            tool_result = ActionToolResult(payload={"terminal_action": action.type.value})
            update = VerificationUpdate()
            next_claim_state = state.claim_state.model_copy(
                update={"terminal_status": self._terminal_status(action.type)}
            )
        else:
            registered = self._actions[action.type]
            cost = registered.cost
            tool_result = registered.handler(action, state)
            for ref in tool_result.evidence_refs:
                if ref.release_id != state.claim_state.evidence_release_id:
                    raise ValueError("tool returned evidence from a different release")
            update = self._verifier.verify(
                action=action,
                state=state,
                tool_result=tool_result,
            )
            slots = dict(state.claim_state.slots)
            unknown_slots = set(update.slot_updates) - set(slots)
            if unknown_slots:
                raise ValueError(f"verifier returned unknown slots: {sorted(unknown_slots)}")
            slots.update(update.slot_updates)
            budget = state.claim_state.remaining_budget.consume(
                token=cost.tokens,
                tool_calls=cost.tool_calls,
                human_queries=cost.human_queries,
                wall_clock_seconds=cost.wall_clock_seconds,
            )
            claim_updates: dict[str, Any] = {"slots": slots, "remaining_budget": budget}
            if update.conflict_status is not None:
                claim_updates["conflict_status"] = update.conflict_status
            next_claim_state = ClaimState(**{
                **state.claim_state.model_dump(mode="python"),
                **claim_updates,
                "unresolved_slots": (),
            })

        record = ActionRecord(
            action=action,
            result_digest=self._digest(tool_result),
            cost=cost,
            verifier_delta=update.delta,
        )
        gathered = tuple(
            {
                (ref.release_id, ref.block_id, ref.checksum): ref
                for ref in (*state.gathered_evidence, *tool_result.evidence_refs)
            }.values()
        )
        next_state = ControllerState(
            candidate=state.candidate,
            claim_state=next_claim_state,
            evidence_index=state.evidence_index,
            memory_hints=tool_result.memory_hints or state.memory_hints,
            domain_requirements=state.domain_requirements,
            gathered_evidence=gathered,
            action_history=(*state.action_history, record),
        )
        return ExecutionOutcome(state=next_state, action_record=record, tool_result=tool_result)
