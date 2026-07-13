"""Controller interfaces and deterministic baselines."""

from __future__ import annotations

from typing import Protocol

from .actions import ActionType, CurationAction
from .state import ControllerState


class ControllerPolicy(Protocol):
    def choose_action(
        self,
        state: ControllerState,
        legal_actions: frozenset[ActionType],
    ) -> CurationAction: ...


class HeuristicController:
    """Transparent non-learning baseline for sequential benchmarks."""

    use_memory: bool = True

    @staticmethod
    def _used(state: ControllerState) -> set[ActionType]:
        return {record.action.type for record in state.action_history}

    def choose_action(
        self,
        state: ControllerState,
        legal_actions: frozenset[ActionType],
    ) -> CurationAction:
        unresolved = state.claim_state.unresolved_slots
        used = self._used(state)
        if not unresolved:
            return CurationAction(type=ActionType.REQUEST_PUBLICATION)

        if self.use_memory and not state.memory_hints.memory_ids:
            for memory_action in (
                ActionType.RETRIEVE_REJECTED_MEMORY,
                ActionType.RETRIEVE_VERIFIED_MEMORY,
                ActionType.RETRIEVE_CONFLICT_MEMORY,
            ):
                if memory_action in legal_actions and memory_action not in used:
                    return CurationAction(type=memory_action)

        if any(slot in unresolved for slot in ("value", "unit")):
            if ActionType.RETRIEVE_TABLE in legal_actions and ActionType.RETRIEVE_TABLE not in used:
                claim = state.candidate.claim
                return CurationAction(
                    type=ActionType.RETRIEVE_TABLE,
                    arguments={"query": f"{claim.property_key} {claim.value_raw} {claim.unit_raw or ''}"},
                    rationale_code={"target_slot": "value_unit"},
                )

        target = unresolved[0]
        if state.gathered_evidence and ActionType.REQUEST_SLOT_VERIFICATION in legal_actions:
            verification_repeated = any(
                record.action.type == ActionType.REQUEST_SLOT_VERIFICATION
                and record.action.arguments.get("slot_name") == target
                for record in state.action_history
            )
            if not verification_repeated:
                return CurationAction(
                    type=ActionType.REQUEST_SLOT_VERIFICATION,
                    arguments={"slot_name": target},
                    rationale_code={"target_slot": target},
                )

        if ActionType.RETRIEVE_PASSAGE in legal_actions:
            return CurationAction(
                type=ActionType.RETRIEVE_PASSAGE,
                arguments={"query": f"{state.candidate.claim.property_key} {target}"},
                rationale_code={"target_slot": target},
            )
        if ActionType.ASK_HUMAN in legal_actions:
            return CurationAction(
                type=ActionType.ASK_HUMAN,
                arguments={
                    "slot_name": target,
                    "evidence_bundle": [ref.model_dump(mode="json") for ref in state.gathered_evidence],
                },
            )
        return CurationAction(type=ActionType.DEFER_FOR_REVIEW)


class NoMemoryController(HeuristicController):
    use_memory = False
