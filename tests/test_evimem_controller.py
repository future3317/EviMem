from __future__ import annotations

import pytest

from evimem.controller import (
    ActionExecutor,
    ActionToolResult,
    ActionType,
    CurationAction,
    HeuristicController,
    RegisteredAction,
    SequentialCurationEngine,
    StateBuilder,
    VerificationUpdate,
)
from evimem.controller.state import EvidenceIndexEntry
from evimem.core.contracts import (
    ActionCost,
    CurationBudget,
    SlotStatus,
    VerificationSlot,
    VerifierDelta,
)

from .evimem_helpers import candidate, evidence_ref


class _Verifier:
    def verify(self, *, action, state, tool_result):
        if action.type == ActionType.REQUEST_SLOT_VERIFICATION:
            slot = action.arguments["slot_name"]
            refs = tool_result.evidence_refs or state.gathered_evidence
            return VerificationUpdate(
                slot_updates={
                    slot: VerificationSlot(status=SlotStatus.VERIFIED, evidence_refs=refs)
                },
                delta=VerifierDelta(newly_verified_slots=(slot,)),
            )
        if tool_result.evidence_refs:
            updates = {
                name: VerificationSlot(
                    status=SlotStatus.VERIFIED,
                    evidence_refs=tool_result.evidence_refs,
                )
                for name in state.claim_state.unresolved_slots
            }
            return VerificationUpdate(
                slot_updates=updates,
                delta=VerifierDelta(newly_verified_slots=tuple(updates)),
            )
        return VerificationUpdate()


def _state(tool_calls: int = 3):
    return StateBuilder.build(
        candidate=candidate(),
        required_slots=("property", "value", "unit", "material"),
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        budget=CurationBudget(
            token=1000,
            tool_calls=tool_calls,
            human_queries=0,
            wall_clock_seconds=10,
        ),
        evidence_index=(EvidenceIndexEntry(evidence_ref=evidence_ref(), kind="passage"),),
    )


def _executor(ref=None):
    returned_ref = ref or evidence_ref()
    return ActionExecutor(
        actions={
            ActionType.RETRIEVE_TABLE: RegisteredAction(
                handler=lambda action, state: ActionToolResult(evidence_refs=(returned_ref,)),
                cost=ActionCost(tool_calls=1, wall_clock_seconds=0.1),
            ),
            ActionType.RETRIEVE_PASSAGE: RegisteredAction(
                handler=lambda action, state: ActionToolResult(evidence_refs=(returned_ref,)),
                cost=ActionCost(tool_calls=1, wall_clock_seconds=0.1),
            ),
            ActionType.REQUEST_SLOT_VERIFICATION: RegisteredAction(
                handler=lambda action, state: ActionToolResult(),
                cost=ActionCost(tool_calls=1, wall_clock_seconds=0.1),
            ),
        },
        verifier=_Verifier(),
    )


def test_state_builder_marks_present_values_as_candidates() -> None:
    state = _state()
    assert all(slot.status == SlotStatus.CANDIDATE for slot in state.claim_state.slots.values())
    assert state.claim_state.unresolved_slots == ("material", "property", "unit", "value")


def test_executor_updates_state_only_from_verifier() -> None:
    state = _state()
    action = CurationAction(type=ActionType.RETRIEVE_TABLE, arguments={"query": "d33"})
    outcome = _executor().execute(action, state)
    assert not outcome.state.claim_state.unresolved_slots
    assert outcome.state.claim_state.remaining_budget.tool_calls == 2
    assert len(outcome.state.action_history) == 1


def test_executor_rejects_evidence_from_another_release() -> None:
    wrong = evidence_ref(release_id="release-2")
    with pytest.raises(ValueError, match="different release"):
        _executor(wrong).execute(
            CurationAction(type=ActionType.RETRIEVE_TABLE, arguments={"query": "d33"}),
            _state(),
        )


def test_controller_cannot_register_publication_writer_action() -> None:
    with pytest.raises(ValueError, match="terminal actions"):
        ActionExecutor(
            actions={
                ActionType.REQUEST_PUBLICATION: RegisteredAction(
                    handler=lambda action, state: ActionToolResult(),
                    cost=ActionCost(),
                )
            },
            verifier=_Verifier(),
        )


def test_sequential_engine_outputs_request_without_committing() -> None:
    outcome = SequentialCurationEngine(executor=_executor(), max_steps=4).run(
        run_id="episode-1",
        initial_state=_state(),
        policy=HeuristicController(),
    )
    assert outcome.publication_requested
    assert outcome.final_state.claim_state.terminal_status == "request_publication"
    assert outcome.trajectory.terminal_action == "REQUEST_PUBLICATION"
    assert [step.step for step in outcome.trajectory.steps] == list(
        range(len(outcome.trajectory.steps))
    )


def test_budget_exhaustion_forces_defer() -> None:
    outcome = SequentialCurationEngine(executor=_executor(), max_steps=4).run(
        run_id="episode-2",
        initial_state=_state(tool_calls=0),
        policy=HeuristicController(),
    )
    assert not outcome.publication_requested
    assert outcome.trajectory.terminal_action == "DEFER_FOR_REVIEW"
