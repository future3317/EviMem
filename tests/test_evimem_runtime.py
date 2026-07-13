from __future__ import annotations

from evimem.contracts import (
    ActionCost,
    CurationBudget,
    SlotStatus,
    VerificationSlot,
    VerifierDelta,
)
from evimem.controller import (
    ActionExecutor,
    ActionToolResult,
    ActionType,
    HeuristicController,
    RegisteredAction,
    SequentialCurationEngine,
    StateBuilder,
    VerificationUpdate,
)
from evimem.memory import GovernedMemoryStore
from evimem.rl import TrajectoryReplayBuffer
from evimem.runtime import EviMemRuntime

from .evimem_helpers import candidate, certificate, evidence_ref


class _Verifier:
    def verify(self, *, action, state, tool_result):
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


class _Harness:
    def certify(self, *, outcome):
        return certificate()


def test_runtime_records_reward_and_consolidates_without_committing(tmp_path) -> None:
    executor = ActionExecutor(
        actions={
            ActionType.RETRIEVE_TABLE: RegisteredAction(
                handler=lambda action, state: ActionToolResult(evidence_refs=(evidence_ref(),)),
                cost=ActionCost(tool_calls=1),
            )
        },
        verifier=_Verifier(),
    )
    state = StateBuilder.build(
        candidate=candidate(),
        required_slots=("property", "value", "unit", "material"),
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        budget=CurationBudget(tool_calls=2, wall_clock_seconds=10),
    )
    replay = TrajectoryReplayBuffer(tmp_path / "replay.sqlite")
    memory = GovernedMemoryStore(tmp_path / "memory.sqlite")
    result = EviMemRuntime(
        engine=SequentialCurationEngine(executor=executor, max_steps=3),
        harness=_Harness(),
        replay_buffer=replay,
        memory_store=memory,
    ).run_candidate(
        run_id="run-1",
        initial_state=state,
        policy=HeuristicController(),
        domain="piezoelectric",
        material_family="PZT",
    )
    assert result.publication_authorized
    assert result.trajectory.total_reward is not None
    assert replay.get(result.trajectory.trajectory_id) == result.trajectory
    assert result.memory_id is not None
    assert memory.get(result.memory_id) is not None
