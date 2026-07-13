from __future__ import annotations

from evimem.benchmark import (
    BenchmarkEpisode,
    OracleAnnotation,
    SequentialBenchmarkRunner,
    build_episode_stream,
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
from evimem.core.contracts import (
    ActionCost,
    CurationBudget,
    SlotStatus,
    VerificationSlot,
    VerifierDelta,
)

from .evimem_helpers import candidate, certificate, evidence_ref


class _Verifier:
    def verify(self, *, action, state, tool_result):
        if not tool_result.evidence_refs:
            return VerificationUpdate()
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


def _episode(episode_id: str, position: int) -> BenchmarkEpisode:
    state = StateBuilder.build(
        candidate=candidate().model_copy(update={"candidate_id": f"candidate-{episode_id}"}),
        required_slots=("property", "value", "unit", "material"),
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        budget=CurationBudget(tool_calls=2, wall_clock_seconds=10),
    )
    return BenchmarkEpisode(
        episode_id=episode_id,
        stream_position=position,
        initial_state=state,
    )


def _runner() -> SequentialBenchmarkRunner:
    executor = ActionExecutor(
        actions={
            ActionType.RETRIEVE_TABLE: RegisteredAction(
                handler=lambda action, state: ActionToolResult(evidence_refs=(evidence_ref(),)),
                cost=ActionCost(tool_calls=1),
            ),
            ActionType.RETRIEVE_PASSAGE: RegisteredAction(
                handler=lambda action, state: ActionToolResult(evidence_refs=(evidence_ref(),)),
                cost=ActionCost(tool_calls=1),
            ),
        },
        verifier=_Verifier(),
    )
    return SequentialBenchmarkRunner(
        engine=SequentialCurationEngine(executor=executor, max_steps=3)
    )


def test_stream_builder_orders_without_oracle_payload() -> None:
    stream = build_episode_stream([_episode("b", 1), _episode("a", 0)])
    assert [episode.episode_id for episode in stream] == ["a", "b"]
    assert "gold" not in BenchmarkEpisode.model_fields


def test_runner_keeps_oracle_out_of_policy_state() -> None:
    episode = _episode("one", 0)
    cert = certificate().model_copy(update={"candidate_id": "candidate-one"})
    runs = _runner().run(
        episodes=[episode],
        controllers={"heuristic": HeuristicController()},
        certificate_evaluator=lambda episode_id, outcome: cert,
        oracle_annotations={
            "one": OracleAnnotation(
                episode_id="one",
                expected_terminal_action="REQUEST_PUBLICATION",
                gold_evidence_refs=(evidence_ref(),),
            )
        },
    )
    run = runs["heuristic"]
    assert run.metrics.episode_count == 1
    assert run.metrics.verified_strong_count == 1
    assert run.metrics.terminal_action_accuracy == 1.0
    assert run.metrics.gold_evidence_hit_rate == 1.0
