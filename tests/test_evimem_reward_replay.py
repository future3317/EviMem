from __future__ import annotations

from evimem.contracts import ActionCost, CurationStep, CurationTrajectory, VerifierDelta
from evimem.rl import TrajectoryReplayBuffer, VerifierShapedReward

from .evimem_helpers import certificate


def _trajectory() -> CurationTrajectory:
    step = CurationStep(
        step=0,
        state_hash="state",
        action="RETRIEVE_TABLE",
        action_args={"query": "d33"},
        result_digest="digest",
        cost=ActionCost(tool_calls=1, tokens=100),
        verifier_delta=VerifierDelta(newly_verified_slots=("value", "unit")),
    )
    return CurationTrajectory(
        trajectory_id="trajectory-1",
        run_id="run-1",
        candidate_id="candidate-1",
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        initial_state_hash="initial",
        steps=(step,),
        terminal_action="REQUEST_PUBLICATION",
        final_certificate_id="certificate-1",
    )


def test_reward_uses_verifier_delta_and_certificate() -> None:
    result = VerifierShapedReward().compute(_trajectory(), certificate())
    assert result.step_reward > 0
    assert result.terminal_reward > 0
    assert result.total_reward == result.step_reward + result.terminal_reward


def test_replay_buffer_is_append_only_and_roundtrips(tmp_path) -> None:
    buffer = TrajectoryReplayBuffer(tmp_path / "replay.sqlite")
    trajectory = _trajectory()
    assert buffer.append(trajectory)
    assert not buffer.append(trajectory)
    assert buffer.get(trajectory.trajectory_id) == trajectory
    assert buffer.list_ids(evidence_release_id="release-1") == ["trajectory-1"]
