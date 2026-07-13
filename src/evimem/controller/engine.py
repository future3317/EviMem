"""Sequential episode engine; returns publication requests, never commits."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.core.contracts import CurationTrajectory
from evimem.rl.trajectory import TrajectoryRecorder

from .actions import ActionType, CurationAction
from .executor import ActionExecutor
from .policies import ControllerPolicy
from .state import ControllerState
from .termination import evaluate_termination


@dataclass(frozen=True)
class EpisodeOutcome:
    final_state: ControllerState
    trajectory: CurationTrajectory
    publication_requested: bool


class SequentialCurationEngine:
    def __init__(self, *, executor: ActionExecutor, max_steps: int = 12):
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        self.executor = executor
        self.max_steps = max_steps

    def run(
        self,
        *,
        run_id: str,
        initial_state: ControllerState,
        policy: ControllerPolicy,
    ) -> EpisodeOutcome:
        state = initial_state
        recorder = TrajectoryRecorder(run_id=run_id, initial_state=initial_state)
        while state.claim_state.terminal_status == "active":
            termination = evaluate_termination(state, max_steps=self.max_steps)
            if termination.terminal:
                action = CurationAction(type=ActionType.DEFER_FOR_REVIEW)
            else:
                action = policy.choose_action(state, self.executor.legal_actions(state))
            prior = state
            outcome = self.executor.execute(action, state)
            state = outcome.state
            recorder.append(prior_state=prior, record=outcome.action_record)

        terminal_action = state.action_history[-1].action.type.value
        trajectory = recorder.build(terminal_action=terminal_action)
        return EpisodeOutcome(
            final_state=state,
            trajectory=trajectory,
            publication_requested=terminal_action == ActionType.REQUEST_PUBLICATION.value,
        )
