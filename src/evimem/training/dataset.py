"""Leakage-safe next-action datasets compiled from replayable trajectories."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from evimem.controller import ActionExecutor, ActionType, CurationAction
from evimem.controller.state import ControllerState
from evimem.core.contracts import CurationTrajectory
from evimem.core.ids import deterministic_id

from .codec import ActionCodec, PolicyPromptRenderer

TrajectorySource = Literal[
    "heuristic",
    "oracle_evidence_path",
    "human",
    "successful_run",
    "counterfactual_repair",
]


class OracleActionExample(BaseModel):
    """One policy-visible state paired with an expert next action."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    episode_id: str
    document_id: str
    state: ControllerState
    legal_actions: tuple[ActionType, ...]
    target_action: CurationAction
    source: TrajectorySource

    @model_validator(mode="after")
    def _target_must_be_legal(self) -> OracleActionExample:
        if self.target_action.type not in self.legal_actions:
            raise ValueError("target action is not legal in the recorded state")
        if not self.document_id.strip():
            raise ValueError("document_id is required for leakage-safe splitting")
        if self.document_id != self.state.candidate.doi:
            raise ValueError("document_id must match the policy-visible candidate DOI")
        return self

    def sft_record(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "document_id": self.document_id,
            "prompt": PolicyPromptRenderer.render_messages(self.state, self.legal_actions),
            "completion": [
                {"role": "assistant", "content": ActionCodec.encode(self.target_action)}
            ],
        }

    def grpo_record(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "episode_id": self.episode_id,
            "document_id": self.document_id,
            "legal_actions": [action.value for action in self.legal_actions],
            "prompt": PolicyPromptRenderer.render_messages(self.state, self.legal_actions),
        }


class OracleTrajectoryCompiler:
    """Replay a deterministic trajectory and recover state/action supervision."""

    @staticmethod
    def compile(
        *,
        initial_state: ControllerState,
        trajectory: CurationTrajectory,
        executor: ActionExecutor,
        source: TrajectorySource,
    ) -> list[OracleActionExample]:
        if trajectory.candidate_id != initial_state.candidate.candidate_id:
            raise ValueError("trajectory and initial state candidates differ")
        if trajectory.evidence_release_id != initial_state.claim_state.evidence_release_id:
            raise ValueError("trajectory and initial state releases differ")

        state = initial_state
        examples: list[OracleActionExample] = []
        for step in trajectory.steps:
            if state.state_hash() != step.state_hash:
                raise ValueError(f"trajectory state hash mismatch at step {step.step}")
            action = CurationAction(
                type=ActionType(step.action),
                arguments=step.action_args,
                rationale_code=step.rationale_code,
            )
            legal = tuple(sorted(executor.legal_actions(state), key=lambda item: item.value))
            examples.append(
                OracleActionExample(
                    example_id=deterministic_id(
                        trajectory.trajectory_id,
                        step.step,
                        step.state_hash,
                        namespace="oracle-step",
                        length=32,
                    ),
                    episode_id=trajectory.run_id,
                    document_id=initial_state.candidate.doi,
                    state=state,
                    legal_actions=legal,
                    target_action=action,
                    source=source,
                )
            )
            outcome = executor.execute(action, state)
            if outcome.action_record.result_digest != step.result_digest:
                raise ValueError(f"trajectory result mismatch at step {step.step}")
            state = outcome.state
        return examples


def split_examples_by_document(
    examples: Iterable[OracleActionExample],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict[str, list[OracleActionExample]]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave a test split")

    splits: dict[str, list[OracleActionExample]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for example in examples:
        digest = hashlib.sha256(f"{seed}|{example.document_id}".encode()).digest()
        fraction = int.from_bytes(digest[:8], "big") / float(2**64)
        if fraction < train_fraction:
            split = "train"
        elif fraction < train_fraction + validation_fraction:
            split = "validation"
        else:
            split = "test"
        splits[split].append(example)
    return splits


def build_sft_dataset(examples: Iterable[OracleActionExample]):
    from datasets import Dataset

    return Dataset.from_list([example.sft_record() for example in examples])


def build_grpo_dataset(examples: Iterable[OracleActionExample]):
    from datasets import Dataset

    return Dataset.from_list([example.grpo_record() for example in examples])


def write_examples_jsonl(path: str | Path, examples: Iterable[OracleActionExample]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(example.model_dump_json() + "\n")


def load_examples_jsonl(path: str | Path) -> list[OracleActionExample]:
    examples: list[OracleActionExample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(OracleActionExample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid oracle example at line {line_number}: {exc}") from exc
    return examples
