"""TRL reward adapters backed only by external deterministic verifier outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evimem.controller import ActionType, CurationAction

from .codec import ActionCodec, ActionDecodingError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from .dataset import OracleActionExample


class ActionRewardOracle(Protocol):
    """Authority that scores an action from deterministic cached or live verification."""

    def score(self, example_id: str, action: CurationAction) -> float: ...


class CertifiedRewardRecord(BaseModel):
    """Policy-hidden rewards produced by an external verifier harness."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.certified_action_rewards.v1"

    example_id: str
    candidate_id: str
    evidence_release_id: str
    domain_pack_id: str
    domain_pack_version: str
    domain_pack_hash: str
    verifier_version: str
    action_rewards: dict[str, float] = Field(min_length=1)
    artifact_hash: str

    @staticmethod
    def _compute_hash(payload: dict[str, Any]) -> str:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def create(
        cls,
        *,
        example: OracleActionExample,
        action_rewards: dict[str, float],
        verifier_version: str,
    ) -> CertifiedRewardRecord:
        state = example.state
        payload = {
            "example_id": example.example_id,
            "candidate_id": state.candidate.candidate_id,
            "evidence_release_id": state.claim_state.evidence_release_id,
            "domain_pack_id": state.claim_state.domain_pack_id,
            "domain_pack_version": state.claim_state.domain_pack_version,
            "domain_pack_hash": state.claim_state.domain_pack_hash,
            "verifier_version": verifier_version,
            "action_rewards": action_rewards,
        }
        return cls(**payload, artifact_hash=cls._compute_hash(payload))

    @model_validator(mode="after")
    def _validate_integrity(self) -> CertifiedRewardRecord:
        payload = self.model_dump(mode="json", exclude={"artifact_hash"})
        if self.artifact_hash != self._compute_hash(payload):
            raise ValueError("certified reward artifact hash mismatch")
        if any(not math.isfinite(reward) for reward in self.action_rewards.values()):
            raise ValueError("certified rewards must be finite")
        return self


class CachedVerifierOracle:
    def __init__(
        self,
        records: list[CertifiedRewardRecord],
        *,
        unsupported_action_penalty: float = -2.0,
        examples: Iterable[OracleActionExample] | None = None,
    ) -> None:
        self._records = {record.example_id: record for record in records}
        if len(self._records) != len(records):
            raise ValueError("duplicate certified reward example_id")
        self.unsupported_action_penalty = unsupported_action_penalty
        if examples is not None:
            self.validate_examples(examples)

    def validate_examples(self, examples: Iterable[OracleActionExample]) -> None:
        for example in examples:
            record = self._records.get(example.example_id)
            if record is None:
                raise ValueError(f"certified rewards missing for example {example.example_id}")
            state = example.state
            expected = (
                state.candidate.candidate_id,
                state.claim_state.evidence_release_id,
                state.claim_state.domain_pack_id,
                state.claim_state.domain_pack_version,
                state.claim_state.domain_pack_hash,
            )
            actual = (
                record.candidate_id,
                record.evidence_release_id,
                record.domain_pack_id,
                record.domain_pack_version,
                record.domain_pack_hash,
            )
            if actual != expected:
                raise ValueError(f"certified reward identity mismatch for {example.example_id}")
            legal = frozenset(example.legal_actions)
            for action_key in record.action_rewards:
                try:
                    raw = json.loads(action_key)
                    action = CurationAction(
                        type=ActionType(raw["type"]),
                        arguments=raw.get("arguments", {}),
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid certified action key for {example.example_id}"
                    ) from exc
                if action.type not in legal:
                    raise ValueError(
                        f"certified reward contains illegal action for {example.example_id}"
                    )

    def score(self, example_id: str, action: CurationAction) -> float:
        record = self._records.get(example_id)
        if record is None:
            raise KeyError(f"no certified rewards for example {example_id}")
        return float(
            record.action_rewards.get(
                ActionCodec.canonical_key(action),
                self.unsupported_action_penalty,
            )
        )


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict) and "content" in last:
            return str(last["content"])
    if isinstance(completion, dict) and "content" in completion:
        return str(completion["content"])
    return str(completion)


class VerifierRewardAdapter:
    """Callable with the reward-function protocol expected by TRL GRPOTrainer."""

    def __init__(
        self,
        oracle: ActionRewardOracle,
        *,
        invalid_action_penalty: float = -3.0,
    ) -> None:
        self.oracle = oracle
        self.invalid_action_penalty = invalid_action_penalty

    def __call__(
        self,
        completions: list[Any],
        example_id: list[str],
        legal_actions: list[list[str]],
        **_: Any,
    ) -> list[float]:
        if not (len(completions) == len(example_id) == len(legal_actions)):
            raise ValueError("TRL reward columns have inconsistent lengths")
        rewards: list[float] = []
        for completion, item_id, legal in zip(completions, example_id, legal_actions):
            try:
                action = ActionCodec.decode(
                    _completion_text(completion),
                    legal_actions=(ActionType(value) for value in legal),
                )
                rewards.append(float(self.oracle.score(item_id, action)))
            except (ActionDecodingError, KeyError, ValueError):
                rewards.append(self.invalid_action_penalty)
        return rewards


def load_reward_records_jsonl(path: str | Path) -> list[CertifiedRewardRecord]:
    records: list[CertifiedRewardRecord] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(CertifiedRewardRecord.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid reward record at line {line_number}: {exc}") from exc
    return records


def write_reward_records_jsonl(
    path: str | Path,
    records: list[CertifiedRewardRecord],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
