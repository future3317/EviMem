"""Replayable candidate-level curation trajectory."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ActionCost(BaseModel):
    model_config = ConfigDict(frozen=True)

    tokens: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    human_queries: int = Field(default=0, ge=0)
    wall_clock_seconds: float = Field(default=0.0, ge=0.0)


class VerifierDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    newly_bound_slots: tuple[str, ...] = ()
    newly_verified_slots: tuple[str, ...] = ()
    newly_ambiguous_slots: tuple[str, ...] = ()
    newly_conflicting_slots: tuple[str, ...] = ()
    ambiguity_reduction: int = 0
    conflict_resolution: Literal[
        "none", "distinct_context", "resolved", "unresolved"
    ] = "none"


class CurationStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int = Field(ge=0)
    state_hash: str
    action: str
    action_args: dict[str, Any] = Field(default_factory=dict)
    rationale_code: dict[str, str] = Field(default_factory=dict)
    result_digest: str
    cost: ActionCost = Field(default_factory=ActionCost)
    verifier_delta: VerifierDelta = Field(default_factory=VerifierDelta)


class CurationTrajectory(BaseModel):
    """Immutable action trace used for audit, replay and policy learning."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.trajectory.v2"

    trajectory_id: str
    run_id: str
    candidate_id: str
    evidence_release_id: str
    domain_pack_id: str
    domain_pack_version: str
    domain_pack_hash: str
    initial_state_hash: str
    steps: tuple[CurationStep, ...] = ()
    terminal_action: Literal[
        "REQUEST_PUBLICATION", "DEFER_FOR_REVIEW", "REJECT_CANDIDATE", "STOP_NO_RECORD"
    ]
    final_certificate_id: str | None = None
    total_reward: float | None = None

    @field_validator(
        "trajectory_id",
        "run_id",
        "candidate_id",
        "evidence_release_id",
        "domain_pack_id",
        "domain_pack_version",
        "domain_pack_hash",
        "initial_state_hash",
    )
    @classmethod
    def _require_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("trajectory identity fields must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_step_order(self) -> CurationTrajectory:
        if [item.step for item in self.steps] != list(range(len(self.steps))):
            raise ValueError("trajectory steps must be contiguous and zero-based")
        return self
