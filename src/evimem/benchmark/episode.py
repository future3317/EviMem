"""Inference-visible episode data and separately held oracle annotations."""

from __future__ import annotations

from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from evimem.contracts import CurationTrajectory, EvidenceRef, VerificationCertificate
from evimem.controller.state import ControllerState


class HardCaseType(StrEnum):
    SINGLE_BLOCK = "single_block"
    TABLE_ONLY = "table_only"
    CAPTION_ONLY = "caption_only"
    DISTRIBUTED_EVIDENCE = "distributed_evidence"
    MULTIPLE_MATERIALS = "same_value_multiple_materials"
    MULTIPLE_CONDITIONS = "same_material_multiple_conditions"
    CROSS_PAPER_CORRECTION = "cross_paper_correction"
    CONFLICTING_MEASUREMENTS = "conflicting_measurements"
    OBSOLETE_POLICY = "obsolete_policy"
    NEGATIVE_CONTROL = "negative_control"
    REVIEW_OR_PREDICTION = "review_or_prediction"
    UNSUPPORTED_MEMORY = "unsupported_memory"
    MISSING_CONDITION = "missing_condition"
    UNIT_CONVERSION = "unit_conversion"
    MATERIAL_ALIAS_DRIFT = "material_alias_drift"


class BenchmarkEpisode(BaseModel):
    """The complete state visible to an inference policy."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.benchmark_episode.v1"

    episode_id: str
    stream_position: int = Field(ge=0)
    initial_state: ControllerState
    hard_cases: tuple[HardCaseType, ...] = ()
    negative_control: bool = False


class OracleAnnotation(BaseModel):
    """Gold data held outside the policy-facing episode object."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    expected_terminal_action: str
    gold_evidence_refs: tuple[EvidenceRef, ...] = ()
    gold_certificate_id: str | None = None


class EpisodeEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    controller_name: str
    episode_id: str
    negative_control: bool
    trajectory: CurationTrajectory
    certificate: VerificationCertificate
    terminal_action_correct: bool | None = None
    gold_evidence_hit: bool | None = None
