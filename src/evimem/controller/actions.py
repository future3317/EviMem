"""Small, typed action space for evidence acquisition and deferral."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(StrEnum):
    RETRIEVE_PASSAGE = "RETRIEVE_PASSAGE"
    RETRIEVE_TABLE = "RETRIEVE_TABLE"
    INSPECT_TABLE_CELL = "INSPECT_TABLE_CELL"
    INSPECT_CAPTION = "INSPECT_CAPTION"
    EXPAND_LOCAL_WINDOW = "EXPAND_LOCAL_WINDOW"
    FOLLOW_CROSS_REFERENCE = "FOLLOW_CROSS_REFERENCE"
    RETRIEVE_VERIFIED_MEMORY = "RETRIEVE_VERIFIED_MEMORY"
    RETRIEVE_REJECTED_MEMORY = "RETRIEVE_REJECTED_MEMORY"
    RETRIEVE_CONFLICT_MEMORY = "RETRIEVE_CONFLICT_MEMORY"
    CHECK_POLICY_HISTORY = "CHECK_POLICY_HISTORY"
    REQUEST_SLOT_VERIFICATION = "REQUEST_SLOT_VERIFICATION"
    REQUEST_CONFLICT_CHECK = "REQUEST_CONFLICT_CHECK"
    REQUEST_DOMAIN_VALIDATION = "REQUEST_DOMAIN_VALIDATION"
    ASK_HUMAN = "ASK_HUMAN"
    REQUEST_PUBLICATION = "REQUEST_PUBLICATION"
    DEFER_FOR_REVIEW = "DEFER_FOR_REVIEW"
    REJECT_CANDIDATE = "REJECT_CANDIDATE"
    STOP_NO_RECORD = "STOP_NO_RECORD"


EVIDENCE_ACTIONS = frozenset(
    {
        ActionType.RETRIEVE_PASSAGE,
        ActionType.RETRIEVE_TABLE,
        ActionType.INSPECT_TABLE_CELL,
        ActionType.INSPECT_CAPTION,
        ActionType.EXPAND_LOCAL_WINDOW,
        ActionType.FOLLOW_CROSS_REFERENCE,
    }
)
MEMORY_ACTIONS = frozenset(
    {
        ActionType.RETRIEVE_VERIFIED_MEMORY,
        ActionType.RETRIEVE_REJECTED_MEMORY,
        ActionType.RETRIEVE_CONFLICT_MEMORY,
        ActionType.CHECK_POLICY_HISTORY,
    }
)
VERIFICATION_ACTIONS = frozenset(
    {
        ActionType.REQUEST_SLOT_VERIFICATION,
        ActionType.REQUEST_CONFLICT_CHECK,
        ActionType.REQUEST_DOMAIN_VALIDATION,
    }
)
TERMINAL_ACTIONS = frozenset(
    {
        ActionType.REQUEST_PUBLICATION,
        ActionType.DEFER_FOR_REVIEW,
        ActionType.REJECT_CANDIDATE,
        ActionType.STOP_NO_RECORD,
    }
)


_REQUIRED_ARGUMENTS: dict[ActionType, tuple[str, ...]] = {
    ActionType.RETRIEVE_PASSAGE: ("query",),
    ActionType.RETRIEVE_TABLE: ("query",),
    ActionType.INSPECT_TABLE_CELL: ("table_id", "row", "column"),
    ActionType.INSPECT_CAPTION: ("reference_id",),
    ActionType.EXPAND_LOCAL_WINDOW: ("evidence_ref", "radius"),
    ActionType.FOLLOW_CROSS_REFERENCE: ("reference_id",),
    ActionType.REQUEST_SLOT_VERIFICATION: ("slot_name",),
    ActionType.ASK_HUMAN: ("slot_name", "evidence_bundle"),
}


class CurationAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    type: ActionType
    arguments: dict[str, Any] = Field(default_factory=dict)
    rationale_code: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_arguments(self) -> CurationAction:
        missing = [
            name
            for name in _REQUIRED_ARGUMENTS.get(self.type, ())
            if name not in self.arguments
        ]
        if missing:
            raise ValueError(f"{self.type.value} missing arguments: {', '.join(missing)}")
        return self

    @property
    def is_terminal(self) -> bool:
        return self.type in TERMINAL_ACTIONS
