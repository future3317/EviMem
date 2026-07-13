"""Strict structured-action encoding and policy-visible prompt rendering."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evimem.controller import ActionType, CurationAction
from evimem.controller.state import ControllerState


class ActionDecodingError(ValueError):
    """Raised when model output is not one valid, legal structured action."""


class _ActionBody(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: ActionType
    arguments: dict[str, Any] = Field(default_factory=dict)


class _ActionEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: _ActionBody
    rationale_code: dict[str, str] = Field(default_factory=dict)


class ActionCodec:
    """Canonical JSON protocol between a language model and the executor."""

    @staticmethod
    def encode(action: CurationAction) -> str:
        envelope = _ActionEnvelope(
            action=_ActionBody(type=action.type, arguments=action.arguments),
            rationale_code=action.rationale_code,
        )
        return json.dumps(
            envelope.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def canonical_key(action: CurationAction) -> str:
        """Return the rationale-independent identity used by reward oracles."""

        return json.dumps(
            {
                "arguments": action.arguments,
                "type": action.type.value,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _unwrap_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith("```"):
            return stripped
        lines = stripped.splitlines()
        if len(lines) < 3 or lines[-1].strip() != "```":
            raise ActionDecodingError("unterminated JSON code fence")
        if lines[0].strip().lower() not in {"```", "```json"}:
            raise ActionDecodingError("only a JSON code fence is accepted")
        return "\n".join(lines[1:-1]).strip()

    @classmethod
    def decode(
        cls,
        text: str,
        *,
        legal_actions: Iterable[ActionType] | None = None,
    ) -> CurationAction:
        payload = cls._unwrap_fence(text)
        try:
            raw = json.loads(payload)
            envelope = _ActionEnvelope.model_validate(raw)
            action = CurationAction(
                type=envelope.action.type,
                arguments=envelope.action.arguments,
                rationale_code=envelope.rationale_code,
            )
        except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
            raise ActionDecodingError(f"invalid action JSON: {exc}") from exc
        if legal_actions is not None and action.type not in frozenset(legal_actions):
            raise ActionDecodingError(f"illegal action for current state: {action.type.value}")
        return action

    @staticmethod
    def json_schema() -> dict[str, Any]:
        return _ActionEnvelope.model_json_schema()


class PolicyPromptRenderer:
    """Render only the state that is explicitly visible to the controller."""

    SYSTEM_PROMPT = (
        "You are the EviMem scientific-curation controller. Select exactly one "
        "legal action that best improves deterministic verification within the "
        "remaining budget. Return one JSON object matching the supplied schema. "
        "Do not claim that a fact is verified, do not write publication data, "
        "and do not include prose outside the JSON object."
    )

    @classmethod
    def render_messages(
        cls,
        state: ControllerState,
        legal_actions: Iterable[ActionType],
    ) -> list[dict[str, str]]:
        legal = sorted(action.value for action in frozenset(legal_actions))
        policy_state = state.model_dump(mode="json", exclude_none=False)
        user_payload = {
            "action_schema": ActionCodec.json_schema(),
            "legal_actions": legal,
            "policy_visible_state": policy_state,
            "task": "Select the next curation action.",
        }
        return [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    user_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]

