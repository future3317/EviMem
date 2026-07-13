"""Fail-closed decoding for a supervised typed memory manager."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from evimem.contracts import (
    AdmissionAction,
    MemoryManagerAction,
    ScientificMemoryRecord,
    UpdateOperation,
)


class ManagerDecodingError(ValueError):
    pass


class TextGenerator(Protocol):
    def generate(self, prompt: str) -> str: ...


class ManagerActionCodec:
    @staticmethod
    def decode(raw: str) -> MemoryManagerAction:
        try:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise TypeError("manager output must be a JSON object")
            return MemoryManagerAction.model_validate(payload)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ManagerDecodingError(str(exc)) from exc


@dataclass(frozen=True)
class ManagerInput:
    current_record: ScientificMemoryRecord
    retrieved_memories: tuple[ScientificMemoryRecord, ...] = ()

    def render(self) -> str:
        return json.dumps(
            {
                "instruction": "Return exactly one JSON MemoryManagerAction.",
                "current_record": self.current_record.model_dump(mode="json"),
                "retrieved_memories": [
                    record.model_dump(mode="json") for record in self.retrieved_memories
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )


class StructuredMemoryManager:
    """Adapter that can emit only typed actions and fails closed on invalid output."""

    def __init__(self, generator: TextGenerator):
        self.generator = generator
        self.last_decode_error: str | None = None

    def decide(self, manager_input: ManagerInput) -> MemoryManagerAction:
        raw = self.generator.generate(manager_input.render())
        try:
            action = ManagerActionCodec.decode(raw)
            self.last_decode_error = None
            return action
        except ManagerDecodingError as exc:
            self.last_decode_error = str(exc)
            return MemoryManagerAction(
                admission=AdmissionAction.EPHEMERAL_ONLY,
                update_operation=UpdateOperation.IGNORE,
                reason_code="invalid_model_output",
            )
