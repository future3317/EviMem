"""Supervised examples for retrieval and typed memory management."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evimem.contracts import MemoryManagerAction, ScientificMemoryRecord


class RetrievalTrainingExample(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.retrieval_training.v1"

    example_id: str
    dataset_name: str
    split: Literal["train", "validation", "test"]
    query: str
    positive_memory_ids: tuple[str, ...] = Field(min_length=1)
    hard_negative_memory_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _disjoint_labels(self) -> RetrievalTrainingExample:
        overlap = set(self.positive_memory_ids) & set(self.hard_negative_memory_ids)
        if overlap:
            raise ValueError(f"positive and negative memory IDs overlap: {sorted(overlap)}")
        return self


class ManagerTrainingExample(BaseModel):
    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.manager_training.v1"

    example_id: str
    dataset_name: str
    split: Literal["train", "validation", "test"]
    current_record: ScientificMemoryRecord
    retrieved_memories: tuple[ScientificMemoryRecord, ...] = ()
    target: MemoryManagerAction

    def prompt_record(self) -> dict[str, str]:
        visible = {
            "current_record": self.current_record.model_dump(mode="json"),
            "retrieved_memories": [
                record.model_dump(mode="json") for record in self.retrieved_memories
            ],
            "instruction": (
                "Return exactly one JSON MemoryManagerAction. Evidence and certificate "
                "requirements are hard constraints."
            ),
        }
        return {
            "prompt": json.dumps(visible, ensure_ascii=False, sort_keys=True),
            "completion": self.target.model_dump_json(),
        }


def require_official_training_splits(
    examples: Iterable[RetrievalTrainingExample | ManagerTrainingExample],
) -> None:
    invalid = [item.example_id for item in examples if item.split not in {"train", "validation"}]
    if invalid:
        raise ValueError(f"test examples cannot enter optimization: {sorted(invalid)}")


def write_examples_jsonl(
    path: str | Path,
    examples: Iterable[RetrievalTrainingExample | ManagerTrainingExample],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for example in examples:
            stream.write(example.model_dump_json() + "\n")


def load_manager_examples_jsonl(path: str | Path) -> list[ManagerTrainingExample]:
    examples: list[ManagerTrainingExample] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                examples.append(ManagerTrainingExample.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid manager example at line {line_number}: {exc}") from exc
    return examples
