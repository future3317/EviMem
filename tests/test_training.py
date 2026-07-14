from __future__ import annotations

import json

import pytest

from evimem.contracts import (
    AdmissionAction,
    AuthorityRelation,
    EvidenceSufficiency,
    MemoryManagerAction,
    ScopeRelation,
    SemanticRelation,
)
from evimem.training import (
    ManagerActionCodec,
    ManagerInput,
    ManagerTrainingExample,
    RetrievalTrainingExample,
    StructuredMemoryManager,
    require_official_training_splits,
)

from .evimem_helpers import memory_record


class StaticGenerator:
    def __init__(self, output: str):
        self.output = output

    def generate(self, prompt: str) -> str:
        assert "certificate" in prompt
        return self.output


def test_manager_codec_accepts_only_typed_json() -> None:
    action = MemoryManagerAction(
        admission=AdmissionAction.WRITE_VERIFIED,
        semantic_relation=SemanticRelation.UNRELATED,
        scope_relation=ScopeRelation.DIFFERENT_SCOPE,
        authority_relation=AuthorityRelation.NOT_APPLICABLE,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        reason_code="new_verified_claim",
    )
    assert ManagerActionCodec.decode(action.model_dump_json()) == action
    with pytest.raises(ValueError):
        ManagerActionCodec.decode("ADD this memory")


def test_invalid_model_output_fails_closed() -> None:
    manager = StructuredMemoryManager(StaticGenerator("not json"))
    action = manager.decide(ManagerInput(current_record=memory_record()))
    assert action.admission == AdmissionAction.EPHEMERAL_ONLY
    assert action.semantic_relation == SemanticRelation.INSUFFICIENT_CONTEXT
    assert "update_operation" not in action.model_dump()


def test_supervised_example_has_no_reward_or_trajectory_fields() -> None:
    target = MemoryManagerAction(
        admission=AdmissionAction.WRITE_VERIFIED,
        semantic_relation=SemanticRelation.UNRELATED,
        scope_relation=ScopeRelation.DIFFERENT_SCOPE,
        authority_relation=AuthorityRelation.NOT_APPLICABLE,
        evidence_sufficiency=EvidenceSufficiency.SUFFICIENT,
        reason_code="new_verified_claim",
    )
    example = ManagerTrainingExample(
        example_id="example-1",
        dataset_name="SciFact",
        split="train",
        current_record=memory_record(),
        target=target,
    )
    rendered = example.prompt_record()
    assert "update_operation" not in json.loads(rendered["completion"])
    assert "reward" not in example.model_dump()
    assert "trajectory" not in example.model_dump()


def test_test_split_cannot_enter_optimization() -> None:
    example = RetrievalTrainingExample(
        example_id="test-1",
        dataset_name="SciFact",
        split="test",
        query="claim",
        positive_memory_ids=("m1",),
    )
    with pytest.raises(ValueError, match="cannot enter optimization"):
        require_official_training_splits([example])
