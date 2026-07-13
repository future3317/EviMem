from __future__ import annotations

from evimem.contracts import CurationBudget
from evimem.controller import (
    ActionType,
    CurationAction,
    EvidenceIndexEntry,
    EvidenceIndexTools,
    StateBuilder,
    build_standard_action_registry,
)

from .evimem_helpers import candidate, evidence_ref


def _state():
    passage = EvidenceIndexEntry(
        evidence_ref=evidence_ref("passage"),
        kind="passage",
        labels=("Results",),
    )
    table = EvidenceIndexEntry(
        evidence_ref=evidence_ref("table"),
        kind="table",
        labels=("Table 1", "d33"),
    )
    return StateBuilder.build(
        candidate=candidate(),
        required_slots=("property", "value"),
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        budget=CurationBudget(tool_calls=3, wall_clock_seconds=10),
        evidence_index=(passage, table),
    )


def test_evidence_tools_keep_retrieval_inside_episode_index() -> None:
    result = EvidenceIndexTools().retrieve_table(
        CurationAction(type=ActionType.RETRIEVE_TABLE, arguments={"query": "d33"}),
        _state(),
    )
    assert [ref.block_id for ref in result.evidence_refs] == ["table"]


def test_standard_registry_has_one_executor_handler_per_action() -> None:
    registry = build_standard_action_registry()
    assert ActionType.RETRIEVE_PASSAGE in registry
    assert ActionType.REQUEST_SLOT_VERIFICATION in registry
    assert ActionType.REQUEST_PUBLICATION not in registry
