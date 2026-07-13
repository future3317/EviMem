"""Deterministic evidence and governed-memory action implementations."""

from __future__ import annotations

from collections.abc import Callable

from evimem.core.contracts import ActionCost, ClaimSignature, MemoryType
from evimem.memory import MemoryRetriever, RetrievalQuery

from .actions import ActionType, CurationAction
from .executor import ActionToolResult, RegisteredAction
from .state import ControllerState, EvidenceIndexEntry


def _query_text(state: ControllerState) -> str:
    claim = state.candidate.claim
    return " ".join(
        str(value)
        for value in (
            claim.property_key,
            claim.value_raw,
            claim.unit_raw,
            claim.material_raw,
            claim.composition_raw,
            claim.conditions_raw,
        )
        if value not in (None, "")
    )


class EvidenceIndexTools:
    """Bounded retrieval over the episode's immutable evidence index."""

    @staticmethod
    def _rank(
        entries: tuple[EvidenceIndexEntry, ...],
        query: str,
        *,
        kinds: frozenset[str] | None = None,
        limit: int = 4,
    ) -> tuple[EvidenceIndexEntry, ...]:
        query_tokens = {token.lower() for token in query.split() if token}
        scored: list[tuple[float, str, EvidenceIndexEntry]] = []
        for entry in entries:
            if kinds is not None and entry.kind not in kinds:
                continue
            haystack = " ".join((entry.evidence_ref.quote or "", *entry.labels)).lower()
            tokens = set(haystack.split())
            score = len(query_tokens & tokens) / max(1, len(query_tokens))
            scored.append((score, entry.evidence_ref.block_id, entry))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in scored[:limit])

    def retrieve_passage(
        self, action: CurationAction, state: ControllerState
    ) -> ActionToolResult:
        entries = self._rank(
            state.evidence_index,
            str(action.arguments["query"]),
            kinds=frozenset({"passage", "caption", "metadata"}),
        )
        return ActionToolResult(evidence_refs=tuple(entry.evidence_ref for entry in entries))

    def retrieve_table(
        self, action: CurationAction, state: ControllerState
    ) -> ActionToolResult:
        entries = self._rank(
            state.evidence_index,
            str(action.arguments["query"]),
            kinds=frozenset({"table"}),
        )
        return ActionToolResult(evidence_refs=tuple(entry.evidence_ref for entry in entries))

    def inspect_caption(
        self, action: CurationAction, state: ControllerState
    ) -> ActionToolResult:
        reference_id = str(action.arguments["reference_id"])
        refs = tuple(
            entry.evidence_ref
            for entry in state.evidence_index
            if entry.kind == "caption"
            and reference_id in {entry.evidence_ref.block_id, *entry.labels}
        )
        return ActionToolResult(evidence_refs=refs)

    def inspect_table_cell(
        self, action: CurationAction, state: ControllerState
    ) -> ActionToolResult:
        table_id = str(action.arguments["table_id"])
        row = int(action.arguments["row"])
        column = int(action.arguments["column"])
        refs = []
        for entry in state.evidence_index:
            locator = entry.evidence_ref.locator
            if (
                getattr(locator, "table_id", None) == table_id
                and getattr(locator, "row_index", None) == row
                and getattr(locator, "column_index", None) == column
            ):
                refs.append(entry.evidence_ref)
        return ActionToolResult(evidence_refs=tuple(refs))

    def expand_local_window(
        self, action: CurationAction, state: ControllerState
    ) -> ActionToolResult:
        reference = action.arguments["evidence_ref"]
        block_id = reference.get("block_id") if isinstance(reference, dict) else str(reference)
        radius = max(0, int(action.arguments["radius"]))
        positions = [
            index
            for index, entry in enumerate(state.evidence_index)
            if entry.evidence_ref.block_id == block_id
        ]
        if not positions:
            return ActionToolResult()
        center = positions[0]
        selected = state.evidence_index[max(0, center - radius) : center + radius + 1]
        return ActionToolResult(evidence_refs=tuple(entry.evidence_ref for entry in selected))

    def follow_cross_reference(
        self, action: CurationAction, state: ControllerState
    ) -> ActionToolResult:
        reference_id = str(action.arguments["reference_id"])
        refs = tuple(
            entry.evidence_ref
            for entry in state.evidence_index
            if reference_id in entry.labels
        )
        return ActionToolResult(evidence_refs=refs)


class MemoryActionTools:
    def __init__(self, retriever: MemoryRetriever):
        self.retriever = retriever

    def retrieve(self, action: CurationAction, state: ControllerState) -> ActionToolResult:
        memory_type = {
            ActionType.RETRIEVE_VERIFIED_MEMORY: MemoryType.VERIFIED,
            ActionType.RETRIEVE_REJECTED_MEMORY: MemoryType.REJECTED,
            ActionType.RETRIEVE_CONFLICT_MEMORY: MemoryType.CONFLICT,
            ActionType.CHECK_POLICY_HISTORY: MemoryType.POLICY,
        }[action.type]
        claim = state.candidate.claim
        signature = ClaimSignature(
            domain=state.claim_state.domain_pack_id,
            property_key=claim.property_key,
            material_identity=claim.material_normalized or claim.material_raw,
            composition=claim.composition_normalized or claim.composition_raw,
            condition_signature=claim.conditions_raw,
        )
        results = self.retriever.retrieve(
            RetrievalQuery(
                signature=signature,
                policy_version=state.claim_state.domain_pack_version,
                policy_hash=state.claim_state.domain_pack_hash,
                query_text=_query_text(state),
                memory_types=(memory_type,),
            )
        )
        return ActionToolResult(
            payload={"retrieved_memory_ids": [result.item.memory_id for result in results]},
            memory_hints=self.retriever.to_hints(results),
        )


def build_standard_action_registry(
    *,
    evidence_tools: EvidenceIndexTools | None = None,
    memory_tools: MemoryActionTools | None = None,
    human_handler: Callable[[CurationAction, ControllerState], ActionToolResult] | None = None,
) -> dict[ActionType, RegisteredAction]:
    evidence = evidence_tools or EvidenceIndexTools()
    registry = {
        ActionType.RETRIEVE_PASSAGE: RegisteredAction(
            evidence.retrieve_passage, ActionCost(tool_calls=1)
        ),
        ActionType.RETRIEVE_TABLE: RegisteredAction(
            evidence.retrieve_table, ActionCost(tool_calls=1)
        ),
        ActionType.INSPECT_TABLE_CELL: RegisteredAction(
            evidence.inspect_table_cell, ActionCost(tool_calls=1)
        ),
        ActionType.INSPECT_CAPTION: RegisteredAction(
            evidence.inspect_caption, ActionCost(tool_calls=1)
        ),
        ActionType.EXPAND_LOCAL_WINDOW: RegisteredAction(
            evidence.expand_local_window, ActionCost(tool_calls=1)
        ),
        ActionType.FOLLOW_CROSS_REFERENCE: RegisteredAction(
            evidence.follow_cross_reference, ActionCost(tool_calls=1)
        ),
    }
    if memory_tools is not None:
        for action_type in (
            ActionType.RETRIEVE_VERIFIED_MEMORY,
            ActionType.RETRIEVE_REJECTED_MEMORY,
            ActionType.RETRIEVE_CONFLICT_MEMORY,
            ActionType.CHECK_POLICY_HISTORY,
        ):
            registry[action_type] = RegisteredAction(
                memory_tools.retrieve, ActionCost(tool_calls=1)
            )
    def verification_handler(
        action: CurationAction, state: ControllerState
    ) -> ActionToolResult:
        return ActionToolResult()
    for action_type in (
        ActionType.REQUEST_SLOT_VERIFICATION,
        ActionType.REQUEST_CONFLICT_CHECK,
        ActionType.REQUEST_DOMAIN_VALIDATION,
    ):
        registry[action_type] = RegisteredAction(
            verification_handler, ActionCost(tool_calls=1)
        )
    if human_handler is not None:
        registry[ActionType.ASK_HUMAN] = RegisteredAction(
            human_handler, ActionCost(human_queries=1)
        )
    return registry
