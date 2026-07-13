"""Transparent memory baselines under the same chronological store boundary."""

from __future__ import annotations

from .governed_store import GovernedMemoryStore
from .retriever import RetrievalQuery, RetrievedMemory


class NoMemoryBaseline:
    def retrieve(self, query: RetrievalQuery) -> list[RetrievedMemory]:
        return []


class FullHistoryBaseline:
    def __init__(self, store: GovernedMemoryStore):
        self.store = store

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedMemory]:
        records = self.store.query(
            domain=query.signature.domain,
            memory_types=query.memory_types or None,
            observed_before=query.as_of,
            include_superseded=True,
            limit=1_000_000,
        )
        return [
            RetrievedMemory(
                record=record,
                score=0.0,
                semantic_similarity=0.0,
                structure_match=0.0,
                policy_compatibility=0.0,
                authority=record.authority.level / 4.0,
                temporal_relevance=0.0,
                staleness=float(record.status.value == "superseded"),
                conflict_risk=float(record.memory_type.value == "conflict"),
            )
            for record in records
        ]
