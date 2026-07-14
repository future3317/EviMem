"""Fair evaluation primitives for the Phase 1B retrieval validity pilot."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PilotMemoryItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    memory_id: str
    text: str
    token_count: int = Field(ge=1)
    entity_key: str | None = None
    memory_type: Literal["verified", "rejected", "conflict", "unavailable"] = "unavailable"
    stale: bool = False
    policy_compatible: bool = True
    certificate_compatible: bool | None = None


class PilotQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    text: str
    positive_memory_ids: tuple[str, ...]
    entity_key: str | None = None

    def model_input(self) -> dict[str, str | None]:
        """Inference input physically excludes oracle positive IDs."""

        return {"query_id": self.query_id, "text": self.text, "entity_key": self.entity_key}


def assert_pilot_split_isolation(
    *,
    train_document_ids: set[str],
    evaluation_document_ids: set[str],
) -> None:
    overlap = train_document_ids & evaluation_document_ids
    if overlap:
        raise ValueError(f"retrieval pilot document leakage: {sorted(overlap)[:5]}")


def select_fixed_token_budget(
    ranked_memory_ids: Sequence[str],
    memory_by_id: Mapping[str, PilotMemoryItem],
    *,
    token_budget: int,
) -> tuple[str, ...]:
    if token_budget < 1:
        raise ValueError("token budget must be positive")
    selected: list[str] = []
    used = 0
    for memory_id in ranked_memory_ids:
        item = memory_by_id[memory_id]
        if used + item.token_count > token_budget:
            continue
        selected.append(memory_id)
        used += item.token_count
    return tuple(selected)


def _ranking_metrics(
    queries: Sequence[PilotQuery],
    rankings: Mapping[str, Sequence[str]],
) -> dict[str, float | int]:
    eligible = [query for query in queries if query.positive_memory_ids]
    recalls = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    for query in eligible:
        positive = set(query.positive_memory_ids)
        ranked = list(rankings[query.query_id])
        ranks = [index + 1 for index, memory_id in enumerate(ranked) if memory_id in positive]
        for cutoff in recalls:
            recalls[cutoff] += int(any(rank <= cutoff for rank in ranks))
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        gains = [1.0 if memory_id in positive else 0.0 for memory_id in ranked[:10]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_count = min(len(positive), 10)
        idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    denominator = len(eligible)
    return {
        "query_count": len(queries),
        "queries_with_retrieval_gold": denominator,
        "recall_at_1": recalls[1] / denominator if denominator else 0.0,
        "recall_at_5": recalls[5] / denominator if denominator else 0.0,
        "recall_at_10": recalls[10] / denominator if denominator else 0.0,
        "mrr": sum(reciprocal_ranks) / denominator if denominator else 0.0,
        "ndcg_at_10": sum(ndcgs) / denominator if denominator else 0.0,
    }


def evaluate_rankings(
    *,
    queries: Sequence[PilotQuery],
    memory_items: Sequence[PilotMemoryItem],
    rankings: Mapping[str, Sequence[str]],
    token_budget: int,
) -> dict[str, Any]:
    memory_by_id = {item.memory_id: item for item in memory_items}
    if set(rankings) != {query.query_id for query in queries}:
        raise ValueError("rankings must cover exactly the pilot queries")
    for query in queries:
        ranked = rankings[query.query_id]
        if len(ranked) != len(set(ranked)) or set(ranked) - set(memory_by_id):
            raise ValueError(f"invalid ranking for {query.query_id}")

    fixed_k = _ranking_metrics(queries, rankings)
    budget_rankings = {
        query.query_id: select_fixed_token_budget(
            rankings[query.query_id], memory_by_id, token_budget=token_budget
        )
        for query in queries
    }
    fixed_budget = _ranking_metrics(queries, budget_rankings)
    selected = [
        memory_by_id[memory_id]
        for query in queries
        for memory_id in list(rankings[query.query_id])[:10]
    ]
    budget_selected = [
        memory_by_id[memory_id]
        for query in queries
        for memory_id in budget_rankings[query.query_id]
    ]
    wrong_entity = 0
    hard_negative_count = 0
    for query in queries:
        positives = set(query.positive_memory_ids)
        for memory_id in list(rankings[query.query_id])[:10]:
            item = memory_by_id[memory_id]
            if memory_id in positives:
                continue
            if query.entity_key is not None and item.entity_key is not None:
                hard_negative_count += 1
                wrong_entity += int(query.entity_key != item.entity_key)

    typed_recall: dict[str, float | None] = {}
    for memory_type in ("verified", "rejected", "conflict"):
        subset = [
            query
            for query in queries
            if any(
                memory_by_id[memory_id].memory_type == memory_type
                for memory_id in query.positive_memory_ids
            )
        ]
        subset_metrics = _ranking_metrics(subset, rankings) if subset else None
        for cutoff in (1, 5, 10):
            typed_recall[f"{memory_type}_recall_at_{cutoff}"] = (
                subset_metrics[f"recall_at_{cutoff}"] if subset_metrics else None
            )

    certificate_known = [item for item in selected if item.certificate_compatible is not None]
    fixed_budget.update(
        {
            "token_budget": token_budget,
            "average_selected_items": (
                len(budget_selected) / len(queries) if queries else 0.0
            ),
            "average_selected_tokens": (
                sum(item.token_count for item in budget_selected) / len(queries)
                if queries
                else 0.0
            ),
        }
    )
    return {
        "fixed_k": fixed_k,
        "fixed_token_budget": fixed_budget,
        "typed_recall": typed_recall,
        "stale_retrieval_rate": (
            sum(item.stale for item in selected) / len(selected) if selected else 0.0
        ),
        "policy_incompatible_retrieval_rate": (
            sum(not item.policy_compatible for item in selected) / len(selected)
            if selected
            else 0.0
        ),
        "certificate_mismatch_retrieval_rate": (
            sum(not bool(item.certificate_compatible) for item in certificate_known)
            / len(certificate_known)
            if certificate_known
            else None
        ),
        "wrong_entity_hard_negative_rate": (
            wrong_entity / hard_negative_count if hard_negative_count else None
        ),
        "wrong_entity_metric_note": (
            "SciREX uses native method entity; SciFact/QASPER use source-document identity "
            "as the available wrong-entity proxy."
        ),
    }
