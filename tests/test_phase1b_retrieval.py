from __future__ import annotations

import pytest

from evimem.phase1b.retrieval import (
    PilotMemoryItem,
    PilotQuery,
    assert_pilot_split_isolation,
    evaluate_rankings,
    select_fixed_token_budget,
)


def test_fixed_budget_selection_is_baseline_independent() -> None:
    items = {
        "a": PilotMemoryItem(memory_id="a", text="a", token_count=4),
        "b": PilotMemoryItem(memory_id="b", text="b", token_count=7),
        "c": PilotMemoryItem(memory_id="c", text="c", token_count=5),
    }
    ranking = ("a", "b", "c")
    first = select_fixed_token_budget(ranking, items, token_budget=10)
    second = select_fixed_token_budget(ranking, items, token_budget=10)
    assert first == second == ("a", "c")
    assert sum(items[item].token_count for item in first) <= 10


def test_fixed_k_and_budget_use_same_query_and_memory_pool() -> None:
    memory = [
        PilotMemoryItem(memory_id="a", text="alpha", token_count=4, entity_key="x"),
        PilotMemoryItem(memory_id="b", text="beta", token_count=7, entity_key="y"),
    ]
    query = PilotQuery(
        query_id="q", text="alpha", positive_memory_ids=("a",), entity_key="x"
    )
    result = evaluate_rankings(
        queries=[query], memory_items=memory, rankings={"q": ("a", "b")}, token_budget=4
    )
    assert result["fixed_k"]["recall_at_1"] == 1.0
    assert result["fixed_token_budget"]["recall_at_1"] == 1.0
    assert "positive_memory_ids" not in query.model_input()


def test_retrieval_pilot_split_leakage_is_rejected() -> None:
    with pytest.raises(ValueError, match="document leakage"):
        assert_pilot_split_isolation(
            train_document_ids={"train-only", "leaked"},
            evaluation_document_ids={"eval-only", "leaked"},
        )
