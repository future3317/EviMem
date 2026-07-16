"""Capacity-matched non-learning memory baselines for MatMem evaluation."""

from __future__ import annotations

from collections.abc import Iterable

from .cards import MaterialMemoryCard, MaterialQuery
from .residual import cosine_similarity


class FIFOBoundedMemory:
    """Chronological baseline that retains the most recently observed cards."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("FIFO active witness capacity cannot be negative")
        self.capacity = capacity
        self._cards: list[MaterialMemoryCard] = []

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards)

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery] = (),
    ) -> None:
        del query_pool
        self._cards = [item for item in self._cards if item.card_id != card.card_id]
        self._cards.append(card)
        self._cards = self._cards[-self.capacity :] if self.capacity else []


class DiversityBoundedMemory:
    """Query-coverage baseline independent of oracle residual values."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("diversity capacity cannot be negative")
        self.capacity = capacity
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards[key] for key in sorted(self._cards))

    @staticmethod
    def _coverage(
        cards: Iterable[MaterialMemoryCard],
        queries: Iterable[MaterialQuery],
    ) -> float:
        memory = tuple(cards)
        if not memory:
            return 0.0
        return sum(
            max(max(0.0, cosine_similarity(query.embedding, card.embedding)) for card in memory)
            for query in queries
        )

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> None:
        self._cards[card.card_id] = card
        queries = tuple(query_pool)
        while len(self._cards) > self.capacity:
            choices = []
            for card_id in self._cards:
                retained = [item for key, item in self._cards.items() if key != card_id]
                choices.append((self._coverage(retained, queries), card_id))
            _, evicted = max(choices, key=lambda item: (item[0], item[1]))
            del self._cards[evicted]
