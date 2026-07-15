"""Capacity-matched non-learning memory baselines for MatMem evaluation."""

from __future__ import annotations

import hashlib
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


class FullHistoryMemory:
    """Unbounded-history diagnostic upper bound, never a deployable policy."""

    def __init__(self, maximum_calls: int) -> None:
        if maximum_calls < 1:
            raise ValueError("full-history maximum calls must be positive")
        self.capacity = maximum_calls
        self._cards: list[MaterialMemoryCard] = []

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards)

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery] = (),
    ) -> None:
        del query_pool
        self._cards.append(card)


class ResidualPriorityMemory:
    """Strong simple baseline retaining the largest absolute residuals."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("residual-priority capacity cannot be negative")
        self.capacity = capacity
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards[key] for key in sorted(self._cards))

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery] = (),
    ) -> None:
        del query_pool
        self._cards[card.card_id] = card
        ranked = sorted(
            self._cards.values(),
            key=lambda item: (-abs(item.oracle_residual_ev_per_atom), item.card_id),
        )[: self.capacity]
        self._cards = {item.card_id: item for item in ranked}


class DeterministicReservoirMemory:
    """Seeded process-stable reservoir baseline."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity < 0:
            raise ValueError("reservoir capacity cannot be negative")
        self.capacity = capacity
        self.seed = seed
        self.seen_count = 0
        self._cards: list[MaterialMemoryCard] = []

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards)

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery] = (),
    ) -> None:
        del query_pool
        self.seen_count += 1
        if self.capacity == 0:
            return
        if len(self._cards) < self.capacity:
            self._cards.append(card)
            return
        digest = hashlib.sha256(
            f"{self.seed}:{self.seen_count}:{card.card_id}".encode()
        ).digest()
        index = int.from_bytes(digest[:8], "big") % self.seen_count
        if index < self.capacity:
            self._cards[index] = card


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
