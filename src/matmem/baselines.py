"""Capacity-matched non-learning memory baselines for MatMem evaluation."""

from __future__ import annotations

from collections.abc import Iterable

from .cards import MaterialMemoryCard, MaterialQuery
from .residual import cosine_similarity
from .residual_posterior import FixedKernelResidualGP


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


class GPVarianceOneSwapMemory:
    """Minimize summed GP posterior variance in the streaming neighborhood.

    The selector compares rejection with every legal one-swap after an
    arrival.  Its objective depends on query locations and compatible witness
    certificates, but not on unqueried outcomes.  This is a capacity-matched
    GP coreset baseline, not a decision-aware method.
    """

    def __init__(self, capacity: int, posterior_template: FixedKernelResidualGP) -> None:
        if capacity < 0:
            raise ValueError("GP-variance capacity cannot be negative")
        self.capacity = capacity
        self.posterior_template = posterior_template
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards.values())

    def _objective(
        self,
        cards: tuple[MaterialMemoryCard, ...],
        queries: tuple[MaterialQuery, ...],
    ) -> float:
        if not queries:
            return 0.0
        prediction = self.posterior_template.clone_unfit().fit(cards).predict(queries)
        return sum(value * value for value in prediction.std_ev_per_atom)

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> None:
        if card.card_id in self._cards:
            raise ValueError("new GP-variance card is already active")
        current_ids = tuple(self._cards)
        candidates: list[tuple[str, ...]] = [current_ids]
        if self.capacity > 0:
            if len(current_ids) < self.capacity:
                candidates.append((*current_ids, card.card_id))
            else:
                candidates.extend(
                    tuple(
                        card.card_id if index == evicted else card_id
                        for index, card_id in enumerate(current_ids)
                    )
                    for evicted in range(len(current_ids))
                )
        cards_by_id = {**self._cards, card.card_id: card}
        queries = tuple(query_pool)
        objectives = {
            candidate: self._objective(
                tuple(cards_by_id[card_id] for card_id in candidate), queries
            )
            for candidate in candidates
        }
        current_objective = objectives[current_ids]
        improving = [
            candidate
            for candidate in candidates[1:]
            if objectives[candidate] < current_objective
        ]
        if not improving:
            return
        selected = min(improving, key=lambda ids: (objectives[ids], tuple(sorted(ids))))
        self._cards = {card_id: cards_by_id[card_id] for card_id in selected}
