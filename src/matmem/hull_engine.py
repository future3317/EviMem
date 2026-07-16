"""Protocol for causal-hull transitions used by acquisition fantasies."""

from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Protocol

from .cards import MaterialMemoryCard, MaterialQuery


class CausalHullEngine(Protocol):
    """State transition used after reveal and before calibration retention."""

    def update_after_observation(
        self,
        remaining: MutableMapping[str, MaterialQuery],
        observed: MaterialMemoryCard,
        *,
        call_index: int,
    ) -> int: ...

    def preview_after_fantasy(
        self,
        remaining_queries: Iterable[MaterialQuery],
        hypothetical: MaterialMemoryCard,
        *,
        call_index: int,
    ) -> tuple[MaterialQuery, ...]: ...

    def final_stability(
        self, selected_cards: Iterable[MaterialMemoryCard]
    ) -> dict[str, bool]: ...
