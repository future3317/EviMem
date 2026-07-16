"""Explicit causal-hull engines for chronological discovery evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, MutableMapping
from typing import Protocol

from .cards import HullSnapshot, MaterialMemoryCard, MaterialQuery


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


class SyntheticMinHullEngine:
    """Scalar minimum-hull approximation restricted to synthetic controls.

    This deliberately preserves the historical stress-test transition.  It is
    scientifically invalid for a multi-composition phase diagram and is never
    used by the WBM runner.
    """

    @staticmethod
    def _revised_snapshot(
        query: MaterialQuery,
        observed: MaterialMemoryCard,
        call_index: int,
    ) -> HullSnapshot:
        old = query.hull_snapshot
        if old.chemical_system != observed.hull_snapshot.chemical_system:
            return old
        reference = min(
            old.reference_hull_energy_ev_per_atom,
            observed.formation_energy_ev_per_atom,
        )
        if reference == old.reference_hull_energy_ev_per_atom:
            return old
        payload = (
            f"{old.phase_set_checksum}:synthetic-min:{observed.card_id}:"
            f"{call_index}:{reference:.12g}"
        )
        built_at = max(old.built_at, observed.observed_at)
        return old.model_copy(
            update={
                "snapshot_id": f"{old.snapshot_id}:synthetic:{call_index}",
                "reference_hull_energy_ev_per_atom": reference,
                "phase_set_checksum": "sha256:"
                + hashlib.sha256(payload.encode()).hexdigest(),
                "known_through": built_at,
                "built_at": built_at,
            }
        )

    def update_after_observation(
        self,
        remaining: MutableMapping[str, MaterialQuery],
        observed: MaterialMemoryCard,
        *,
        call_index: int,
    ) -> int:
        changed = 0
        for query_id, query in tuple(remaining.items()):
            snapshot = self._revised_snapshot(query, observed, call_index)
            if snapshot.snapshot_id == query.hull_snapshot.snapshot_id:
                continue
            remaining[query_id] = query.model_copy(
                update={"hull_snapshot": snapshot, "as_of": snapshot.built_at}
            )
            changed += 1
        return changed

    def preview_after_fantasy(
        self,
        remaining_queries: Iterable[MaterialQuery],
        hypothetical: MaterialMemoryCard,
        *,
        call_index: int,
    ) -> tuple[MaterialQuery, ...]:
        copied = {query.query_id: query for query in remaining_queries}
        self.update_after_observation(
            copied, hypothetical, call_index=call_index
        )
        return tuple(copied.values())

    def final_stability(
        self, selected_cards: Iterable[MaterialMemoryCard]
    ) -> dict[str, bool]:
        cards = tuple(selected_cards)
        by_system: dict[tuple[str, ...], list[MaterialMemoryCard]] = {}
        for card in cards:
            by_system.setdefault(card.hull_snapshot.chemical_system, []).append(card)
        result: dict[str, bool] = {}
        for group in by_system.values():
            reference = min(card.formation_energy_ev_per_atom for card in group)
            result.update(
                {
                    card.material_id: card.formation_energy_ev_per_atom - reference <= 0
                    for card in group
                }
            )
        return result
