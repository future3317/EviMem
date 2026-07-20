"""Fail-closed activation of certified cross-protocol evidence.

The immutable target-protocol archive is never bounded by this module. Only
evidence that requires an explicit directed transport certificate can be
gated, and the gate is based on certificate validity rather than subset size.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from .cards import MaterialMemoryCard, MaterialQuery
from .protocols import CompatibilityKind, ProtocolCompatibilityResolver


class ProtocolActivationAudit(BaseModel):
    """Observable explanation of one query-specific evidence view."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    archive_card_ids: tuple[str, ...]
    active_card_ids: tuple[str, ...]
    direct_card_ids: tuple[str, ...]
    transported_card_ids: tuple[str, ...]
    rejected_incompatible_card_ids: tuple[str, ...]
    rejected_uncertainty_card_ids: tuple[str, ...]

    @property
    def full_history_equivalent(self) -> bool:
        return self.active_card_ids == self.archive_card_ids


@dataclass(frozen=True)
class ProtocolActivation:
    """Selected cards plus their immutable activation audit."""

    cards: tuple[MaterialMemoryCard, ...]
    audit: ProtocolActivationAudit


class ProtocolAwareActivator:
    """Activate direct history and every certified transported card.

    All cards already calculated under the target protocol remain active. Every
    transported card that passes the frozen certificate gate also remains
    active; this class cannot use a capacity limit to delete outcome
    contributions under another name.
    """

    def __init__(
        self,
        resolver: ProtocolCompatibilityResolver,
        *,
        max_transport_uncertainty_ev_per_atom: float | None = None,
    ) -> None:
        if (
            max_transport_uncertainty_ev_per_atom is not None
            and (
                not math.isfinite(max_transport_uncertainty_ev_per_atom)
                or max_transport_uncertainty_ev_per_atom < 0
            )
        ):
            raise ValueError("transport uncertainty gate must be finite and non-negative")
        self.resolver = resolver
        self.max_transport_uncertainty_ev_per_atom = (
            max_transport_uncertainty_ev_per_atom
        )

    def activate(
        self,
        query: MaterialQuery,
        archive: Sequence[MaterialMemoryCard],
    ) -> ProtocolActivation:
        cards = tuple(archive)
        ids = tuple(card.card_id for card in cards)
        if len(set(ids)) != len(ids):
            raise ValueError("protocol activation requires unique archive card IDs")

        direct: list[MaterialMemoryCard] = []
        transported: list[MaterialMemoryCard] = []
        incompatible: list[str] = []
        uncertain: list[str] = []
        for card in cards:
            resolution = self.resolver.resolve(card.protocol, query.protocol)
            if resolution.kind is CompatibilityKind.REJECT:
                incompatible.append(card.card_id)
                continue
            if resolution.kind is CompatibilityKind.DIRECT:
                direct.append(card)
                continue
            radius = resolution.uncertainty_radius_ev_per_atom
            if (
                self.max_transport_uncertainty_ev_per_atom is not None
                and radius > self.max_transport_uncertainty_ev_per_atom
            ):
                uncertain.append(card.card_id)
                continue
            transported.append(card)

        direct_ids = {card.card_id for card in direct}
        transported_ids = {card.card_id for card in transported}
        active = tuple(
            card
            for card in cards
            if card.card_id in direct_ids or card.card_id in transported_ids
        )
        audit = ProtocolActivationAudit(
            query_id=query.query_id,
            archive_card_ids=ids,
            active_card_ids=tuple(card.card_id for card in active),
            direct_card_ids=tuple(card.card_id for card in cards if card.card_id in direct_ids),
            transported_card_ids=tuple(
                card.card_id for card in cards if card.card_id in transported_ids
            ),
            rejected_incompatible_card_ids=tuple(sorted(incompatible)),
            rejected_uncertainty_card_ids=tuple(sorted(uncertain)),
        )
        return ProtocolActivation(cards=active, audit=audit)
