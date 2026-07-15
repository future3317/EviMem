"""Certificate-compatible residual retrieval with explicit abstention."""

from __future__ import annotations

import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from .cards import MaterialMemoryCard, MaterialQuery
from .protocols import ProtocolCompatibilityResolver


def cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise ValueError("structure embeddings must have matching dimensions")
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    if norm_left == 0 or norm_right == 0:
        raise ValueError("structure embeddings must have non-zero norm")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (norm_left * norm_right)


class ResidualCorrection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str
    corrected_formation_energy_ev_per_atom: float | None = None
    corrected_hull_distance_ev_per_atom: float | None = None
    residual_shift_ev_per_atom: float | None = None
    uncertainty_radius_ev_per_atom: float | None = None
    contributing_card_ids: tuple[str, ...] = ()
    rejected_card_ids: tuple[str, ...] = ()


class ResidualCorrector:
    """No compatible card means abstention; base predictions are not silently reused."""

    def __init__(
        self,
        resolver: ProtocolCompatibilityResolver,
        *,
        minimum_similarity: float = 0.05,
        max_neighbors: int = 8,
    ) -> None:
        if not 0 <= minimum_similarity <= 1:
            raise ValueError("minimum_similarity must be within [0, 1]")
        self.resolver = resolver
        self.minimum_similarity = minimum_similarity
        self.max_neighbors = max_neighbors

    def correct(
        self,
        query: MaterialQuery,
        cards: Iterable[MaterialMemoryCard],
    ) -> ResidualCorrection:
        compatible: list[tuple[float, float, float, str]] = []
        rejected: list[str] = []
        for card in cards:
            compatibility = self.resolver.resolve(card.protocol, query.protocol)
            transferred = compatibility.transfer_residual(card.oracle_residual_ev_per_atom)
            if transferred is None:
                rejected.append(card.card_id)
                continue
            similarity = max(0.0, cosine_similarity(query.embedding, card.embedding))
            if similarity < self.minimum_similarity:
                continue
            protocol_weight = 1.0 / (1.0 + compatibility.uncertainty_radius_ev_per_atom)
            weight = similarity * protocol_weight * card.quality_weight
            compatible.append((weight, transferred, compatibility.uncertainty_radius_ev_per_atom, card.card_id))
        compatible.sort(key=lambda item: (-item[0], item[3]))
        compatible = compatible[: self.max_neighbors]
        total_weight = sum(item[0] for item in compatible)
        if total_weight == 0:
            return ResidualCorrection(
                status="abstain_no_certificate_compatible_neighbor",
                rejected_card_ids=tuple(sorted(rejected)),
            )
        shift = sum(weight * residual for weight, residual, _, _ in compatible) / total_weight
        uncertainty = sum(weight * radius for weight, _, radius, _ in compatible) / total_weight
        corrected_energy = query.base_predicted_formation_energy_ev_per_atom + shift
        return ResidualCorrection(
            status="corrected",
            corrected_formation_energy_ev_per_atom=corrected_energy,
            corrected_hull_distance_ev_per_atom=query.hull_distance(corrected_energy),
            residual_shift_ev_per_atom=shift,
            uncertainty_radius_ev_per_atom=uncertainty,
            contributing_card_ids=tuple(item[3] for item in compatible),
            rejected_card_ids=tuple(sorted(rejected)),
        )
