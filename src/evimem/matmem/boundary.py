"""Observable boundary intervals and certified active-witness selection.

All oracle observations remain available to an audit archive owned by the
evaluator.  This module selects at most ``K`` *active* residual witnesses for
online correction.  It never interprets ``K`` as permission to destroy a DFT
result.

The boundary potential compares working sets with a weight fixed by the query
view and its causal hull.  Compatible prior/witness intervals are intersected;
an empty intersection is a calibration conflict and fails closed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from .cards import MaterialMemoryCard, MaterialQuery
from .protocols import ProtocolCertificate, ProtocolCompatibilityResolver
from .residual import cosine_similarity


class BoundaryRiskConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    residual_lipschitz_ev_per_atom: float = Field(default=0.08, ge=0)
    prior_radius_ev_per_atom: float = Field(default=0.15, gt=0)
    calibration_radius_ev_per_atom: float = Field(default=0.01, ge=0)
    boundary_bandwidth_ev_per_atom: float = Field(default=0.05, gt=0)
    minimum_boundary_weight: float = Field(default=0.05, gt=0, le=1)
    false_stable_cost: float = Field(default=5.0, gt=0)
    false_unstable_cost: float = Field(default=1.0, gt=0)
    minimum_similarity: float = Field(default=0.05, ge=0, le=1)


class BoundaryWitness(BaseModel):
    """Minimal residual view used for online and hypothetical calculations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    witness_id: str
    embedding: tuple[float, ...]
    residual_ev_per_atom: float
    protocol: ProtocolCertificate
    quality_weight: float = Field(default=1.0, gt=0, le=1)

    @classmethod
    def from_card(cls, card: MaterialMemoryCard) -> BoundaryWitness:
        return cls(
            witness_id=card.card_id,
            embedding=card.embedding,
            residual_ev_per_atom=card.oracle_residual_ev_per_atom,
            protocol=card.protocol,
            quality_weight=card.quality_weight,
        )


class BoundaryEstimate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    center_hull_distance_ev_per_atom: float
    radius_ev_per_atom: float = Field(ge=0)
    lower_hull_distance_ev_per_atom: float
    upper_hull_distance_ev_per_atom: float
    scenario_stable_weight: float = Field(ge=0, le=1)
    predicted_stable: bool
    boundary_ambiguous: bool
    interval_conflict: bool
    abstained: bool
    boundary_weight: float = Field(gt=0, le=1)
    weighted_risk_upper_bound: float = Field(ge=0)
    source_witness_ids: tuple[str, ...] = ()

    @property
    def source_witness_id(self) -> str | None:
        """Compatibility alias for callers that only need presence/absence."""

        return self.source_witness_ids[0] if self.source_witness_ids else None


class BoundaryPotentialValue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    total: float = Field(ge=0)
    per_query: dict[str, float]
    ambiguous_query_count: int = Field(ge=0)
    conflict_query_count: int = Field(default=0, ge=0)


class BoundaryRetentionSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_card_ids: tuple[str, ...]
    potential_before: float = Field(ge=0)
    potential_after: float = Field(ge=0)
    evicted_card_ids: tuple[str, ...]


class BoundaryRiskPotential:
    """A common, memory-comparable upper-bound surrogate.

    The weight ``omega_t(x)`` depends only on the observable base prediction and
    causal hull stored in ``query``.  Working sets can change the interval, but
    cannot change which candidates the objective considers important.
    """

    def __init__(
        self,
        resolver: ProtocolCompatibilityResolver,
        config: BoundaryRiskConfig | None = None,
    ) -> None:
        self.resolver = resolver
        self.config = config or BoundaryRiskConfig()

    def fixed_boundary_weight(self, query: MaterialQuery) -> float:
        margin = abs(
            query.base_hull_distance_ev_per_atom
            - query.stability_threshold_ev_per_atom
        )
        return max(
            self.config.minimum_boundary_weight,
            math.exp(-margin / self.config.boundary_bandwidth_ev_per_atom),
        )

    def _witness_interval(
        self,
        query: MaterialQuery,
        witness: BoundaryWitness,
    ) -> tuple[float, float] | None:
        compatibility = self.resolver.resolve(witness.protocol, query.protocol)
        residual = compatibility.transfer_residual(witness.residual_ev_per_atom)
        if residual is None:
            return None
        similarity = max(0.0, cosine_similarity(query.embedding, witness.embedding))
        if similarity < self.config.minimum_similarity:
            return None
        structure_distance = math.sqrt(max(0.0, 2.0 - 2.0 * similarity))
        radius = (
            self.config.residual_lipschitz_ev_per_atom * structure_distance
            + compatibility.uncertainty_radius_ev_per_atom
            + self.config.calibration_radius_ev_per_atom
        ) / witness.quality_weight
        center = query.hull_distance(
            query.base_predicted_formation_energy_ev_per_atom + residual
        )
        return center - radius, center + radius

    def estimate_from_witnesses(
        self,
        query: MaterialQuery,
        witnesses: Iterable[BoundaryWitness],
    ) -> BoundaryEstimate:
        base_center = query.base_hull_distance_ev_per_atom
        lower = base_center - self.config.prior_radius_ev_per_atom
        upper = base_center + self.config.prior_radius_ev_per_atom
        source_ids: list[str] = []
        for witness in sorted(witnesses, key=lambda item: item.witness_id):
            interval = self._witness_interval(query, witness)
            if interval is None:
                continue
            witness_lower, witness_upper = interval
            lower = max(lower, witness_lower)
            upper = min(upper, witness_upper)
            source_ids.append(witness.witness_id)

        conflict = lower > upper
        if conflict:
            center = base_center
            radius = 0.0
            scenario_stable_weight = 0.5
            predicted_stable = False
            ambiguous = True
            abstained = True
        else:
            center = 0.5 * (lower + upper)
            radius = 0.5 * (upper - lower)
            threshold = query.stability_threshold_ev_per_atom
            predicted_stable = center <= threshold
            ambiguous = lower <= threshold < upper
            abstained = False
            if upper == lower:
                scenario_stable_weight = float(center <= threshold)
            else:
                scenario_stable_weight = max(
                    0.0,
                    min(1.0, (threshold - lower) / (upper - lower)),
                )

        boundary_weight = self.fixed_boundary_weight(query)
        if conflict:
            risk = boundary_weight * max(
                self.config.false_stable_cost,
                self.config.false_unstable_cost,
            )
        elif predicted_stable:
            risk = (
                boundary_weight * self.config.false_stable_cost
                if upper > query.stability_threshold_ev_per_atom
                else 0.0
            )
        else:
            risk = (
                boundary_weight * self.config.false_unstable_cost
                if lower <= query.stability_threshold_ev_per_atom
                else 0.0
            )
        return BoundaryEstimate(
            query_id=query.query_id,
            center_hull_distance_ev_per_atom=center,
            radius_ev_per_atom=radius,
            lower_hull_distance_ev_per_atom=lower,
            upper_hull_distance_ev_per_atom=upper,
            scenario_stable_weight=scenario_stable_weight,
            predicted_stable=predicted_stable,
            boundary_ambiguous=ambiguous,
            interval_conflict=conflict,
            abstained=abstained,
            boundary_weight=boundary_weight,
            weighted_risk_upper_bound=risk,
            source_witness_ids=tuple(source_ids),
        )

    def estimate(
        self,
        query: MaterialQuery,
        cards: Iterable[MaterialMemoryCard],
    ) -> BoundaryEstimate:
        return self.estimate_from_witnesses(
            query,
            (BoundaryWitness.from_card(card) for card in cards),
        )

    def evaluate_witnesses(
        self,
        queries: Iterable[MaterialQuery],
        witnesses: Iterable[BoundaryWitness],
    ) -> BoundaryPotentialValue:
        memory = tuple(witnesses)
        estimates = [self.estimate_from_witnesses(query, memory) for query in queries]
        per_query = {
            estimate.query_id: estimate.weighted_risk_upper_bound for estimate in estimates
        }
        return BoundaryPotentialValue(
            total=sum(per_query.values()),
            per_query=per_query,
            ambiguous_query_count=sum(estimate.boundary_ambiguous for estimate in estimates),
            conflict_query_count=sum(estimate.interval_conflict for estimate in estimates),
        )

    def evaluate(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> BoundaryPotentialValue:
        return self.evaluate_witnesses(
            queries,
            (BoundaryWitness.from_card(card) for card in cards),
        )


class BruteForceRetentionSolver:
    """Exact solver over every legal subset with cardinality at most ``K``."""

    def select(
        self,
        witnesses: Iterable[BoundaryWitness],
        queries: Iterable[MaterialQuery],
        *,
        capacity: int,
        potential: BoundaryRiskPotential,
    ) -> tuple[BoundaryWitness, ...]:
        if capacity < 0:
            raise ValueError("active witness capacity cannot be negative")
        candidates = tuple(sorted(witnesses, key=lambda item: item.witness_id))
        query_pool = tuple(queries)
        choices: list[
            tuple[float, int, tuple[str, ...], tuple[BoundaryWitness, ...]]
        ] = []
        for size in range(min(capacity, len(candidates)) + 1):
            for retained in itertools.combinations(candidates, size):
                value = potential.evaluate_witnesses(query_pool, retained).total
                ids = tuple(item.witness_id for item in retained)
                choices.append((value, -size, ids, retained))
        return min(choices, key=lambda item: (item[0], item[1], item[2]))[3]


class BoundaryRiskRetention:
    """Select a certified active witness working set; archive ownership is external."""

    def __init__(
        self,
        capacity: int,
        potential: BoundaryRiskPotential,
        solver: BruteForceRetentionSolver | None = None,
    ) -> None:
        if capacity < 0:
            raise ValueError("active witness capacity cannot be negative")
        self.capacity = capacity
        self.potential = potential
        self.solver = solver or BruteForceRetentionSolver()
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards[key] for key in sorted(self._cards))

    def select_witnesses(
        self,
        witnesses: Iterable[BoundaryWitness],
        queries: Iterable[MaterialQuery],
    ) -> tuple[BoundaryWitness, ...]:
        return self.solver.select(
            witnesses,
            queries,
            capacity=self.capacity,
            potential=self.potential,
        )

    def select(
        self,
        candidates: Iterable[MaterialMemoryCard],
        query_pool: Iterable[MaterialQuery],
    ) -> BoundaryRetentionSelection:
        candidate_list = tuple(candidates)
        cards = {card.card_id: card for card in candidate_list}
        if len(cards) != len(candidate_list):
            raise ValueError("active witness candidates must have unique card IDs")
        queries = tuple(query_pool)
        before = self.potential.evaluate(queries, cards.values()).total
        witnesses = self.select_witnesses(
            (BoundaryWitness.from_card(card) for card in cards.values()),
            queries,
        )
        selected_ids = tuple(witness.witness_id for witness in witnesses)
        retained = [cards[card_id] for card_id in selected_ids]
        after = self.potential.evaluate(queries, retained).total
        return BoundaryRetentionSelection(
            selected_card_ids=selected_ids,
            potential_before=before,
            potential_after=after,
            evicted_card_ids=tuple(sorted(set(cards) - set(selected_ids))),
        )

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> BoundaryRetentionSelection:
        all_cards = {**self._cards, card.card_id: card}
        selection = self.select(all_cards.values(), query_pool)
        self._cards = {card_id: all_cards[card_id] for card_id in selection.selected_card_ids}
        return selection
