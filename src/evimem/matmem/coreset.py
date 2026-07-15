"""Decision-aware streaming coreset selection for screening, not residual magnitude."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from .cards import MaterialMemoryCard, MaterialQuery
from .protocols import ProtocolCompatibilityResolver
from .residual import cosine_similarity


class CoresetSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_card_ids: tuple[str, ...]
    objective_value: float = Field(ge=0)
    baseline_screening_risk: float = Field(ge=0)
    selected_screening_risk: float = Field(ge=0)
    marginal_gains: dict[str, float]
    rejected_card_ids: tuple[str, ...] = ()


class DecisionAwareOnlineCoreset:
    """Greedy maximizer of a monotone screening-utility coverage objective.

    For a recent candidate pool ``Q``, the policy selects ``M`` (``|M| <= K``)
    to maximize ``sum_x max_i k_z(x,i) k_c(x,i) Delta_l_screen(x;i)``.
    The greedy cardinality algorithm has the standard ``1 - 1/e`` guarantee
    for this non-negative monotone submodular objective; no such claim applies
    if callers alter the utility to include negative values.
    """

    def __init__(
        self,
        capacity: int,
        resolver: ProtocolCompatibilityResolver,
        *,
        false_stable_cost: float = 5.0,
        false_unstable_cost: float = 1.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("coreset capacity must be positive")
        if false_stable_cost <= 0 or false_unstable_cost <= 0:
            raise ValueError("screening costs must be positive")
        self.capacity = capacity
        self.resolver = resolver
        self.false_stable_cost = false_stable_cost
        self.false_unstable_cost = false_unstable_cost
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards[key] for key in sorted(self._cards))

    def _screening_loss(self, predicted_hull_distance: float, proxy_hull_distance: float, threshold: float) -> float:
        predicted_stable = predicted_hull_distance <= threshold
        proxy_stable = proxy_hull_distance <= threshold
        if predicted_stable and not proxy_stable:
            return self.false_stable_cost
        if not predicted_stable and proxy_stable:
            return self.false_unstable_cost
        return 0.0

    def _query_utilities(
        self,
        query: MaterialQuery,
        cards: dict[str, MaterialMemoryCard],
    ) -> tuple[float, dict[str, float]] | None:
        """Estimate a common local screening-risk distribution for one query.

        Each compatible card induces a local residual outcome. Their normalized
        structure/protocol/quality weights form an empirical probability of an
        unstable screen. A card's utility is the decrease in this *same*
        expected asymmetric risk, not its residual magnitude or a hand-coded
        boundary score.
        """

        witnesses: list[tuple[str, float, float]] = []
        for card_id, card in cards.items():
            compatibility = self.resolver.resolve(card.protocol, query.protocol)
            transferred = compatibility.transfer_residual(card.oracle_residual_ev_per_atom)
            if transferred is None:
                continue
            structure = max(0.0, cosine_similarity(query.embedding, card.embedding))
            protocol = 1.0 / (1.0 + compatibility.uncertainty_radius_ev_per_atom)
            weight = structure * protocol * card.quality_weight
            if weight == 0:
                continue
            proxy = query.hull_distance(query.base_predicted_formation_energy_ev_per_atom + transferred)
            witnesses.append((card_id, weight, proxy))
        total_weight = sum(weight for _, weight, _ in witnesses)
        if total_weight == 0:
            return None
        threshold = query.stability_threshold_ev_per_atom
        unstable_probability = sum(
            weight * float(proxy > threshold) for _, weight, proxy in witnesses
        ) / total_weight
        base_stable = query.base_hull_distance_ev_per_atom <= threshold
        baseline_risk = (
            self.false_stable_cost * unstable_probability
            if base_stable
            else self.false_unstable_cost * (1.0 - unstable_probability)
        )
        max_weight = max(weight for _, weight, _ in witnesses)
        utilities: dict[str, float] = {}
        for card_id, weight, proxy in witnesses:
            card_stable = proxy <= threshold
            card_risk = (
                self.false_stable_cost * unstable_probability
                if card_stable
                else self.false_unstable_cost * (1.0 - unstable_probability)
            )
            utilities[card_id] = (weight / max_weight) * max(0.0, baseline_risk - card_risk)
        return baseline_risk, utilities

    def select(
        self,
        candidates: Iterable[MaterialMemoryCard],
        query_pool: Iterable[MaterialQuery],
    ) -> CoresetSelection:
        candidate_list = list(candidates)
        candidate_items = {card.card_id: card for card in candidate_list}
        if len(candidate_items) != len(candidate_list):
            raise ValueError("duplicate card IDs are not allowed")
        queries = tuple(query_pool)
        if not queries:
            raise ValueError("decision-aware selection requires a non-empty recent candidate pool")
        values: dict[str, list[float]] = {}
        rejected: list[str] = []
        baseline_risks: list[float] = []
        candidate_risk_coverage = {card_id: 0 for card_id in candidate_items}
        per_query_utilities: list[dict[str, float]] = []
        for query in queries:
            result = self._query_utilities(query, candidate_items)
            if result is None:
                baseline_risks.append(0.0)
                per_query_utilities.append({})
                continue
            baseline_risk, utilities = result
            baseline_risks.append(baseline_risk)
            per_query_utilities.append(utilities)
            for card_id in utilities:
                candidate_risk_coverage[card_id] += 1
        for card_id, card in candidate_items.items():
            if candidate_risk_coverage[card_id] == 0:
                rejected.append(card_id)
                continue
            values[card_id] = [utilities.get(card_id, 0.0) for utilities in per_query_utilities]
        covered = [0.0] * len(queries)
        selected: list[str] = []
        gains: dict[str, float] = {}
        while len(selected) < self.capacity:
            choices = [
                (
                    sum(max(current, proposed) - current for current, proposed in zip(covered, row, strict=True)),
                    card_id,
                )
                for card_id, row in values.items()
                if card_id not in selected
            ]
            if not choices:
                break
            gain, card_id = max(choices, key=lambda item: (item[0], item[1]))
            if gain <= 0:
                break
            selected.append(card_id)
            gains[card_id] = gain
            covered = [max(current, proposed) for current, proposed in zip(covered, values[card_id], strict=True)]
        return CoresetSelection(
            selected_card_ids=tuple(selected),
            objective_value=sum(covered),
            baseline_screening_risk=sum(baseline_risks),
            selected_screening_risk=sum(baseline_risks) - sum(covered),
            marginal_gains=gains,
            rejected_card_ids=tuple(sorted(rejected)),
        )

    def admit(self, card: MaterialMemoryCard, query_pool: Iterable[MaterialQuery]) -> CoresetSelection:
        """Reoptimize the bounded state after a newly observed oracle result."""

        selection = self.select([*self._cards.values(), card], query_pool)
        selected = set(selection.selected_card_ids)
        all_cards = {**self._cards, card.card_id: card}
        self._cards = {card_id: all_cards[card_id] for card_id in selected}
        return selection
