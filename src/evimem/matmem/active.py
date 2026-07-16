"""Pool-based closed-loop evaluation under coupled oracle and memory budgets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .acquisition import AcquisitionScore
from .cards import MaterialMemoryCard, MaterialQuery
from .hull_engine import CausalHullEngine


class AcquisitionPolicy(Protocol):
    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]: ...


class RetentionPolicy(Protocol):
    capacity: int

    def cards(self) -> tuple[MaterialMemoryCard, ...]: ...

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> object: ...


class CardOracleVault:
    """Synthetic-only card vault; policies never receive this object."""

    def __init__(self, cards_by_query_id: Mapping[str, MaterialMemoryCard]) -> None:
        self._cards = dict(cards_by_query_id)
        if not self._cards or any(not query_id for query_id in self._cards):
            raise ValueError("synthetic oracle vault requires nonempty query IDs")
        self._revealed: list[str] = []

    @property
    def revealed_query_ids(self) -> tuple[str, ...]:
        return tuple(self._revealed)

    def benchmark_card(self, query: MaterialQuery) -> MaterialMemoryCard:
        card = self._cards.get(query.query_id)
        if card is None:
            raise KeyError("query is absent from synthetic oracle vault")
        self._validate_alignment(query, card)
        return card

    def reveal(self, query: MaterialQuery) -> MaterialMemoryCard:
        if query.query_id in self._revealed:
            raise ValueError("synthetic oracle result has already been revealed")
        card = self.benchmark_card(query)
        self._revealed.append(query.query_id)
        return card

    @staticmethod
    def _validate_alignment(query: MaterialQuery, card: MaterialMemoryCard) -> None:
        if query.structure_hash != card.structure_hash:
            raise ValueError("query and oracle card must share a structure hash")
        if query.protocol.scientific_fingerprint != card.protocol.scientific_fingerprint:
            raise ValueError("oracle card must use the query scientific protocol")
        if query.hull_snapshot.chemical_system != card.hull_snapshot.chemical_system:
            raise ValueError("query and oracle card must share a chemical-system hull")


class ActiveStep(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    oracle_call_index: int = Field(ge=1)
    query_id: str
    acquisition_score: float
    stable_score: float = Field(ge=0, le=1)
    stable_score_kind: str
    predicted_stable: bool
    actual_stable: bool
    final_hull_stable: bool | None = None
    oracle_cost: float = Field(gt=0)
    cumulative_oracle_cost: float = Field(gt=0)
    discovery_regret: float = Field(ge=0)
    information_seeking: bool = False
    downstream_risk_reduction: float = Field(default=0.0, ge=0)
    memory_size_after_observation: int = Field(ge=0)
    archive_size_after_observation: int = Field(ge=1)
    active_witness_ids_after_observation: tuple[str, ...]
    active_witness_state_checksum_after_observation: str
    selected_hull_snapshot_id_before_observation: str
    selected_hull_phase_checksum_before_observation: str
    remaining_hull_state_checksum_after_observation: str
    causal_discoveries_after_observation: int = Field(ge=0)
    causal_hull_transition_after_observation: bool = False


class ActiveDiscoveryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_count: int
    oracle_budget: float
    oracle_cost_spent: float = Field(ge=0)
    oracle_calls: int
    active_witness_budget: int
    archive_size: int = Field(ge=0)
    archive_card_ids: tuple[str, ...]
    available_stable_count: int
    cumulative_true_discoveries: int
    query_time_causal_discoveries: int
    final_hull_confirmed_discoveries: int
    invalidated_provisional_discoveries: int
    discovery_recall_at_budget: float
    unstable_oracle_calls: int
    unstable_oracle_cost: float = Field(ge=0)
    false_stable_oracle_calls: int
    information_seeking_unstable_calls: int
    cumulative_discovery_regret: float
    discovery_regret_cost: float = Field(ge=0)
    cost_per_true_discovery: float | None
    average_memory_size: float
    hull_revision_count: int = Field(ge=0)
    selected_query_ids: tuple[str, ...]
    steps: tuple[ActiveStep, ...]


class ActiveDiscoveryEvaluator:
    """Reveal exactly one selected oracle result per call, then update memory."""

    def __init__(
        self,
        acquisition: AcquisitionPolicy,
        retention: RetentionPolicy,
        *,
        oracle_budget: float,
        hull_engine: CausalHullEngine | None = None,
    ) -> None:
        if oracle_budget <= 0:
            raise ValueError("oracle budget must be positive")
        self.acquisition = acquisition
        self.retention = retention
        self.oracle_budget = oracle_budget
        self.hull_engine = hull_engine
        self.active_witness_budget = int(
            getattr(acquisition, "active_witness_budget", retention.capacity)
        )

    def evaluate(
        self,
        candidates: Iterable[MaterialQuery],
        oracle_vault: CardOracleVault,
    ) -> ActiveDiscoveryMetrics:
        items = list(candidates)
        by_query = {item.query_id: item for item in items}
        if len(by_query) != len(items):
            raise ValueError("candidate pool query IDs must be unique")
        if not items:
            raise ValueError("candidate pool must be non-empty")
        remaining = dict(by_query)
        available_stable = sum(
            self._actual_stable(query, oracle_vault.benchmark_card(query))
            for query in items
        )
        discoveries = unstable_calls = false_stable_calls = 0
        information_seeking_unstable_calls = 0
        cumulative_regret = 0.0
        regret_cost = unstable_cost = spent = 0.0
        hull_revisions = 0
        sizes: list[int] = []
        steps: list[ActiveStep] = []
        archive: list[MaterialMemoryCard] = []
        selected_cards: list[MaterialMemoryCard] = []
        call_index = 0
        while remaining:
            affordable = tuple(
                query
                for query in remaining.values()
                if query.oracle_cost <= self.oracle_budget - spent + 1e-12
            )
            if not affordable:
                break
            affordable_ids = {query.query_id for query in affordable}
            ranked = self.acquisition.rank(
                affordable,
                self.retention.cards(),
            )
            if not ranked:
                raise RuntimeError("acquisition policy returned an empty ranking")
            chosen_score = ranked[0]
            if chosen_score.query_id not in remaining:
                raise RuntimeError("acquisition policy selected a query outside the candidate pool")
            chosen = remaining.pop(chosen_score.query_id)
            revealed_card = oracle_vault.reveal(chosen)
            call_index += 1
            spent += chosen.oracle_cost
            actual_stable = self._actual_stable(chosen, revealed_card)
            any_stable_before_selection = actual_stable or any(
                self._actual_stable(query, oracle_vault.benchmark_card(query))
                for query_id, query in remaining.items()
                if query_id in affordable_ids
            )
            regret = float(any_stable_before_selection and not actual_stable)
            cumulative_regret += regret
            regret_cost += chosen.oracle_cost * regret
            discoveries += int(actual_stable)
            unstable_calls += int(not actual_stable)
            unstable_cost += chosen.oracle_cost * int(not actual_stable)
            false_stable_calls += int(chosen_score.predicted_stable and not actual_stable)
            information_seeking = chosen_score.downstream_risk_reduction > 0
            information_seeking_unstable_calls += int(information_seeking and not actual_stable)
            archive.append(revealed_card)
            selected_cards.append(revealed_card)
            revised_query_count = 0
            if self.hull_engine is not None:
                revised_query_count = self.hull_engine.update_after_observation(
                    remaining,
                    revealed_card,
                    call_index=call_index,
                )
                hull_revisions += revised_query_count
            future_queries = tuple(remaining.values())
            self.retention.admit(
                revealed_card,
                future_queries or (chosen,),
            )
            memory_size = min(len(self.retention.cards()), self.active_witness_budget)
            sizes.append(memory_size)
            active_cards = self.retention.cards()
            steps.append(
                ActiveStep(
                    oracle_call_index=call_index,
                    query_id=chosen.query_id,
                    acquisition_score=chosen_score.score,
                    stable_score=chosen_score.stable_score,
                    stable_score_kind=chosen_score.stable_score_kind,
                    predicted_stable=chosen_score.predicted_stable,
                    actual_stable=actual_stable,
                    oracle_cost=chosen.oracle_cost,
                    cumulative_oracle_cost=spent,
                    discovery_regret=regret,
                    information_seeking=information_seeking,
                    downstream_risk_reduction=chosen_score.downstream_risk_reduction,
                    memory_size_after_observation=memory_size,
                    archive_size_after_observation=len(archive),
                    active_witness_ids_after_observation=tuple(
                        card.card_id for card in active_cards
                    ),
                    active_witness_state_checksum_after_observation=(
                        self._active_witness_state_checksum(active_cards)
                    ),
                    selected_hull_snapshot_id_before_observation=(
                        chosen.hull_snapshot.snapshot_id
                    ),
                    selected_hull_phase_checksum_before_observation=(
                        chosen.hull_snapshot.phase_set_checksum
                    ),
                    remaining_hull_state_checksum_after_observation=(
                        self._remaining_hull_state_checksum(remaining)
                    ),
                    causal_discoveries_after_observation=discoveries,
                    causal_hull_transition_after_observation=revised_query_count > 0,
                )
            )
        calls = len(steps)
        final_by_query = (
            self.hull_engine.final_stability(selected_cards)
            if self.hull_engine is not None
            else {}
        )
        if self.hull_engine is not None and set(final_by_query) != {
            item.material_id for item in selected_cards
        }:
            raise ValueError("causal hull engine did not finalize exactly the selected queries")
        finalized_steps: list[ActiveStep] = []
        final_discoveries = invalidated = 0
        for step, selected in zip(steps, selected_cards, strict=True):
            final_stable = (
                final_by_query[selected.material_id]
                if self.hull_engine is not None
                else step.actual_stable
            )
            final_discoveries += int(final_stable)
            invalidated += int(step.actual_stable and not final_stable)
            finalized_steps.append(step.model_copy(update={"final_hull_stable": final_stable}))
        return ActiveDiscoveryMetrics(
            candidate_count=len(items),
            oracle_budget=self.oracle_budget,
            oracle_cost_spent=spent,
            oracle_calls=calls,
            active_witness_budget=self.active_witness_budget,
            archive_size=len(archive),
            archive_card_ids=tuple(card.card_id for card in archive),
            available_stable_count=available_stable,
            cumulative_true_discoveries=discoveries,
            query_time_causal_discoveries=discoveries,
            final_hull_confirmed_discoveries=final_discoveries,
            invalidated_provisional_discoveries=invalidated,
            discovery_recall_at_budget=discoveries / available_stable if available_stable else 0.0,
            unstable_oracle_calls=unstable_calls,
            unstable_oracle_cost=unstable_cost,
            false_stable_oracle_calls=false_stable_calls,
            information_seeking_unstable_calls=information_seeking_unstable_calls,
            cumulative_discovery_regret=cumulative_regret,
            discovery_regret_cost=regret_cost,
            cost_per_true_discovery=spent / discoveries if discoveries else None,
            average_memory_size=sum(sizes) / calls if calls else 0.0,
            hull_revision_count=hull_revisions,
            selected_query_ids=tuple(step.query_id for step in steps),
            steps=tuple(finalized_steps),
        )

    @staticmethod
    def _actual_stable(query: MaterialQuery, card: MaterialMemoryCard) -> bool:
        return (
            card.hull_distance(query.hull_snapshot)
            <= query.stability_threshold_ev_per_atom
        )

    @staticmethod
    def _active_witness_state_checksum(cards: Iterable[MaterialMemoryCard]) -> str:
        payload = [
            {
                "card_id": card.card_id,
                "protocol": card.protocol.scientific_fingerprint,
                "hull_snapshot_id": card.hull_snapshot.snapshot_id,
                "hull_phase_checksum": card.hull_snapshot.phase_set_checksum,
            }
            for card in cards
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _remaining_hull_state_checksum(remaining: dict[str, MaterialQuery]) -> str:
        payload = [
            {
                "query_id": query_id,
                "snapshot_id": query.hull_snapshot.snapshot_id,
                "phase_checksum": query.hull_snapshot.phase_set_checksum,
                "reference": query.hull_snapshot.reference_hull_energy_ev_per_atom,
            }
            for query_id, query in sorted(remaining.items())
        ]
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()
