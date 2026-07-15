"""Pool-based closed-loop evaluation under coupled oracle and memory budgets."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .acquisition import AcquisitionScore
from .cards import MaterialMemoryCard, MaterialQuery


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


class CandidatePoolItem(BaseModel):
    """Evaluation-only pairing; acquisition receives only ``query``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: MaterialQuery
    oracle_card: MaterialMemoryCard

    @model_validator(mode="after")
    def _aligned_oracle(self) -> CandidatePoolItem:
        if self.query.structure_hash != self.oracle_card.structure_hash:
            raise ValueError("candidate query and oracle card must share a structure hash")
        if self.query.protocol.scientific_fingerprint != self.oracle_card.protocol.scientific_fingerprint:
            raise ValueError("candidate oracle must use the query scientific protocol")
        if self.query.hull_snapshot.chemical_system != self.oracle_card.hull_snapshot.chemical_system:
            raise ValueError("candidate query and oracle card must share a chemical-system hull")
        return self


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
        causal_hull_updates: bool = False,
    ) -> None:
        if oracle_budget <= 0:
            raise ValueError("oracle budget must be positive")
        self.acquisition = acquisition
        self.retention = retention
        self.oracle_budget = oracle_budget
        self.causal_hull_updates = causal_hull_updates
        self.active_witness_budget = int(
            getattr(acquisition, "active_witness_budget", retention.capacity)
        )

    def evaluate(self, candidates: Iterable[CandidatePoolItem]) -> ActiveDiscoveryMetrics:
        items = list(candidates)
        by_query = {item.query.query_id: item for item in items}
        if len(by_query) != len(items):
            raise ValueError("candidate pool query IDs must be unique")
        if not items:
            raise ValueError("candidate pool must be non-empty")
        remaining = dict(by_query)
        available_stable = sum(self._actual_stable(item) for item in items)
        discoveries = unstable_calls = false_stable_calls = 0
        information_seeking_unstable_calls = 0
        cumulative_regret = 0.0
        regret_cost = unstable_cost = spent = 0.0
        hull_revisions = 0
        sizes: list[int] = []
        steps: list[ActiveStep] = []
        archive: list[MaterialMemoryCard] = []
        selected_items: list[CandidatePoolItem] = []
        call_index = 0
        while remaining:
            affordable = tuple(
                item.query
                for item in remaining.values()
                if item.query.oracle_cost <= self.oracle_budget - spent + 1e-12
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
            call_index += 1
            spent += chosen.query.oracle_cost
            actual_stable = self._actual_stable(chosen)
            any_stable_before_selection = actual_stable or any(
                self._actual_stable(item)
                for query_id, item in remaining.items()
                if query_id in affordable_ids
            )
            regret = float(any_stable_before_selection and not actual_stable)
            cumulative_regret += regret
            regret_cost += chosen.query.oracle_cost * regret
            discoveries += int(actual_stable)
            unstable_calls += int(not actual_stable)
            unstable_cost += chosen.query.oracle_cost * int(not actual_stable)
            false_stable_calls += int(chosen_score.predicted_stable and not actual_stable)
            information_seeking = chosen_score.downstream_risk_reduction > 0
            information_seeking_unstable_calls += int(information_seeking and not actual_stable)
            archive.append(chosen.oracle_card)
            selected_items.append(chosen)
            future_queries = tuple(item.query for item in remaining.values())
            self.retention.admit(
                chosen.oracle_card,
                future_queries or (chosen.query,),
            )
            memory_size = min(len(self.retention.cards()), self.active_witness_budget)
            sizes.append(memory_size)
            steps.append(
                ActiveStep(
                    oracle_call_index=call_index,
                    query_id=chosen.query.query_id,
                    acquisition_score=chosen_score.score,
                    stable_score=chosen_score.stable_score,
                    stable_score_kind=chosen_score.stable_score_kind,
                    predicted_stable=chosen_score.predicted_stable,
                    actual_stable=actual_stable,
                    oracle_cost=chosen.query.oracle_cost,
                    cumulative_oracle_cost=spent,
                    discovery_regret=regret,
                    information_seeking=information_seeking,
                    downstream_risk_reduction=chosen_score.downstream_risk_reduction,
                    memory_size_after_observation=memory_size,
                    archive_size_after_observation=len(archive),
                )
            )
            if self.causal_hull_updates:
                hull_revisions += self._revise_remaining_hulls(
                    remaining,
                    chosen.oracle_card,
                    call_index,
                )
        calls = len(steps)
        final_references: dict[tuple[str, ...], float] = {}
        for selected in selected_items:
            system = selected.query.hull_snapshot.chemical_system
            reference = final_references.get(
                system,
                selected.query.hull_snapshot.reference_hull_energy_ev_per_atom,
            )
            final_references[system] = min(
                reference,
                selected.oracle_card.formation_energy_ev_per_atom,
            )
        finalized_steps: list[ActiveStep] = []
        final_discoveries = invalidated = 0
        for step, selected in zip(steps, selected_items, strict=True):
            final_reference = final_references[selected.query.hull_snapshot.chemical_system]
            final_stable = (
                selected.oracle_card.formation_energy_ev_per_atom - final_reference
                <= selected.query.stability_threshold_ev_per_atom
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
    def _actual_stable(item: CandidatePoolItem) -> bool:
        return (
            item.oracle_card.hull_distance(item.query.hull_snapshot)
            <= item.query.stability_threshold_ev_per_atom
        )

    @staticmethod
    def _revise_remaining_hulls(
        remaining: dict[str, CandidatePoolItem],
        observed: MaterialMemoryCard,
        call_index: int,
    ) -> int:
        """Causally lower same-system reference hulls after an observed phase."""

        revisions = 0
        for query_id, item in tuple(remaining.items()):
            old = item.query.hull_snapshot
            if old.chemical_system != observed.hull_snapshot.chemical_system:
                continue
            new_reference = min(
                old.reference_hull_energy_ev_per_atom,
                observed.formation_energy_ev_per_atom,
            )
            if new_reference == old.reference_hull_energy_ev_per_atom:
                continue
            built_at = max(old.built_at, observed.observed_at)
            checksum_payload = f"{old.phase_set_checksum}:{observed.card_id}:{call_index}"
            revised = old.model_copy(
                update={
                    "snapshot_id": f"{old.snapshot_id}:after:{observed.card_id}",
                    "reference_hull_energy_ev_per_atom": new_reference,
                    "phase_set_checksum": "sha256:"
                    + hashlib.sha256(checksum_payload.encode()).hexdigest(),
                    "known_through": built_at,
                    "built_at": built_at,
                }
            )
            revised_query = item.query.model_copy(
                update={"hull_snapshot": revised, "as_of": built_at}
            )
            remaining[query_id] = item.model_copy(update={"query": revised_query})
            revisions += 1
        return revisions
