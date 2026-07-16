"""Decision-aware calibration coresets with exact streaming one-swap updates."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from .calibration_utility import CalibrationUtilityBuilder, CalibrationUtilityMatrix
from .cards import MaterialMemoryCard, MaterialQuery


class CoresetSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_card_ids: tuple[str, ...]
    objective_value: float = Field(ge=0)
    baseline_decision_risk: float = Field(ge=0)
    selected_decision_risk: float = Field(ge=0)
    marginal_gains: dict[str, float]
    rejected_card_ids: tuple[str, ...] = ()
    admitted_new_card: bool = False
    evicted_card_ids: tuple[str, ...] = ()
    objective_improvement: float = Field(default=0.0, ge=0)


class FacilityLocationCoresetPlanner:
    """Plan bounded calibration sets under ``F_t(M)=sum_u max_m G_t(u,m)``."""

    def __init__(
        self,
        capacity: int,
        utility_builder: CalibrationUtilityBuilder,
        *,
        min_admission_gain: float = 0.0,
    ) -> None:
        if capacity < 0:
            raise ValueError("calibration coreset capacity cannot be negative")
        if min_admission_gain < 0:
            raise ValueError("minimum admission gain cannot be negative")
        self.capacity = capacity
        self.utility_builder = utility_builder
        self.min_admission_gain = min_admission_gain

    def build_utility_matrix(
        self,
        queries: Iterable[MaterialQuery],
        witnesses: Iterable[MaterialMemoryCard],
    ) -> tuple[CalibrationUtilityMatrix, float]:
        return self.utility_builder.build(queries, witnesses)

    @staticmethod
    def _marginal_gains(
        matrix: CalibrationUtilityMatrix,
        selected_ids: tuple[str, ...],
    ) -> dict[str, float]:
        running: list[str] = []
        result: dict[str, float] = {}
        for card_id in selected_ids:
            result[card_id] = matrix.marginal_gain(running, card_id)
            running.append(card_id)
        return result

    @staticmethod
    def _selection(
        matrix: CalibrationUtilityMatrix,
        baseline_risk: float,
        selected_ids: tuple[str, ...],
        *,
        admitted_new_card: bool = False,
        evicted_card_ids: tuple[str, ...] = (),
        objective_improvement: float = 0.0,
    ) -> CoresetSelection:
        value = matrix.value(selected_ids)
        return CoresetSelection(
            selected_card_ids=selected_ids,
            objective_value=value,
            baseline_decision_risk=baseline_risk,
            selected_decision_risk=max(0.0, baseline_risk - value),
            marginal_gains=FacilityLocationCoresetPlanner._marginal_gains(
                matrix, selected_ids
            ),
            rejected_card_ids=tuple(
                sorted(set(matrix.witness_ids) - set(selected_ids))
            ),
            admitted_new_card=admitted_new_card,
            evicted_card_ids=evicted_card_ids,
            objective_improvement=objective_improvement,
        )

    def preview_admit(
        self,
        current_cards: Iterable[MaterialMemoryCard],
        new_card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> CoresetSelection:
        """Exactly optimize the current active set plus one new observation.

        At full capacity the only legal streaming choices are rejection or one
        swap.  This is exact for that ``M_{t-1} union {m_t}`` neighborhood, not
        a claim of global optimality over the immutable archive.
        """

        current = tuple(current_cards)
        if len(current) > self.capacity:
            raise ValueError("current calibration set exceeds planner capacity")
        if new_card.card_id in {card.card_id for card in current}:
            raise ValueError("new calibration card is already active")
        candidates = (*current, new_card)
        matrix, baseline = self.build_utility_matrix(query_pool, candidates)
        current_ids = tuple(card.card_id for card in current)
        current_value = matrix.value(current_ids)
        if self.capacity == 0:
            return self._selection(matrix, baseline, current_ids)
        if len(current) < self.capacity:
            proposals = [(*current_ids, new_card.card_id)]
        else:
            proposals = [
                tuple(
                    new_card.card_id if index == evicted else card.card_id
                    for index, card in enumerate(current)
                )
                for evicted in range(len(current))
            ]
        best_ids = current_ids
        best_value = current_value
        for proposal in proposals:
            value = matrix.value(proposal)
            if value > best_value:
                best_ids = proposal
                best_value = value
            elif value == best_value and best_ids != current_ids:
                best_ids = min(best_ids, proposal)
        improvement = best_value - current_value
        if improvement <= self.min_admission_gain:
            return self._selection(matrix, baseline, current_ids)
        evicted = tuple(sorted(set(current_ids) - set(best_ids)))
        return self._selection(
            matrix,
            baseline,
            best_ids,
            admitted_new_card=True,
            evicted_card_ids=evicted,
            objective_improvement=improvement,
        )

    def select_from_archive_greedy(
        self,
        archive: Iterable[MaterialMemoryCard],
        query_pool: Iterable[MaterialQuery],
    ) -> CoresetSelection:
        """Greedy full-archive selection with the standard ``1-1/e`` bound."""

        cards = tuple(archive)
        matrix, baseline = self.build_utility_matrix(query_pool, cards)
        selected: list[str] = []
        while len(selected) < self.capacity:
            choices = [
                (matrix.marginal_gain(selected, card_id), card_id)
                for card_id in matrix.witness_ids
                if card_id not in selected
            ]
            if not choices:
                break
            gain, card_id = min(choices, key=lambda item: (-item[0], item[1]))
            if gain <= self.min_admission_gain:
                break
            selected.append(card_id)
        return self._selection(matrix, baseline, tuple(selected))


class StreamingCalibrationCoreset:
    """Bounded active calibration state backed by exact one-step previews."""

    def __init__(self, planner: FacilityLocationCoresetPlanner) -> None:
        self.planner = planner
        self.capacity = planner.capacity
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards.values())

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> CoresetSelection:
        selection = self.planner.preview_admit(
            self.cards(), card, tuple(query_pool)
        )
        if selection.admitted_new_card:
            candidates = {**self._cards, card.card_id: card}
            self._cards = {
                card_id: candidates[card_id]
                for card_id in selection.selected_card_ids
            }
        return selection
