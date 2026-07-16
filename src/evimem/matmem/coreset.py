"""Decision-aware calibration coresets with exact streaming one-swap updates."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

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


class JointPosteriorRiskSelection(BaseModel):
    """Exact minimum-risk choice in one streaming one-swap neighborhood."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_card_ids: tuple[str, ...]
    weighted_joint_risk: float = Field(ge=0)
    previous_weighted_joint_risk: float = Field(ge=0)
    candidate_weighted_joint_risks: dict[str, float]
    admitted_new_card: bool = False
    evicted_card_ids: tuple[str, ...] = ()
    objective_improvement: float = Field(default=0.0, ge=0)


class ObjectiveFidelityCandidate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_key: str
    selected_card_ids: tuple[str, ...]
    facility_location_value: float = Field(ge=0)
    weighted_joint_risk: float = Field(ge=0)


class ObjectiveFidelityDiagnostic(BaseModel):
    """Agreement of facility location with true joint posterior risk."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidates: tuple[ObjectiveFidelityCandidate, ...]
    facility_selected_card_ids: tuple[str, ...]
    joint_risk_selected_card_ids: tuple[str, ...]
    selections_agree: bool
    spearman_facility_vs_negative_joint_risk: float | None = Field(
        default=None, ge=-1, le=1
    )
    facility_joint_risk_regret: float = Field(ge=0)


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


def _candidate_subsets(
    current_ids: tuple[str, ...],
    new_id: str,
    capacity: int,
) -> tuple[tuple[str, ...], ...]:
    if capacity < 0 or len(current_ids) > capacity:
        raise ValueError("invalid one-swap neighborhood capacity")
    candidates: list[tuple[str, ...]] = [current_ids]
    if capacity == 0:
        return tuple(candidates)
    if len(current_ids) < capacity:
        candidates.append((*current_ids, new_id))
    else:
        candidates.extend(
            tuple(
                new_id if index == evicted else card_id
                for index, card_id in enumerate(current_ids)
            )
            for evicted in range(len(current_ids))
        )
    return tuple(dict.fromkeys(candidates))


def _candidate_key(card_ids: tuple[str, ...]) -> str:
    return "|".join(sorted(card_ids)) if card_ids else "<empty>"


class JointPosteriorRiskOneSwapPlanner:
    """Minimize actual weighted joint-GP Bayes risk after one arrival.

    This has no submodular or global-optimality claim. It is exact only over
    rejection and the legal one-swap choices in ``current union {new}``.
    """

    def __init__(
        self,
        capacity: int,
        utility_builder: CalibrationUtilityBuilder,
        *,
        min_risk_improvement: float = 0.0,
    ) -> None:
        if capacity < 0:
            raise ValueError("joint-risk coreset capacity cannot be negative")
        if min_risk_improvement < 0:
            raise ValueError("minimum risk improvement cannot be negative")
        self.capacity = capacity
        self.utility_builder = utility_builder
        self.min_risk_improvement = min_risk_improvement

    def preview_admit(
        self,
        current_cards: Iterable[MaterialMemoryCard],
        new_card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> JointPosteriorRiskSelection:
        current = tuple(current_cards)
        current_ids = tuple(card.card_id for card in current)
        if len(set(current_ids)) != len(current_ids):
            raise ValueError("current joint-risk cards must have unique IDs")
        if new_card.card_id in set(current_ids):
            raise ValueError("new joint-risk card is already active")
        candidate_sets = _candidate_subsets(current_ids, new_card.card_id, self.capacity)
        cards_by_id = {card.card_id: card for card in (*current, new_card)}
        queries = tuple(query_pool)
        risks = {
            _candidate_key(card_ids): self.utility_builder.weighted_decision_risk(
                queries,
                tuple(cards_by_id[card_id] for card_id in card_ids),
            )
            for card_ids in candidate_sets
        }
        previous = risks[_candidate_key(current_ids)]
        best_ids = current_ids
        best_risk = previous
        for candidate_ids in candidate_sets[1:]:
            risk = risks[_candidate_key(candidate_ids)]
            if risk < best_risk:
                best_ids = candidate_ids
                best_risk = risk
            elif risk == best_risk and best_ids != current_ids:
                best_ids = min(best_ids, candidate_ids)
        improvement = previous - best_risk
        if improvement <= self.min_risk_improvement:
            best_ids = current_ids
            best_risk = previous
            improvement = 0.0
        return JointPosteriorRiskSelection(
            selected_card_ids=best_ids,
            weighted_joint_risk=best_risk,
            previous_weighted_joint_risk=previous,
            candidate_weighted_joint_risks=risks,
            admitted_new_card=new_card.card_id in best_ids,
            evicted_card_ids=tuple(sorted(set(current_ids) - set(best_ids))),
            objective_improvement=improvement,
        )


def compare_facility_and_joint_objectives(
    current_cards: Iterable[MaterialMemoryCard],
    new_card: MaterialMemoryCard,
    query_pool: Iterable[MaterialQuery],
    facility_planner: FacilityLocationCoresetPlanner,
    joint_planner: JointPosteriorRiskOneSwapPlanner,
) -> ObjectiveFidelityDiagnostic:
    """Score the same streaming neighborhood under both observable objectives."""

    if facility_planner.capacity != joint_planner.capacity:
        raise ValueError("objective-fidelity planners must share capacity")
    current = tuple(current_cards)
    queries = tuple(query_pool)
    current_ids = tuple(card.card_id for card in current)
    candidates = _candidate_subsets(
        current_ids, new_card.card_id, facility_planner.capacity
    )
    cards_by_id = {card.card_id: card for card in (*current, new_card)}
    matrix, _ = facility_planner.build_utility_matrix(queries, cards_by_id.values())
    rows = tuple(
        ObjectiveFidelityCandidate(
            candidate_key=_candidate_key(card_ids),
            selected_card_ids=card_ids,
            facility_location_value=matrix.value(card_ids),
            weighted_joint_risk=joint_planner.utility_builder.weighted_decision_risk(
                queries,
                tuple(cards_by_id[card_id] for card_id in card_ids),
            ),
        )
        for card_ids in candidates
    )
    current_row = rows[0]
    facility_row = current_row
    joint_row = current_row
    for row in rows[1:]:
        if row.facility_location_value > facility_row.facility_location_value:
            facility_row = row
        elif (
            row.facility_location_value == facility_row.facility_location_value
            and facility_row is not current_row
            and row.selected_card_ids < facility_row.selected_card_ids
        ):
            facility_row = row
        if row.weighted_joint_risk < joint_row.weighted_joint_risk:
            joint_row = row
        elif (
            row.weighted_joint_risk == joint_row.weighted_joint_risk
            and joint_row is not current_row
            and row.selected_card_ids < joint_row.selected_card_ids
        ):
            joint_row = row
    if (
        facility_row.facility_location_value
        - current_row.facility_location_value
        <= facility_planner.min_admission_gain
    ):
        facility_row = current_row
    if (
        current_row.weighted_joint_risk - joint_row.weighted_joint_risk
        <= joint_planner.min_risk_improvement
    ):
        joint_row = current_row
    facility_values = np.asarray([row.facility_location_value for row in rows])
    negative_risks = -np.asarray([row.weighted_joint_risk for row in rows])
    correlation: float | None = None
    if len(rows) >= 2 and np.ptp(facility_values) > 0 and np.ptp(negative_risks) > 0:
        statistic = float(spearmanr(facility_values, negative_risks).statistic)
        if np.isfinite(statistic):
            correlation = statistic
    return ObjectiveFidelityDiagnostic(
        candidates=rows,
        facility_selected_card_ids=facility_row.selected_card_ids,
        joint_risk_selected_card_ids=joint_row.selected_card_ids,
        selections_agree=set(facility_row.selected_card_ids)
        == set(joint_row.selected_card_ids),
        spearman_facility_vs_negative_joint_risk=correlation,
        facility_joint_risk_regret=max(
            0.0,
            facility_row.weighted_joint_risk - joint_row.weighted_joint_risk,
        ),
    )


class StreamingJointPosteriorRiskCoreset:
    """Bounded active state selected by observable joint posterior risk."""

    def __init__(self, planner: JointPosteriorRiskOneSwapPlanner) -> None:
        self.planner = planner
        self.capacity = planner.capacity
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards.values())

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> JointPosteriorRiskSelection:
        selection = self.planner.preview_admit(self.cards(), card, tuple(query_pool))
        if selection.admitted_new_card:
            candidates = {**self._cards, card.card_id: card}
            self._cards = {
                card_id: candidates[card_id]
                for card_id in selection.selected_card_ids
            }
        return selection
