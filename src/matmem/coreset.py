"""Decision-aware calibration coresets with exact streaming one-swap updates."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from itertools import combinations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.stats import spearmanr

from .calibration_utility import (
    CalibrationUtilityBuilder,
    CalibrationUtilityMatrix,
    ProperPosteriorDivergence,
    ReferencePosteriorSnapshot,
    bernoulli_log_divergence,
    reference_decision_regret,
)
from .cards import MaterialMemoryCard, MaterialQuery
from .residual_posterior import FixedKernelResidualGP


class CoresetSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_card_ids: tuple[str, ...]
    objective_value: float = Field(ge=0)
    baseline_decision_risk: float = Field(ge=0)
    facility_proxy_risk: float = Field(ge=0)
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
    spearman_facility_vs_negative_joint_risk: float | None = Field(default=None, ge=-1, le=1)
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
            facility_proxy_risk=max(0.0, baseline_risk - value),
            marginal_gains=FacilityLocationCoresetPlanner._marginal_gains(matrix, selected_ids),
            rejected_card_ids=tuple(sorted(set(matrix.witness_ids) - set(selected_ids))),
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
        selection = self.planner.preview_admit(self.cards(), card, tuple(query_pool))
        if selection.admitted_new_card:
            candidates = {**self._cards, card.card_id: card}
            self._cards = {card_id: candidates[card_id] for card_id in selection.selected_card_ids}
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
                new_id if index == evicted else card_id for index, card_id in enumerate(current_ids)
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
    candidates = _candidate_subsets(current_ids, new_card.card_id, facility_planner.capacity)
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
        facility_row.facility_location_value - current_row.facility_location_value
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
        selections_agree=set(facility_row.selected_card_ids) == set(joint_row.selected_card_ids),
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
            self._cards = {card_id: candidates[card_id] for card_id in selection.selected_card_ids}
        return selection


class PosteriorProjectionCandidate(BaseModel):
    """One candidate projection of a fixed full-evidence posterior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_key: str
    selected_card_ids: tuple[str, ...]
    proper_divergence: float = Field(ge=0)
    reference_decision_regret: float = Field(ge=0)
    reference_log_divergence: float = Field(ge=0)
    reactivation_cost: float = Field(default=0.0, ge=0)
    feasible: bool
    normalized_constraint_violation: float = Field(ge=0)


class PosteriorProjectionSelection(BaseModel):
    """Exact optimum for one explicit posterior-projection candidate space."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    divergence_kind: str
    reference_card_ids: tuple[str, ...]
    selected_card_ids: tuple[str, ...]
    candidates: tuple[PosteriorProjectionCandidate, ...]
    selected_proper_divergence: float = Field(ge=0)
    selected_decision_regret: float = Field(ge=0)
    selected_log_divergence: float = Field(ge=0)
    admitted_new_card: bool = False
    evicted_card_ids: tuple[str, ...] = ()
    used_constraint_fallback: bool = False
    reference_fit_seconds: float = Field(default=0.0, ge=0)
    reference_prediction_seconds: float = Field(default=0.0, ge=0)
    candidate_projection_seconds: float = Field(default=0.0, ge=0)
    candidate_enumeration_seconds: float = Field(default=0.0, ge=0)


class PosteriorProjectionScorer:
    """Score explicit candidate subsets against an independently chosen reference.

    This public diagnostic surface is the single implementation used by both
    online one-swap and archive-exact P3C.  It permits factorial reference and
    search-space audits without copying posterior or divergence formulas.
    """

    def __init__(
        self,
        posterior_template: FixedKernelResidualGP,
        divergence: ProperPosteriorDivergence,
        *,
        false_stable_cost: float,
        false_unstable_cost: float,
        max_decision_regret: float | None,
        max_log_divergence: float | None,
        reactivation_weight: float,
    ) -> None:
        if min(false_stable_cost, false_unstable_cost) <= 0:
            raise ValueError("posterior-projection decision costs must be positive")
        if max_decision_regret is not None and max_decision_regret < 0:
            raise ValueError("decision-regret constraint must be nonnegative")
        if max_log_divergence is not None and max_log_divergence < 0:
            raise ValueError("log-divergence constraint must be nonnegative")
        if reactivation_weight < 0:
            raise ValueError("reactivation weight must be nonnegative")
        self.posterior_template = posterior_template
        self.divergence = divergence
        self.false_stable_cost = false_stable_cost
        self.false_unstable_cost = false_unstable_cost
        self.max_decision_regret = max_decision_regret
        self.max_log_divergence = max_log_divergence
        self.reactivation_weight = reactivation_weight

    @staticmethod
    def _violation(value: float, limit: float | None) -> float:
        if limit is None or value <= limit:
            return 0.0
        return (value - limit) / limit if limit > 0 else value

    def score(
        self,
        *,
        reference_cards: tuple[MaterialMemoryCard, ...],
        candidate_sets: tuple[tuple[str, ...], ...],
        cards_by_id: Mapping[str, MaterialMemoryCard],
        queries: tuple[MaterialQuery, ...],
        current_ids: tuple[str, ...],
        reactivation_cost_by_card: Mapping[str, float] | None = None,
    ) -> PosteriorProjectionSelection:
        if not candidate_sets:
            raise ValueError("posterior projection requires at least one candidate set")
        reference_fit_started = time.perf_counter()
        reference_posterior = self.posterior_template.clone_unfit().fit(reference_cards)
        reference_fit_seconds = time.perf_counter() - reference_fit_started
        reference_prediction_started = time.perf_counter()
        reference_prediction = reference_posterior.predict(queries)
        reference_prediction_seconds = time.perf_counter() - reference_prediction_started
        reference = ReferencePosteriorSnapshot.from_prediction(queries, reference_prediction)
        weights = self.divergence.reference_weights(
            reference,
            false_stable_cost=self.false_stable_cost,
            false_unstable_cost=self.false_unstable_cost,
        )
        cost_by_card = reactivation_cost_by_card or {}
        current = set(current_ids)
        candidates: list[PosteriorProjectionCandidate] = []
        candidate_projection_started = time.perf_counter()
        for card_ids in candidate_sets:
            if len(set(card_ids)) != len(card_ids):
                raise ValueError("posterior projection candidate IDs must be unique")
            missing = set(card_ids) - cards_by_id.keys()
            if missing:
                raise ValueError(
                    f"posterior projection candidate IDs are missing: {sorted(missing)}"
                )
            prediction = (
                self.posterior_template.clone_unfit()
                .fit(tuple(cards_by_id[card_id] for card_id in card_ids))
                .predict(queries)
            )
            per_query = self.divergence.per_query(reference, prediction)
            candidate_probability = np.asarray(prediction.stable_probability, dtype=float)
            proper = float(np.sum(weights * per_query))
            log_divergence = float(
                np.sum(
                    [
                        weights[index]
                        * bernoulli_log_divergence(
                            float(reference.stable_probability[index]),
                            float(candidate_probability[index]),
                        )
                        for index in range(len(queries))
                    ]
                )
            )
            decision_regret = float(
                np.sum(
                    [
                        weights[index]
                        * reference_decision_regret(
                            float(reference.stable_probability[index]),
                            float(candidate_probability[index]),
                            false_stable_cost=self.false_stable_cost,
                            false_unstable_cost=self.false_unstable_cost,
                        )
                        for index in range(len(queries))
                    ]
                )
            )
            violation = self._violation(
                decision_regret, self.max_decision_regret
            ) + self._violation(log_divergence, self.max_log_divergence)
            reactivation_cost = self.reactivation_weight * sum(
                float(cost_by_card.get(card_id, 0.0)) for card_id in set(card_ids) - current
            )
            candidates.append(
                PosteriorProjectionCandidate(
                    candidate_key=_candidate_key(card_ids),
                    selected_card_ids=card_ids,
                    proper_divergence=proper,
                    reference_decision_regret=decision_regret,
                    reference_log_divergence=log_divergence,
                    reactivation_cost=reactivation_cost,
                    feasible=violation <= 1e-15,
                    normalized_constraint_violation=max(0.0, violation),
                )
            )
        candidate_projection_seconds = time.perf_counter() - candidate_projection_started
        feasible = [item for item in candidates if item.feasible]
        used_fallback = not feasible
        eligible = feasible or candidates
        if used_fallback:
            selected = min(
                eligible,
                key=lambda item: (
                    item.normalized_constraint_violation,
                    item.proper_divergence + item.reactivation_cost,
                    tuple(sorted(item.selected_card_ids)),
                ),
            )
        else:
            selected = min(
                eligible,
                key=lambda item: (
                    item.proper_divergence + item.reactivation_cost,
                    tuple(sorted(item.selected_card_ids)),
                ),
            )
        return PosteriorProjectionSelection(
            divergence_kind=self.divergence.kind,
            reference_card_ids=tuple(card.card_id for card in reference_cards),
            selected_card_ids=selected.selected_card_ids,
            candidates=tuple(candidates),
            selected_proper_divergence=selected.proper_divergence,
            selected_decision_regret=selected.reference_decision_regret,
            selected_log_divergence=selected.reference_log_divergence,
            admitted_new_card=False,
            evicted_card_ids=tuple(sorted(current - set(selected.selected_card_ids))),
            used_constraint_fallback=used_fallback,
            reference_fit_seconds=reference_fit_seconds,
            reference_prediction_seconds=reference_prediction_seconds,
            candidate_projection_seconds=candidate_projection_seconds,
        )


class PosteriorProjectionOneSwapPlanner:
    """Exactly project the current K+1 posterior onto its drop-one subsets."""

    def __init__(
        self,
        capacity: int,
        posterior_template: FixedKernelResidualGP,
        divergence: ProperPosteriorDivergence,
        *,
        false_stable_cost: float = 5.0,
        false_unstable_cost: float = 1.0,
        max_decision_regret: float | None = None,
        max_log_divergence: float | None = None,
    ) -> None:
        if capacity < 0:
            raise ValueError("posterior-projection capacity cannot be negative")
        self.capacity = capacity
        self.scorer = PosteriorProjectionScorer(
            posterior_template,
            divergence,
            false_stable_cost=false_stable_cost,
            false_unstable_cost=false_unstable_cost,
            max_decision_regret=max_decision_regret,
            max_log_divergence=max_log_divergence,
            reactivation_weight=0.0,
        )

    def preview_admit(
        self,
        current_cards: Iterable[MaterialMemoryCard],
        new_card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> PosteriorProjectionSelection:
        current = tuple(current_cards)
        current_ids = tuple(card.card_id for card in current)
        if len(set(current_ids)) != len(current_ids) or len(current) > self.capacity:
            raise ValueError("invalid current posterior-projection working set")
        if new_card.card_id in set(current_ids):
            raise ValueError("new posterior-projection card is already active")
        reference_cards = (*current, new_card)
        cards_by_id = {card.card_id: card for card in reference_cards}
        enumeration_started = time.perf_counter()
        candidate_sets = _candidate_subsets(current_ids, new_card.card_id, self.capacity)
        enumeration_seconds = time.perf_counter() - enumeration_started
        selection = self.scorer.score(
            reference_cards=reference_cards,
            candidate_sets=candidate_sets,
            cards_by_id=cards_by_id,
            queries=tuple(query_pool),
            current_ids=current_ids,
        )
        return selection.model_copy(
            update={
                "admitted_new_card": new_card.card_id in selection.selected_card_ids,
                "candidate_enumeration_seconds": enumeration_seconds,
            }
        )


class ExactArchivePosteriorProjectionPlanner:
    """Exact archive projection diagnostic for the small frozen B/K regime."""

    def __init__(
        self,
        capacity: int,
        posterior_template: FixedKernelResidualGP,
        divergence: ProperPosteriorDivergence,
        *,
        false_stable_cost: float = 5.0,
        false_unstable_cost: float = 1.0,
        max_decision_regret: float | None = None,
        max_log_divergence: float | None = None,
        reactivation_weight: float = 0.0,
    ) -> None:
        if capacity < 0:
            raise ValueError("archive posterior-projection capacity cannot be negative")
        self.capacity = capacity
        self.scorer = PosteriorProjectionScorer(
            posterior_template,
            divergence,
            false_stable_cost=false_stable_cost,
            false_unstable_cost=false_unstable_cost,
            max_decision_regret=max_decision_regret,
            max_log_divergence=max_log_divergence,
            reactivation_weight=reactivation_weight,
        )

    def select(
        self,
        archive: Iterable[MaterialMemoryCard],
        query_pool: Iterable[MaterialQuery],
        *,
        current_cards: Iterable[MaterialMemoryCard] = (),
        reactivation_cost_by_card: Mapping[str, float] | None = None,
    ) -> PosteriorProjectionSelection:
        cards = tuple(archive)
        card_ids = tuple(card.card_id for card in cards)
        if len(set(card_ids)) != len(card_ids):
            raise ValueError("archive posterior-projection cards must be unique")
        current_ids = tuple(card.card_id for card in current_cards)
        if not set(current_ids).issubset(card_ids):
            raise ValueError("active cards must be contained in the archive")
        enumeration_started = time.perf_counter()
        candidate_sets = tuple(
            subset
            for size in range(min(self.capacity, len(cards)) + 1)
            for subset in combinations(card_ids, size)
        )
        enumeration_seconds = time.perf_counter() - enumeration_started
        selection = self.scorer.score(
            reference_cards=cards,
            candidate_sets=candidate_sets,
            cards_by_id={card.card_id: card for card in cards},
            queries=tuple(query_pool),
            current_ids=current_ids,
            reactivation_cost_by_card=reactivation_cost_by_card,
        )
        return selection.model_copy(update={"candidate_enumeration_seconds": enumeration_seconds})


class StreamingPosteriorProjectionCoreset:
    """Bounded active state selected by proper posterior projection."""

    def __init__(self, planner: PosteriorProjectionOneSwapPlanner) -> None:
        self.planner = planner
        self.capacity = planner.capacity
        self._cards: dict[str, MaterialMemoryCard] = {}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards.values())

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> PosteriorProjectionSelection:
        selection = self.planner.preview_admit(self.cards(), card, tuple(query_pool))
        candidates = {**self._cards, card.card_id: card}
        self._cards = {card_id: candidates[card_id] for card_id in selection.selected_card_ids}
        return selection
