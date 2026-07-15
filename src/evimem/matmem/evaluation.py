"""Leakage-safe, chronological evaluation of bounded materials memory."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cards import MaterialMemoryCard, MaterialQuery
from .coreset import DecisionAwareOnlineCoreset
from .residual import ResidualCorrector
from .risk import ProtocolRiskController, ScreeningDecision


class DeploymentStrategy(StrEnum):
    """Deployment semantics reported separately to avoid abstention artifacts."""

    BASE_ONLY = "base_only"
    FALLBACK_BASE = "fallback_base"
    STRICT_ABSTAIN = "strict_abstain"


class StreamEvent(BaseModel):
    """The oracle card is visible only after the query has been screened."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: MaterialQuery
    oracle_card: MaterialMemoryCard

    @model_validator(mode="after")
    def _same_structure(self) -> StreamEvent:
        if self.query.structure_hash != self.oracle_card.structure_hash:
            raise ValueError("stream event query and oracle card must share a structure hash")
        if self.query.protocol.scientific_fingerprint != self.oracle_card.protocol.scientific_fingerprint:
            raise ValueError("stream event oracle must use the query scientific protocol")
        if self.query.hull_snapshot.chemical_system != self.oracle_card.hull_snapshot.chemical_system:
            raise ValueError("stream event query and oracle card must share a chemical-system hull")
        return self


class DiscoveryMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_count: int
    deployment_strategy: DeploymentStrategy
    coverage: float
    screened_stable_count: int
    actual_stable_count: int
    cumulative_true_stable_discoveries: int
    discovery_recall: float
    false_stable_count: int
    false_stable_rate: float
    false_unstable_count: int
    selective_screening_risk: float
    abstention_rate: float
    average_memory_size: float
    final_memory_size: int
    capacity: int


class ScreeningOutcome(BaseModel):
    """Per-query outcome for coverage-matched reporting, never a training label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    decision: ScreeningDecision
    actual_stable: bool
    confidence: float | None = None


class RiskCoveragePoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: float = Field(ge=0, le=1)
    selective_screening_risk: float = Field(ge=0, le=1)
    false_stable_rate: float = Field(ge=0, le=1)
    discovery_recall: float = Field(ge=0, le=1)


class OnlineDiscoveryEvaluator:
    """Evaluates one fixed oracle budget and one fixed memory capacity."""

    def __init__(
        self,
        coreset: DecisionAwareOnlineCoreset,
        corrector: ResidualCorrector,
        risk_controller: ProtocolRiskController,
        *,
        recent_query_window: int = 64,
        deployment_strategy: DeploymentStrategy = DeploymentStrategy.STRICT_ABSTAIN,
    ) -> None:
        if recent_query_window < 1:
            raise ValueError("recent_query_window must be positive")
        self.coreset = coreset
        self.corrector = corrector
        self.risk_controller = risk_controller
        self.recent_query_window = recent_query_window
        self.deployment_strategy = deployment_strategy

    def _deployment_decision(self, event: StreamEvent) -> tuple[ScreeningDecision, float | None]:
        base_hull = event.query.base_hull_distance_ev_per_atom
        if self.deployment_strategy == DeploymentStrategy.BASE_ONLY:
            return (
                ScreeningDecision.STABLE
                if base_hull <= event.query.stability_threshold_ev_per_atom
                else ScreeningDecision.NOT_STABLE,
                event.query.stability_threshold_ev_per_atom - base_hull,
            )
        correction = self.corrector.correct(event.query, self.coreset.cards())
        risk_decision = self.risk_controller.screen(event.query, correction)
        confidence = (
            event.query.stability_threshold_ev_per_atom - risk_decision.upper_hull_distance_ev_per_atom
            if risk_decision.upper_hull_distance_ev_per_atom is not None
            else None
        )
        if (
            self.deployment_strategy == DeploymentStrategy.FALLBACK_BASE
            and risk_decision.decision == ScreeningDecision.ABSTAIN
        ):
            return (
                ScreeningDecision.STABLE
                if base_hull <= event.query.stability_threshold_ev_per_atom
                else ScreeningDecision.NOT_STABLE,
                event.query.stability_threshold_ev_per_atom - base_hull,
            )
        return risk_decision.decision, confidence

    def evaluate(self, events: Iterable[StreamEvent]) -> DiscoveryMetrics:
        metrics, _ = self.evaluate_with_outcomes(events)
        return metrics

    def evaluate_with_outcomes(
        self,
        events: Iterable[StreamEvent],
    ) -> tuple[DiscoveryMetrics, tuple[ScreeningOutcome, ...]]:
        history: list[MaterialQuery] = []
        stable_screens = true_stable = actual_stable_count = false_stable = false_unstable = abstained = 0
        sizes: list[int] = []
        outcomes: list[ScreeningOutcome] = []
        previous_query_time = None
        for event in events:
            if previous_query_time is not None and event.query.as_of < previous_query_time:
                raise ValueError("online discovery events must be chronological")
            previous_query_time = event.query.as_of
            decision, confidence = self._deployment_decision(event)
            actual_stable = (
                event.oracle_card.hull_distance(event.query.hull_snapshot)
                <= event.query.stability_threshold_ev_per_atom
            )
            actual_stable_count += int(actual_stable)
            if decision == ScreeningDecision.ABSTAIN:
                abstained += 1
            elif decision == ScreeningDecision.STABLE:
                stable_screens += 1
                if actual_stable:
                    true_stable += 1
                else:
                    false_stable += 1
            elif actual_stable:
                false_unstable += 1
            outcomes.append(
                ScreeningOutcome(
                    query_id=event.query.query_id,
                    decision=decision,
                    actual_stable=actual_stable,
                    confidence=confidence,
                )
            )
            history.append(event.query)
            self.coreset.admit(event.oracle_card, history[-self.recent_query_window :])
            sizes.append(len(self.coreset.cards()))
        count = len(sizes)
        accepted = count - abstained
        metrics = DiscoveryMetrics(
            event_count=count,
            deployment_strategy=self.deployment_strategy,
            coverage=accepted / count if count else 0.0,
            screened_stable_count=stable_screens,
            actual_stable_count=actual_stable_count,
            cumulative_true_stable_discoveries=true_stable,
            discovery_recall=true_stable / actual_stable_count if actual_stable_count else 0.0,
            false_stable_count=false_stable,
            false_stable_rate=false_stable / stable_screens if stable_screens else 0.0,
            false_unstable_count=false_unstable,
            selective_screening_risk=(false_stable + false_unstable) / accepted if accepted else 0.0,
            abstention_rate=abstained / count if count else 0.0,
            average_memory_size=sum(sizes) / count if count else 0.0,
            final_memory_size=len(self.coreset.cards()),
            capacity=self.coreset.capacity,
        )
        return metrics, tuple(outcomes)


def risk_coverage_curve(
    outcomes: Iterable[ScreeningOutcome],
    coverages: Iterable[float],
) -> tuple[RiskCoveragePoint, ...]:
    """Compute selective risk and discovery recall at shared coverage levels."""

    items = list(outcomes)
    if not items:
        return ()
    ranked = sorted(
        (item for item in items if item.confidence is not None),
        key=lambda item: (-float(item.confidence), item.query_id),
    )
    total_stable = sum(item.actual_stable for item in items)
    points: list[RiskCoveragePoint] = []
    for coverage in coverages:
        if not 0 <= coverage <= 1:
            raise ValueError("requested coverage must be in [0, 1]")
        retained = ranked[: min(len(ranked), round(coverage * len(items)))]
        stable_predictions = [item for item in retained if item.decision == ScreeningDecision.STABLE]
        false_stable = sum(not item.actual_stable for item in stable_predictions)
        false_unstable = sum(
            item.actual_stable for item in retained if item.decision == ScreeningDecision.NOT_STABLE
        )
        discoveries = sum(item.actual_stable for item in stable_predictions)
        points.append(
            RiskCoveragePoint(
                coverage=len(retained) / len(items),
                selective_screening_risk=(false_stable + false_unstable) / len(retained)
                if retained
                else 0.0,
                false_stable_rate=false_stable / len(stable_predictions)
                if stable_predictions
                else 0.0,
                discovery_recall=discoveries / total_stable if total_stable else 0.0,
            )
        )
    return tuple(points)
