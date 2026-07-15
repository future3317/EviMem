"""Oracle-blind acquisition policies for active-witness-budgeted discovery.

The policies in this module may inspect candidate queries and previously
observed memory cards. They never receive the oracle card of an unqueried
candidate. Oracle outcomes are used only by the evaluator after selection.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .boundary import BoundaryRiskPotential, BoundaryWitness, BruteForceRetentionSolver
from .cards import MaterialMemoryCard, MaterialQuery
from .protocols import ProtocolCompatibilityResolver
from .residual import cosine_similarity


class AcquisitionScore(BaseModel):
    """Auditable score computed without access to a candidate's oracle result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    score: float
    stable_score: float = Field(ge=0, le=1)
    stable_score_kind: Literal["model_probability", "two_scenario_weight", "fixed_prior"] = (
        "model_probability"
    )
    expected_discovery_utility: float
    exploration_bonus: float = Field(ge=0)
    downstream_risk_reduction: float = Field(default=0.0, ge=0)
    novelty: float = Field(ge=0, le=1)
    compatible_witness_count: int = Field(ge=0)
    predicted_stable: bool


class ProtocolAwareBoundaryAcquisition:
    """Cost-sensitive acquisition from protocol-compatible residual witnesses.

    Past residuals induce a kernel-weighted probability that a candidate lies
    below the current causal hull threshold. The acquisition score combines
    expected discovery utility with an uncertainty/novelty bonus. Unsupported
    protocols contribute no witness and therefore cannot cause silent transfer.
    """

    def __init__(
        self,
        resolver: ProtocolCompatibilityResolver,
        *,
        discovery_reward: float = 5.0,
        false_stable_cost: float = 1.0,
        false_unstable_cost: float = 1.0,
        exploration_weight: float = 0.5,
        prior_strength: float = 1.0,
        boundary_temperature_ev_per_atom: float = 0.04,
        minimum_similarity: float = 0.05,
    ) -> None:
        if min(discovery_reward, false_stable_cost, false_unstable_cost) <= 0:
            raise ValueError("acquisition rewards and costs must be positive")
        if exploration_weight < 0 or prior_strength <= 0:
            raise ValueError("exploration weight must be non-negative and prior strength positive")
        if boundary_temperature_ev_per_atom <= 0:
            raise ValueError("boundary temperature must be positive")
        if not 0 <= minimum_similarity <= 1:
            raise ValueError("minimum similarity must be within [0, 1]")
        self.resolver = resolver
        self.discovery_reward = discovery_reward
        self.false_stable_cost = false_stable_cost
        self.false_unstable_cost = false_unstable_cost
        self.exploration_weight = exploration_weight
        self.prior_strength = prior_strength
        self.boundary_temperature_ev_per_atom = boundary_temperature_ev_per_atom
        self.minimum_similarity = minimum_similarity

    def _base_prior(self, query: MaterialQuery) -> float:
        margin = (
            query.stability_threshold_ev_per_atom - query.base_hull_distance_ev_per_atom
        ) / self.boundary_temperature_ev_per_atom
        if margin >= 0:
            return 1.0 / (1.0 + math.exp(-min(margin, 50.0)))
        exp_margin = math.exp(max(margin, -50.0))
        return exp_margin / (1.0 + exp_margin)

    def score(
        self,
        query: MaterialQuery,
        cards: Iterable[MaterialMemoryCard],
    ) -> AcquisitionScore:
        prior = self._base_prior(query)
        stable_mass = self.prior_strength * prior
        total_mass = self.prior_strength
        strongest_support = 0.0
        witness_count = 0
        for card in cards:
            compatibility = self.resolver.resolve(card.protocol, query.protocol)
            residual = compatibility.transfer_residual(card.oracle_residual_ev_per_atom)
            if residual is None:
                continue
            similarity = max(0.0, cosine_similarity(query.embedding, card.embedding))
            if similarity < self.minimum_similarity:
                continue
            protocol_weight = 1.0 / (1.0 + compatibility.uncertainty_radius_ev_per_atom)
            weight = similarity * protocol_weight * card.quality_weight
            if weight <= 0:
                continue
            proxy_energy = query.base_predicted_formation_energy_ev_per_atom + residual
            proxy_stable = query.hull_distance(proxy_energy) <= query.stability_threshold_ev_per_atom
            stable_mass += weight * float(proxy_stable)
            total_mass += weight
            strongest_support = max(strongest_support, min(1.0, weight))
            witness_count += 1
        stable_score = stable_mass / total_mass
        expected_utility = (
            self.discovery_reward * stable_score
            - self.false_stable_cost * (1.0 - stable_score)
        )
        novelty = 1.0 - strongest_support
        uncertainty = 4.0 * stable_score * (1.0 - stable_score)
        exploration_bonus = self.exploration_weight * uncertainty * novelty
        stable_threshold = self.false_stable_cost / (
            self.false_stable_cost + self.false_unstable_cost
        )
        return AcquisitionScore(
            query_id=query.query_id,
            score=(expected_utility + exploration_bonus) / query.oracle_cost,
            stable_score=stable_score,
            expected_discovery_utility=expected_utility,
            exploration_bonus=exploration_bonus,
            novelty=novelty,
            compatible_witness_count=witness_count,
            predicted_stable=stable_score >= stable_threshold,
        )

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        memory = tuple(cards)
        scores = [self.score(query, memory) for query in queries]
        return tuple(sorted(scores, key=lambda item: (-item.score, item.query_id)))


class OnDemandKNNArchiveAcquisition:
    """Strong archive baseline retrieving at most ``K`` witnesses per query.

    This policy keeps every observed result in the supplied archive and pays no
    persistent-retention penalty.  It is deliberately strong: if it dominates
    a persistent working set, the scientific case for retention is weakened.
    """

    def __init__(
        self,
        base: ProtocolAwareBoundaryAcquisition,
        *,
        active_witness_budget: int,
    ) -> None:
        if active_witness_budget < 0:
            raise ValueError("active witness budget cannot be negative")
        self.base = base
        self.active_witness_budget = active_witness_budget

    def _retrieve(
        self,
        query: MaterialQuery,
        archive: tuple[MaterialMemoryCard, ...],
    ) -> tuple[MaterialMemoryCard, ...]:
        compatible: list[tuple[float, str, MaterialMemoryCard]] = []
        for card in archive:
            transfer = self.base.resolver.resolve(card.protocol, query.protocol)
            if transfer.transfer_residual(card.oracle_residual_ev_per_atom) is None:
                continue
            similarity = max(0.0, cosine_similarity(query.embedding, card.embedding))
            if similarity < self.base.minimum_similarity:
                continue
            compatible.append((-similarity, card.card_id, card))
        compatible.sort(key=lambda item: (item[0], item[1]))
        return tuple(item[2] for item in compatible[: self.active_witness_budget])

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        archive = tuple(cards)
        scores = [self.base.score(query, self._retrieve(query, archive)) for query in queries]
        return tuple(sorted(scores, key=lambda item: (-item.score, item.query_id)))


class BaseBoundaryAcquisition:
    """Frozen-predictor baseline with no use of memory."""

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        del cards
        scores = []
        for query in queries:
            margin = query.stability_threshold_ev_per_atom - query.base_hull_distance_ev_per_atom
            scaled_margin = max(-50.0, min(50.0, margin / 0.04))
            probability = 1.0 / (1.0 + math.exp(-scaled_margin))
            scores.append(
                AcquisitionScore(
                    query_id=query.query_id,
                    score=margin / query.oracle_cost,
                    stable_score=probability,
                    expected_discovery_utility=margin,
                    exploration_bonus=0.0,
                    novelty=1.0,
                    compatible_witness_count=0,
                    predicted_stable=margin >= 0,
                )
            )
        return tuple(sorted(scores, key=lambda item: (-item.score, item.query_id)))


class SeededRandomAcquisition:
    """Deterministic random-ranking control, stable across Python processes."""

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        del cards
        scores = []
        for query in queries:
            digest = hashlib.sha256(f"{self.seed}:{query.query_id}".encode()).digest()
            score = int.from_bytes(digest[:8], "big") / (2**64 - 1)
            scores.append(
                AcquisitionScore(
                    query_id=query.query_id,
                    score=score,
                    stable_score=0.5,
                    stable_score_kind="fixed_prior",
                    expected_discovery_utility=0.0,
                    exploration_bonus=score,
                    novelty=1.0,
                    compatible_witness_count=0,
                    predicted_stable=True,
                )
            )
        return tuple(sorted(scores, key=lambda item: (-item.score, item.query_id)))


class BoundaryUncertaintyAcquisition:
    """Decoupled active-learning baseline that ignores retention competition."""

    def __init__(self, potential: BoundaryRiskPotential) -> None:
        self.potential = potential

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        memory = tuple(cards)
        scores = []
        for query in queries:
            estimate = self.potential.estimate(query, memory)
            uncertainty = 4.0 * estimate.scenario_stable_weight * (
                1.0 - estimate.scenario_stable_weight
            )
            scores.append(
                AcquisitionScore(
                    query_id=query.query_id,
                    score=uncertainty,
                    stable_score=estimate.scenario_stable_weight,
                    expected_discovery_utility=0.0,
                    exploration_bonus=uncertainty,
                    novelty=1.0 if estimate.source_witness_id is None else 0.0,
                    compatible_witness_count=int(estimate.source_witness_id is not None),
                    predicted_stable=estimate.predicted_stable,
                )
            )
        return tuple(sorted(scores, key=lambda item: (-item.score, item.query_id)))


class RetentionAwareBoundaryAcquisition:
    """One-step two-scenario lookahead after active-witness competition.

    The interval endpoints induce two deliberately heuristic scenario weights;
    they are not a calibrated residual distribution.  Both sides of the
    information-value difference use the same pool ``U_t \\ {x}``.  Each
    hypothetical residual competes with existing active witnesses for ``K``
    slots before downstream risk is evaluated. No unqueried oracle value enters
    this calculation.
    """

    def __init__(
        self,
        potential: BoundaryRiskPotential,
        *,
        active_witness_budget: int,
        discovery_weight: float = 1.0,
        information_weight: float = 1.0,
        oracle_cost: float = 1.0,
        outcome_margin_ev_per_atom: float = 0.01,
    ) -> None:
        if active_witness_budget < 0:
            raise ValueError("active witness capacity cannot be negative")
        if min(discovery_weight, information_weight, oracle_cost) <= 0:
            raise ValueError("acquisition weights and oracle cost must be positive")
        if outcome_margin_ev_per_atom <= 0:
            raise ValueError("hypothetical outcome margin must be positive")
        self.potential = potential
        self.active_witness_budget = active_witness_budget
        self.discovery_weight = discovery_weight
        self.information_weight = information_weight
        self.oracle_cost = oracle_cost
        self.outcome_margin_ev_per_atom = outcome_margin_ev_per_atom
        self.retention_solver = BruteForceRetentionSolver()

    def _retained_witnesses(
        self,
        witnesses: tuple[BoundaryWitness, ...],
        queries: tuple[MaterialQuery, ...],
    ) -> tuple[BoundaryWitness, ...]:
        return self.retention_solver.select(
            witnesses,
            queries,
            capacity=self.active_witness_budget,
            potential=self.potential,
        )

    @staticmethod
    def _hypothetical_hull_transition(
        query: MaterialQuery,
        residual_ev_per_atom: float,
        remaining_queries: tuple[MaterialQuery, ...],
    ) -> tuple[MaterialQuery, ...]:
        """Apply a causal, evaluation-only hull transition to the common pool."""

        hypothetical_energy = (
            query.base_predicted_formation_energy_ev_per_atom + residual_ev_per_atom
        )
        transitioned: list[MaterialQuery] = []
        for future in remaining_queries:
            old = future.hull_snapshot
            if old.chemical_system != query.hull_snapshot.chemical_system:
                transitioned.append(future)
                continue
            new_reference = min(old.reference_hull_energy_ev_per_atom, hypothetical_energy)
            if new_reference == old.reference_hull_energy_ev_per_atom:
                transitioned.append(future)
                continue
            built_at = max(old.built_at, query.as_of)
            checksum_payload = (
                f"{old.phase_set_checksum}:hypothetical:{query.query_id}:"
                f"{residual_ev_per_atom:.12g}"
            )
            revised = old.model_copy(
                update={
                    "snapshot_id": f"{old.snapshot_id}:hypothetical:{query.query_id}",
                    "reference_hull_energy_ev_per_atom": new_reference,
                    "phase_set_checksum": "sha256:"
                    + hashlib.sha256(checksum_payload.encode()).hexdigest(),
                    "known_through": built_at,
                    "built_at": built_at,
                }
            )
            transitioned.append(
                future.model_copy(update={"hull_snapshot": revised, "as_of": built_at})
            )
        return tuple(transitioned)

    def information_value(
        self,
        query: MaterialQuery,
        remaining_queries: tuple[MaterialQuery, ...],
        cards: tuple[MaterialMemoryCard, ...],
    ) -> float:
        """Common-pool two-scenario reduction, excluding queried-item removal."""

        estimate = self.potential.estimate(query, cards)
        current_witnesses = tuple(BoundaryWitness.from_card(card) for card in cards)
        risk_before = self.potential.evaluate_witnesses(
            remaining_queries,
            current_witnesses,
        ).total
        threshold_residual = (
            query.stability_threshold_ev_per_atom - query.base_hull_distance_ev_per_atom
        )
        outcomes = (
            (
                estimate.scenario_stable_weight,
                threshold_residual - self.outcome_margin_ev_per_atom,
                "stable",
            ),
            (
                1.0 - estimate.scenario_stable_weight,
                threshold_residual + self.outcome_margin_ev_per_atom,
                "unstable",
            ),
        )
        reduction = 0.0
        for scenario_weight, residual, label in outcomes:
            if scenario_weight == 0:
                continue
            hypothetical = BoundaryWitness(
                witness_id=f"hypothetical:{query.query_id}:{label}",
                embedding=query.embedding,
                residual_ev_per_atom=residual,
                protocol=query.protocol,
            )
            transitioned_queries = self._hypothetical_hull_transition(
                query,
                residual,
                remaining_queries,
            )
            retained = self._retained_witnesses(
                (*current_witnesses, hypothetical),
                transitioned_queries,
            )
            risk_after = self.potential.evaluate_witnesses(
                transitioned_queries,
                retained,
            ).total
            reduction += scenario_weight * max(0.0, risk_before - risk_after)
        return reduction

    def score(
        self,
        query: MaterialQuery,
        future_queries: tuple[MaterialQuery, ...],
        cards: tuple[MaterialMemoryCard, ...],
    ) -> AcquisitionScore:
        estimate = self.potential.estimate(query, cards)
        expected_reduction = self.information_value(query, future_queries, cards)
        immediate_value = self.discovery_weight * estimate.scenario_stable_weight
        information_value = self.information_weight * expected_reduction
        return AcquisitionScore(
            query_id=query.query_id,
            score=(immediate_value + information_value)
            / (self.oracle_cost * query.oracle_cost),
            stable_score=estimate.scenario_stable_weight,
            stable_score_kind="two_scenario_weight",
            expected_discovery_utility=immediate_value,
            exploration_bonus=information_value,
            downstream_risk_reduction=expected_reduction,
            novelty=1.0 if estimate.source_witness_id is None else 0.0,
            compatible_witness_count=int(estimate.source_witness_id is not None),
            predicted_stable=estimate.predicted_stable,
        )

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        candidates = tuple(queries)
        memory = tuple(cards)
        scores = [
            self.score(
                query,
                tuple(item for item in candidates if item.query_id != query.query_id),
                memory,
            )
            for query in candidates
        ]
        return tuple(sorted(scores, key=lambda item: (-item.score, item.query_id)))
