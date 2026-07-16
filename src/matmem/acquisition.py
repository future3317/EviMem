"""Oracle-blind acquisition policies for calibration-aware discovery.

Policies may inspect candidate queries and previously revealed memory cards.
They never receive the oracle record of an unqueried candidate.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .cards import MaterialMemoryCard, MaterialQuery
from .coreset import FacilityLocationCoresetPlanner
from .hull_engine import CausalHullEngine
from .residual_posterior import ResidualPosterior


class AcquisitionScore(BaseModel):
    """Auditable score computed without access to a candidate's oracle result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    score: float
    stable_score: float = Field(ge=0, le=1)
    stable_score_kind: Literal[
        "model_probability",
        "posterior_probability",
        "fixed_prior",
    ] = "model_probability"
    expected_discovery_utility: float
    exploration_bonus: float = Field(ge=0)
    downstream_risk_reduction: float = Field(default=0.0, ge=0)
    novelty: float = Field(ge=0, le=1)
    compatible_witness_count: int = Field(ge=0)
    predicted_stable: bool


class FrozenHullDistanceAcquisition:
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
                    score=-query.base_hull_distance_ev_per_atom / query.oracle_cost,
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


class PosteriorUncertaintyAcquisition:
    """Fixed-GP stability uncertainty with no retention lookahead."""

    def __init__(self, posterior: ResidualPosterior) -> None:
        self.posterior = posterior

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        candidates = tuple(queries)
        memory = tuple(cards)
        self.posterior.fit(memory)
        prediction = self.posterior.predict(candidates)
        scores = []
        for query, probability, count in zip(
            candidates,
            prediction.stable_probability,
            prediction.compatible_witness_count,
            strict=True,
        ):
            uncertainty = 4.0 * probability * (1.0 - probability)
            scores.append(
                AcquisitionScore(
                    query_id=query.query_id,
                    score=uncertainty / query.oracle_cost,
                    stable_score=probability,
                    stable_score_kind="posterior_probability",
                    expected_discovery_utility=probability,
                    exploration_bonus=uncertainty,
                    novelty=float(count == 0),
                    compatible_witness_count=count,
                    predicted_stable=probability >= 0.5,
                )
            )
        return tuple(sorted(scores, key=lambda item: (-item.score, item.query_id)))


class SurvivalConditionedAcquisition:
    """Rerank a small proposal set by value that survives compression.

    The posterior is fitted only to currently active calibration cards. Every
    fantasy is passed to ``preview_admit`` and is never written to the archive.
    Setting ``survival_weight`` to zero returns the proposal ranking verbatim.
    """

    def __init__(
        self,
        proposal: object,
        posterior: ResidualPosterior,
        coreset_planner: FacilityLocationCoresetPlanner,
        *,
        proposal_size: int = 32,
        num_fantasies: int = 8,
        survival_weight: float = 1.0,
        seed: int = 0,
        fantasy_hull_engine: CausalHullEngine | None = None,
    ) -> None:
        if proposal_size < 1 or num_fantasies < 1:
            raise ValueError("survival acquisition sizes must be positive")
        if survival_weight < 0:
            raise ValueError("survival weight cannot be negative")
        if not hasattr(proposal, "rank"):
            raise TypeError("survival acquisition proposal must implement rank")
        self.proposal = proposal
        self.posterior = posterior
        self.coreset_planner = coreset_planner
        self.proposal_size = proposal_size
        self.num_fantasies = num_fantasies
        self.survival_weight = survival_weight
        self.seed = seed
        self.fantasy_hull_engine = fantasy_hull_engine
        self.active_witness_budget = coreset_planner.capacity

    def _fantasy_seed(self, query_id: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{query_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big")

    @staticmethod
    def _fantasy_card(
        query: MaterialQuery,
        residual_ev_per_atom: float,
        index: int,
    ) -> MaterialMemoryCard:
        from .cards import SourceProvenance

        return MaterialMemoryCard(
            card_id=f"fantasy:{query.query_id}:{index}",
            material_id=f"fantasy:{query.query_id}",
            structure_hash=query.structure_hash,
            identity=query.identity,
            composition=query.composition,
            embedding=query.embedding,
            protocol=query.protocol,
            provenance=SourceProvenance(
                source_name="posterior-fantasy",
                source_version="survival-conditioned-v1",
                record_locator=f"{query.query_id}:{index}",
                retrieved_at=query.as_of,
            ),
            formation_energy_ev_per_atom=(
                query.base_predicted_formation_energy_ev_per_atom + residual_ev_per_atom
            ),
            base_predicted_formation_energy_ev_per_atom=(
                query.base_predicted_formation_energy_ev_per_atom
            ),
            oracle_residual_ev_per_atom=residual_ev_per_atom,
            hull_snapshot=query.hull_snapshot,
            observed_at=query.as_of,
        )

    def _survival_bonus(
        self,
        query: MaterialQuery,
        future_queries: tuple[MaterialQuery, ...],
        cards: tuple[MaterialMemoryCard, ...],
    ) -> float:
        residuals = self.posterior.sample_residuals(
            query,
            num_samples=self.num_fantasies,
            seed=self._fantasy_seed(query.query_id),
        )
        bonuses: list[float] = []
        for index, residual in enumerate(residuals):
            hypothetical = self._fantasy_card(query, float(residual), index)
            transitioned = (
                self.fantasy_hull_engine.preview_after_fantasy(
                    future_queries,
                    hypothetical,
                    call_index=len(cards) + 1,
                )
                if self.fantasy_hull_engine is not None
                else future_queries
            )
            selection = self.coreset_planner.preview_admit(
                cards,
                hypothetical,
                transitioned,
            )
            bonuses.append(selection.objective_improvement)
        return sum(bonuses) / len(bonuses)

    def rank(
        self,
        queries: Iterable[MaterialQuery],
        cards: Iterable[MaterialMemoryCard],
    ) -> tuple[AcquisitionScore, ...]:
        candidates = tuple(queries)
        memory = tuple(cards)
        base_ranking = tuple(self.proposal.rank(candidates, memory))
        if self.survival_weight == 0 or not candidates:
            return base_ranking
        self.posterior.fit(memory)
        proposed_ids = {item.query_id for item in base_ranking[: self.proposal_size]}
        by_id = {query.query_id: query for query in candidates}
        posterior = self.posterior.predict(
            tuple(by_id[item.query_id] for item in base_ranking)
        )
        probabilities = dict(
            zip(posterior.query_ids, posterior.stable_probability, strict=True)
        )
        rescored: list[AcquisitionScore] = []
        for base in base_ranking:
            if base.query_id not in proposed_ids:
                rescored.append(base)
                continue
            query = by_id[base.query_id]
            future = tuple(item for item in candidates if item.query_id != query.query_id)
            bonus = self._survival_bonus(query, future, memory)
            weighted = self.survival_weight * bonus
            rescored.append(
                base.model_copy(
                    update={
                        "score": base.score + weighted,
                        "stable_score": probabilities[base.query_id],
                        "stable_score_kind": "posterior_probability",
                        "exploration_bonus": base.exploration_bonus + weighted,
                        "downstream_risk_reduction": bonus,
                        "predicted_stable": probabilities[base.query_id] >= 0.5,
                    }
                )
            )
        return tuple(sorted(rescored, key=lambda item: (-item.score, item.query_id)))
