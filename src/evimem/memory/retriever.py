"""Structured retrieval for evidence-warranted memories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from evimem.core.contracts import ClaimSignature, MemoryType, WarrantedMemoryItem

from .governed_store import GovernedMemoryStore


class SemanticScorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class TfidfSemanticScorer:
    """Local sparse semantic baseline backed by scikit-learn."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit_transform(
            [query, *documents]
        )
        return [float(value) for value in cosine_similarity(matrix[:1], matrix[1:])[0]]


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    signature: ClaimSignature
    policy_version: str
    policy_hash: str
    query_text: str = ""
    memory_types: tuple[MemoryType, ...] = ()
    limit: int = 8


@dataclass(frozen=True)
class RetrievedMemory:
    item: WarrantedMemoryItem
    score: float
    semantic_similarity: float
    structure_match: float
    policy_compatibility: float
    authority: float
    staleness: float
    conflict_risk: float


class MemoryHints(BaseModel):
    model_config = ConfigDict(frozen=True)

    memory_ids: tuple[str, ...] = ()
    known_aliases: tuple[str, ...] = ()
    likely_evidence_locations: tuple[str, ...] = ()
    prior_failure_modes: tuple[str, ...] = ()
    possible_conflicts: tuple[str, ...] = ()
    required_checks: tuple[str, ...] = ()


class MemoryRetriever:
    def __init__(
        self,
        store: GovernedMemoryStore,
        *,
        semantic_scorer: SemanticScorer | None = None,
    ) -> None:
        self.store = store
        self.semantic_scorer = semantic_scorer

    @staticmethod
    def _structure_match(query: ClaimSignature, item: WarrantedMemoryItem) -> float:
        candidate = item.claim_signature
        fields = (
            "property_key",
            "material_family",
            "material_identity",
            "composition",
            "condition_signature",
        )
        comparisons = [
            (getattr(query, field), getattr(candidate, field))
            for field in fields
            if getattr(query, field) is not None
        ]
        if not comparisons:
            return 0.0
        return sum(left == right for left, right in comparisons) / len(comparisons)

    @staticmethod
    def _document(item: WarrantedMemoryItem) -> str:
        signature = item.claim_signature
        values = [
            signature.property_key,
            signature.material_family or "",
            signature.material_identity or "",
            signature.composition or "",
            signature.condition_signature or "",
            item.decision.reason,
        ]
        return " ".join(value for value in values if value)

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedMemory]:
        candidates = self.store.query(
            domain=query.signature.domain,
            property_key=query.signature.property_key,
            memory_types=query.memory_types or None,
            include_superseded=True,
            limit=max(query.limit * 8, 32),
        )
        semantic = [0.0] * len(candidates)
        if self.semantic_scorer is not None and query.query_text:
            semantic = self.semantic_scorer.score(
                query.query_text,
                [self._document(item) for item in candidates],
            )

        now = datetime.now(UTC)
        ranked: list[RetrievedMemory] = []
        for item, semantic_score in zip(candidates, semantic):
            structure = self._structure_match(query.signature, item)
            policy = 1.0 if (
                item.policy_version == query.policy_version
                and item.policy_hash == query.policy_hash
            ) else 0.0
            authority = item.authority.level / 4.0
            age_days = max(0.0, (now - item.valid_from).total_seconds() / 86400.0)
            staleness = min(1.0, age_days / 3650.0)
            if item.status == "superseded" or item.valid_until is not None:
                staleness = 1.0
            conflict_risk = 1.0 if item.memory_type == MemoryType.CONFLICT else 0.0
            score = (
                0.20 * semantic_score
                + 0.35 * structure
                + 0.25 * policy
                + 0.20 * authority
                - 0.20 * staleness
                - 0.10 * conflict_risk
            )
            ranked.append(
                RetrievedMemory(
                    item=item,
                    score=score,
                    semantic_similarity=semantic_score,
                    structure_match=structure,
                    policy_compatibility=policy,
                    authority=authority,
                    staleness=staleness,
                    conflict_risk=conflict_risk,
                )
            )
        ranked.sort(key=lambda result: (-result.score, result.item.memory_id))
        return ranked[: query.limit]

    @staticmethod
    def to_hints(results: Sequence[RetrievedMemory]) -> MemoryHints:
        aliases: set[str] = set()
        locations: set[str] = set()
        failures: set[str] = set()
        conflicts: set[str] = set()
        checks: set[str] = set()
        for result in results:
            item = result.item
            aliases.update(str(value) for value in item.normalized_content.get("aliases", []) or [])
            locations.update(
                str(value) for value in item.normalized_content.get("evidence_locations", []) or []
            )
            if item.memory_type == MemoryType.REJECTED:
                failures.add(item.decision.reason)
            if item.memory_type == MemoryType.CONFLICT:
                conflicts.add(item.memory_id)
            checks.update(
                str(value) for value in item.normalized_content.get("required_checks", []) or []
            )
        return MemoryHints(
            memory_ids=tuple(result.item.memory_id for result in results),
            known_aliases=tuple(sorted(aliases)),
            likely_evidence_locations=tuple(sorted(locations)),
            prior_failure_modes=tuple(sorted(failures)),
            possible_conflicts=tuple(sorted(conflicts)),
            required_checks=tuple(sorted(checks)),
        )
