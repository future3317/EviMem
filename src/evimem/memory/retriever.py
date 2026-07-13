"""Evidence-certified retrieval with explicit score components."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from evimem.contracts import ClaimSignature, MemoryType, ScientificMemoryRecord

from .governed_store import GovernedMemoryStore


class SemanticScorer(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> list[float]: ...


class TfidfSemanticScorer:
    """Reproducible sparse baseline backed by scikit-learn."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        matrix = TfidfVectorizer(lowercase=True, ngram_range=(1, 2)).fit_transform(
            [query, *documents]
        )
        return [float(value) for value in cosine_similarity(matrix[:1], matrix[1:])[0]]


class BM25SemanticScorer:
    """Lexical BM25 baseline backed by the maintained rank-bm25 package."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        from rank_bm25 import BM25Okapi

        tokenized = [document.casefold().split() for document in documents]
        raw = [float(value) for value in BM25Okapi(tokenized).get_scores(query.casefold().split())]
        maximum = max(raw, default=0.0)
        return [value / maximum if maximum > 0 else 0.0 for value in raw]


class SentenceTransformerSemanticScorer:
    """Dense bi-encoder scorer using a caller-supplied Sentence Transformer."""

    def __init__(self, model: object):
        self.model = model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        embeddings = self.model.encode(
            [query, *documents],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [float(embeddings[0] @ vector) for vector in embeddings[1:]]


class RetrievalWeights(BaseModel):
    model_config = ConfigDict(frozen=True)

    semantic: float = Field(default=0.25, ge=0)
    structure: float = Field(default=0.30, ge=0)
    authority: float = Field(default=0.15, ge=0)
    temporal: float = Field(default=0.10, ge=0)
    policy: float = Field(default=0.20, ge=0)
    conflict_penalty: float = Field(default=0.10, ge=0)
    stale_penalty: float = Field(default=0.25, ge=0)


class RetrievalQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    signature: ClaimSignature
    policy_version: str
    policy_hash: str
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))
    query_text: str = ""
    memory_types: tuple[MemoryType, ...] = ()
    limit: int = Field(default=8, ge=1, le=100)


@dataclass(frozen=True)
class RetrievedMemory:
    """A complete certified record plus an auditable retrieval score."""

    record: ScientificMemoryRecord
    score: float
    semantic_similarity: float
    structure_match: float
    policy_compatibility: float
    authority: float
    temporal_relevance: float
    staleness: float
    conflict_risk: float


class MemoryRetriever:
    def __init__(
        self,
        store: GovernedMemoryStore,
        *,
        semantic_scorer: SemanticScorer | None = None,
        weights: RetrievalWeights | None = None,
    ) -> None:
        self.store = store
        self.semantic_scorer = semantic_scorer
        self.weights = weights or RetrievalWeights()

    @staticmethod
    def _structure_match(query: ClaimSignature, record: ScientificMemoryRecord) -> float:
        candidate = record.claim_signature
        fields = ("subject", "relation", "object", "unit", "condition_signature", "measurement_setting")
        comparisons = [
            (getattr(query, field), getattr(candidate, field))
            for field in fields
            if getattr(query, field) is not None
        ]
        return (
            sum(left.casefold() == right.casefold() if isinstance(left, str) and isinstance(right, str) else left == right for left, right in comparisons)
            / len(comparisons)
            if comparisons
            else 0.0
        )

    @staticmethod
    def _document(record: ScientificMemoryRecord) -> str:
        claim = record.claim
        return " ".join(
            str(value)
            for value in (
                claim.subject,
                claim.relation,
                claim.object,
                claim.value,
                claim.unit,
                claim.condition,
                record.decision.reason,
            )
            if value not in (None, "", {})
        )

    def retrieve(self, query: RetrievalQuery) -> list[RetrievedMemory]:
        candidates = self.store.query(
            domain=query.signature.domain,
            relation=None,
            memory_types=query.memory_types or None,
            observed_before=query.as_of,
            include_superseded=True,
            limit=max(query.limit * 16, 64),
        )
        semantic = [0.0] * len(candidates)
        if self.semantic_scorer is not None and query.query_text:
            semantic = self.semantic_scorer.score(
                query.query_text,
                [self._document(record) for record in candidates],
            )

        ranked: list[RetrievedMemory] = []
        for record, semantic_score in zip(candidates, semantic):
            structure = self._structure_match(query.signature, record)
            policy = float(
                record.policy_version == query.policy_version and record.policy_hash == query.policy_hash
            )
            authority = record.authority.level / 4.0
            age_days = max(0.0, (query.as_of - record.observed_at).total_seconds() / 86400.0)
            temporal = 1.0 / (1.0 + age_days / 365.0)
            stale = float(record.status.value == "superseded" or record.valid_until is not None)
            conflict = float(record.memory_type == MemoryType.CONFLICT)
            weights = self.weights
            score = (
                weights.semantic * semantic_score
                + weights.structure * structure
                + weights.authority * authority
                + weights.temporal * temporal
                + weights.policy * policy
                - weights.conflict_penalty * conflict
                - weights.stale_penalty * stale
            )
            ranked.append(
                RetrievedMemory(
                    record=record,
                    score=score,
                    semantic_similarity=semantic_score,
                    structure_match=structure,
                    policy_compatibility=policy,
                    authority=authority,
                    temporal_relevance=temporal,
                    staleness=stale,
                    conflict_risk=conflict,
                )
            )
        ranked.sort(key=lambda result: (-result.score, result.record.memory_id))
        return ranked[: query.limit]
