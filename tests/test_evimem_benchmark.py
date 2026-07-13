from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from evimem.benchmark import (
    BenchmarkEpisode,
    DatasetRegistry,
    EpisodePrediction,
    MemoryQuery,
    OracleAnnotation,
    ScientificDocument,
    SciFactAdapter,
    build_episode_stream,
    compute_benchmark_metrics,
)
from evimem.contracts import AdmissionAction, ScientificClaimRecord, UpdateOperation

ROOT = Path(__file__).resolve().parents[1]


def _episode(position: int, year: int) -> BenchmarkEpisode:
    return BenchmarkEpisode(
        episode_id=f"episode-{position}",
        stream_position=position,
        current_document=ScientificDocument(
            document_id=f"doc-{position}",
            text="claim text",
            timestamp=datetime(year, 1, 1, tzinfo=UTC),
            dataset_name="SciFact",
            split="train",
        ),
        query=MemoryQuery(query_id=f"query-{position}", text="claim"),
    )


def test_dataset_manifest_separates_training_ood_scale_and_case_study() -> None:
    registry = DatasetRegistry.load(ROOT / "configs" / "datasets.json")
    assert registry.audit()["ok"] is True
    assert registry.audit()["training_ready"] is False
    assert "QASPER" in registry.audit()["blocked_core_training"]
    registry.assert_split_allowed("SciREX", "train")
    registry.assert_training_allowed("SciREX", "retrieval_view")
    with pytest.raises(ValueError, match="blocked for training"):
        registry.assert_training_allowed("QASPER", "retrieval_view")
    with pytest.raises(ValueError, match="not in its official protocol"):
        registry.assert_split_allowed("Materials-150-DOI", "train")


def test_stream_rejects_future_ordering() -> None:
    with pytest.raises(ValueError, match="future ordering"):
        build_episode_stream([_episode(0, 2026), _episode(1, 2025)])


def test_scifact_refute_is_not_promoted_to_rejected_memory() -> None:
    episode, oracle = SciFactAdapter().convert(
        {
            "id": "claim-1",
            "document_id": "doc-1",
            "claim": "X increases Y",
            "label": "CONTRADICT",
            "abstract": "Evidence refutes the claim.",
        },
        split="train",
        stream_position=0,
    )
    assert "gold" not in episode.model_dump()
    assert episode.current_document.timestamp is None
    assert oracle.final_record is None
    assert oracle.admission is None
    assert oracle.memory_operation is None


def test_metrics_cover_retrieval_update_and_publication_safety() -> None:
    claim = ScientificClaimRecord(subject="X", relation="increases", object="Y")
    oracle = OracleAnnotation(
        episode_id="e1",
        relevant_memory_ids=("m1",),
        final_record=claim,
        admission=AdmissionAction.WRITE_VERIFIED,
        memory_operation=UpdateOperation.LINK,
        target_memory_ids=("m1",),
    )
    prediction = EpisodePrediction(
        episode_id="e1",
        retrieved_memory_ids=("m1", "m2"),
        predicted_record=claim,
        admission=AdmissionAction.WRITE_VERIFIED,
        memory_operation=UpdateOperation.LINK,
        target_memory_ids=("m1",),
        publication_requested=True,
        publication_authorized=True,
        certificate_id="cert-1",
        memory_size=10,
        retrieval_tokens=20,
    )
    metrics = compute_benchmark_metrics([prediction], {"e1": oracle})
    assert metrics.tuple_f1 == 1.0
    assert metrics.recall_at_1 == 1.0
    assert metrics.update_operation_accuracy == 1.0
    assert metrics.unsupported_publication_rate == 0.0
