"""Deterministic chronological streams that preserve official dataset splits."""

from __future__ import annotations

from collections.abc import Iterable

from .datasets import DatasetRegistry
from .episode import BenchmarkEpisode


def build_episode_stream(
    episodes: Iterable[BenchmarkEpisode],
    *,
    registry: DatasetRegistry | None = None,
) -> tuple[BenchmarkEpisode, ...]:
    ordered = tuple(sorted(episodes, key=lambda item: (item.stream_position, item.episode_id)))
    positions = [episode.stream_position for episode in ordered]
    if len(positions) != len(set(positions)):
        raise ValueError("stream positions must be unique")
    dated = [episode for episode in ordered if episode.current_document.timestamp is not None]
    timestamps = [episode.current_document.timestamp for episode in dated]
    if timestamps != sorted(timestamps):
        raise ValueError("stream position must follow document time; future ordering is forbidden")
    split_documents: dict[str, set[str]] = {}
    for episode in ordered:
        document = episode.current_document
        seen_splits = split_documents.setdefault(document.document_id, set())
        seen_splits.add(document.split)
        if len(seen_splits) > 1:
            raise ValueError(
                f"source_document_id {document.document_id!r} crosses official splits"
            )
    if registry is not None:
        for episode in ordered:
            registry.assert_split_allowed(
                episode.current_document.dataset_name,
                episode.current_document.split,
            )
    return ordered
