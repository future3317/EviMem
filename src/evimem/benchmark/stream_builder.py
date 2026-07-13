"""Deterministic paper-stream ordering without oracle exposure."""

from __future__ import annotations

from collections.abc import Iterable

from .episode import BenchmarkEpisode


def build_episode_stream(episodes: Iterable[BenchmarkEpisode]) -> tuple[BenchmarkEpisode, ...]:
    ordered = tuple(sorted(episodes, key=lambda item: (item.stream_position, item.episode_id)))
    positions = [episode.stream_position for episode in ordered]
    if len(positions) != len(set(positions)):
        raise ValueError("stream positions must be unique")
    release_by_episode = {
        episode.episode_id: episode.initial_state.claim_state.evidence_release_id
        for episode in ordered
    }
    if any(not release for release in release_by_episode.values()):
        raise ValueError("every episode must pin an evidence release")
    return ordered
