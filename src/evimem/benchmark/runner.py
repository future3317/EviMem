"""Sequential controller comparison with oracle data isolated from inference."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from evimem.contracts import VerificationCertificate
from evimem.controller import ControllerPolicy, SequentialCurationEngine

from .episode import BenchmarkEpisode, EpisodeEvaluation, OracleAnnotation
from .metrics import BenchmarkMetrics, compute_benchmark_metrics
from .stream_builder import build_episode_stream

CertificateEvaluator = Callable[[str, object], VerificationCertificate]


@dataclass(frozen=True)
class BenchmarkRun:
    controller_name: str
    evaluations: tuple[EpisodeEvaluation, ...]
    metrics: BenchmarkMetrics


class SequentialBenchmarkRunner:
    def __init__(self, *, engine: SequentialCurationEngine):
        self.engine = engine

    def run(
        self,
        *,
        episodes: list[BenchmarkEpisode],
        controllers: dict[str, ControllerPolicy],
        certificate_evaluator: CertificateEvaluator,
        oracle_annotations: dict[str, OracleAnnotation] | None = None,
    ) -> dict[str, BenchmarkRun]:
        stream = build_episode_stream(episodes)
        annotations = oracle_annotations or {}
        unknown_annotations = set(annotations) - {episode.episode_id for episode in stream}
        if unknown_annotations:
            raise ValueError(f"oracle annotations reference unknown episodes: {sorted(unknown_annotations)}")

        runs: dict[str, BenchmarkRun] = {}
        for controller_name, policy in controllers.items():
            evaluations: list[EpisodeEvaluation] = []
            for episode in stream:
                outcome = self.engine.run(
                    run_id=f"{controller_name}:{episode.episode_id}",
                    initial_state=episode.initial_state,
                    policy=policy,
                )
                certificate = certificate_evaluator(episode.episode_id, outcome)
                annotation = annotations.get(episode.episode_id)
                terminal_correct = None
                evidence_hit = None
                if annotation is not None:
                    terminal_correct = (
                        outcome.trajectory.terminal_action == annotation.expected_terminal_action
                    )
                    gathered = {
                        (ref.release_id, ref.block_id, ref.checksum)
                        for ref in outcome.final_state.gathered_evidence
                    }
                    gold = {
                        (ref.release_id, ref.block_id, ref.checksum)
                        for ref in annotation.gold_evidence_refs
                    }
                    evidence_hit = bool(gold & gathered) if gold else None
                evaluations.append(
                    EpisodeEvaluation(
                        controller_name=controller_name,
                        episode_id=episode.episode_id,
                        negative_control=episode.negative_control,
                        trajectory=outcome.trajectory,
                        certificate=certificate,
                        terminal_action_correct=terminal_correct,
                        gold_evidence_hit=evidence_hit,
                    )
                )
            runs[controller_name] = BenchmarkRun(
                controller_name=controller_name,
                evaluations=tuple(evaluations),
                metrics=compute_benchmark_metrics(evaluations),
            )
        return runs
