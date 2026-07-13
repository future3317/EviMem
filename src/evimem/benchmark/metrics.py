"""Safety, cost and sequential-controller benchmark metrics."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .episode import EpisodeEvaluation


class BenchmarkMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_count: int
    verified_strong_count: int
    publication_request_count: int
    rejected_publication_request_count: int
    publication_request_rejection_rate: float
    negative_control_false_publication_rate: float
    average_trajectory_length: float
    average_tool_calls: float
    average_tokens: float
    human_query_count: int
    cost_per_verified_record: float | None
    terminal_action_accuracy: float | None
    gold_evidence_hit_rate: float | None


def compute_benchmark_metrics(evaluations: list[EpisodeEvaluation]) -> BenchmarkMetrics:
    count = len(evaluations)
    requests = [
        item for item in evaluations if item.trajectory.terminal_action == "REQUEST_PUBLICATION"
    ]
    rejected_requests = [item for item in requests if item.certificate.final_decision != "publish"]
    verified = [
        item
        for item in evaluations
        if item.certificate.final_decision == "publish"
        and item.certificate.support_tier == "verified_strong"
    ]
    negative = [item for item in evaluations if item.negative_control]
    false_negative_publications = [
        item for item in negative if item.certificate.final_decision == "publish"
    ]
    steps = sum(len(item.trajectory.steps) for item in evaluations)
    tools = sum(
        step.cost.tool_calls for item in evaluations for step in item.trajectory.steps
    )
    tokens = sum(step.cost.tokens for item in evaluations for step in item.trajectory.steps)
    humans = sum(
        step.cost.human_queries for item in evaluations for step in item.trajectory.steps
    )
    total_cost = float(tokens + tools + humans)
    action_labels = [item.terminal_action_correct for item in evaluations if item.terminal_action_correct is not None]
    evidence_labels = [item.gold_evidence_hit for item in evaluations if item.gold_evidence_hit is not None]
    return BenchmarkMetrics(
        episode_count=count,
        verified_strong_count=len(verified),
        publication_request_count=len(requests),
        rejected_publication_request_count=len(rejected_requests),
        publication_request_rejection_rate=len(rejected_requests) / len(requests) if requests else 0.0,
        negative_control_false_publication_rate=(
            len(false_negative_publications) / len(negative) if negative else 0.0
        ),
        average_trajectory_length=steps / count if count else 0.0,
        average_tool_calls=tools / count if count else 0.0,
        average_tokens=tokens / count if count else 0.0,
        human_query_count=humans,
        cost_per_verified_record=total_cost / len(verified) if verified else None,
        terminal_action_accuracy=(
            sum(bool(value) for value in action_labels) / len(action_labels)
            if action_labels
            else None
        ),
        gold_evidence_hit_rate=(
            sum(bool(value) for value in evidence_labels) / len(evidence_labels)
            if evidence_labels
            else None
        ),
    )
