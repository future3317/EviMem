"""Scientific-record, memory and continual-evaluation metrics."""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict

from .episode import EpisodePrediction, OracleAnnotation


class BenchmarkMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    episode_count: int
    tuple_precision: float
    tuple_recall: float
    tuple_f1: float
    evidence_span_f1: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    memory_admission_precision: float
    update_operation_accuracy: float
    conflict_resolution_accuracy: float
    stale_memory_error_rate: float
    unsupported_publication_rate: float
    negative_control_false_publication_rate: float
    average_memory_size: float
    average_retrieval_tokens: float


def _f1(tp: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, score


def _evidence_key(ref: object) -> tuple[object, ...]:
    return (
        getattr(ref, "release_id"),
        getattr(ref, "document_id"),
        getattr(ref, "block_id"),
        getattr(ref, "checksum"),
    )


def compute_benchmark_metrics(
    predictions: list[EpisodePrediction],
    annotations: dict[str, OracleAnnotation],
) -> BenchmarkMetrics:
    if len({item.episode_id for item in predictions}) != len(predictions):
        raise ValueError("duplicate prediction episode_id")
    unknown = {item.episode_id for item in predictions} - set(annotations)
    if unknown:
        raise ValueError(f"predictions reference unknown episodes: {sorted(unknown)}")

    tuple_tp = tuple_predicted = tuple_gold = 0
    evidence_tp = evidence_predicted = evidence_gold = 0
    recall_hits = {1: 0, 5: 0, 10: 0}
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    admission_tp = admission_predicted = 0
    update_correct = update_count = conflict_correct = conflict_count = 0
    stale_errors = stale_predictions = 0
    unsupported_publications = publication_count = 0
    negative_publications = negative_count = 0

    write_actions = {"WRITE_VERIFIED", "WRITE_REJECTED", "WRITE_CONFLICT"}
    for prediction in predictions:
        gold = annotations[prediction.episode_id]
        if gold.final_record is not None:
            tuple_gold += 1
        if prediction.predicted_record is not None:
            tuple_predicted += 1
        if prediction.predicted_record is not None and prediction.predicted_record == gold.final_record:
            tuple_tp += 1

        predicted_evidence = {_evidence_key(ref) for ref in prediction.evidence_refs}
        gold_evidence = {_evidence_key(ref) for ref in gold.evidence_refs}
        evidence_tp += len(predicted_evidence & gold_evidence)
        evidence_predicted += len(predicted_evidence)
        evidence_gold += len(gold_evidence)

        relevant = set(gold.relevant_memory_ids)
        ranked = prediction.retrieved_memory_ids
        ranks = [index + 1 for index, memory_id in enumerate(ranked) if memory_id in relevant]
        for cutoff in recall_hits:
            if relevant and any(rank <= cutoff for rank in ranks):
                recall_hits[cutoff] += 1
        reciprocal_ranks.append(1.0 / min(ranks) if ranks else 0.0)
        gains = [1.0 if memory_id in relevant else 0.0 for memory_id in ranked[:10]]
        dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
        ideal_count = min(len(relevant), 10)
        idcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
        ndcgs.append(dcg / idcg if idcg else 0.0)

        predicted_write = prediction.admission.value in write_actions
        if gold.admission is not None:
            gold_write = gold.admission.value in write_actions
            admission_predicted += int(predicted_write)
            admission_tp += int(predicted_write and gold_write)
        if gold.memory_operation is not None:
            update_count += 1
            update_correct += int(prediction.memory_operation == gold.memory_operation)
        if gold.memory_operation is not None and gold.memory_operation.value == "CONFLICT":
            conflict_count += 1
            conflict_correct += int(
                prediction.memory_operation == gold.memory_operation
                and set(prediction.target_memory_ids) == set(gold.target_memory_ids)
            )
        stale_target = bool(set(prediction.target_memory_ids) - set(gold.target_memory_ids))
        stale_predictions += int(bool(prediction.target_memory_ids))
        stale_errors += int(stale_target)

        publication_count += int(prediction.publication_requested)
        unsupported_publications += int(
            prediction.publication_requested and not prediction.publication_authorized
        )
        is_negative = gold.admission is not None and gold.admission.value in {
            "WRITE_REJECTED",
            "IGNORE",
            "EPHEMERAL_ONLY",
        }
        negative_count += int(is_negative)
        negative_publications += int(is_negative and prediction.publication_authorized)

    count = len(predictions)
    tuple_precision, tuple_recall, tuple_f1 = _f1(tuple_tp, tuple_predicted, tuple_gold)
    _, _, evidence_f1 = _f1(evidence_tp, evidence_predicted, evidence_gold)
    return BenchmarkMetrics(
        episode_count=count,
        tuple_precision=tuple_precision,
        tuple_recall=tuple_recall,
        tuple_f1=tuple_f1,
        evidence_span_f1=evidence_f1,
        recall_at_1=recall_hits[1] / count if count else 0.0,
        recall_at_5=recall_hits[5] / count if count else 0.0,
        recall_at_10=recall_hits[10] / count if count else 0.0,
        mrr=sum(reciprocal_ranks) / count if count else 0.0,
        ndcg_at_10=sum(ndcgs) / count if count else 0.0,
        memory_admission_precision=(admission_tp / admission_predicted if admission_predicted else 0.0),
        update_operation_accuracy=update_correct / update_count if update_count else 0.0,
        conflict_resolution_accuracy=(conflict_correct / conflict_count if conflict_count else 0.0),
        stale_memory_error_rate=stale_errors / stale_predictions if stale_predictions else 0.0,
        unsupported_publication_rate=(
            unsupported_publications / publication_count if publication_count else 0.0
        ),
        negative_control_false_publication_rate=(
            negative_publications / negative_count if negative_count else 0.0
        ),
        average_memory_size=(sum(item.memory_size for item in predictions) / count if count else 0.0),
        average_retrieval_tokens=(
            sum(item.retrieval_tokens for item in predictions) / count if count else 0.0
        ),
    )
