"""Execute the Phase 1B retrieval validity pilot without training an update manager."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import statistics
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from run_phase1a_audit import load_qasper, load_scifact, load_scirex

from evimem.benchmark import ViewSample, sha256_file
from evimem.phase1b.retrieval import (
    PilotMemoryItem,
    PilotQuery,
    assert_pilot_split_isolation,
    evaluate_rankings,
)

BASE_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SCIENTIFIC_DENSE_MODEL = "sentence-transformers/allenai-specter"
SEEDS = (13, 42, 97)
TOP_K = 10
RANK_LIMIT = 200
TOKEN_BUDGET = 256


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _claim_family(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_samples(
    samples: list[ViewSample],
    report: dict[str, Any],
    *,
    splits: set[str],
) -> list[ViewSample]:
    leaked_documents = set(report["source_document_cross_split"])
    leaked_families = set(report["exact_normalized_claim_or_question_family_cross_split"])
    return [
        sample
        for sample in samples
        if sample.split in splits
        and sample.source_document_id not in leaked_documents
        and _claim_family(sample.query_text) not in leaked_families
        and sample.view is not None
    ]


def _evidence_text(sample: ViewSample) -> str | None:
    text = " ".join(span.text.strip() for span in sample.evidence if span.text.strip()).strip()
    return text or None


def _entity_key(sample: ViewSample) -> str:
    if sample.dataset_name == "SciREX" and sample.claim is not None:
        return sample.claim.subject.casefold()
    return sample.source_document_id.casefold()


def _memory_id(dataset_name: str, sample: ViewSample, text: str) -> str:
    payload = f"{dataset_name}\0{sample.source_document_id}\0{text}".encode()
    return f"{dataset_name.lower()}:{hashlib.sha256(payload).hexdigest()[:24]}"


def build_pilot_dataset(
    dataset_name: str,
    samples: list[ViewSample],
) -> tuple[list[PilotQuery], list[PilotMemoryItem]]:
    memory: dict[str, PilotMemoryItem] = {}
    queries: list[PilotQuery] = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        evidence_text = _evidence_text(sample)
        positives: tuple[str, ...] = ()
        if evidence_text is not None:
            memory_id = _memory_id(dataset_name, sample, evidence_text)
            memory.setdefault(
                memory_id,
                PilotMemoryItem(
                    memory_id=memory_id,
                    text=evidence_text,
                    token_count=max(1, len(evidence_text.split())),
                    entity_key=_entity_key(sample),
                    memory_type="unavailable",
                    stale=False,
                    policy_compatible=True,
                    certificate_compatible=None,
                ),
            )
            positives = (memory_id,)
        queries.append(
            PilotQuery(
                query_id=sample.sample_id,
                text=sample.query_text,
                positive_memory_ids=positives,
                entity_key=_entity_key(sample),
            )
        )
    return queries, sorted(memory.values(), key=lambda item: item.memory_id)


def _rank_matrix(
    scores: np.ndarray,
    queries: list[PilotQuery],
    memory: list[PilotMemoryItem],
) -> dict[str, tuple[str, ...]]:
    limit = min(RANK_LIMIT, len(memory))
    rankings: dict[str, tuple[str, ...]] = {}
    ids = np.asarray([item.memory_id for item in memory])
    for index, query in enumerate(queries):
        row = scores[index]
        if limit == len(memory):
            candidates = np.arange(len(memory))
        else:
            candidates = np.argpartition(-row, limit - 1)[:limit]
        ordered = sorted(candidates.tolist(), key=lambda item: (-float(row[item]), ids[item]))
        rankings[query.query_id] = tuple(str(ids[item]) for item in ordered)
    return rankings


def _tfidf_scores(queries: list[PilotQuery], memory: list[PilotMemoryItem]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=1)
    documents = vectorizer.fit_transform(item.text for item in memory)
    query_matrix = vectorizer.transform(query.text for query in queries)
    return (query_matrix @ documents.T).toarray().astype(np.float32)


def _bm25_scores(queries: list[PilotQuery], memory: list[PilotMemoryItem]) -> np.ndarray:
    from rank_bm25 import BM25Okapi

    tokenized = [item.text.casefold().split() for item in memory]
    model = BM25Okapi(tokenized)
    return np.vstack(
        [model.get_scores(query.text.casefold().split()) for query in queries]
    ).astype(np.float32)


def _dense_scores(model: Any, queries: list[PilotQuery], memory: list[PilotMemoryItem]) -> np.ndarray:
    documents = model.encode(
        [item.text for item in memory],
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_embeddings = model.encode(
        [query.text for query in queries],
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(query_embeddings @ documents.T, dtype=np.float32)


def _normalize_rows(scores: np.ndarray) -> np.ndarray:
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    span = maximum - minimum
    span[span == 0] = 1.0
    return (scores - minimum) / span


def _evaluate_scores(
    scores: np.ndarray,
    queries: list[PilotQuery],
    memory: list[PilotMemoryItem],
) -> dict[str, Any]:
    rankings = _rank_matrix(scores, queries, memory)
    return evaluate_rankings(
        queries=queries,
        memory_items=memory,
        rankings=rankings,
        token_budget=TOKEN_BUDGET,
    )


def _training_pairs(samples: list[ViewSample]) -> list[tuple[str, str]]:
    pairs = []
    for sample in sorted(samples, key=lambda item: item.sample_id):
        evidence = _evidence_text(sample)
        if evidence is None:
            raise ValueError(f"training retrieval sample lacks evidence: {sample.sample_id}")
        pairs.append((sample.query_text, evidence))
    return pairs


def _train_dense(pairs: list[tuple[str, str]], seed: int):
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SentenceTransformer(BASE_DENSE_MODEL)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        [InputExample(texts=[query, document]) for query, document in pairs],
        batch_size=32,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    loss = losses.MultipleNegativesRankingLoss(model)
    steps = math.ceil(len(pairs) / 32)
    with tempfile.TemporaryDirectory(prefix=f"evimem-retriever-seed-{seed}-") as temp_dir:
        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir)
            model.fit(
                train_objectives=[(loader, loss)],
                epochs=1,
                warmup_steps=max(1, int(steps * 0.1)),
                optimizer_params={"lr": 2e-5},
                show_progress_bar=True,
                output_path=str(Path(temp_dir) / "model"),
                checkpoint_path=None,
            )
        finally:
            os.chdir(original_cwd)
    return model


def _primary_aggregate(results: dict[str, dict[str, Any]]) -> dict[str, float]:
    datasets = ("SciREX", "SciFact")
    weights = {
        name: results[name]["fixed_k"]["queries_with_retrieval_gold"] for name in datasets
    }
    denominator = sum(weights.values())
    return {
        metric: sum(results[name]["fixed_k"][metric] * weights[name] for name in datasets)
        / denominator
        for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    }


def _evaluate_baseline(
    score_by_dataset: dict[str, np.ndarray],
    datasets: dict[str, tuple[list[PilotQuery], list[PilotMemoryItem]]],
) -> dict[str, Any]:
    result = {
        name: _evaluate_scores(score_by_dataset[name], *datasets[name]) for name in datasets
    }
    result["primary_aggregate"] = _primary_aggregate(result)
    return result


def _markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Phase 1B Retrieval Validity Pilot",
        "",
        "> Pilot results only. These are not formal paper main results.",
        "",
        "All baselines used the same per-dataset query set, evidence-memory pool, top-k, and "
        "256-token selection budget. QASPER is an internal diagnostic and was never used for training.",
        "",
        "## Primary aggregate (SciREX + SciFact)",
        "",
        "| Baseline | R@1 | R@5 | R@10 | MRR | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in report["baselines"].items():
        aggregate = result["primary_aggregate"]
        lines.append(
            f"| {name} | {aggregate['recall_at_1']:.4f} | {aggregate['recall_at_5']:.4f} | "
            f"{aggregate['recall_at_10']:.4f} | {aggregate['mrr']:.4f} | "
            f"{aggregate['ndcg_at_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Primary aggregate at fixed 256-token budget",
            "",
            "| Baseline | R@1 | R@5 | R@10 | MRR | nDCG@10 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in report["baselines"].items():
        weights = {
            dataset: result[dataset]["fixed_token_budget"]["queries_with_retrieval_gold"]
            for dataset in ("SciREX", "SciFact")
        }
        denominator = sum(weights.values())
        aggregate = {
            metric: sum(
                result[dataset]["fixed_token_budget"][metric] * weights[dataset]
                for dataset in weights
            )
            / denominator
            for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
        }
        lines.append(
            f"| {name} | {aggregate['recall_at_1']:.4f} | {aggregate['recall_at_5']:.4f} | "
            f"{aggregate['recall_at_10']:.4f} | {aggregate['mrr']:.4f} | "
            f"{aggregate['ndcg_at_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Three-seed fine-tuning variance",
            "",
            "```json",
            json.dumps(report["fine_tuned_three_seed_summary"], indent=2, sort_keys=True),
            "```",
            "",
            "## Certificate-aware interpretation",
            "",
            report["certificate_aware_effect"]["conclusion"],
            "",
            "Rejected/conflict/certificate-mismatch metrics are null because the authorized "
            "retrieval views contain no certificate or memory-type gold. No such labels were fabricated.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scirex", type=Path, required=True)
    parser.add_argument("--scifact", type=Path, required=True)
    parser.add_argument("--qasper", type=Path, required=True)
    parser.add_argument("--reports-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scirex_all, _, scirex_files = load_scirex(args.scirex)
    scifact_all, _, scifact_files = load_scifact(args.scifact)
    qasper_all, _, qasper_files = load_qasper(args.qasper)
    scirex_leakage = _load_report(args.reports_root / "scirex" / "leakage_report.json")
    scifact_leakage = _load_report(args.reports_root / "scifact" / "leakage_report.json")
    qasper_leakage = _load_report(args.reports_root / "qasper" / "leakage_report.json")

    scirex_train = _safe_samples(scirex_all, scirex_leakage, splits={"train"})
    scirex_eval = _safe_samples(scirex_all, scirex_leakage, splits={"dev", "test"})
    scifact_train = _safe_samples(scifact_all, scifact_leakage, splits={"train"})
    scifact_eval = _safe_samples(scifact_all, scifact_leakage, splits={"dev"})
    qasper_eval = _safe_samples(qasper_all, qasper_leakage, splits={"dev", "test"})
    actual_counts = {
        "SciREX_train": len(scirex_train),
        "SciREX_eval": len(scirex_eval),
        "SciFact_train": len(scifact_train),
        "SciFact_eval": len(scifact_eval),
        "QASPER_internal_diagnostic": len(qasper_eval),
    }
    expected_counts = {
        "SciREX_train": 517,
        "SciREX_eval": 177,
        "SciFact_train": 679,
        "SciFact_eval": 139,
        "QASPER_internal_diagnostic": 4555,
    }
    if actual_counts != expected_counts:
        raise RuntimeError(f"audited split counts changed: {actual_counts} != {expected_counts}")
    assert_pilot_split_isolation(
        train_document_ids={sample.source_document_id for sample in scirex_train},
        evaluation_document_ids={sample.source_document_id for sample in scirex_eval},
    )
    assert_pilot_split_isolation(
        train_document_ids={sample.source_document_id for sample in scifact_train},
        evaluation_document_ids={sample.source_document_id for sample in scifact_eval},
    )

    datasets = {
        "SciREX": build_pilot_dataset("SciREX", scirex_eval),
        "SciFact": build_pilot_dataset("SciFact", scifact_eval),
        "QASPER": build_pilot_dataset("QASPER", qasper_eval),
    }
    training_pairs = _training_pairs(scirex_train) + _training_pairs(scifact_train)
    if len(training_pairs) != 1196:
        raise RuntimeError("retriever optimization must use exactly 517 + 679 examples")

    tfidf_scores = {
        name: _tfidf_scores(*dataset) for name, dataset in datasets.items()
    }
    bm25_scores = {name: _bm25_scores(*dataset) for name, dataset in datasets.items()}

    from sentence_transformers import SentenceTransformer

    frozen_model = SentenceTransformer(BASE_DENSE_MODEL)
    frozen_scores = {
        name: _dense_scores(frozen_model, *dataset) for name, dataset in datasets.items()
    }
    del frozen_model
    scientific_model = SentenceTransformer(SCIENTIFIC_DENSE_MODEL)
    scientific_scores = {
        name: _dense_scores(scientific_model, *dataset) for name, dataset in datasets.items()
    }
    del scientific_model

    baselines: dict[str, Any] = {
        "TF-IDF": _evaluate_baseline(tfidf_scores, datasets),
        "BM25": _evaluate_baseline(bm25_scores, datasets),
        "frozen_dense": _evaluate_baseline(frozen_scores, datasets),
        "frozen_scientific_dense": _evaluate_baseline(scientific_scores, datasets),
    }
    fine_scores_by_seed: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        model = _train_dense(training_pairs, seed)
        scores = {name: _dense_scores(model, *dataset) for name, dataset in datasets.items()}
        fine_scores_by_seed[seed] = scores
        baselines[f"fine_tuned_dense_seed_{seed}"] = _evaluate_baseline(scores, datasets)
        del model

    reference_seed = SEEDS[0]
    certificate_scores = {
        name: scores.copy() for name, scores in fine_scores_by_seed[reference_seed].items()
    }
    baselines["dense_certificate_aware_reranker"] = _evaluate_baseline(
        certificate_scores, datasets
    )
    full_scores = {
        name: (
            0.65 * _normalize_rows(fine_scores_by_seed[reference_seed][name])
            + 0.25 * _normalize_rows(tfidf_scores[name])
            + 0.10
        )
        for name in datasets
    }
    baselines["EviMem_full_retrieval_score"] = _evaluate_baseline(full_scores, datasets)

    seed_aggregates = [
        baselines[f"fine_tuned_dense_seed_{seed}"]["primary_aggregate"] for seed in SEEDS
    ]
    seed_summary = {
        metric: {
            "mean": statistics.mean(item[metric] for item in seed_aggregates),
            "sample_std": statistics.stdev(item[metric] for item in seed_aggregates),
            "values": {str(seed): seed_aggregates[index][metric] for index, seed in enumerate(SEEDS)},
        }
        for metric in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr", "ndcg_at_10")
    }
    reference = baselines[f"fine_tuned_dense_seed_{reference_seed}"]["primary_aggregate"]
    reranked = baselines["dense_certificate_aware_reranker"]["primary_aggregate"]
    deltas = {metric: reranked[metric] - reference[metric] for metric in reference}

    import torch

    package_versions = {
        package: importlib.metadata.version(package)
        for package in (
            "torch",
            "sentence-transformers",
            "transformers",
            "scikit-learn",
            "rank-bm25",
        )
    }
    report = {
        "schema_version": "evimem.phase1b_retrieval_pilot.v1",
        "created_at": datetime.now(UTC).isoformat(),
        "result_scope": "retrieval_validity_pilot_not_formal_paper_results",
        "retrieval_training_executed": True,
        "update_manager_trained": False,
        "qlora_executed": False,
        "models": {
            "frozen_dense": BASE_DENSE_MODEL,
            "frozen_scientific_dense": SCIENTIFIC_DENSE_MODEL,
            "fine_tuned_dense_initialization": BASE_DENSE_MODEL,
            "fine_tuning": {
                "objective": "MultipleNegativesRankingLoss",
                "epochs": 1,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "seeds": list(SEEDS),
                "checkpoint_committed": False,
            },
        },
        "environment": {
            "packages": package_versions,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "torchcodec_removed": True,
            "dependency_note": (
                "Removed optional incompatible torchcodec 0.14.0; no audio/video path is used."
            ),
        },
        "audited_sample_counts": actual_counts,
        "training_sources": ["SciREX train", "SciFact leakage-safe train"],
        "excluded_from_training": [
            "SciREX dev/test",
            "SciFact dev",
            "QASPER",
            "Evidence Inference 2.0",
            "MeasEval",
            "POLYIE",
            "BioRED",
        ],
        "fairness": {
            "same_memory_pool_per_dataset": True,
            "same_queries_per_dataset": True,
            "top_k": TOP_K,
            "ranked_candidates_retained": RANK_LIMIT,
            "fixed_token_budget": TOKEN_BUDGET,
            "oracle_positive_ids_excluded_from_model_input": True,
        },
        "memory_pool_counts": {
            name: {
                "queries": len(dataset[0]),
                "queries_with_retrieval_gold": sum(bool(query.positive_memory_ids) for query in dataset[0]),
                "unique_memory_items": len(dataset[1]),
            }
            for name, dataset in datasets.items()
        },
        "baselines": baselines,
        "fine_tuned_three_seed_summary": seed_summary,
        "certificate_aware_effect": {
            "reference_seed": reference_seed,
            "observed_primary_metric_deltas": deltas,
            "effective": None,
            "conclusion": (
                "Not estimable in this pilot: the authorized retrieval views have no certificate "
                "or memory-type gold. The fail-closed reranker therefore leaves dense scores "
                "unchanged, and no certificate labels were fabricated."
            ),
        },
        "metric_availability": {
            "verified_recall": "unavailable_no_memory_type_gold",
            "rejected_recall": "unavailable_no_memory_type_gold",
            "conflict_recall": "unavailable_no_memory_type_gold",
            "certificate_mismatch_rate": "unavailable_no_certificate_labels",
            "stale_rate": "reported; pool contains no temporal stale labels",
            "policy_incompatible_rate": "reported; licensed in-scope pools are policy-compatible",
        },
        "source_checksums": [
            {"name": path.name, "sha256": sha256_file(path)}
            for path in sorted(
                {path.resolve() for path in scirex_files + scifact_files + qasper_files if path.exists()},
                key=lambda item: str(item),
            )
        ],
    }
    output = args.output
    _write_json(output / "retrieval_results.json", report)
    (output / "retrieval_results.md").write_text(_markdown_report(report), encoding="utf-8")
    print(json.dumps({"counts": actual_counts, "seed_summary": seed_summary}, default=float))


if __name__ == "__main__":
    main()
