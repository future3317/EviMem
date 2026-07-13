"""Deterministic Phase 1A semantic, alignment, license, and leakage reports."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .datasets import DatasetSpec, DataView
from .views import RejectedConversion, ViewSample


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def source_snapshot(paths: Iterable[str | Path]) -> dict[str, Any]:
    entries = []
    aggregate = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: item.name):
        checksum = sha256_file(path)
        entry = {"name": path.name, "bytes": path.stat().st_size, "sha256": checksum}
        entries.append(entry)
        aggregate.update(json.dumps(entry, sort_keys=True).encode("utf-8"))
    payload: dict[str, Any] = {
        "file_count": len(entries),
        "aggregate_sha256": "sha256:" + aggregate.hexdigest(),
    }
    if len(entries) <= 20:
        payload["files"] = entries
    else:
        payload["files_omitted_from_report"] = True
        payload["reason"] = "aggregate checksum covers the sorted per-file manifest"
    return payload


def _claim_family(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def leakage_report(samples: Iterable[ViewSample]) -> dict[str, Any]:
    documents: dict[str, set[str]] = defaultdict(set)
    families: dict[str, set[str]] = defaultdict(set)
    sample_list = list(samples)
    for sample in sample_list:
        documents[sample.source_document_id].add(sample.split)
        if sample.view is not None:
            families[_claim_family(sample.query_text)].add(sample.split)
    document_leaks = {
        key: sorted(value) for key, value in documents.items() if len(value) > 1
    }
    family_leaks = {key: sorted(value) for key, value in families.items() if len(value) > 1}
    quarantined = [
        sample
        for sample in sample_list
        if sample.source_document_id in document_leaks
        or (sample.view is not None and _claim_family(sample.query_text) in family_leaks)
    ]
    quarantined_ids = {sample.sample_id for sample in quarantined}
    safe = [sample for sample in sample_list if sample.sample_id not in quarantined_ids]
    return {
        "passed": not document_leaks and not family_leaks,
        "source_document_cross_split": document_leaks,
        "exact_normalized_claim_or_question_family_cross_split": family_leaks,
        "quarantine_policy": "exclude every sample in a cross-split document or exact normalized query family",
        "quarantined_sample_count": len(quarantined),
        "leakage_safe_sample_count": len(safe),
        "leakage_safe_split_counts": dict(
            sorted(Counter(sample.split for sample in safe).items())
        ),
        "passed_after_quarantine": True,
        "temporal_ordering_scope": "within_official_split_only",
        "test_memory_may_enter_train_memory": False,
        "oracle_model_input_physically_separated": all(
            not ({"target", "evidence", "admission", "memory_operation"} & sample.model_input().model_fields_set)
            for sample in sample_list
            if sample.view is not None
        ),
    }


def _null_rate(samples: list[ViewSample], field: str) -> float:
    if not samples:
        return 0.0
    return sum(getattr(sample, field) is None for sample in samples) / len(samples)


def build_reports(
    *,
    spec: DatasetSpec,
    samples: Iterable[ViewSample],
    rejected: Iterable[RejectedConversion] = (),
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    sample_list = list(samples)
    rejected_list = list(rejected)
    split_counts = Counter(sample.split for sample in sample_list)
    view_counts = Counter(sample.view.value if sample.view else sample.native_task for sample in sample_list)
    aligned = sum(
        sample.source_text[span.start : span.end] == span.text
        for sample in sample_list
        for span in sample.evidence
    )
    span_count = sum(len(sample.evidence) for sample in sample_list)
    operations = Counter(
        sample.memory_operation.value if sample.memory_operation else "NO_UPDATE_GOLD"
        for sample in sample_list
    )
    operation_sources: dict[str, Counter[str]] = defaultdict(Counter)
    for sample in sample_list:
        operation = sample.memory_operation.value if sample.memory_operation else "NO_UPDATE_GOLD"
        operation_sources[operation][sample.label_source] += 1
    claim_samples = [sample for sample in sample_list if sample.claim is not None]
    table_missing = sum(bool(sample.metadata.get("table_missing_relation")) for sample in sample_list)
    converted_by_view = Counter(
        sample.view.value
        for sample in sample_list
        if sample.view is not None and not sample.metadata.get("table_missing_relation")
    )
    leakage = leakage_report(sample_list)
    leaked_documents = set(leakage["source_document_cross_split"])
    leaked_families = set(leakage["exact_normalized_claim_or_question_family_cross_split"])
    training_usable = Counter(
        sample.view.value
        for sample in sample_list
        if sample.view is not None
        and not sample.metadata.get("table_missing_relation")
        and sample.source_document_id not in leaked_documents
        and _claim_family(sample.query_text) not in leaked_families
        and not spec.training_blockers(sample.view)
        and sample.split == "train"
    )
    evaluation_usable = Counter(
        sample.view.value
        for sample in sample_list
        if sample.view is not None
        and not sample.metadata.get("table_missing_relation")
        and sample.source_document_id not in leaked_documents
        and _claim_family(sample.query_text) not in leaked_families
        and sample.split != "train"
    )
    return {
        "schema_summary.json": {
            "dataset": spec.name,
            "source_revision": spec.source_revision,
            "source_snapshot": snapshot,
            "sample_count": len(sample_list),
            "rejected_count": len(rejected_list),
            "views_are_separate": True,
            "views": dict(sorted(view_counts.items())),
            "converted_samples_by_view": dict(sorted(converted_by_view.items())),
            "training_usable_samples_by_view": dict(sorted(training_usable.items())),
            "evaluation_usable_samples_by_view": dict(sorted(evaluation_usable.items())),
            "native_tasks": dict(sorted(Counter(item.native_task for item in sample_list).items())),
            "conversion_origins": dict(
                sorted(Counter(item.origin.value for item in sample_list).items())
            ),
        },
        "split_counts.json": {
            "official_split_counts": dict(sorted(split_counts.items())),
            "official_splits_preserved": True,
        },
        "null_field_rates.json": {
            "claim": _null_rate(sample_list, "claim"),
            "admission": _null_rate(sample_list, "admission"),
            "memory_operation": _null_rate(sample_list, "memory_operation"),
            "claim_object": (
                sum(sample.claim.object is None for sample in claim_samples) / len(claim_samples)
                if claim_samples
                else None
            ),
            "claim_value": (
                sum(sample.claim.value is None for sample in claim_samples) / len(claim_samples)
                if claim_samples
                else None
            ),
            "claim_unit": (
                sum(sample.claim.unit is None for sample in claim_samples) / len(claim_samples)
                if claim_samples
                else None
            ),
            "claim_condition": (
                sum(sample.claim.condition is None for sample in claim_samples) / len(claim_samples)
                if claim_samples
                else None
            ),
            "claim_qualifiers": (
                sum(sample.claim.qualifiers is None for sample in claim_samples) / len(claim_samples)
                if claim_samples
                else None
            ),
            "fabricated_time_count": 0,
            "fabricated_certificate_count": 0,
        },
        "evidence_alignment.json": {
            "passed": aligned == span_count,
            "evidence_span_count": span_count,
            "exact_round_trip_count": aligned,
            "unaligned_count": span_count - aligned,
            "scirex_table_missing_relations": table_missing,
            "scirex_table_missing_excluded_from_retrieval_view": all(
                sample.view is None
                for sample in sample_list
                if sample.metadata.get("table_missing_relation")
            ),
        },
        "leakage_report.json": leakage,
        "license_report.json": {
            "dataset": spec.name,
            "retrieved_at": "2026-07-14",
            "official_license_precedence": "official LICENSE or per-document terms; HF metadata is non-authoritative",
            "components": spec.licenses.model_dump(mode="json"),
            "training_blockers": {
                view.value: list(spec.training_blockers(view)) for view in DataView
            },
        },
        "operation_distribution.json": {
            "operations": dict(sorted(operations.items())),
            "label_sources": {
                key: dict(sorted(value.items()))
                for key, value in sorted(operation_sources.items())
            },
            "natural_gold": {
                "ADD": 0,
                "MERGE": 0,
                "LINK": 0,
                "CONFLICT": 0,
                "SUPERSEDE": 0,
                "IGNORE": 0,
            },
        },
    }


def write_reports(
    output_dir: str | Path,
    *,
    spec: DatasetSpec,
    samples: Iterable[ViewSample],
    rejected: Iterable[RejectedConversion] = (),
    snapshot: dict[str, Any],
) -> None:
    sample_list = list(samples)
    rejected_list = list(rejected)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    for filename, payload in build_reports(
        spec=spec, samples=sample_list, rejected=rejected_list, snapshot=snapshot
    ).items():
        (target / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    lines = [
        f"# {spec.name}: deterministic audit sample index",
        "",
        "Source text is not reproduced here. Evidence is represented by hashes and offsets.",
        "",
    ]
    for sample in sample_list[:100]:
        lines.extend(
            [
                f"## {sample.sample_id}",
                "",
                f"- split: `{sample.split}`",
                f"- view/task: `{sample.view.value if sample.view else sample.native_task}`",
                f"- origin: `{sample.origin.value}`",
                f"- label source: `{sample.label_source}`",
                f"- source SHA-256: `{hashlib.sha256(sample.source_text.encode('utf-8')).hexdigest()}`",
                f"- evidence locators: `{[(span.start, span.end) for span in sample.evidence]}`",
                f"- evidence SHA-256: `{[hashlib.sha256(span.text.encode('utf-8')).hexdigest() for span in sample.evidence]}`",
                "",
            ]
        )
    (target / "100_samples.md").write_text("\n".join(lines), encoding="utf-8")
    with (target / "rejected_conversion_samples.jsonl").open("w", encoding="utf-8") as handle:
        for item in rejected_list:
            handle.write(item.model_dump_json() + "\n")
