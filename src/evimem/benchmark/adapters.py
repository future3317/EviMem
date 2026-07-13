"""Conservative adapters from pinned public schemas into separated task views."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

from evimem.contracts import ScientificClaimRecord, ScientificMemoryRecord

from .datasets import DataView
from .episode import BenchmarkEpisode, MemoryQuery, OracleAnnotation, ScientificDocument
from .views import ConversionOrigin, ViewSample, exact_evidence


def _checksum(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity(row: Mapping[str, Any]) -> str:
    value = row.get("document_id") or row.get("doc_id") or row.get("article_id") or row.get("id")
    if value is None or not str(value).strip():
        raise ValueError("row lacks a source document identity")
    return str(value)


def _timestamp(row: Mapping[str, Any]) -> datetime | None:
    raw = row.get("timestamp") or row.get("publication_date")
    if raw is None or not str(raw).strip():
        return None
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed


def _text(row: Mapping[str, Any]) -> str:
    words = row.get("words")
    if isinstance(words, Sequence) and not isinstance(words, (str, bytes)):
        return " ".join(str(item) for item in words)
    for key in ("text", "document", "abstract"):
        value = row.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return "\n".join(str(item) for item in value)
    return ""


class DatasetAdapter(ABC):
    dataset_name: str

    @abstractmethod
    def convert_views(self, row: Mapping[str, Any], *, split: str) -> tuple[ViewSample, ...]: ...

    def convert(
        self,
        row: Mapping[str, Any],
        *,
        split: str,
        stream_position: int,
    ) -> tuple[BenchmarkEpisode, OracleAnnotation]:
        """Compatibility projection; gold admission/update remain absent unless native."""

        samples = self.convert_views(row, split=split)
        if not samples:
            raise ValueError(f"{self.dataset_name} row produced no auditable sample")
        sample = samples[0]
        document_id = _identity(row)
        episode_id = f"{self.dataset_name}:{split}:{document_id}:{stream_position}"
        history = tuple(
            ScientificMemoryRecord.model_validate(item) for item in (row.get("history") or ())
        )
        episode = BenchmarkEpisode(
            episode_id=episode_id,
            stream_position=stream_position,
            history=history,
            current_document=ScientificDocument(
                document_id=document_id,
                text=sample.source_text,
                timestamp=_timestamp(row),
                dataset_name=self.dataset_name,
                split=split,
                metadata={
                    "source_row_checksum": _checksum(row),
                    "view": sample.view.value if sample.view else None,
                    "oracle_fields_excluded": True,
                },
            ),
            query=MemoryQuery(
                query_id=f"query:{episode_id}",
                text=sample.query_text,
                candidate_claim=sample.claim,
            ),
        )
        return episode, OracleAnnotation(
            episode_id=episode_id,
            final_record=(sample.claim if sample.view != DataView.RETRIEVAL else None),
            admission=sample.admission,
            memory_operation=sample.memory_operation,
        )


class SciRexAdapter(DatasetAdapter):
    dataset_name = "SciREX"
    _entity_types = ("Material", "Method", "Metric", "Task")

    def convert_views(self, row: Mapping[str, Any], *, split: str) -> tuple[ViewSample, ...]:
        document_id = _identity(row)
        source_text = _text(row)
        relations = row.get("n_ary_relations")
        if relations is None and isinstance(row.get("relation"), Mapping):
            relations = [row["relation"]]
        if not isinstance(relations, Sequence) or isinstance(relations, (str, bytes)):
            raise ValueError("SciREX conversion requires n_ary_relations")
        coref = row.get("coref") or row.get("clusters") or {}
        words = list(row.get("words") or [])
        samples: list[ViewSample] = []
        for index, relation in enumerate(relations):
            if not isinstance(relation, Mapping):
                raise ValueError("SciREX relation must be an object")
            missing_types = [
                entity_type
                for entity_type in self._entity_types
                if not (coref.get(relation.get(entity_type)) or [])
            ]
            filtered_out = bool(missing_types)
            evidence = []
            if words and not filtered_out:
                for entity_type in self._entity_types:
                    mention = coref[str(relation[entity_type])][0]
                    start_word, end_word = int(mention[0]), int(mention[1])
                    mention_text = " ".join(str(item) for item in words[start_word:end_word])
                    evidence.append(
                        exact_evidence(
                            source_text,
                            mention_text,
                            source_document_id=document_id,
                            source_field=f"coref.{entity_type}",
                        )
                    )
            task = relation.get("Task") or relation.get("task")
            method = relation.get("Method") or relation.get("method")
            metric = relation.get("Metric") or relation.get("metric")
            material = relation.get("Material") or relation.get("material")
            if not method or not metric:
                raise ValueError("SciREX relation lacks Method or Metric")
            claim = ScientificClaimRecord(
                subject=str(method),
                relation=str(metric),
                object=str(task) if task else None,
                value=relation.get("score") or relation.get("Score"),
                condition={"dataset": material} if material is not None else None,
            )
            samples.append(
                ViewSample(
                    sample_id=f"scirex:{split}:{document_id}:relation:{index}",
                    dataset_name=self.dataset_name,
                    source_document_id=document_id,
                    split=split,
                    view=None if filtered_out else DataView.RETRIEVAL,
                    native_task=(
                        "filtered_relation_table_missing"
                        if filtered_out
                        else "n_ary_relation_retrieval"
                    ),
                    query_text=claim.canonical_key(),
                    source_text=source_text,
                    target={"relation": dict(relation), "retrievable": not filtered_out},
                    claim=claim,
                    evidence=tuple(evidence),
                    origin=ConversionOrigin.DETERMINISTIC_DERIVED,
                    label_source="native_scirex_relation_and_coreference",
                    metadata={
                        "official_filtered_protocol": True,
                        "table_missing_relation": filtered_out,
                        "missing_mention_types": missing_types,
                        "source_row_checksum": _checksum(row),
                    },
                )
            )
        return tuple(samples)


class QasperAdapter(DatasetAdapter):
    dataset_name = "QASPER"

    @staticmethod
    def _article_text(row: Mapping[str, Any]) -> str:
        paragraphs: list[str] = []
        abstract = row.get("abstract")
        if isinstance(abstract, str) and abstract.strip():
            paragraphs.append(abstract.replace("\n", " ").strip())
        for section in row.get("full_text") or ():
            if not isinstance(section, Mapping):
                continue
            name = section.get("section_name")
            if name:
                paragraphs.append(str(name).replace("\n", " ").strip())
            for paragraph in section.get("paragraphs") or ():
                normalized = str(paragraph).replace("\n", " ").strip()
                if normalized:
                    paragraphs.append(normalized)
        return "\n".join(paragraphs) or _text(row)

    def convert_views(self, row: Mapping[str, Any], *, split: str) -> tuple[ViewSample, ...]:
        document_id = _identity(row)
        source_text = self._article_text(row)
        qas = row.get("qas")
        if qas is None and row.get("question") is not None:
            qas = [row]
        if not isinstance(qas, Sequence) or isinstance(qas, (str, bytes)):
            raise ValueError("QASPER conversion requires qas")
        samples: list[ViewSample] = []
        for question_index, qa in enumerate(qas):
            question = str(qa.get("question") or "").strip()
            if not question:
                raise ValueError("QASPER row lacks a question")
            answers = qa.get("answers")
            if answers is None:
                answers = [{"answer": {"evidence": qa.get("evidence") or []}}]
            for answer_index, annotation in enumerate(answers):
                answer = annotation.get("answer", annotation)
                spans = []
                for evidence_text in answer.get("evidence") or ():
                    normalized = str(evidence_text).replace("\n", " ").strip()
                    if not normalized or "FLOAT SELECTED" in normalized:
                        continue
                    spans.append(
                        exact_evidence(
                            source_text,
                            normalized,
                            source_document_id=document_id,
                            source_field="answer.evidence",
                        )
                    )
                answer_target = {
                    key: answer.get(key)
                    for key in ("unanswerable", "extractive_spans", "yes_no", "free_form_answer")
                    if key in answer
                }
                samples.append(
                    ViewSample(
                        sample_id=(
                            f"qasper:{split}:{document_id}:"
                            f"{qa.get('question_id', question_index)}:"
                            f"{annotation.get('annotation_id', answer_index)}"
                        ),
                        dataset_name=self.dataset_name,
                        source_document_id=document_id,
                        split=split,
                        view=DataView.RETRIEVAL,
                        native_task="question_evidence_retrieval",
                        query_text=question,
                        source_text=source_text,
                        target={"answer": answer_target},
                        evidence=tuple(spans),
                        origin=ConversionOrigin.NATIVE,
                        label_source="native_qasper_answer_evidence",
                        metadata={
                            "annotation_id": annotation.get("annotation_id"),
                            "source_row_checksum": _checksum(row),
                            "admission_label_available": False,
                            "update_label_available": False,
                        },
                    )
                )
        return tuple(samples)


class SciFactAdapter(DatasetAdapter):
    dataset_name = "SciFact"

    def convert_views(self, row: Mapping[str, Any], *, split: str) -> tuple[ViewSample, ...]:
        claim_text = str(row.get("claim") or "").strip()
        if not claim_text:
            raise ValueError("SciFact row lacks a claim")
        document_id = _identity(row)
        source_text = _text(row)
        label = str(row.get("label") or row.get("veracity_label") or "").upper() or None
        sentence_ids = row.get("evidence_sentence_ids") or row.get("sentences") or ()
        abstract_sentences = row.get("abstract_sentences") or row.get("abstract")
        evidence = []
        if isinstance(abstract_sentences, Sequence) and not isinstance(
            abstract_sentences, (str, bytes)
        ):
            source_text = "\n".join(str(item) for item in abstract_sentences)
            for sentence_id in sentence_ids:
                sentence_text = str(abstract_sentences[int(sentence_id)])
                evidence.append(
                    exact_evidence(
                        source_text,
                        sentence_text,
                        source_document_id=document_id,
                        source_field=f"abstract[{sentence_id}]",
                    )
                )
        else:
            for evidence_text in row.get("evidence_texts") or ():
                evidence.append(
                    exact_evidence(
                        source_text,
                        str(evidence_text),
                        source_document_id=document_id,
                        source_field="evidence_texts",
                    )
                )
        claim = ScientificClaimRecord(subject=claim_text, relation="scientific_claim_veracity")
        return (
            ViewSample(
                sample_id=f"scifact:{split}:{document_id}:{row.get('id', 'claim')}",
                dataset_name=self.dataset_name,
                source_document_id=document_id,
                split=split,
                view=DataView.RETRIEVAL,
                native_task="claim_rationale_retrieval",
                query_text=claim_text,
                source_text=source_text,
                target={"veracity_label": label},
                claim=claim,
                evidence=tuple(evidence),
                origin=ConversionOrigin.NATIVE,
                label_source="native_scifact_rationale",
                metadata={
                    "source_row_checksum": _checksum(row),
                    "support_refute_is_not_admission": True,
                    "contradiction_is_not_supersede": True,
                },
            ),
        )


class EvidenceInferenceAdapter(DatasetAdapter):
    dataset_name = "Evidence Inference 2.0"

    def convert_views(self, row: Mapping[str, Any], *, split: str) -> tuple[ViewSample, ...]:
        intervention = str(row.get("intervention") or "").strip()
        comparator = str(row.get("comparator") or "").strip()
        outcome = str(row.get("outcome") or "").strip()
        if not intervention or not outcome:
            raise ValueError("Evidence Inference row lacks intervention or outcome")
        document_id = _identity(row)
        source_text = _text(row)
        evidence_text = row.get("evidence_text") or row.get("evidence")
        evidence = ()
        if evidence_text:
            evidence = (
                exact_evidence(
                    source_text,
                    str(evidence_text),
                    source_document_id=document_id,
                    source_field="evidence_text",
                    start_hint=(int(row["evidence_start"]) if row.get("evidence_start") is not None else None),
                ),
            )
        claim = ScientificClaimRecord(
            subject=intervention,
            relation=outcome,
            object=comparator or None,
            value=row.get("label") or row.get("direction"),
            condition=(dict(row["condition"]) if row.get("condition") is not None else None),
        )
        return (
            ViewSample(
                sample_id=f"evidence-inference:{split}:{document_id}:{row.get('id', 'prompt')}",
                dataset_name=self.dataset_name,
                source_document_id=document_id,
                split=split,
                view=DataView.RETRIEVAL,
                native_task="ico_evidence_retrieval",
                query_text=f"{intervention} versus {comparator} on {outcome}",
                source_text=source_text,
                target={"direction": row.get("label") or row.get("direction")},
                claim=claim,
                evidence=evidence,
                origin=ConversionOrigin.NATIVE,
                label_source="native_evidence_inference_span",
                metadata={
                    "source_row_checksum": _checksum(row),
                    "source_article_license_resolved": bool(row.get("source_article_license")),
                },
            ),
        )


class MeasEvalAdapter(DatasetAdapter):
    dataset_name = "MeasEval"

    def convert_views(self, row: Mapping[str, Any], *, split: str) -> tuple[ViewSample, ...]:
        document_id = _identity(row)
        source_text = _text(row)
        entity = row.get("measured_entity")
        prop = row.get("measured_property")
        quantity = row.get("quantity")
        if entity is None and prop is None and quantity is None:
            raise ValueError("MeasEval row lacks native slot annotations")
        spans = []
        for field in ("measured_entity", "measured_property", "quantity"):
            value = row.get(field)
            if value is not None and str(value):
                spans.append(
                    exact_evidence(
                        source_text,
                        str(value),
                        source_document_id=document_id,
                        source_field=field,
                        start_hint=(
                            int(row[f"{field}_start"])
                            if row.get(f"{field}_start") is not None
                            else None
                        ),
                    )
                )
        return (
            ViewSample(
                sample_id=f"measeval:{split}:{document_id}:{row.get('id', 'annotation')}",
                dataset_name=self.dataset_name,
                source_document_id=document_id,
                split=split,
                view=None,
                native_task="slot_extraction_only",
                query_text="extract measurement slots",
                source_text=source_text,
                target={
                    "measured_entity": entity,
                    "measured_property": prop,
                    "quantity": quantity,
                    "unit": row.get("unit"),
                    "context": row.get("context"),
                },
                evidence=tuple(spans),
                origin=ConversionOrigin.NATIVE,
                label_source="native_measeval_slot_annotation",
                metadata={
                    "source_row_checksum": _checksum(row),
                    "license_blocked": True,
                    "admission_label_available": False,
                    "update_label_available": False,
                },
            ),
        )


ADAPTERS: dict[str, type[DatasetAdapter]] = {
    adapter.dataset_name: adapter
    for adapter in (
        SciRexAdapter,
        SciFactAdapter,
        QasperAdapter,
        EvidenceInferenceAdapter,
        MeasEvalAdapter,
    )
}
