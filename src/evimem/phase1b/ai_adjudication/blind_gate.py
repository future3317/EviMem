"""Fail-closed local gate for independently produced blind model candidates.

The gate never chooses a semantic label, compiles a memory operation, or sends
one model's annotation to another model.  External reviewers must receive only
safe packets and produce their own blind candidates.  This module compares
those candidates locally and decides whether a record is merely a provisional
consensus candidate or must remain in the human/high-risk review queue.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evimem.contracts.memory import EvidenceSufficiency, ScopeRelation, SemanticRelation
from evimem.phase1b.ai_adjudication.schema import AdjudicationPacket, validate_ai_adjudication_label
from evimem.phase1b.ai_adjudication.validate import read_jsonl_records
from evimem.phase1b.candidates import sha256_json

GATE_SCHEMA_VERSION = "phase1b-blind-gate-v1"
AXIS_FIELDS = (
    "semantic_relation",
    "scope_relation",
    "authority_relation",
    "evidence_sufficiency",
)


class BlindGateError(ValueError):
    """Raised when candidates are unsafe, incomplete, or incomparable."""


@dataclass(frozen=True)
class CandidateExport:
    """A validated blind candidate set from one model/run."""

    source_name: str
    source_checksum: str
    records: Mapping[str, dict[str, Any]]


def _file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def load_blind_candidate_export(
    path: Path,
    packets: Mapping[str, AdjudicationPacket],
    *,
    allow_superset: bool = False,
) -> CandidateExport:
    """Load a candidate export and reject anything other than blind juror labels."""
    records: dict[str, dict[str, Any]] = {}
    for line_number, record in enumerate(read_jsonl_records(path), start=1):
        task_id = record.get("task_id")
        if not isinstance(task_id, str):
            raise BlindGateError(f"invalid or duplicate task_id at {path}:{line_number}")
        packet = packets.get(task_id)
        if packet is None:
            if allow_superset:
                continue
            raise BlindGateError(f"candidate task_id is absent from packet set: {task_id}")
        if task_id in records:
            raise BlindGateError(f"invalid or duplicate task_id at {path}:{line_number}")
        try:
            normalized = validate_ai_adjudication_label(
                record, packet=packet
            )
        except ValueError as exc:
            raise BlindGateError(f"invalid candidate at {path}:{line_number}") from exc
        if normalized["annotation_provenance"] != "ai_juror":
            raise BlindGateError("blind gate accepts ai_juror candidates only")
        records[task_id] = normalized

    expected = set(packets)
    observed = set(records)
    if observed != expected:
        missing = len(expected - observed)
        unexpected = len(observed - expected)
        raise BlindGateError(
            f"candidate task set mismatch for {path}: missing={missing}, unexpected={unexpected}"
        )
    return CandidateExport(
        source_name=path.name,
        source_checksum=_file_checksum(path),
        records=records,
    )


def _candidate_proof(record: Mapping[str, Any]) -> dict[str, str]:
    """Keep local provenance without embedding labels or notes in gate exports."""
    return {
        "model_id": str(record["model_id"]),
        "juror_run_id": str(record["juror_run_id"]),
        "annotation_checksum": sha256_json(dict(record)),
    }


def _risk_reasons(
    records: Iterable[Mapping[str, Any]],
    *,
    distinct_model_count: int,
    min_distinct_models: int,
    require_model_diversity: bool,
) -> list[str]:
    candidates = list(records)
    reasons: list[str] = []
    if require_model_diversity and distinct_model_count < min_distinct_models:
        reasons.append("insufficient_distinct_model_diversity")
    for field in AXIS_FIELDS:
        if len({candidate[field] for candidate in candidates}) != 1:
            reasons.append(f"model_disagreement:{field}")
    if any(
        candidate["evidence_sufficiency"] != EvidenceSufficiency.SUFFICIENT.value
        for candidate in candidates
    ):
        reasons.append("evidence_not_unanimously_sufficient")
    if any(
        candidate["semantic_relation"] == SemanticRelation.INSUFFICIENT_CONTEXT.value
        for candidate in candidates
    ):
        reasons.append("insufficient_context_candidate")
    if any(
        candidate["semantic_relation"] == SemanticRelation.CONTRADICTORY.value
        and candidate["scope_relation"] == ScopeRelation.SAME_SCOPE.value
        for candidate in candidates
    ):
        reasons.append("same_scope_contradiction_requires_higher_review")
    return reasons


def build_blind_gate_records(
    packets: Mapping[str, AdjudicationPacket],
    candidates: Iterable[CandidateExport],
    *,
    min_distinct_models: int = 3,
    same_model_repeat_mode: bool = False,
) -> list[dict[str, Any]]:
    """Build conservative local routing records without selecting a label winner.

    ``same_model_repeat_mode`` is deliberately a stability check, not a proxy for
    independent-model agreement.  It accepts repeated calls only when they have
    distinct blind juror-run identifiers.  Ledger verification of each export
    remains a separate prerequisite for interpreting such a check.
    """
    candidate_exports = list(candidates)
    if len(candidate_exports) < 2:
        raise BlindGateError("at least two blind candidate exports are required")
    if min_distinct_models < 2:
        raise BlindGateError("min_distinct_models must be at least two")

    records: list[dict[str, Any]] = []
    for task_id, packet in sorted(packets.items()):
        task_candidates = [candidate.records[task_id] for candidate in candidate_exports]
        proofs = [_candidate_proof(candidate) for candidate in task_candidates]
        distinct_models = {proof["model_id"] for proof in proofs}
        if same_model_repeat_mode and len(distinct_models) != 1:
            raise BlindGateError(
                "same_model_repeat_mode requires every candidate to have the same model_id"
            )
        distinct_blind_runs = {proof["juror_run_id"] for proof in proofs}
        if same_model_repeat_mode and len(distinct_blind_runs) != len(proofs):
            raise BlindGateError(
                "same_model_repeat_mode requires distinct blind juror_run_id values"
            )
        reasons = _risk_reasons(
            task_candidates,
            distinct_model_count=len(distinct_models),
            min_distinct_models=min_distinct_models,
            require_model_diversity=not same_model_repeat_mode,
        )
        if packet.source_dataset == "Crossref/Retraction Watch":
            reasons.append("source_level_metadata_only")
        status = (
            (
                "same_model_repeat_consistent_candidate"
                if same_model_repeat_mode
                else "provisional_multi_model_consensus_candidate"
            )
            if not reasons
            else "requires_human_review"
        )
        records.append(
            {
                "schema_version": GATE_SCHEMA_VERSION,
                "task_id": task_id,
                "packet_checksum": packet.packet_checksum,
                "candidate_export_checksums": [
                    candidate.source_checksum for candidate in candidate_exports
                ],
                "candidate_proofs": proofs,
                "candidate_count": len(task_candidates),
                "distinct_model_count": len(distinct_models),
                "distinct_blind_run_count": len(distinct_blind_runs),
                "axis_agreement": {
                    field: len({candidate[field] for candidate in task_candidates}) == 1
                    for field in AXIS_FIELDS
                },
                "risk_reasons": reasons,
                "gate_status": status,
                "gold_status": "not_gold",
            }
        )
    return records


def summarize_gate_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize routing only; do not compute a selected annotation label."""
    items = list(records)
    status_counts = Counter(str(item["gate_status"]) for item in items)
    risk_counts = Counter(
        reason for item in items for reason in item.get("risk_reasons", [])
    )
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "record_count": len(items),
        "gate_status_counts": dict(sorted(status_counts.items())),
        "risk_reason_counts": dict(sorted(risk_counts.items())),
        "interpretation": (
            "Routing metadata only: no label winner, gold label, training target, "
            "memory operation, or final AI adjudication is produced."
        ),
    }
