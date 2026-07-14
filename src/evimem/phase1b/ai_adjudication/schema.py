"""Contracts for AI-adjudicated SciMem-Update silver annotation.

These models define the canonical schema for packets, juror annotations,
critic reviews, and adjudicated silver labels. They are intentionally
separate from human-annotation workflows and from compiled memory operations.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from evimem.contracts.memory import (
    AuthorityRelation,
    EvidenceSufficiency,
    ScopeRelation,
    SemanticRelation,
)
from evimem.phase1b.candidates import sha256_json

SCHEMA_VERSION: str = "phase1b-v3"
GOLD_STATUS: Literal["not_gold"] = "not_gold"

FORBIDDEN_OPERATION_LABELS: frozenset[str] = frozenset(
    {"ADD", "MERGE", "LINK", "CONFLICT", "SUPERSEDE", "IGNORE"}
)
FORBIDDEN_OPERATION_KEYS: frozenset[str] = frozenset(
    {"compiled_operation", "update_operation", "operation", "memory_operation"}
)
FORBIDDEN_PROVENANCE_TERMS: frozenset[str] = frozenset(
    {"human-reviewed", "adjudicated human evidence", "gold", "sciMem-update gold"}
)


class PacketSide(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_text: str
    source_document_id: str
    evidence_locator: str
    evidence_checksum: str

    @field_validator("claim_text", "source_document_id", "evidence_locator")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("packet side fields must be non-empty")
        return normalized

    @field_validator("evidence_checksum")
    @classmethod
    def _sha256_checksum(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("evidence_checksum must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"


class AdjudicationPacket(BaseModel):
    """A single minimal task packet for blind AI adjudication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    source_dataset: Literal["SciREX", "SciFact", "Crossref/Retraction Watch"]
    left: PacketSide
    right: PacketSide
    source_level_update_type: str
    source_level_update_notice: str
    packet_provenance: str
    packet_checksum: str

    @field_validator("task_id", "packet_provenance")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("packet identity/provenance fields must be non-empty")
        return normalized

    @field_validator("packet_checksum")
    @classmethod
    def _sha256_checksum(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("packet_checksum must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"

    @model_validator(mode="after")
    def _checksum_matches_content(self) -> AdjudicationPacket:
        canonical = self._canonical_dict_for_checksum()
        expected = sha256_json(canonical)
        if self.packet_checksum != expected:
            raise ValueError(
                f"packet_checksum mismatch: expected {expected}, got {self.packet_checksum}"
            )
        return self

    def _canonical_dict_for_checksum(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_dataset": self.source_dataset,
            "left": self.left.model_dump(),
            "right": self.right.model_dump(),
            "source_level_update_type": self.source_level_update_type,
            "source_level_update_notice": self.source_level_update_notice,
        }

    @classmethod
    def from_external_safe_record(
        cls,
        record: dict[str, Any],
        *,
        provenance: str,
    ) -> AdjudicationPacket:
        data = record.get("data", {})
        meta = record.get("meta", {})
        task_id = str(record.get("id", data.get("pair_id")))
        if not task_id:
            raise ValueError("external-safe record missing id/pair_id")

        left = PacketSide(
            claim_text=data["left_claim"],
            source_document_id=data["left_source"],
            evidence_locator=data["left_evidence_locator"],
            evidence_checksum=meta["left_evidence_checksum"],
        )
        right = PacketSide(
            claim_text=data["right_claim"],
            source_document_id=data["right_source"],
            evidence_locator=data["right_evidence_locator"],
            evidence_checksum=meta["right_evidence_checksum"],
        )
        source_dataset = meta["source_dataset"]
        if source_dataset not in {"SciREX", "SciFact", "Crossref/Retraction Watch"}:
            raise ValueError(f"unsupported source_dataset: {source_dataset}")

        canonical = {
            "task_id": task_id,
            "source_dataset": source_dataset,
            "left": left.model_dump(),
            "right": right.model_dump(),
            "source_level_update_type": data.get("source_level_update_type", "none"),
            "source_level_update_notice": data.get("source_level_update_notice", "none"),
        }
        return cls(
            task_id=task_id,
            source_dataset=source_dataset,  # type: ignore[arg-type]
            left=left,
            right=right,
            source_level_update_type=data.get("source_level_update_type", "none"),
            source_level_update_notice=data.get("source_level_update_notice", "none"),
            packet_provenance=provenance,
            packet_checksum=sha256_json(canonical),
        )


class BaseAnnotation(BaseModel):
    """Shared fields for every canonical annotation record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    semantic_relation: SemanticRelation
    scope_relation: ScopeRelation
    authority_relation: AuthorityRelation
    evidence_sufficiency: EvidenceSufficiency
    evidence_note: str
    uncertainty_note: str
    annotation_provenance: str
    annotator_id: str
    model_id: str
    prompt_checksum: str
    packet_checksum: str
    schema_version: Literal["phase1b-v3"] = SCHEMA_VERSION
    gold_status: Literal["not_gold"] = GOLD_STATUS

    @field_validator("task_id", "annotation_provenance", "annotator_id", "model_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("annotation identity/provenance fields must be non-empty")
        return normalized

    @field_validator("prompt_checksum", "packet_checksum")
    @classmethod
    def _sha256_checksum(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("checksum fields must be SHA-256 digests")
        return f"sha256:{digest.lower()}"


class JurorAnnotation(BaseAnnotation):
    """A single juror's four-axis assessment of one packet."""

    juror_run_id: str | None = None

    @field_validator("juror_run_id")
    @classmethod
    def _non_empty_run_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("juror_run_id must be non-empty if provided")
        return normalized


class CriticIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    axis: Literal["semantic", "scope", "authority", "evidence", "note"]
    issue_type: str
    evidence_locator_ref: str
    explanation: str

    @field_validator("issue_type", "evidence_locator_ref", "explanation")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("critic issue fields must be non-empty")
        return normalized


class CriticReview(BaseModel):
    """A critic review of two juror annotations for one packet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str
    critic_run_id: str
    juror_run_ids: tuple[str, str]
    packet_checksum: str
    issues: tuple[CriticIssue, ...]
    prompt_checksum: str
    annotation_provenance: str
    schema_version: Literal["phase1b-v3"] = SCHEMA_VERSION

    @field_validator("task_id", "critic_run_id", "annotation_provenance")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("critic identity/provenance fields must be non-empty")
        return normalized

    @field_validator("prompt_checksum", "packet_checksum")
    @classmethod
    def _sha256_checksum(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("checksum fields must be SHA-256 digests")
        return f"sha256:{digest.lower()}"

    @field_validator("annotation_provenance")
    @classmethod
    def _ai_critic_only(cls, value: str) -> str:
        if value != "ai_critic":
            raise ValueError("critic reviews must use annotation_provenance=ai_critic")
        return value

    @field_validator("issues")
    @classmethod
    def _no_operation_language_in_issues(
        cls, issues: tuple[CriticIssue, ...]
    ) -> tuple[CriticIssue, ...]:
        for issue in issues:
            combined = f"{issue.issue_type} {issue.explanation}"
            for label in FORBIDDEN_OPERATION_LABELS:
                if re.search(rf"\b{re.escape(label)}\b", combined, flags=re.IGNORECASE):
                    raise ValueError(f"forbidden operation label in critic issue: {label}")
        return issues


class AiAdjudicatedSilverLabel(BaseAnnotation):
    """Judge output: ai-adjudicated silver label, not gold."""

    juror_run_ids: tuple[str, str]
    critic_run_id: str
    adjudication_path: str
    evidence_locator_refs: tuple[str, ...]
    requires_higher_tier_ai_review: bool

    @field_validator("adjudication_path")
    @classmethod
    def _non_empty_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("adjudication_path must be non-empty")
        return normalized


def _detect_method(claim_text: str) -> str | None:
    """Extract a SciREX method token from a normalized claim string.

    SciREX claims rendered by the candidate pool have the shape:
    "<METHOD> <METRIC> for <TASK> = <VALUE> under {...}".
    This helper is used only by the fail-closed validator, never for
    label generation.
    """
    normalized = claim_text.strip()
    if not normalized:
        return None
    first_token = normalized.split()[0]
    if first_token and first_token.replace("_", "").replace("-", "").isalnum():
        return first_token
    return None


def _has_forbidden_operation_text(text: str) -> str | None:
    """Return first forbidden operation label found in text, else None."""
    for label in FORBIDDEN_OPERATION_LABELS:
        if re.search(rf"\b{re.escape(label)}\b", text, flags=re.IGNORECASE):
            return label
    return None


def validate_ai_adjudication_label(
    record: dict[str, Any],
    *,
    packet: AdjudicationPacket | None = None,
    allow_label_studio_nested: bool = False,
) -> dict[str, Any]:
    """Fail-closed validation of a canonical annotation record.

    Returns the normalized canonical record on success. Raises ValueError on
    any violation. The validator never auto-fixes labels or emits operations.
    """
    if allow_label_studio_nested:
        record = _normalize_label_studio_record(record)

    canonical_keys = {
        "task_id",
        "semantic_relation",
        "scope_relation",
        "authority_relation",
        "evidence_sufficiency",
        "evidence_note",
        "uncertainty_note",
        "annotation_provenance",
        "annotator_id",
        "model_id",
        "prompt_checksum",
        "packet_checksum",
        "schema_version",
        "gold_status",
    }
    judge_keys = {
        "juror_run_ids",
        "critic_run_id",
        "adjudication_path",
        "evidence_locator_refs",
        "requires_higher_tier_ai_review",
    }
    juror_keys = {"juror_run_id"}
    normalization_keys = {"original_checksum", "normalization_provenance", "transformation_version"}
    allowed_keys = canonical_keys | judge_keys | juror_keys | normalization_keys

    missing = canonical_keys - record.keys()
    if missing:
        raise ValueError(f"missing canonical fields: {sorted(missing)}")

    for key in FORBIDDEN_OPERATION_KEYS:
        if key in record:
            raise ValueError(f"forbidden operation key present: {key}")

    extra = set(record.keys()) - allowed_keys
    if extra:
        raise ValueError(f"unexpected extra fields: {sorted(extra)}")

    note_text = f"{record.get('evidence_note', '')} {record.get('uncertainty_note', '')}"
    forbidden_label = _has_forbidden_operation_text(note_text)
    if forbidden_label:
        raise ValueError(f"forbidden operation label in note: {forbidden_label}")

    for term in FORBIDDEN_PROVENANCE_TERMS:
        combined_text = f"{record.get('annotation_provenance', '')} {note_text}"
        if term.lower() in combined_text.lower():
            raise ValueError(
                f"annotation_provenance or note contains forbidden provenance term: {term}"
            )

    if packet is not None:
        if record["task_id"] != packet.task_id:
            raise ValueError("annotation task_id does not match supplied packet")
        if record["packet_checksum"] != packet.packet_checksum:
            raise ValueError("annotation packet_checksum does not match supplied packet")

    semantic = SemanticRelation(record["semantic_relation"])
    scope = ScopeRelation(record["scope_relation"])
    authority = AuthorityRelation(record["authority_relation"])
    evidence = EvidenceSufficiency(record["evidence_sufficiency"])

    if semantic == SemanticRelation.CONTRADICTORY and scope == ScopeRelation.SAME_SCOPE:
        if authority == AuthorityRelation.NOT_APPLICABLE:
            raise ValueError(
                "CONTRADICTORY + SAME_SCOPE requires an authority assessment; "
                "NOT_APPLICABLE is invalid"
            )

    source_dataset = packet.source_dataset if packet else None
    if source_dataset == "Crossref/Retraction Watch":
        if semantic != SemanticRelation.INSUFFICIENT_CONTEXT:
            raise ValueError(
                "Crossref/Retraction Watch without claim-level evidence must be "
                "INSUFFICIENT_CONTEXT"
            )
        if scope != ScopeRelation.UNKNOWN_SCOPE:
            raise ValueError(
                "Crossref/Retraction Watch without claim-level evidence must be "
                "UNKNOWN_SCOPE"
            )
        if authority != AuthorityRelation.UNRESOLVED:
            raise ValueError(
                "Crossref/Retraction Watch without claim-level evidence must be "
                "UNRESOLVED"
            )
        if evidence != EvidenceSufficiency.INSUFFICIENT:
            raise ValueError(
                "Crossref/Retraction Watch without claim-level evidence must be "
                "INSUFFICIENT"
            )

    if source_dataset == "SciREX" and scope == ScopeRelation.SAME_SCOPE:
        left_method = _detect_method(packet.left.claim_text) if packet else None
        right_method = _detect_method(packet.right.claim_text) if packet else None
        if left_method and right_method and left_method != right_method:
            raise ValueError(
                f"SciREX claims with different methods cannot be SAME_SCOPE: "
                f"{left_method} vs {right_method}"
            )

    if evidence in {EvidenceSufficiency.PARTIAL, EvidenceSufficiency.INSUFFICIENT}:
        if semantic == SemanticRelation.EQUIVALENT and scope == ScopeRelation.SAME_SCOPE:
            raise ValueError(
                "EQUIVALENT + SAME_SCOPE with partial/insufficient evidence is inconsistent"
            )

    if record.get("requires_higher_tier_ai_review") is True:
        if evidence == EvidenceSufficiency.SUFFICIENT:
            raise ValueError(
                "requires_higher_tier_ai_review cannot be true when evidence is SUFFICIENT"
            )

    # Re-validate through Pydantic depending on record shape and only allow
    # the machine provenance appropriate to that annotation role.
    if "juror_run_ids" in record and "critic_run_id" in record:
        if record["annotation_provenance"] != "ai_adjudicated_silver":
            raise ValueError(
                "judge labels must use annotation_provenance=ai_adjudicated_silver"
            )
        normalized = AiAdjudicatedSilverLabel.model_validate(record)
    else:
        if record["annotation_provenance"] != "ai_juror":
            raise ValueError("juror labels must use annotation_provenance=ai_juror")
        normalized = JurorAnnotation.model_validate(record)

    return normalized.model_dump(mode="json")


def _normalize_label_studio_record(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten a Label Studio nested result record without changing labels."""
    if "data" not in record or "annotations" not in record:
        raise ValueError("Label Studio record must contain 'data' and 'annotations'")

    data = record["data"]
    annotations = record["annotations"]
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Label Studio record must contain at least one annotation")

    result = annotations[0].get("result", [])
    if not isinstance(result, list):
        raise ValueError("Label Studio annotation result must be a list")

    flattened: dict[str, Any] = {
        "task_id": record.get("id", data.get("pair_id")),
    }
    for item in result:
        if not isinstance(item, dict):
            continue
        from_name = item.get("from_name")
        value = item.get("value", {})
        if from_name == "semantic_relation":
            choices = value.get("choices", [])
            flattened["semantic_relation"] = choices[0] if choices else None
        elif from_name == "scope_relation":
            choices = value.get("choices", [])
            flattened["scope_relation"] = choices[0] if choices else None
        elif from_name == "authority_relation":
            choices = value.get("choices", [])
            flattened["authority_relation"] = choices[0] if choices else None
        elif from_name == "evidence_sufficiency":
            choices = value.get("choices", [])
            flattened["evidence_sufficiency"] = choices[0] if choices else None
        elif from_name == "evidence_note":
            flattened["evidence_note"] = value.get("text", [""])[0] if isinstance(value.get("text"), list) else value.get("text", "")
        elif from_name == "uncertainty_note":
            flattened["uncertainty_note"] = value.get("text", [""])[0] if isinstance(value.get("text"), list) else value.get("text", "")

    # Preserve provenance/checksum fields if present in meta or top-level.
    meta = record.get("meta", {})
    for key in [
        "annotation_provenance",
        "annotator_id",
        "model_id",
        "prompt_checksum",
        "packet_checksum",
        "schema_version",
        "gold_status",
    ]:
        if key not in flattened:
            flattened[key] = record.get(key, meta.get(key))

    if "schema_version" not in flattened or flattened["schema_version"] is None:
        flattened["schema_version"] = SCHEMA_VERSION
    if "gold_status" not in flattened or flattened["gold_status"] is None:
        flattened["gold_status"] = GOLD_STATUS

    return flattened


def canonical_json_checksum(record: dict[str, Any]) -> str:
    """Compute a deterministic checksum for a canonical record."""
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
