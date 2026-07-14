"""Contracts for an unlabeled SciMem-Update annotation candidate pool."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CandidateSide(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    claim_text: str
    source_document_id: str
    source_timestamp: str | None = None
    evidence_locator: str
    evidence_checksum: str

    @field_validator("claim_text", "source_document_id", "evidence_locator")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("candidate text and source identities must be non-empty")
        return normalized


class SourceLevelUpdate(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source: Literal["publisher", "retraction-watch"]
    update_type: str
    timestamp: str
    original_doi: str
    update_doi: str
    api_response_checksum: str
    status_scope: Literal["source_level_only"] = "source_level_only"
    claim_level_status: Literal[
        "awaiting_human_evidence_annotation"
    ] = "awaiting_human_evidence_annotation"


class UpdatePilotCandidate(BaseModel):
    """A pair for human labeling; no compiled operation is representable."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    source_dataset: Literal["SciREX", "SciFact", "Crossref/Retraction Watch"]
    split: Literal["train", "metadata"]
    left: CandidateSide
    right: CandidateSide
    sampling_stratum: str
    license_components: tuple[str, ...] = Field(min_length=1)
    source_level_update: SourceLevelUpdate | None = None
    annotation_status: Literal["unlabeled"] = "unlabeled"
    candidate_is_gold: Literal[False] = False

    @model_validator(mode="after")
    def _source_update_is_metadata_only(self) -> UpdatePilotCandidate:
        if self.source_dataset == "Crossref/Retraction Watch":
            if self.split != "metadata" or self.source_level_update is None:
                raise ValueError("Crossref candidates require source-level metadata")
        elif self.source_level_update is not None:
            raise ValueError("dataset candidates cannot carry source update metadata")
        return self

    def label_studio_task(self) -> dict[str, Any]:
        update = self.source_level_update
        data: dict[str, Any] = {
            "pair_id": self.pair_id,
            "left_claim": self.left.claim_text,
            "right_claim": self.right.claim_text,
            "left_source": self.left.source_document_id,
            "right_source": self.right.source_document_id,
            "left_evidence_locator": self.left.evidence_locator,
            "right_evidence_locator": self.right.evidence_locator,
            "source_level_update_type": update.update_type if update else "none",
            "source_level_update_notice": (
                "Document-level metadata only; never claim-level supersession authority."
                if update
                else "none"
            ),
        }
        return {
            "id": self.pair_id,
            "data": data,
            "meta": {
                "source_dataset": self.source_dataset,
                "sampling_stratum_not_gold": self.sampling_stratum,
                "candidate_is_gold": False,
                "license_components": list(self.license_components),
                "left_evidence_checksum": self.left.evidence_checksum,
                "right_evidence_checksum": self.right.evidence_checksum,
                "source_level_update": update.model_dump(mode="json") if update else None,
            },
        }


def _crossref_timestamp(update: Mapping[str, Any]) -> str:
    updated = update.get("updated")
    if isinstance(updated, Mapping) and updated.get("date-time"):
        return str(updated["date-time"])
    return "unknown"


def crossref_update_candidates(
    response: Mapping[str, Any],
    *,
    response_checksum: str,
    limit: int,
) -> list[UpdatePilotCandidate]:
    """Map Crossref update metadata to source-status pairs without update gold."""

    message = response.get("message")
    items = message.get("items", ()) if isinstance(message, Mapping) else ()
    output: list[UpdatePilotCandidate] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        updates = item.get("update-to") or ()
        if isinstance(updates, Mapping):
            updates = (updates,)
        title_values = item.get("title") or ()
        title = str(title_values[0] if title_values else "Untitled Crossref work").strip()
        update_doi = str(item.get("DOI") or "").lower()
        for update in updates:
            if not isinstance(update, Mapping):
                continue
            source = str(update.get("source") or "publisher").lower()
            if source not in {"publisher", "retraction-watch"}:
                continue
            original_doi = str(update.get("DOI") or "").lower()
            update_type = str(update.get("type") or update.get("label") or "update").lower()
            timestamp = _crossref_timestamp(update)
            source_update = SourceLevelUpdate(
                source=source,
                update_type=update_type,
                timestamp=timestamp,
                original_doi=original_doi,
                update_doi=update_doi,
                api_response_checksum=response_checksum,
            )
            identity = sha256_json(
                {
                    "original_doi": original_doi,
                    "update_doi": update_doi,
                    "type": update_type,
                    "source": source,
                    "timestamp": timestamp,
                }
            ).split(":", 1)[1][:16]
            left_text = f"Work metadata title: {title}"
            right_text = (
                f"Source-level {update_type} status recorded for DOI {original_doi}; "
                "claim-level effect is not yet annotated."
            )
            output.append(
                UpdatePilotCandidate(
                    pair_id=f"crossref:{identity}",
                    source_dataset="Crossref/Retraction Watch",
                    split="metadata",
                    left=CandidateSide(
                        claim_text=left_text,
                        source_document_id=f"doi:{original_doi}",
                        source_timestamp=None,
                        evidence_locator="Crossref REST item.update-to",
                        evidence_checksum=sha256_json(item),
                    ),
                    right=CandidateSide(
                        claim_text=right_text,
                        source_document_id=f"doi:{update_doi}",
                        source_timestamp=timestamp,
                        evidence_locator="Crossref REST item.update-to",
                        evidence_checksum=sha256_json(update),
                    ),
                    sampling_stratum=f"source_level_{update_type}",
                    license_components=("Crossref metadata", "Retraction Watch metadata"),
                    source_level_update=source_update,
                )
            )
            if len(output) >= limit:
                return output
    return output


def validate_candidate_pool(
    candidates: Iterable[UpdatePilotCandidate],
    *,
    forbidden_document_ids: set[str] | None = None,
) -> dict[str, int]:
    items = list(candidates)
    if not 300 <= len(items) <= 400:
        raise ValueError("pilot candidate pool must contain 300-400 pairs")
    ids = [item.pair_id for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("pilot candidate pair IDs must be unique")
    forbidden = forbidden_document_ids or set()
    leaked = [
        item.pair_id
        for item in items
        if item.left.source_document_id in forbidden
        or item.right.source_document_id in forbidden
    ]
    if leaked:
        raise ValueError(f"pilot candidate pool contains forbidden eval documents: {leaked[:5]}")
    invalid_split = [
        item.pair_id
        for item in items
        if item.source_dataset != "Crossref/Retraction Watch" and item.split != "train"
    ]
    if invalid_split:
        raise ValueError(f"non-training dataset candidates found: {invalid_split[:5]}")
    return {
        "candidate_count": len(items),
        "unique_pair_ids": len(set(ids)),
        "forbidden_document_overlap": len(leaked),
    }
