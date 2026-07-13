"""Canonical immutable evidence reference.

An evidence reference is meaningful only inside a specific immutable release.
The release, document, block and locator identities are therefore part of the
contract instead of being supplied by ambient runtime state.
"""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .locators import CaptionLocator, TableCellLocator, TextSpanLocator


class EvidenceRef(BaseModel):
    """A durable anchor from a claim to source evidence."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    release_id: str
    document_id: str
    block_id: str
    checksum: str
    quote: str | None = None
    locator: TextSpanLocator | TableCellLocator | CaptionLocator = Field(
        ..., discriminator="locator_type"
    )

    @field_validator("release_id", "document_id", "block_id", "checksum")
    @classmethod
    def _require_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("evidence identity fields must be non-empty")
        return value

    @field_validator("checksum")
    @classmethod
    def _validate_checksum(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in digest):
            raise ValueError("checksum must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"

    @model_validator(mode="after")
    def _validate_locator_block(self) -> EvidenceRef:
        locator_block = getattr(self.locator, "block_id", None)
        if locator_block is not None and locator_block != self.block_id:
            raise ValueError("locator block_id must match evidence block_id")
        return self


def evidence_ref_from_block(
    block: dict[str, object],
    *,
    release_id: str,
    document_id: str,
    locator: TextSpanLocator | TableCellLocator | CaptionLocator | None = None,
    quote: str | None = None,
) -> EvidenceRef:
    """Create a canonical ref directly from an immutable release block."""

    block_id = str(block.get("block_id", "") or "").strip()
    if not block_id:
        raise ValueError("block must have a stable block_id")
    text = str(block.get("text", block.get("content", "")) or "")
    selected_quote = text if quote is None else quote
    if locator is None:
        start = text.find(selected_quote) if selected_quote else 0
        start = max(0, start)
        locator = TextSpanLocator(
            block_id=block_id,
            start=start,
            end=start + len(selected_quote),
        )
    checksum = str(block.get("evidence_block_checksum", "") or "")
    if not checksum:
        payload = json.dumps(
            block,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        )
        checksum = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return EvidenceRef(
        release_id=release_id,
        document_id=document_id,
        block_id=block_id,
        checksum=checksum,
        quote=selected_quote or None,
        locator=locator,
    )
