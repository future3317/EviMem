"""Typed evidence locators for V2 canonical contracts.

Locators describe *where* a claim is anchored in source evidence.  They are
intentionally simple, serializable, and deterministic so that evidence refs can
be hashed into certificate and observation IDs.
"""

from __future__ import annotations

from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict


class TextSpanLocator(BaseModel):
    """A character span inside an evidence block."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    locator_type: Literal["text_span"] = "text_span"
    block_id: str
    start: int
    end: int


class TableCellLocator(BaseModel):
    """A concrete cell inside a parsed table."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    locator_type: Literal["table_cell"] = "table_cell"
    table_id: str
    row_index: int
    column_index: int
    row_label: str | None = None
    column_header: str | None = None
    raw_cell_text: str | None = None


class CaptionLocator(BaseModel):
    """A table or figure caption block."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    locator_type: Literal["caption"] = "caption"
    block_id: str
    caption_type: Literal["table", "figure"]
