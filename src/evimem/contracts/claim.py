"""V2 canonical scientific claim model.

A ``ScientificClaim`` is the smallest publishable fact: a property/value/unit
statement about a material under conditions.  It deliberately does *not*
contain evidence refs or run IDs so that the same claim can be proposed by
multiple proposers and verified independently.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ScientificClaim(BaseModel):
    """A normalized, evidence-free scientific claim."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    property_key: str
    value_raw: str
    value_num: float | None = None
    unit_raw: str | None = None
    unit_canonical: str | None = None
    material_raw: str | None = None
    material_normalized: str | None = None
    composition_raw: str | None = None
    composition_normalized: str | None = None
    sample_id: str | None = None
    conditions_raw: str | None = None
    conditions: dict[str, Any] = Field(default_factory=dict)
