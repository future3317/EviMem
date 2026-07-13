"""Canonical scientific claim model for the retained materials safety case.

A ``ScientificClaim`` is the smallest publishable fact: a property/value/unit
statement about a material under conditions.  It deliberately does *not*
contain evidence refs or run IDs so that the same claim can be proposed by
multiple proposers and verified independently.
"""

from __future__ import annotations

import hashlib
import json
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

    def memory_claim_payload(self) -> dict[str, Any]:
        """Canonical cross-domain projection certified for long-term memory."""

        return {
            "subject": self.material_normalized or self.material_raw or "unknown",
            "relation": self.property_key,
            "object": None,
            "value": self.value_num if self.value_num is not None else self.value_raw,
            "unit": self.unit_canonical or self.unit_raw,
            "condition": self.conditions,
            "qualifiers": {
                key: value
                for key, value in {
                    "composition": self.composition_normalized or self.composition_raw,
                    "sample_id": self.sample_id,
                    "conditions_raw": self.conditions_raw,
                }.items()
                if value is not None
            },
        }

    def memory_fingerprint(self) -> str:
        payload = json.dumps(
            self.memory_claim_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
