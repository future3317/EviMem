"""Typed, versioned scientific-domain configuration."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OntologyTerm(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    canonical_name: str
    aliases: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    description: str = ""
    status: str = "known"


class PropertyDefinition(OntologyTerm):
    value_type: str = "float"
    expected_range: tuple[float, float] | None = None
    required_context: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered_range(self) -> PropertyDefinition:
        if self.expected_range is not None and self.expected_range[0] > self.expected_range[1]:
            raise ValueError("property expected_range must be ordered")
        return self


class FalsePositivePattern(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    pattern_id: str
    keywords: tuple[str, ...]
    reason_code: str
    risk_level: Literal["low", "medium", "high"]
    description: str = ""


class PublicationPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")

    blocked_materialization_properties: tuple[str, ...] = ()
    strict_materialization_allowed: bool = False
    requires_verified_strong_evidence: bool = True
    allow_structured_prompt_support: bool = False


class DomainPack(BaseModel):
    """Domain ontology plus the policy identity used by certificates."""

    model_config = ConfigDict(frozen=True, extra="allow")

    domain_id: str
    version: str
    display_name: str
    description: str = ""
    status: str
    properties: dict[str, PropertyDefinition]
    conditions: dict[str, OntologyTerm] = Field(default_factory=dict)
    processes: dict[str, OntologyTerm] = Field(default_factory=dict)
    false_positive_patterns: dict[str, FalsePositivePattern] = Field(default_factory=dict)
    material_families: dict[str, dict[str, Any]] = Field(default_factory=dict)
    publication_policy: PublicationPolicy = Field(default_factory=PublicationPolicy)
    content_hash: str = Field(exclude=True)

    @field_validator("domain_id", "version", "display_name", "status", "content_hash")
    @classmethod
    def _non_empty_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("DomainPack identity fields must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_property_keys(self) -> DomainPack:
        if not self.properties:
            raise ValueError("DomainPack must define at least one property")
        return self

    @property
    def pack_hash(self) -> str:
        return self.content_hash
