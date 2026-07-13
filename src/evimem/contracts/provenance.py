"""Provenance records for proposers and deterministic verifiers.

Provenance is split into two immutable structures: one describing *who*
proposed a candidate, and one describing *which* verifier certified it.
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class ProposerProvenance(BaseModel):
    """Identity of the agent/model that proposed a candidate."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    provider: str
    model: str
    extraction_schema_version: str
    prompt_hash: str
    extraction_timestamp: datetime


class VerifierProvenance(BaseModel):
    """Identity of the harness verifier that issued a certificate."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    verifier_version: str
    domain_pack_id: str
    domain_pack_hash: str
    evidence_release_id: str
    verification_timestamp: datetime
