"""V2 published observation contract.

A ``PublishedObservation`` is a durable, citation-ready observation tuple. It
references the certificate that authorized publication and can be revised
without changing its stable ``observation_id``.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from .claim import ScientificClaim
from .evidence import EvidenceRef


class PublishedObservation(BaseModel):
    """A published, auditable scientific observation."""

    model_config = ConfigDict(frozen=True)
    schema_version: ClassVar[str] = "evimem.v1"

    observation_id: str
    observation_key: str
    doi: str
    claim: ScientificClaim
    evidence: list[EvidenceRef]
    certificate_id: str
    first_published_run_id: str
    current_revision: int = 1
    publication_policy_version: str
