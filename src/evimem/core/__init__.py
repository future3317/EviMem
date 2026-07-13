"""Canonical EviMem contracts and immutable evidence releases."""

from __future__ import annotations

from .evidence_release_manager import (
    CURRENT_POINTER_NAME,
    RELEASE_SCHEMA_VERSION,
    EvidenceRelease,
    EvidenceReleaseManager,
)

__all__ = [
    "CURRENT_POINTER_NAME",
    "EvidenceRelease",
    "EvidenceReleaseManager",
    "RELEASE_SCHEMA_VERSION",
]
