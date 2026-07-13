"""Stable, deterministic ID generation for V2 canonical contracts.

All IDs are SHA-256 digests over a canonical JSON serialization (sorted keys,
no whitespace) of the input fields.  Where appropriate the hex digest is
truncated to 32 characters for compactness while retaining collision
resistance suitable for dataset-scale work.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .claim import ScientificClaim
from .evidence import EvidenceRef

_HASH_BITS = 256
_TRUNCATED_LEN = 32


def _canonical_json(payload: Any) -> str:
    """Return a compact, key-sorted JSON serialization."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_hex(data: str) -> str:
    """Return the full SHA-256 hex digest of ``data``."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _truncated_sha256_hex(data: str) -> str:
    """Return a 32-character SHA-256 hex digest of ``data``."""
    return _sha256_hex(data)[:_TRUNCATED_LEN]


def make_candidate_id(run_id: str, candidate_index: int, payload_hash: str) -> str:
    """Return a stable candidate ID for a proposer run."""
    payload = {
        "run_id": run_id,
        "candidate_index": candidate_index,
        "payload_hash": payload_hash,
    }
    return _truncated_sha256_hex(_canonical_json(payload))


def make_candidate_fingerprint(claim: ScientificClaim) -> str:
    """Return a stable fingerprint of the *claim content*.

    The fingerprint is independent of evidence refs and run IDs so that the
    same logical claim proposed by different agents hashes identically.
    """
    payload = {
        "property_key": claim.property_key,
        "value_raw": claim.value_raw,
        "value_num": claim.value_num,
        "unit_raw": claim.unit_raw,
        "unit_canonical": claim.unit_canonical,
        "material_raw": claim.material_raw,
        "material_normalized": claim.material_normalized,
        "composition_raw": claim.composition_raw,
        "composition_normalized": claim.composition_normalized,
        "sample_id": claim.sample_id,
        "conditions_raw": claim.conditions_raw,
        "conditions": sorted(
            ((str(k), str(v)) for k, v in claim.conditions.items()),
            key=lambda item: item[0],
        ),
    }
    return _truncated_sha256_hex(_canonical_json(payload))


def make_observation_key(doi: str, domain_name: str, claim: ScientificClaim) -> str:
    """Return a stable observation key for deduplication.

    The key is derived from DOI, domain, and claim identity fields only.
    It deliberately excludes evidence refs and run IDs so that equivalent
    claims from different extraction runs collide and can be reconciled.
    """
    payload = {
        "doi": doi,
        "domain_name": domain_name,
        "property_key": claim.property_key,
        "value_raw": claim.value_raw,
        "unit_raw": claim.unit_raw,
        "material_raw": claim.material_raw,
        "composition_raw": claim.composition_raw,
        "sample_id": claim.sample_id,
        "conditions": sorted(
            ((str(k), str(v)) for k, v in claim.conditions.items()),
            key=lambda item: item[0],
        ),
    }
    return _truncated_sha256_hex(_canonical_json(payload))


def make_observation_id(observation_key: str, first_published_run_id: str) -> str:
    """Return a stable observation ID from a key and first publication run."""
    payload = {
        "observation_key": observation_key,
        "first_published_run_id": first_published_run_id,
    }
    return _truncated_sha256_hex(_canonical_json(payload))


def make_certificate_id(
    candidate_id: str,
    resolved_evidence: list[EvidenceRef],
    verifier_version: str,
) -> str:
    """Return a certificate ID bound to candidate, evidence, and verifier."""
    payload = {
        "candidate_id": candidate_id,
        "resolved_evidence": [
            ev.model_dump(mode="json", by_alias=True) for ev in resolved_evidence
        ],
        "verifier_version": verifier_version,
    }
    return _truncated_sha256_hex(_canonical_json(payload))


def make_publication_commit_id(
    run_id: str, doi: str, artifact_hash: str, policy_version: str
) -> str:
    """Return a deterministic publication commit ID."""
    payload = {
        "run_id": run_id,
        "doi": doi,
        "artifact_hash": artifact_hash,
        "policy_version": policy_version,
    }
    return _truncated_sha256_hex(_canonical_json(payload))
