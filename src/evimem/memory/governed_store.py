"""Append-only SQLite store with certificate-governed memory admission."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evimem.core.contracts import (
    MemoryType,
    SlotStatus,
    VerificationCertificate,
    WarrantedMemoryItem,
)


class MemoryAdmissionError(ValueError):
    """Raised when an item cannot cross the long-term memory boundary."""


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason_codes: tuple[str, ...]
    idempotent: bool = False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS warranted_memory_items (
    memory_id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    property_key TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    evidence_release_id TEXT NOT NULL,
    certificate_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    admitted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_signature
ON warranted_memory_items(domain, property_key, memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_policy
ON warranted_memory_items(policy_version, policy_hash);

CREATE TABLE IF NOT EXISTS warranted_memory_supersessions (
    superseded_memory_id TEXT PRIMARY KEY,
    successor_memory_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(superseded_memory_id) REFERENCES warranted_memory_items(memory_id),
    FOREIGN KEY(successor_memory_id) REFERENCES warranted_memory_items(memory_id)
);
"""


def _canonical_payload(item: WarrantedMemoryItem) -> str:
    return item.model_dump_json(exclude_none=False)


def _payload_hash(payload: str) -> str:
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class GovernedMemoryStore:
    """Durable memory store, deliberately separate from publication storage."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def assess(
        item: WarrantedMemoryItem,
        certificate: VerificationCertificate,
    ) -> AdmissionDecision:
        reasons: list[str] = []
        if item.certificate_id != certificate.certificate_id:
            reasons.append("certificate_id_mismatch")
        if item.evidence_release_id != certificate.evidence_release_id:
            reasons.append("evidence_release_mismatch")
        if item.policy_version != certificate.domain_pack_version:
            reasons.append("policy_version_mismatch")
        if item.policy_hash != certificate.domain_pack_hash:
            reasons.append("policy_hash_mismatch")
        if {ref.model_dump_json() for ref in item.evidence_refs} - {
            ref.model_dump_json() for ref in certificate.resolved_evidence
        }:
            reasons.append("memory_evidence_not_certified")

        if item.memory_type == MemoryType.VERIFIED:
            if certificate.final_decision != "publish":
                reasons.append("verified_memory_requires_publish_decision")
            if certificate.support_tier != "verified_strong":
                reasons.append("verified_memory_requires_verified_strong")
            if certificate.constraint_result != "pass":
                reasons.append("verified_memory_requires_constraint_pass")
            if certificate.conflict_result not in {
                "pass", "distinct_context", "exact_duplicate"
            }:
                reasons.append("verified_memory_requires_conflict_clearance")
            if any(status != SlotStatus.VERIFIED for status in certificate.slot_verification.values()):
                reasons.append("verified_memory_has_unverified_slots")
            if item.decision.status != "published":
                reasons.append("verified_memory_decision_not_published")
        elif item.memory_type == MemoryType.REJECTED:
            if certificate.final_decision != "reject":
                reasons.append("rejected_memory_requires_reject_decision")
            if not certificate.exclusion_reasons:
                reasons.append("rejected_memory_requires_stable_reason")
            if item.decision.status != "rejected":
                reasons.append("rejected_memory_decision_mismatch")
        elif item.memory_type == MemoryType.CONFLICT:
            if certificate.conflict_result != "unresolved_conflict":
                reasons.append("conflict_memory_requires_unresolved_conflict")
            if certificate.final_decision not in {"review", "defer"}:
                reasons.append("conflict_memory_requires_review_or_defer")
        elif item.memory_type in {MemoryType.CORRECTION, MemoryType.POLICY}:
            if item.authority.source != "human_curator":
                reasons.append("human_authority_required")

        return AdmissionDecision(admitted=not reasons, reason_codes=tuple(reasons))

    def admit(
        self,
        item: WarrantedMemoryItem,
        certificate: VerificationCertificate,
    ) -> AdmissionDecision:
        """Admit an item iff its deterministic warrant is internally consistent."""

        decision = self.assess(item, certificate)
        if not decision.admitted:
            raise MemoryAdmissionError(",".join(decision.reason_codes))

        payload = _canonical_payload(item)
        digest = _payload_hash(payload)
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_hash FROM warranted_memory_items WHERE memory_id = ?",
                (item.memory_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != digest:
                    raise MemoryAdmissionError("memory_id_collision")
                return AdmissionDecision(admitted=True, reason_codes=(), idempotent=True)
            connection.execute(
                """
                INSERT INTO warranted_memory_items(
                    memory_id, memory_type, domain, property_key,
                    policy_version, policy_hash, evidence_release_id,
                    certificate_id, payload_json, payload_hash, admitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.memory_id,
                    item.memory_type.value,
                    item.claim_signature.domain,
                    item.claim_signature.property_key,
                    item.policy_version,
                    item.policy_hash,
                    item.evidence_release_id,
                    item.certificate_id,
                    payload,
                    digest,
                    now,
                ),
            )
        return decision

    def get(self, memory_id: str) -> WarrantedMemoryItem | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM warranted_memory_items WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            supersession = connection.execute(
                """
                SELECT successor_memory_id, reason
                FROM warranted_memory_supersessions
                WHERE superseded_memory_id = ?
                """,
                (memory_id,),
            ).fetchone()
        item = WarrantedMemoryItem.model_validate_json(row["payload_json"])
        if supersession is None:
            return item
        return item.model_copy(
            update={
                "status": "superseded",
                "decision": item.decision.model_copy(update={"status": "superseded"}),
                "contradicted_by": tuple(
                    dict.fromkeys((*item.contradicted_by, supersession["successor_memory_id"]))
                ),
                "supersession_reason": supersession["reason"],
            }
        )

    def query(
        self,
        *,
        domain: str,
        property_key: str | None = None,
        memory_types: Iterable[MemoryType] | None = None,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> list[WarrantedMemoryItem]:
        clauses = ["m.domain = ?"]
        params: list[object] = [domain]
        if property_key is not None:
            clauses.append("m.property_key = ?")
            params.append(property_key)
        selected_types = tuple(memory_types or ())
        if selected_types:
            marks = ",".join("?" for _ in selected_types)
            clauses.append(f"m.memory_type IN ({marks})")
            params.extend(item.value for item in selected_types)
        if not include_superseded:
            clauses.append("s.superseded_memory_id IS NULL")
        params.append(max(1, limit))
        sql = f"""
            SELECT m.memory_id
            FROM warranted_memory_items m
            LEFT JOIN warranted_memory_supersessions s
              ON s.superseded_memory_id = m.memory_id
            WHERE {' AND '.join(clauses)}
            ORDER BY m.admitted_at DESC, m.memory_id ASC
            LIMIT ?
        """
        with self._connect() as connection:
            ids = [row["memory_id"] for row in connection.execute(sql, params).fetchall()]
        return [item for memory_id in ids if (item := self.get(memory_id)) is not None]

    def register_supersession(
        self,
        *,
        superseded_memory_id: str,
        successor_memory_id: str,
        reason: str,
    ) -> bool:
        if superseded_memory_id == successor_memory_id:
            raise MemoryAdmissionError("memory cannot supersede itself")
        reason = reason.strip()
        if not reason:
            raise MemoryAdmissionError("supersession requires a reason")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT memory_id FROM warranted_memory_items WHERE memory_id IN (?, ?)",
                (superseded_memory_id, successor_memory_id),
            ).fetchall()
            if len(rows) != 2:
                raise MemoryAdmissionError("both memories must be admitted before supersession")
            existing = connection.execute(
                "SELECT successor_memory_id, reason FROM warranted_memory_supersessions WHERE superseded_memory_id = ?",
                (superseded_memory_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["successor_memory_id"] == successor_memory_id
                    and existing["reason"] == reason
                ):
                    return False
                raise MemoryAdmissionError("memory already has a different successor")
            connection.execute(
                """
                INSERT INTO warranted_memory_supersessions(
                    superseded_memory_id, successor_memory_id, reason, created_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    superseded_memory_id,
                    successor_memory_id,
                    reason,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return True

    def verify_integrity(self) -> dict[str, object]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT memory_id, payload_json, payload_hash FROM warranted_memory_items"
            ).fetchall()
        corrupt = [
            row["memory_id"]
            for row in rows
            if _payload_hash(row["payload_json"]) != row["payload_hash"]
        ]
        return {"ok": not corrupt, "item_count": len(rows), "corrupt_memory_ids": corrupt}
