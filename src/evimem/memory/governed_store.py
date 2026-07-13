"""Append-only storage with certificate- and action-governed memory admission."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evimem.contracts import (
    AdmissionAction,
    MemoryStatus,
    MemoryType,
    ScientificMemoryRecord,
    SlotStatus,
    UpdateOperation,
)


class MemoryAdmissionError(ValueError):
    """Raised when a record cannot cross the long-term memory boundary."""


@dataclass(frozen=True)
class AdmissionDecision:
    admitted: bool
    reason_codes: tuple[str, ...]
    idempotent: bool = False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scientific_memory_records (
    memory_id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    subject TEXT NOT NULL,
    relation TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    policy_hash TEXT NOT NULL,
    evidence_release_id TEXT NOT NULL,
    certificate_id TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    admitted_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_signature
ON scientific_memory_records(domain, subject, relation, memory_type);
CREATE INDEX IF NOT EXISTS idx_memory_time
ON scientific_memory_records(observed_at);

CREATE TABLE IF NOT EXISTS memory_relations (
    source_memory_id TEXT NOT NULL,
    target_memory_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(source_memory_id, target_memory_id, operation),
    FOREIGN KEY(source_memory_id) REFERENCES scientific_memory_records(memory_id),
    FOREIGN KEY(target_memory_id) REFERENCES scientific_memory_records(memory_id)
);
"""


def _canonical_payload(record: ScientificMemoryRecord) -> str:
    return record.model_dump_json(exclude_none=False)


def _payload_hash(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MemoryAdmissionGate:
    """Hard constraints that a learned admission prediction cannot override."""

    _EXPECTED_ACTION = {
        MemoryType.VERIFIED: AdmissionAction.WRITE_VERIFIED,
        MemoryType.REJECTED: AdmissionAction.WRITE_REJECTED,
        MemoryType.CONFLICT: AdmissionAction.WRITE_CONFLICT,
    }

    @classmethod
    def assess(
        cls,
        record: ScientificMemoryRecord,
        requested_action: AdmissionAction,
    ) -> AdmissionDecision:
        if requested_action in {AdmissionAction.EPHEMERAL_ONLY, AdmissionAction.IGNORE}:
            return AdmissionDecision(False, (requested_action.value.lower(),))

        reasons: list[str] = []
        certificate = record.certificate
        if requested_action != cls._EXPECTED_ACTION[record.memory_type]:
            reasons.append("admission_memory_type_mismatch")
        if not certificate.resolved_evidence:
            reasons.append("certificate_has_no_resolved_evidence")
        if certificate.support_tier == "unbound":
            reasons.append("unbound_certificate_not_admissible")
        if record.memory_type == MemoryType.VERIFIED:
            if certificate.final_decision != "publish":
                reasons.append("verified_memory_requires_publish_decision")
            if certificate.support_tier != "verified_strong":
                reasons.append("verified_memory_requires_verified_strong")
            if certificate.constraint_result != "pass":
                reasons.append("verified_memory_requires_constraint_pass")
            if certificate.conflict_result not in {"pass", "distinct_context", "exact_duplicate"}:
                reasons.append("verified_memory_requires_conflict_clearance")
            if any(status != SlotStatus.VERIFIED for status in certificate.slot_verification.values()):
                reasons.append("verified_memory_has_unverified_slots")
            if record.decision.status != "published":
                reasons.append("verified_memory_decision_not_published")
        elif record.memory_type == MemoryType.REJECTED:
            if certificate.final_decision != "reject":
                reasons.append("rejected_memory_requires_reject_decision")
            if not certificate.exclusion_reasons:
                reasons.append("rejected_memory_requires_stable_reason")
            if record.decision.status != "rejected":
                reasons.append("rejected_memory_decision_mismatch")
        elif record.memory_type == MemoryType.CONFLICT:
            if certificate.conflict_result != "unresolved_conflict":
                reasons.append("conflict_memory_requires_unresolved_conflict")
            if certificate.final_decision not in {"review", "defer"}:
                reasons.append("conflict_memory_requires_review_or_defer")
            if record.decision.status != "conflict":
                reasons.append("conflict_memory_decision_mismatch")
        return AdmissionDecision(not reasons, tuple(reasons))


class GovernedMemoryStore:
    """Durable memory store, physically separate from publication storage."""

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

    def admit(
        self,
        record: ScientificMemoryRecord,
        requested_action: AdmissionAction,
    ) -> AdmissionDecision:
        decision = MemoryAdmissionGate.assess(record, requested_action)
        if not decision.admitted:
            if requested_action in {AdmissionAction.EPHEMERAL_ONLY, AdmissionAction.IGNORE}:
                return decision
            raise MemoryAdmissionError(",".join(decision.reason_codes))

        payload = _canonical_payload(record)
        digest = _payload_hash(payload)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_hash FROM scientific_memory_records WHERE memory_id = ?",
                (record.memory_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_hash"] != digest:
                    raise MemoryAdmissionError("memory_id_collision")
                return AdmissionDecision(True, (), idempotent=True)
            connection.execute(
                """
                INSERT INTO scientific_memory_records(
                    memory_id, memory_type, domain, subject, relation, observed_at,
                    policy_version, policy_hash, evidence_release_id, certificate_id,
                    payload_json, payload_hash, admitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.memory_id,
                    record.memory_type.value,
                    record.claim_signature.domain,
                    record.claim.subject,
                    record.claim.relation,
                    record.observed_at.isoformat(),
                    record.policy_version,
                    record.policy_hash,
                    record.evidence_release_id,
                    record.certificate_id,
                    payload,
                    digest,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return decision

    def get(self, memory_id: str) -> ScientificMemoryRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM scientific_memory_records WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
            if row is None:
                return None
            supersession = connection.execute(
                """
                SELECT source_memory_id, reason, created_at
                FROM memory_relations
                WHERE target_memory_id = ? AND operation = ?
                ORDER BY created_at ASC, source_memory_id ASC
                """,
                (memory_id, UpdateOperation.SUPERSEDE.value),
            ).fetchall()
        record = ScientificMemoryRecord.model_validate_json(row["payload_json"])
        if not supersession:
            return record
        return record.model_copy(
            update={
                "status": MemoryStatus.SUPERSEDED,
                "decision": record.decision.model_copy(update={"status": "superseded"}),
                "valid_until": datetime.fromisoformat(supersession[0]["created_at"]),
                "superseded_by": tuple(item["source_memory_id"] for item in supersession),
                "supersession_reason": supersession[0]["reason"],
            }
        )

    def query(
        self,
        *,
        domain: str,
        relation: str | None = None,
        subject: str | None = None,
        memory_types: Iterable[MemoryType] | None = None,
        observed_before: datetime | None = None,
        include_superseded: bool = False,
        limit: int = 100,
    ) -> list[ScientificMemoryRecord]:
        clauses = ["m.domain = ?"]
        params: list[object] = [domain]
        if relation is not None:
            clauses.append("m.relation = ?")
            params.append(relation)
        if subject is not None:
            clauses.append("m.subject = ?")
            params.append(subject)
        selected_types = tuple(memory_types or ())
        if selected_types:
            marks = ",".join("?" for _ in selected_types)
            clauses.append(f"m.memory_type IN ({marks})")
            params.extend(item.value for item in selected_types)
        if observed_before is not None:
            clauses.append("m.observed_at <= ?")
            params.append(observed_before.isoformat())
        if not include_superseded:
            clauses.append("s.target_memory_id IS NULL")
        params.append(max(1, limit))
        sql = f"""
            SELECT DISTINCT m.memory_id
            FROM scientific_memory_records m
            LEFT JOIN memory_relations s
              ON s.target_memory_id = m.memory_id AND s.operation = 'SUPERSEDE'
            WHERE {' AND '.join(clauses)}
            ORDER BY m.observed_at DESC, m.memory_id ASC
            LIMIT ?
        """
        with self._connect() as connection:
            ids = [row["memory_id"] for row in connection.execute(sql, params).fetchall()]
        return [record for memory_id in ids if (record := self.get(memory_id)) is not None]

    def register_relation(
        self,
        *,
        source_memory_id: str,
        target_memory_id: str,
        operation: UpdateOperation,
        reason: str,
    ) -> bool:
        if operation in {UpdateOperation.ADD, UpdateOperation.IGNORE}:
            raise MemoryAdmissionError(f"{operation.value} is not a binary memory relation")
        if source_memory_id == target_memory_id:
            raise MemoryAdmissionError("memory relation cannot target itself")
        reason = reason.strip()
        if not reason:
            raise MemoryAdmissionError("memory relation requires a reason")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT memory_id FROM scientific_memory_records WHERE memory_id IN (?, ?)",
                (source_memory_id, target_memory_id),
            ).fetchall()
            if len(rows) != 2:
                raise MemoryAdmissionError("both memories must be admitted before linking")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO memory_relations(
                    source_memory_id, target_memory_id, operation, reason, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source_memory_id,
                    target_memory_id,
                    operation.value,
                    reason,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return cursor.rowcount == 1

    def verify_integrity(self) -> dict[str, object]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT memory_id, payload_json, payload_hash FROM scientific_memory_records"
            ).fetchall()
        corrupt = [
            row["memory_id"]
            for row in rows
            if _payload_hash(row["payload_json"]) != row["payload_hash"]
        ]
        return {"ok": not corrupt, "item_count": len(rows), "corrupt_memory_ids": corrupt}
