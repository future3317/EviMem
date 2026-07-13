"""Separate append-only audit store for non-published certificates."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from evimem.contracts import VerificationCertificate

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rejected_publication_audit (
    certificate_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    candidate_id TEXT NOT NULL,
    final_decision TEXT NOT NULL,
    reason_codes TEXT NOT NULL,
    certificate_json TEXT NOT NULL,
    recorded_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class AuditRecordResult:
    recorded: bool
    idempotent: bool
    certificate_id: str


class RejectionAuditStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def record(self, certificate: VerificationCertificate) -> AuditRecordResult:
        if certificate.final_decision == "publish":
            raise ValueError("publish certificates belong in the publication transaction")
        payload = certificate.model_dump_json(exclude_none=False)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT certificate_json FROM rejected_publication_audit WHERE certificate_id = ?",
                (certificate.certificate_id,),
            ).fetchone()
            if existing is not None:
                if existing["certificate_json"] != payload:
                    raise ValueError("audit certificate_id collision")
                return AuditRecordResult(True, True, certificate.certificate_id)
            connection.execute(
                """
                INSERT INTO rejected_publication_audit(
                    certificate_id, run_id, candidate_id, final_decision,
                    reason_codes, certificate_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    certificate.certificate_id,
                    certificate.run_id,
                    certificate.candidate_id,
                    certificate.final_decision,
                    ",".join(certificate.exclusion_reasons),
                    payload,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return AuditRecordResult(True, False, certificate.certificate_id)

    def count(self) -> int:
        with self._connect() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM rejected_publication_audit").fetchone()
            return int(row["count"])

    def get(self, certificate_id: str) -> VerificationCertificate | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT certificate_json FROM rejected_publication_audit WHERE certificate_id = ?",
                (certificate_id,),
            ).fetchone()
        return None if row is None else VerificationCertificate.model_validate_json(row["certificate_json"])

