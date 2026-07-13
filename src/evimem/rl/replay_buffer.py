"""Append-only trajectory buffer with integrity hashes."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from evimem.core.contracts import CurationTrajectory


class TrajectoryReplayBuffer:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS curation_trajectories (
                    trajectory_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    evidence_release_id TEXT NOT NULL,
                    domain_pack_version TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    recorded_at TEXT NOT NULL
                )
                """
            )

    def append(self, trajectory: CurationTrajectory) -> bool:
        payload = trajectory.model_dump_json(exclude_none=False)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT payload_hash FROM curation_trajectories WHERE trajectory_id = ?",
                (trajectory.trajectory_id,),
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise ValueError("trajectory_id collision")
                return False
            connection.execute(
                """
                INSERT INTO curation_trajectories VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trajectory.trajectory_id,
                    trajectory.run_id,
                    trajectory.candidate_id,
                    trajectory.evidence_release_id,
                    trajectory.domain_pack_version,
                    payload,
                    digest,
                    datetime.now(UTC).isoformat(),
                ),
            )
        return True

    def get(self, trajectory_id: str) -> CurationTrajectory | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT payload_json, payload_hash FROM curation_trajectories WHERE trajectory_id = ?",
                (trajectory_id,),
            ).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0].encode("utf-8")).hexdigest() != row[1]:
            raise ValueError("trajectory payload integrity check failed")
        return CurationTrajectory.model_validate_json(row[0])

    def list_ids(self, *, evidence_release_id: str | None = None) -> list[str]:
        sql = "SELECT trajectory_id FROM curation_trajectories"
        params: tuple[str, ...] = ()
        if evidence_release_id is not None:
            sql += " WHERE evidence_release_id = ?"
            params = (evidence_release_id,)
        sql += " ORDER BY recorded_at, trajectory_id"
        with sqlite3.connect(self.db_path) as connection:
            return [row[0] for row in connection.execute(sql, params).fetchall()]
