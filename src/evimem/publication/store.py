"""SQLAlchemy persistence models and read-only publication queries."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import String, Text, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from evimem.contracts import PublishedObservation, VerificationCertificate


class _Base(DeclarativeBase):
    pass


class _ObservationRow(_Base):
    __tablename__ = "published_observations"

    observation_id: Mapped[str] = mapped_column(String, primary_key=True)
    observation_key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    certificate_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class _CertificateRow(_Base):
    __tablename__ = "verification_certificates"

    certificate_id: Mapped[str] = mapped_column(String, primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class _CommitRow(_Base):
    __tablename__ = "publication_commits"

    commit_id: Mapped[str] = mapped_column(String, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    observation_id: Mapped[str] = mapped_column(String, nullable=False)
    certificate_id: Mapped[str] = mapped_column(String, nullable=False)
    artifact_hash: Mapped[str] = mapped_column(String, nullable=False)
    committed_at: Mapped[str] = mapped_column(String, nullable=False)


class _RunStateRow(_Base):
    __tablename__ = "publication_run_states"

    run_key: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    candidate_id: Mapped[str] = mapped_column(String, nullable=False)
    state: Mapped[str] = mapped_column(String, nullable=False)
    commit_id: Mapped[str] = mapped_column(String, nullable=False)


class PublicationStore:
    """Read interface; mutations are owned by :class:`PublicationCommitService`."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        _Base.metadata.create_all(self._engine)
        self._sessions = sessionmaker(self._engine, expire_on_commit=False)

    def _session(self) -> Session:
        return self._sessions()

    def count_observations(self) -> int:
        with self._session() as session:
            return int(session.scalar(select(func.count()).select_from(_ObservationRow)) or 0)

    def count_certificates(self) -> int:
        with self._session() as session:
            return int(session.scalar(select(func.count()).select_from(_CertificateRow)) or 0)

    def count_commits(self) -> int:
        with self._session() as session:
            return int(session.scalar(select(func.count()).select_from(_CommitRow)) or 0)

    def get_observation(self, observation_id: str) -> PublishedObservation | None:
        with self._session() as session:
            row = session.get(_ObservationRow, observation_id)
            return None if row is None else PublishedObservation.model_validate_json(row.payload_json)

    def get_certificate(self, certificate_id: str) -> VerificationCertificate | None:
        with self._session() as session:
            row = session.get(_CertificateRow, certificate_id)
            return None if row is None else VerificationCertificate.model_validate_json(row.payload_json)


__all__ = ["PublicationStore"]

