"""Single atomic and idempotent publication write authority."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from evimem.contracts import (
    PublishedObservation,
    VerificationCertificate,
    make_observation_id,
    make_observation_key,
    make_publication_commit_id,
)

from .store import (
    PublicationStore,
    _CertificateRow,
    _CommitRow,
    _ObservationRow,
    _RunStateRow,
)


@dataclass(frozen=True)
class PublicationCommitResult:
    committed: bool
    idempotent: bool
    commit_id: str
    observation: PublishedObservation


class PublicationCommitError(ValueError):
    pass


class PublicationCommitService:
    """The only production component authorized to mutate publication tables."""

    def __init__(self, store: PublicationStore, *, policy_version: str):
        self.store = store
        self.policy_version = policy_version

    def commit(
        self,
        *,
        doi: str,
        domain_name: str,
        certificate: VerificationCertificate,
        idempotency_key: str | None = None,
    ) -> PublicationCommitResult:
        if certificate.final_decision != "publish":
            raise PublicationCommitError("only publish certificates can be committed")
        if not certificate.resolved_evidence:
            raise PublicationCommitError("publication requires resolved immutable evidence")
        observation_key = make_observation_key(doi, domain_name, certificate.normalized_claim)
        artifact = {
            "doi": doi,
            "domain_name": domain_name,
            "certificate": certificate.model_dump(mode="json"),
            "observation_key": observation_key,
            "policy_version": self.policy_version,
        }
        artifact_json = json.dumps(artifact, sort_keys=True, separators=(",", ":"))
        artifact_hash = hashlib.sha256(artifact_json.encode("utf-8")).hexdigest()
        commit_id = make_publication_commit_id(
            certificate.run_id,
            doi,
            artifact_hash,
            self.policy_version,
        )
        key = (idempotency_key or commit_id).strip()
        if not key:
            raise PublicationCommitError("idempotency_key must be non-empty")

        with self.store._session() as session, session.begin():
            prior = session.scalar(select(_CommitRow).where(_CommitRow.idempotency_key == key))
            if prior is not None:
                if prior.artifact_hash != artifact_hash:
                    raise PublicationCommitError("idempotency_key_collision")
                row = session.get(_ObservationRow, prior.observation_id)
                if row is None:
                    raise PublicationCommitError("committed observation is missing")
                return PublicationCommitResult(
                    committed=True,
                    idempotent=True,
                    commit_id=prior.commit_id,
                    observation=PublishedObservation.model_validate_json(row.payload_json),
                )

            cert_row = session.get(_CertificateRow, certificate.certificate_id)
            cert_payload = certificate.model_dump_json(exclude_none=False)
            if cert_row is None:
                session.add(
                    _CertificateRow(
                        certificate_id=certificate.certificate_id,
                        payload_json=cert_payload,
                    )
                )
            elif cert_row.payload_json != cert_payload:
                raise PublicationCommitError("certificate_id_collision")

            existing = session.scalar(
                select(_ObservationRow).where(_ObservationRow.observation_key == observation_key)
            )
            if existing is None:
                observation_id = make_observation_id(observation_key, certificate.run_id)
                observation = PublishedObservation(
                    observation_id=observation_id,
                    observation_key=observation_key,
                    doi=doi,
                    claim=certificate.normalized_claim,
                    evidence=certificate.resolved_evidence,
                    certificate_id=certificate.certificate_id,
                    first_published_run_id=certificate.run_id,
                    publication_policy_version=self.policy_version,
                )
                session.add(
                    _ObservationRow(
                        observation_id=observation_id,
                        observation_key=observation_key,
                        certificate_id=certificate.certificate_id,
                        payload_json=observation.model_dump_json(exclude_none=False),
                    )
                )
            else:
                observation = PublishedObservation.model_validate_json(existing.payload_json)

            session.add(
                _CommitRow(
                    commit_id=commit_id,
                    idempotency_key=key,
                    run_id=certificate.run_id,
                    observation_id=observation.observation_id,
                    certificate_id=certificate.certificate_id,
                    artifact_hash=artifact_hash,
                    committed_at=datetime.now(UTC).isoformat(),
                )
            )
            session.add(
                _RunStateRow(
                    run_key=f"{certificate.run_id}:{certificate.candidate_id}",
                    run_id=certificate.run_id,
                    candidate_id=certificate.candidate_id,
                    state="published",
                    commit_id=commit_id,
                )
            )
        return PublicationCommitResult(True, False, commit_id, observation)

