"""Leakage-safe WBM infrastructure gates.

This module deliberately contains no downloader and no benchmark policy matrix.
Dataset files and derived feature caches must live outside the repository and
enter through checksummed manifests. WBM oracle energies use a separate vault;
policy-facing records contain no energy or final-hull label.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .cards import HullSnapshot, MaterialQuery
from .identity import MaterialIdentity
from .protocols import ProtocolCertificate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _canonical_checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class DataLicenseDecision(BaseModel):
    """Explicit human-reviewed license decision for one external artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_name: str
    release_id: str
    source_url: str
    license_spdx: str
    license_url: str
    research_use_permitted: bool
    redistribution_permitted: bool
    attribution_required: bool
    reviewed_by: str
    reviewed_at: datetime

    @field_validator(
        "dataset_name",
        "release_id",
        "source_url",
        "license_spdx",
        "license_url",
        "reviewed_by",
    )
    @classmethod
    def _nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("license decision fields must be non-empty")
        return value.strip()


class ExternalDataArtifact(BaseModel):
    """Checksummed local artifact that must remain outside the Git repository."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["wbm", "materials_project", "prediction", "structure", "other"]
    path: Path
    expected_sha256: str
    license: DataLicenseDecision

    @field_validator("expected_sha256")
    @classmethod
    def _checksum(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError("artifact checksum must be SHA-256")
        return "sha256:" + digest.lower()


class DataAuditFinding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    code: str
    message: str


class DataLicenseAuditReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    artifact_count: int = Field(ge=0)
    findings: tuple[DataAuditFinding, ...]
    manifest_checksum: str


def audit_external_data_artifacts(
    artifacts: Iterable[ExternalDataArtifact],
    *,
    repository_root: Path,
) -> DataLicenseAuditReport:
    """Fail closed on missing license authority, local files, or checksums."""

    items = tuple(artifacts)
    findings: list[DataAuditFinding] = []
    root = repository_root.resolve()
    required_roles = {"wbm", "materials_project", "prediction", "structure"}
    present_roles = {item.role for item in items}
    for missing in sorted(required_roles - present_roles):
        findings.append(
            DataAuditFinding(
                role=missing,
                code="missing-required-artifact",
                message=f"required {missing} artifact is absent",
            )
        )
    for item in items:
        path = item.path.resolve()
        if path.is_relative_to(root):
            findings.append(
                DataAuditFinding(
                    role=item.role,
                    code="artifact-inside-repository",
                    message=f"dataset/cache must remain outside repository: {path}",
                )
            )
        if not item.license.research_use_permitted:
            findings.append(
                DataAuditFinding(
                    role=item.role,
                    code="research-use-not-approved",
                    message="explicit license review did not approve research use",
                )
            )
        if not path.is_file():
            findings.append(
                DataAuditFinding(
                    role=item.role,
                    code="artifact-missing",
                    message=f"artifact does not exist: {path}",
                )
            )
            continue
        actual = _sha256_file(path)
        if actual != item.expected_sha256:
            findings.append(
                DataAuditFinding(
                    role=item.role,
                    code="checksum-mismatch",
                    message=f"expected {item.expected_sha256}, observed {actual}",
                )
            )
    manifest_payload = [item.model_dump(mode="json") for item in items]
    return DataLicenseAuditReport(
        passed=not findings,
        artifact_count=len(items),
        findings=tuple(findings),
        manifest_checksum=_canonical_checksum(manifest_payload),
    )


class MPPhaseRecord(BaseModel):
    """One MP-only phase used to construct an initial causal hull."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase_id: str
    composition_fractions: dict[str, float]
    formation_energy_ev_per_atom: float
    source_release: str
    protocol: ProtocolCertificate
    known_at: datetime

    @field_validator("phase_id", "source_release")
    @classmethod
    def _phase_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("MP phase identity must be non-empty")
        return value.strip()

    @field_validator("composition_fractions")
    @classmethod
    def _composition(cls, values: dict[str, float]) -> dict[str, float]:
        normalized = {key.strip(): float(value) for key, value in values.items()}
        if not normalized or any(not key or value < 0 or not math.isfinite(value) for key, value in normalized.items()):
            raise ValueError("phase composition requires finite non-negative fractions")
        total = sum(normalized.values())
        if not math.isclose(total, 1.0, abs_tol=1e-9):
            raise ValueError("phase composition fractions must sum to one")
        return dict(sorted(normalized.items()))

    @field_validator("formation_energy_ev_per_atom")
    @classmethod
    def _finite_energy(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("MP phase energy must be finite")
        return value


class MPCausalHullBuilder:
    """Build composition-specific references using only a frozen MP phase set."""

    def build(
        self,
        target_composition: Mapping[str, float],
        phases: Iterable[MPPhaseRecord],
        *,
        built_at: datetime,
        source_release: str,
        protocol: ProtocolCertificate,
    ) -> HullSnapshot:
        try:
            from scipy.optimize import linprog
        except ImportError as exc:  # pragma: no cover - exercised only without optional runtime
            raise RuntimeError("MP causal hull construction requires scipy") from exc

        target = {key.strip(): float(value) for key, value in target_composition.items()}
        if not target or any(not key or value < 0 or not math.isfinite(value) for key, value in target.items()):
            raise ValueError("target composition requires finite non-negative fractions")
        total = sum(target.values())
        if total <= 0:
            raise ValueError("target composition must have positive mass")
        target = {key: value / total for key, value in target.items() if value > 0}
        system = tuple(sorted(target))
        eligible = tuple(
            sorted(
                (
                    phase
                    for phase in phases
                    if set(phase.composition_fractions) <= set(system)
                    and phase.source_release == source_release
                    and phase.protocol.scientific_fingerprint
                    == protocol.scientific_fingerprint
                ),
                key=lambda phase: phase.phase_id,
            )
        )
        if not eligible:
            raise ValueError(
                "no frozen MP phases support the target chemical system and protocol"
            )
        known_through = max(phase.known_at for phase in eligible)
        if built_at < known_through:
            raise ValueError("MP hull cannot be built before all frozen phases are known")
        matrix = [
            [phase.composition_fractions.get(element, 0.0) for phase in eligible]
            for element in system
        ]
        result = linprog(
            [phase.formation_energy_ev_per_atom for phase in eligible],
            A_eq=matrix,
            b_eq=[target[element] for element in system],
            bounds=(0.0, None),
            method="highs",
        )
        if not result.success or result.fun is None:
            raise ValueError("frozen MP phases cannot span the target composition")
        phase_payload = [phase.model_dump(mode="json") for phase in eligible]
        phase_checksum = _canonical_checksum(phase_payload)
        target_checksum = _canonical_checksum(target).removeprefix("sha256:")[:16]
        system_id = "-".join(system)
        return HullSnapshot(
            snapshot_id=f"mp:{source_release}:{system_id}:{target_checksum}",
            chemical_system=system,
            reference_hull_energy_ev_per_atom=float(result.fun),
            phase_set_checksum=phase_checksum,
            known_through=known_through,
            built_at=built_at,
            source_version=f"MaterialsProject:{source_release}",
        )


class SOAPCacheConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cutoff_angstrom: float = Field(default=5.0, gt=0)
    n_max: int = Field(default=8, ge=1)
    l_max: int = Field(default=6, ge=0)
    periodic: bool = True
    species: tuple[str, ...]

    @field_validator("species")
    @classmethod
    def _species(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(value.strip() for value in values))
        if not normalized or any(not value for value in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("SOAP species vocabulary must be non-empty and unique")
        return normalized


class WBMObservableRecord(BaseModel):
    """Policy-safe WBM record; oracle energy and labels are structurally absent."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    structure_hash: str
    identity: MaterialIdentity
    composition: str
    chemical_system: tuple[str, ...]
    protocol: ProtocolCertificate
    as_of: datetime

    @field_validator("query_id", "structure_hash", "composition")
    @classmethod
    def _observable_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("WBM observable identity must be non-empty")
        return value.strip()

    @field_validator("chemical_system")
    @classmethod
    def _system(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(value.strip() for value in values))
        if not normalized or any(not value for value in normalized):
            raise ValueError("WBM observable requires a chemical system")
        return normalized


class FrozenPredictionSOAPRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    structure_hash: str
    predicted_formation_energy_ev_per_atom: float
    soap_vector: tuple[float, ...]

    @field_validator("query_id", "structure_hash")
    @classmethod
    def _feature_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("feature record identity must be non-empty")
        return value.strip()

    @field_validator("predicted_formation_energy_ev_per_atom")
    @classmethod
    def _prediction(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("frozen prediction must be finite")
        return value

    @field_validator("soap_vector")
    @classmethod
    def _soap(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if len(values) < 2 or any(not math.isfinite(value) for value in values):
            raise ValueError("SOAP vector must contain finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError("SOAP vector must be normalized")
        return values


class FrozenPredictionSOAPCache(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    predictor_id: str
    predictor_artifact_sha256: str
    config: SOAPCacheConfig
    records: tuple[FrozenPredictionSOAPRecord, ...]
    cache_checksum: str

    @field_validator("predictor_id")
    @classmethod
    def _predictor_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("predictor identity must be non-empty")
        return value.strip()

    @field_validator("predictor_artifact_sha256", "cache_checksum")
    @classmethod
    def _cache_digest(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError("cache identities must be SHA-256")
        return "sha256:" + digest.lower()

    @model_validator(mode="after")
    def _consistent_cache(self) -> FrozenPredictionSOAPCache:
        keys = [(record.query_id, record.structure_hash) for record in self.records]
        if len(set(keys)) != len(keys):
            raise ValueError("feature cache keys must be unique")
        expected = self.compute_checksum(
            self.predictor_id,
            self.predictor_artifact_sha256,
            self.config,
            self.records,
        )
        if self.cache_checksum != expected:
            raise ValueError("feature cache checksum does not match its contents")
        return self

    @staticmethod
    def compute_checksum(
        predictor_id: str,
        predictor_artifact_sha256: str,
        config: SOAPCacheConfig,
        records: Iterable[FrozenPredictionSOAPRecord],
    ) -> str:
        payload = {
            "predictor_id": predictor_id,
            "predictor_artifact_sha256": predictor_artifact_sha256,
            "config": config.model_dump(mode="json"),
            "records": [
                record.model_dump(mode="json")
                for record in sorted(records, key=lambda item: (item.query_id, item.structure_hash))
            ],
        }
        return _canonical_checksum(payload)

    @classmethod
    def create(
        cls,
        *,
        predictor_id: str,
        predictor_artifact_sha256: str,
        config: SOAPCacheConfig,
        records: Iterable[FrozenPredictionSOAPRecord],
    ) -> FrozenPredictionSOAPCache:
        items = tuple(sorted(records, key=lambda item: (item.query_id, item.structure_hash)))
        checksum = cls.compute_checksum(
            predictor_id,
            predictor_artifact_sha256,
            config,
            items,
        )
        return cls(
            predictor_id=predictor_id,
            predictor_artifact_sha256=predictor_artifact_sha256,
            config=config,
            records=items,
            cache_checksum=checksum,
        )

    def query(
        self,
        observable: WBMObservableRecord,
        hull_snapshot: HullSnapshot,
        *,
        stability_threshold_ev_per_atom: float = 0.0,
    ) -> MaterialQuery:
        if observable.chemical_system != hull_snapshot.chemical_system:
            raise ValueError("observable and MP hull chemical systems differ")
        matches = [record for record in self.records if record.query_id == observable.query_id]
        if len(matches) != 1 or matches[0].structure_hash != observable.structure_hash:
            raise KeyError("frozen prediction/SOAP cache identity mismatch")
        record = matches[0]
        return MaterialQuery(
            query_id=observable.query_id,
            structure_hash=observable.structure_hash,
            identity=observable.identity,
            composition=observable.composition,
            embedding=record.soap_vector,
            protocol=observable.protocol,
            hull_snapshot=hull_snapshot,
            base_predicted_formation_energy_ev_per_atom=(
                record.predicted_formation_energy_ev_per_atom
            ),
            stability_threshold_ev_per_atom=stability_threshold_ev_per_atom,
            as_of=observable.as_of,
        )


class WBMOracleRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    structure_hash: str
    corrected_total_energy_ev: float
    corrected_formation_energy_ev_per_atom: float
    source_record_locator: str
    observed_at: datetime

    @field_validator("query_id", "structure_hash", "source_record_locator")
    @classmethod
    def _oracle_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("WBM oracle identity must be non-empty")
        return value.strip()

    @field_validator(
        "corrected_total_energy_ev",
        "corrected_formation_energy_ev_per_atom",
    )
    @classmethod
    def _oracle_energy(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("WBM oracle energies must be finite")
        return value
