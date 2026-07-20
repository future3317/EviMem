"""Native materials-memory records and dynamic convex-hull references."""

from __future__ import annotations

import math
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .identity import MaterialIdentity, StructureArtifactIdentity, StructureStage
from .protocols import ProtocolCertificate


class SourceProvenance(BaseModel):
    """Database identity, kept distinct from the scientific protocol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_name: str
    source_version: str
    record_locator: str
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("source_name", "source_version", "record_locator")
    @classmethod
    def _require_identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("source provenance fields must be non-empty")
        return value


class HullSnapshot(BaseModel):
    """A versioned phase-diagram reference for a single chemical system.

    ``reference_hull_energy_ev_per_atom`` is the relevant composition-specific
    hull interpolation, not a universal elemental reference.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    chemical_system: tuple[str, ...]
    reference_hull_energy_ev_per_atom: float
    phase_set_checksum: str
    known_through: datetime
    built_at: datetime
    source_version: str

    @field_validator("snapshot_id", "source_version")
    @classmethod
    def _require_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("hull snapshot identity must be non-empty")
        return value.strip()

    @field_validator("phase_set_checksum")
    @classmethod
    def _require_checksum(cls, value: str) -> str:
        digest = value.removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
            raise ValueError("phase set checksum must be a SHA-256 digest")
        return f"sha256:{digest.lower()}"

    @field_validator("chemical_system")
    @classmethod
    def _chemical_system(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values or any(not element.strip() for element in values):
            raise ValueError("hull snapshot requires a non-empty chemical system")
        return tuple(sorted(values))

    @field_validator("reference_hull_energy_ev_per_atom")
    @classmethod
    def _finite_reference(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("hull reference energy must be finite")
        return value

    @model_validator(mode="after")
    def _causal_snapshot(self) -> HullSnapshot:
        if self.built_at < self.known_through:
            raise ValueError("hull snapshot cannot be built before its phase set is known")
        return self


class MaterialQuery(BaseModel):
    """A candidate queried before its oracle result can enter memory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    structure_hash: str
    structure_identity: StructureArtifactIdentity
    identity: MaterialIdentity
    composition: str
    embedding: tuple[float, ...]
    protocol: ProtocolCertificate
    hull_snapshot: HullSnapshot
    base_predicted_formation_energy_ev_per_atom: float
    stability_threshold_ev_per_atom: float = Field(default=0.0, ge=0)
    oracle_cost: float = Field(default=1.0, gt=0)
    as_of: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("query_id", "structure_hash", "composition")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query identity fields must be non-empty")
        return value.strip()

    @field_validator("embedding")
    @classmethod
    def _embedding(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2 or any(not math.isfinite(item) for item in value):
            raise ValueError("embedding must contain at least two finite values")
        return value

    @field_validator("base_predicted_formation_energy_ev_per_atom")
    @classmethod
    def _finite_energy(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("predicted formation energy must be finite")
        return value

    @model_validator(mode="after")
    def _query_snapshot_is_causal(self) -> MaterialQuery:
        if (
            self.structure_identity.query_id != self.query_id
            or self.structure_identity.structure_hash != self.structure_hash
        ):
            raise ValueError("query and structure artifact identities disagree")
        if (
            self.structure_identity.stage is not StructureStage.INITIAL
            or not self.structure_identity.causal_available_before_query
        ):
            raise ValueError("material query requires a pre-query initial structure")
        if self.hull_snapshot.built_at > self.as_of:
            raise ValueError("query cannot use a hull snapshot built in its future")
        return self

    def hull_distance(self, formation_energy_ev_per_atom: float) -> float:
        return formation_energy_ev_per_atom - self.hull_snapshot.reference_hull_energy_ev_per_atom

    @property
    def base_hull_distance_ev_per_atom(self) -> float:
        return self.hull_distance(self.base_predicted_formation_energy_ev_per_atom)


class MaterialMemoryCard(BaseModel):
    """One oracle-observed material outcome, retaining raw and derived values.

    The raw protocol-aligned formation energy is authoritative.  Hull distance
    is retained only with its snapshot and must be recomputed for a newer hull.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    card_id: str
    material_id: str
    structure_hash: str
    structure_identity: StructureArtifactIdentity
    identity: MaterialIdentity
    composition: str
    embedding: tuple[float, ...]
    protocol: ProtocolCertificate
    provenance: SourceProvenance
    formation_energy_ev_per_atom: float
    base_predicted_formation_energy_ev_per_atom: float
    oracle_residual_ev_per_atom: float
    hull_snapshot: HullSnapshot
    recorded_hull_distance_ev_per_atom: float | None = None
    quality_weight: float = Field(default=1.0, gt=0, le=1)
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("card_id", "material_id", "structure_hash", "composition")
    @classmethod
    def _require_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("card identity fields must be non-empty")
        return value.strip()

    @field_validator("embedding")
    @classmethod
    def _embedding(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if len(value) < 2 or any(not math.isfinite(item) for item in value):
            raise ValueError("embedding must contain at least two finite values")
        return value

    @field_validator(
        "formation_energy_ev_per_atom",
        "base_predicted_formation_energy_ev_per_atom",
        "oracle_residual_ev_per_atom",
        "recorded_hull_distance_ev_per_atom",
    )
    @classmethod
    def _finite_energy(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("energy values must be finite")
        return value

    @model_validator(mode="after")
    def _validate_derived_values(self) -> MaterialMemoryCard:
        if (
            self.structure_identity.query_id != self.material_id
            or self.structure_identity.structure_hash != self.structure_hash
            or self.structure_identity.stage is not StructureStage.INITIAL
        ):
            raise ValueError("memory card and initial structure identities disagree")
        residual = (
            self.formation_energy_ev_per_atom - self.base_predicted_formation_energy_ev_per_atom
        )
        if not math.isclose(self.oracle_residual_ev_per_atom, residual, abs_tol=1e-9):
            raise ValueError("oracle residual must equal oracle minus frozen base prediction")
        if self.recorded_hull_distance_ev_per_atom is not None and not math.isclose(
            self.recorded_hull_distance_ev_per_atom,
            self.hull_distance(self.hull_snapshot),
            abs_tol=1e-9,
        ):
            raise ValueError("recorded hull distance must match its hull snapshot")
        if self.hull_snapshot.built_at > self.observed_at:
            raise ValueError("card cannot cite a hull snapshot built after observation")
        return self

    def hull_distance(self, snapshot: HullSnapshot | None = None) -> float:
        current = snapshot or self.hull_snapshot
        if current.chemical_system != self.hull_snapshot.chemical_system:
            raise ValueError("cannot evaluate a card against a different chemical system hull")
        return self.formation_energy_ev_per_atom - current.reference_hull_energy_ev_per_atom
