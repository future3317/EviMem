"""Identity levels used to prevent cross-database materials leakage."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, field_validator


class MaterialIdentity(BaseModel):
    """Calculation, canonical-structure, and chemical-family identities.

    A structure hash alone is not a sufficient split key: equivalent relaxed
    structures and database mirrors can carry different hashes.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    exact_calculation_id: str
    canonical_structure_id: str
    composition_family: str
    prototype_family: str | None = None

    @field_validator("exact_calculation_id", "canonical_structure_id", "composition_family")
    @classmethod
    def _require_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("material identity fields must be non-empty")
        return value.strip()

    @field_validator("prototype_family")
    @classmethod
    def _normalize_optional(cls, value: str | None) -> str | None:
        return value.strip() if value and value.strip() else None


class CanonicalGroupSplit(BaseModel):
    """An explicit partition for transport fitting, memory, and evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    calibration_groups: tuple[str, ...]
    memory_groups: tuple[str, ...]
    evaluation_groups: tuple[str, ...]

    @field_validator("calibration_groups", "memory_groups", "evaluation_groups")
    @classmethod
    def _unique_groups(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in values)
        if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
            raise ValueError("split groups must be non-empty and unique")
        return normalized

    def model_post_init(self, __context: object) -> None:
        partitions = {
            "calibration": set(self.calibration_groups),
            "memory": set(self.memory_groups),
            "evaluation": set(self.evaluation_groups),
        }
        overlaps = [
            f"{left}/{right}:{sorted(partitions[left] & partitions[right])}"
            for left, right in (("calibration", "memory"), ("calibration", "evaluation"), ("memory", "evaluation"))
            if partitions[left] & partitions[right]
        ]
        if overlaps:
            raise ValueError("canonical structure groups cross partitions: " + "; ".join(overlaps))

    @staticmethod
    def checksum(groups: Iterable[str]) -> str:
        payload = "\n".join(sorted(set(groups)))
        return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def assert_partition(self, identity: MaterialIdentity, partition: str) -> None:
        expected = {
            "calibration": self.calibration_groups,
            "memory": self.memory_groups,
            "evaluation": self.evaluation_groups,
        }.get(partition)
        if expected is None:
            raise ValueError("unknown material split partition")
        if identity.canonical_structure_id not in expected:
            raise ValueError(f"canonical structure is not assigned to {partition}")
