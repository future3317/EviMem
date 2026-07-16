"""Raw WBM source validation and oracle isolation.

The Materials Cloud release contains the *raw* five substitution rounds.  It
is intentionally not treated as the 256,963-record Matbench Discovery
benchmark: the latter additionally fixes ID alignment, removes pathological
records and formation-energy outliers, and applies MP2020 compatibility.

This module is a phase-one ingestion gate only.  It makes raw structures and
compositions observable while retaining raw total energies in a
single-use oracle vault.  No pool construction or acquisition policy is
implemented here.
"""

from __future__ import annotations

import bz2
import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

RAW_WBM_FILENAMES = tuple(f"step_{step}.json.bz2" for step in range(1, 6))
RAW_WBM_EXPECTED_ENTRY_COUNTS = (61_848, 52_800, 79_205, 40_328, 23_308)


def _canonical_checksum(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class WBMRawReleaseReport(BaseModel):
    """Validation result for the uncurated Materials Cloud WBM release."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root: Path
    entry_counts: tuple[int, ...]
    raw_entry_total: int = Field(ge=0)
    file_checksums: tuple[str, ...]

    @property
    def matches_official_raw_counts(self) -> bool:
        return self.entry_counts == RAW_WBM_EXPECTED_ENTRY_COUNTS


class WBMRawObservableRecord(BaseModel):
    """A raw WBM record visible to infrastructure code, never its energy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_record_locator: str
    composition_amounts: dict[str, float]
    chemical_system: tuple[str, ...]
    structure_checksum: str

    @field_validator("source_record_locator", "structure_checksum")
    @classmethod
    def _text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("raw WBM observable fields must be non-empty")
        return value.strip()

    @field_validator("composition_amounts")
    @classmethod
    def _composition(cls, values: dict[str, float]) -> dict[str, float]:
        normalized = {element.strip(): float(amount) for element, amount in values.items()}
        if not normalized or any(
            not element or not math.isfinite(amount) or amount <= 0
            for element, amount in normalized.items()
        ):
            raise ValueError("raw WBM composition requires positive element amounts")
        return dict(sorted(normalized.items()))

    @field_validator("chemical_system")
    @classmethod
    def _system(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted(element.strip() for element in values))
        if not normalized or any(not element for element in normalized):
            raise ValueError("raw WBM chemical system must be non-empty")
        return normalized


class WBMRawOracleOutcome(BaseModel):
    """One raw total-energy outcome revealed only after selecting its locator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_record_locator: str
    total_energy_ev: float

    @field_validator("total_energy_ev")
    @classmethod
    def _finite_energy(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("raw WBM total energy must be finite")
        return value


class WBMRawOracleVault:
    """Single-use raw energy oracle keyed by an observable source locator."""

    def __init__(self, records: Iterable[tuple[WBMRawObservableRecord, float]]) -> None:
        pairs = tuple(records)
        locators = [observable.source_record_locator for observable, _ in pairs]
        if len(set(locators)) != len(locators):
            raise ValueError("raw WBM oracle source locators must be unique")
        self._observables = {observable.source_record_locator: observable for observable, _ in pairs}
        self._energies = {observable.source_record_locator: float(energy) for observable, energy in pairs}
        self._revealed: list[str] = []

    def observable(self, source_record_locator: str) -> WBMRawObservableRecord:
        try:
            return self._observables[source_record_locator]
        except KeyError as exc:
            raise KeyError("raw WBM source locator is absent") from exc

    @property
    def revealed_source_record_locators(self) -> tuple[str, ...]:
        return tuple(self._revealed)

    def reveal(self, source_record_locator: str) -> WBMRawOracleOutcome:
        if source_record_locator in self._revealed:
            raise ValueError("raw WBM oracle outcome has already been revealed")
        if source_record_locator not in self._energies:
            raise KeyError("raw WBM source locator is absent")
        self._revealed.append(source_record_locator)
        return WBMRawOracleOutcome(
            source_record_locator=source_record_locator,
            total_energy_ev=self._energies[source_record_locator],
        )


def validate_raw_wbm_release(
    root: Path,
    *,
    expected_counts: tuple[int, ...] = RAW_WBM_EXPECTED_ENTRY_COUNTS,
) -> WBMRawReleaseReport:
    """Decode every raw step file and fail on a missing file or count mismatch."""

    if len(expected_counts) != len(RAW_WBM_FILENAMES):
        raise ValueError("expected WBM raw counts must cover all five source steps")
    counts: list[int] = []
    checksums: list[str] = []
    for filename, expected_count in zip(RAW_WBM_FILENAMES, expected_counts, strict=True):
        path = root / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing raw WBM file: {path}")
        compressed = path.read_bytes()
        try:
            payload = json.loads(bz2.decompress(compressed))
            entries = payload["entries"]
        except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid raw WBM step file: {path}") from exc
        if not isinstance(entries, list):
            raise ValueError(f"raw WBM entries must be a list: {path}")
        count = len(entries)
        if count != expected_count:
            raise ValueError(f"raw WBM count mismatch for {filename}: {count} != {expected_count}")
        counts.append(count)
        checksums.append("sha256:" + hashlib.sha256(compressed).hexdigest())
    return WBMRawReleaseReport(
        root=root.resolve(),
        entry_counts=tuple(counts),
        raw_entry_total=sum(counts),
        file_checksums=tuple(checksums),
    )


def raw_wbm_records_from_payload(
    payload: dict[str, object],
    *,
    step: int,
) -> tuple[tuple[WBMRawObservableRecord, float], ...]:
    """Build observable/oracle pairs for one already-decoded raw WBM step.

    The locator is deliberately source-native rather than a Matbench ID.  The
    cleaned benchmark's ID alignment must be applied by its pinned compiler
    before any pool construction.
    """

    if step < 1 or step > 5:
        raise ValueError("raw WBM substitution step must be in [1, 5]")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError("raw WBM payload requires an entries list")
    records: list[tuple[WBMRawObservableRecord, float]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError("raw WBM entry must be an object")
        composition = entry.get("composition")
        structure = entry.get("structure")
        energy = entry.get("energy")
        if not isinstance(composition, dict) or not isinstance(structure, dict):
            raise ValueError("raw WBM entry requires composition and structure objects")
        if not isinstance(energy, int | float) or not math.isfinite(float(energy)):
            raise ValueError("raw WBM entry requires a finite numeric total energy")
        amounts = {str(element): float(amount) for element, amount in composition.items()}
        observable = WBMRawObservableRecord(
            source_record_locator=f"raw-wbm-step-{step}-index-{index}",
            composition_amounts=amounts,
            chemical_system=tuple(amounts),
            structure_checksum=_canonical_checksum(structure),
        )
        records.append((observable, float(energy)))
    return tuple(records)
