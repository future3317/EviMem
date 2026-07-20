"""Streaming, protocol-neutral identities for paired MatPES releases."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MATPES_SPLITS = ("train", "valid", "test")
MATPES_PBE_STEM = "MatPES-PBE-2025.2"
MATPES_R2SCAN_STEM = "MatPES-R2SCAN-2025.2"


@dataclass(frozen=True, slots=True)
class MatPESCompactConfiguration:
    """Fields needed to prove a same-configuration cross-protocol pair."""

    split: str
    nsites: int
    chemsys: str
    composition_key: str
    exact_geometry_sha256: str
    rounded_geometry_sha256: str
    raw_structure_sha256: str
    energy_ev_per_atom: float
    formation_energy_ev_per_atom: float | None
    original_mp_id: str | None


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _composition_key(composition: Any) -> str:
    if not isinstance(composition, dict) or not composition:
        raise ValueError("MatPES composition must be a nonempty mapping")
    canonical = tuple(
        (str(element), round(float(amount), 12))
        for element, amount in sorted(composition.items())
    )
    if any(not math.isfinite(amount) or amount <= 0 for _, amount in canonical):
        raise ValueError("MatPES composition contains an invalid amount")
    return hashlib.sha256(canonical_json_bytes(canonical)).hexdigest()


def _site_species(site: dict[str, Any]) -> tuple[tuple[str, float], ...]:
    species = site.get("species")
    if not isinstance(species, list) or not species:
        raise ValueError("MatPES structure site is missing species")
    return tuple(
        sorted(
            (str(item["element"]), float(item.get("occu", 1.0)))
            for item in species
        )
    )


def _wrapped_fractional(
    values: Iterable[Any], *, decimals: int | None
) -> tuple[float, ...]:
    result = []
    for raw in values:
        value = float(raw) % 1.0
        if math.isclose(value, 1.0, abs_tol=1e-12):
            value = 0.0
        result.append(round(value, decimals) if decimals is not None else value)
    if len(result) != 3 or not all(math.isfinite(value) for value in result):
        raise ValueError("MatPES fractional coordinate must contain three finite values")
    return tuple(result)


def canonical_geometry_payload(
    structure: Any,
    *,
    decimals: int | None,
) -> dict[str, Any]:
    """Canonicalize geometry while excluding protocol-dependent site properties."""

    if not isinstance(structure, dict):
        raise ValueError("MatPES structure must be a mapping")
    lattice = structure.get("lattice", {}).get("matrix")
    sites = structure.get("sites")
    if not isinstance(lattice, list) or len(lattice) != 3 or not isinstance(sites, list):
        raise ValueError("MatPES structure has no valid lattice/sites")
    canonical_lattice = tuple(
        tuple(
            round(float(value), decimals) if decimals is not None else float(value)
            for value in row
        )
        for row in lattice
    )
    if any(len(row) != 3 for row in canonical_lattice):
        raise ValueError("MatPES lattice must be 3x3")
    canonical_sites = sorted(
        (
            _site_species(site),
            _wrapped_fractional(site.get("abc", ()), decimals=decimals),
        )
        for site in sites
    )
    return {"lattice": canonical_lattice, "sites": canonical_sites}


def compact_matpes_configuration(
    row: dict[str, Any], *, split: str
) -> MatPESCompactConfiguration:
    """Validate one MatPES row and reduce it to pairing-relevant identity."""

    identifier = str(row.get("matpes_id", "")).strip()
    if not identifier:
        raise ValueError("MatPES row is missing matpes_id")
    nsites = int(row["nsites"])
    energy = float(row["energy"])
    if nsites <= 0 or not math.isfinite(energy):
        raise ValueError(f"MatPES row {identifier} has invalid energy or nsites")
    structure = row["structure"]
    provenance = row.get("provenance") or {}
    original_mp_id = provenance.get("original_mp_id")
    formation_raw = row.get("formation_energy_per_atom")
    formation = float(formation_raw) if formation_raw is not None else None
    if formation is not None and not math.isfinite(formation):
        raise ValueError(f"MatPES row {identifier} has invalid formation energy")
    return MatPESCompactConfiguration(
        split=split,
        nsites=nsites,
        chemsys=str(row["chemsys"]),
        composition_key=_composition_key(row["composition"]),
        exact_geometry_sha256=_sha256_json(
            canonical_geometry_payload(structure, decimals=None)
        ),
        rounded_geometry_sha256=_sha256_json(
            canonical_geometry_payload(structure, decimals=10)
        ),
        raw_structure_sha256=_sha256_json(structure),
        energy_ev_per_atom=energy / nsites,
        formation_energy_ev_per_atom=formation,
        original_mp_id=(str(original_mp_id) if original_mp_id is not None else None),
    )


def iter_matpes_jsonl(path: Path):
    """Yield validated JSON objects without retaining the release in memory."""

    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid MatPES JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"MatPES row is not an object at {path}:{line_number}")
            yield row
