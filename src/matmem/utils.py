"""Shared, stateless utility helpers used across protocol and WBM modules.

These functions are deliberately small and have no domain dependencies so they
can be imported by any module without creating circular imports.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any


def _checksum(payload: object) -> str:
    """Return a deterministic SHA-256 checksum for any JSON-serializable object."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validated_composition(values: dict[str, float]) -> dict[str, float]:
    """Return a cleaned composition dict with finite non-negative fractions.

    The total is not normalized here so callers that need atom counts can keep
    them intact.
    """

    cleaned = {str(key).strip(): float(value) for key, value in values.items()}
    if (
        not cleaned
        or any(not key or not math.isfinite(value) or value < 0 for key, value in cleaned.items())
        or sum(cleaned.values()) <= 0
    ):
        raise ValueError("composition fractions must be finite and non-negative")
    return dict(sorted(cleaned.items()))


def _normalized_composition(values: dict[str, float]) -> dict[str, float]:
    """Return a normalized composition dict summing to one."""

    cleaned = _validated_composition(values)
    total = sum(cleaned.values())
    return dict(sorted((key, value / total) for key, value in cleaned.items()))


def _normalized_composition_key(composition: dict[str, float]) -> tuple[tuple[str, float], ...]:
    """Return a hashable, normalized key for a composition dict."""

    total = float(sum(composition.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("hull composition must have positive finite mass")
    return tuple(
        (element, round(float(amount) / total, 12))
        for element, amount in sorted(composition.items())
        if float(amount) > 0
    )


def _as_finite_float(value: Any) -> float:
    """Coerce a scalar to a finite float or raise a clear error."""

    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return result
