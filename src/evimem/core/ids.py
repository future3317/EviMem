"""
Shared deterministic ID generation utility.

Provides a stable, hash-based identifier builder extracted from duplicate
implementations across the codebase.  Each call site controls its own
key composition and output length.

Design:
- Pure function: same inputs → same output across processes and platforms
- Uses SHA-256 truncated to caller-chosen length
- Separator between parts for collision resistance
- Optional namespace prefix for ID scoping
"""

from __future__ import annotations

import hashlib
from typing import Any


def deterministic_id(
    *parts: Any,
    length: int = 16,
    namespace: str | None = None,
    sep: str = "|",
) -> str:
    """Generate a deterministic SHA-256 hex digest from key parts.

    Args:
        *parts: Ordered key components (converted to str via ``str()``).
        length: Output hex digest length (1-64).  Default 16.
        namespace: Optional prefix namespace for scoping.
        sep: Separator between parts in the hash input.

    Returns:
        Hex digest string of the requested length.

    Examples:
        >>> deterministic_id("decision", "prop_1", "bundle_a", "10.1000/x", length=32)
        'a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6'
    """
    key_parts = [str(p) for p in parts]
    payload = sep.join(key_parts)
    if namespace:
        payload = f"{namespace}{sep}{payload}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
