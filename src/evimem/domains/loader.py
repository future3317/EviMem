"""Package-resource loader for standalone DomainPack configurations."""

from __future__ import annotations

import hashlib
import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from .models import DomainPack


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_domain_pack(domain_id: str, *, path: str | Path | None = None) -> DomainPack:
    """Load and validate one DomainPack from package data or an explicit file."""

    if path is None:
        resource = files("evimem.domains").joinpath("configs", f"{domain_id}.json")
        if not resource.is_file():
            raise ValueError(
                f"unknown domain {domain_id!r}; available domains: {', '.join(list_domain_packs())}"
            )
        text = resource.read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    payload = json.loads(text)
    if payload.get("domain_id") != domain_id:
        raise ValueError("DomainPack filename/request identity does not match domain_id")
    return DomainPack.model_validate({**payload, "content_hash": _canonical_hash(payload)})


def list_domain_packs() -> tuple[str, ...]:
    config_dir = files("evimem.domains").joinpath("configs")
    return tuple(sorted(item.name.removesuffix(".json") for item in config_dir.iterdir() if item.name.endswith(".json")))

