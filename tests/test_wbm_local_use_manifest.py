from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "build_wbm_local_use_manifest.py")
)


def test_local_research_and_redistribution_are_separate(tmp_path: Path) -> None:
    registry = tmp_path / "datasets.yml"
    registry.write_text("WBM:\n  license: CC-BY-4.0\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"technical_gate_passed": True}), encoding="utf-8")
    manifest = MODULE["build_manifest"](registry=registry, artifact_audit=audit)
    assert manifest["local_research_gate_passed"] is True
    assert manifest["publication_redistribution_gate_passed"] is False
    assert all(
        item["redistribution_permitted_by_this_manifest"] is False
        for item in manifest["components"]
    )


def test_local_research_gate_rejects_unverified_artifacts(tmp_path: Path) -> None:
    registry = tmp_path / "datasets.yml"
    registry.write_text("WBM: {}\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"technical_gate_passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity gate"):
        MODULE["build_manifest"](registry=registry, artifact_audit=audit)
