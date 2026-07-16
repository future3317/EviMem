"""Freeze a conservative WBM local-research license decision outside Git.

The decision intentionally does not authorize redistribution.  It separates
the ability to run local, non-commercial research on official public artifacts
from the later author/legal sign-off needed to redistribute any source data or
derived candidate-level files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_manifest(*, registry: Path, artifact_audit: Path) -> dict[str, object]:
    audit = json.loads(artifact_audit.read_text(encoding="utf-8"))
    if audit.get("technical_gate_passed") is not True:
        raise ValueError("official artifact integrity gate has not passed")
    components = [
        {
            "component": "WBM structures, energies, and locally derived SOAP",
            "release": "Materials Cloud 2021.68 / cleaned Matbench Discovery WBM",
            "license": "CC-BY-4.0",
            "license_source": "Matbench Discovery frozen datasets.yml WBM record",
            "local_research_use_permitted": True,
            "redistribution_permitted_by_this_manifest": False,
            "attribution_required": True,
        },
        {
            "component": "Materials Project 2022.10.28 CSE and PPD",
            "release": "2023-02-07 registry artifacts",
            "license": "CC-BY-4.0",
            "license_source": "Matbench Discovery frozen datasets.yml MP record",
            "local_research_use_permitted": True,
            "redistribution_permitted_by_this_manifest": False,
            "attribution_required": True,
        },
        {
            "component": "official CHGNet 0.3.0 WBM predictions",
            "release": "Figshare file 66646268",
            "license": "local-research-only composite decision",
            "license_source": (
                "official Matbench Discovery registry artifact over CC-BY-4.0 WBM; "
                "CHGNet implementation/checkpoint BSD-3-Clause"
            ),
            "local_research_use_permitted": True,
            "redistribution_permitted_by_this_manifest": False,
            "attribution_required": True,
        },
    ]
    return {
        "schema_version": 1,
        "scope": "local_noncommercial_research_execution_only",
        "not_legal_advice": True,
        "decision_basis": (
            "project owner authorized method-adaptive local experimentation; "
            "official public artifacts remain outside Git"
        ),
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "registry_path": str(registry.resolve()),
        "registry_sha256": _sha256(registry),
        "official_artifact_audit_path": str(artifact_audit.resolve()),
        "official_artifact_audit_sha256": _sha256(artifact_audit),
        "components": components,
        "local_research_gate_passed": all(
            item["local_research_use_permitted"] for item in components
        ),
        "publication_redistribution_gate_passed": False,
        "publication_requirements": [
            "human author signs the final attribution and redistribution table",
            "raw WBM, MP, predictions, SOAP vectors, and outcomes remain outside Git",
            "publish scripts, IDs, checksums, and citations instead of source artifacts",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--artifact-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    for path in (args.registry, args.artifact_audit, args.output):
        if path.resolve().is_relative_to(repo_root):
            raise ValueError("license evidence and manifests must remain outside Git")
    manifest = build_manifest(registry=args.registry, artifact_audit=args.artifact_audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"local_research_gate_passed={manifest['local_research_gate_passed']}")
    print("publication_redistribution_gate_passed=False")
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
