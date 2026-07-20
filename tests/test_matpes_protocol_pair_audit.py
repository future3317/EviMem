from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audit_matpes_protocol_pairs import run


def _row(*, identifier: str, functional: str, energy: float, original_mp_id: str) -> dict:
    return {
        "matpes_id": identifier,
        "functional": functional,
        "nsites": 2,
        "chemsys": "Fe-O",
        "composition": {"Fe": 1.0, "O": 1.0},
        "energy": energy,
        "formation_energy_per_atom": None,
        "provenance": {"original_mp_id": original_mp_id},
        "structure": {
            "lattice": {"matrix": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]},
            "properties": {},
            "sites": [
                {
                    "species": [{"element": "Fe", "occu": 1}],
                    "abc": [0.0, 0.0, 0.0],
                    "properties": {"magmom": 1.0 if functional == "PBE" else 2.0},
                },
                {
                    "species": [{"element": "O", "occu": 1}],
                    "abc": [0.5, 0.5, 0.5],
                    "properties": {},
                },
            ],
        },
    }


def _write_release(root: Path, *, stem: str, functional: str, perturb: bool = False) -> None:
    root.mkdir(parents=True)
    for split, suffix in (("train", "a"), ("valid", "b"), ("test", "c")):
        row = _row(
            identifier=f"matpes-{suffix}",
            functional=functional,
            energy=-4.0 if functional == "PBE" else -4.2,
            original_mp_id=f"mp-{suffix}",
        )
        if perturb and split == "test":
            row["structure"]["sites"][1]["abc"][0] = 0.25
        path = root / f"{stem}-{split}.jsonl"
        raw = (json.dumps(row, sort_keys=True) + "\n").encode()
        path.write_bytes(raw)
        metadata = root / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        import hashlib

        metadata.write_text(f"fixture-revision\n{hashlib.sha256(raw).hexdigest()}\n")


def test_strict_pair_audit_accepts_same_geometry_and_ignores_site_properties(
    tmp_path: Path,
) -> None:
    pbe = tmp_path / "pbe"
    r2scan = tmp_path / "r2scan"
    _write_release(pbe, stem="MatPES-PBE-2025.2", functional="PBE")
    _write_release(r2scan, stem="MatPES-R2SCAN-2025.2", functional="r2SCAN")
    result = run(pbe_root=pbe, r2scan_root=r2scan, output=tmp_path / "audit.json")
    assert result["decision"]["release_one_to_one_split_parity_gate_pass"] is True
    assert result["decision"]["same_configuration_pair_gate_pass"] is True
    assert result["decision"]["same_configuration_protocol_task_supported"] is False
    assert result["pairing"]["same_configuration_pair_count"] == 3
    assert result["pairing"]["mismatch_counts"]["raw_structure"] == 3
    assert result["pairing"]["mismatch_counts"]["rounded_geometry_1e-10"] == 0
    assert result["decision"]["formation_energy_labels_available"] is False


def test_strict_pair_audit_fails_on_geometry_mismatch(tmp_path: Path) -> None:
    pbe = tmp_path / "pbe"
    r2scan = tmp_path / "r2scan"
    _write_release(pbe, stem="MatPES-PBE-2025.2", functional="PBE")
    _write_release(
        r2scan,
        stem="MatPES-R2SCAN-2025.2",
        functional="r2SCAN",
        perturb=True,
    )
    result = run(pbe_root=pbe, r2scan_root=r2scan, output=tmp_path / "audit.json")
    assert result["decision"]["release_one_to_one_split_parity_gate_pass"] is False
    assert result["decision"]["same_configuration_pair_gate_pass"] is False
    assert result["pairing"]["mismatch_counts"]["rounded_geometry_1e-10"] == 1


def test_audit_refuses_to_overwrite(tmp_path: Path) -> None:
    pbe = tmp_path / "pbe"
    r2scan = tmp_path / "r2scan"
    _write_release(pbe, stem="MatPES-PBE-2025.2", functional="PBE")
    _write_release(r2scan, stem="MatPES-R2SCAN-2025.2", functional="r2SCAN")
    output = tmp_path / "audit.json"
    output.write_text("{}")
    with pytest.raises(FileExistsError):
        run(pbe_root=pbe, r2scan_root=r2scan, output=output)
