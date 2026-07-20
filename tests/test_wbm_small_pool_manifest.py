from __future__ import annotations

import bz2
import json
import runpy
from pathlib import Path

import pytest

MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "build_wbm_small_pool_manifest.py"))
SOAP_MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "build_wbm_small_pool_soap_cache.py")
)
Candidate = MODULE["ObservableCandidate"]


def _candidate(system: tuple[str, ...], index: int) -> object:
    return Candidate(
        query_id=f"id-{''.join(system)}-{index}", chemical_system=system,
        composition=tuple((element, 1.0) for element in system),
        exact_structure_sha256=f"sha256:{''.join(system)}-{index}",
    )


def test_exact_duplicate_filter_is_deterministic() -> None:
    first = _candidate(("A", "B"), 1)
    second = Candidate(
        query_id="id-later", chemical_system=("A", "B"), composition=(("A", 1.0), ("B", 1.0)),
        exact_structure_sha256=first.exact_structure_sha256,
    )
    retained, duplicates = MODULE["deduplicate_exact_structures"]([second, first])
    assert duplicates == 1
    assert retained == [first]


def test_selector_fails_instead_of_silently_replacing_an_empty_stratum() -> None:
    candidates = [_candidate(("A", "B"), index) for index in range(20)]
    with pytest.raises(ValueError, match="no eligible systems"):
        MODULE["select_small_pools"](candidates, pool_size=16, calibration_fraction=0.01)


def test_policy_identity_and_soap_source_use_initial_not_relaxed_structure(
    tmp_path: Path,
) -> None:
    cse_root = tmp_path / "cse"
    structures_root = tmp_path / "initial"
    cse_root.mkdir()
    structures_root.mkdir()
    initial = {
        "@module": "pymatgen.core.structure",
        "@class": "Structure",
        "charge": 0,
        "lattice": {"matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]], "pbc": [True] * 3},
        "properties": {},
        "sites": [
            {"species": [{"element": "Li", "occu": 1}], "abc": [0, 0, 0], "properties": {}, "label": "Li"}
        ],
    }
    relaxed = json.loads(json.dumps(initial))
    relaxed["lattice"]["matrix"][0][0] = 3
    for step in range(1, 6):
        entries = []
        structures = {}
        if step == 1:
            entries = [{"composition": {"Li": 1.0}, "structure": relaxed}]
            structures = {"step_1_0": {"org": initial, "opt": relaxed}}
        (cse_root / f"step_{step}.json.bz2").write_bytes(
            bz2.compress(json.dumps({"entries": entries}).encode())
        )
        (structures_root / f"wbm-structures-step-{step}.json.bz2").write_bytes(
            bz2.compress(json.dumps(structures).encode())
        )

    candidates = MODULE["read_observable_candidates"](
        cse_root=cse_root,
        structures_root=structures_root,
        cleaned_ids={"wbm-1-1"},
    )
    assert len(candidates) == 1
    assert candidates[0].exact_structure_sha256 == MODULE["_structure_checksum"](initial)
    assert candidates[0].exact_structure_sha256 != MODULE["_structure_checksum"](relaxed)

    cleaned_ids = tmp_path / "cleaned.txt"
    cleaned_ids.write_text("wbm-1-1\n", encoding="utf-8")
    recovered = SOAP_MODULE["_initial_structures_by_id"](
        cse_root,
        structures_root,
        cleaned_ids,
        {"wbm-1-1"},
    )
    assert recovered == {"wbm-1-1": initial}

    changed_relaxed = json.loads(json.dumps(relaxed))
    changed_relaxed["lattice"]["matrix"][0][0] = 9
    wrapper = {"step_1_0": {"org": initial, "opt": changed_relaxed}}
    (structures_root / "wbm-structures-step-1.json.bz2").write_bytes(
        bz2.compress(json.dumps(wrapper).encode())
    )
    counterfactual = MODULE["read_observable_candidates"](
        cse_root=cse_root,
        structures_root=structures_root,
        cleaned_ids={"wbm-1-1"},
    )
    counterfactual_recovered = SOAP_MODULE["_initial_structures_by_id"](
        cse_root,
        structures_root,
        cleaned_ids,
        {"wbm-1-1"},
    )
    assert counterfactual == candidates
    assert counterfactual_recovered == recovered


def test_initial_structure_reader_never_falls_back_to_relaxed_opt(tmp_path: Path) -> None:
    cse_root = tmp_path / "cse"
    structures_root = tmp_path / "structures"
    cse_root.mkdir()
    structures_root.mkdir()
    relaxed = {
        "lattice": {"matrix": [[3, 0, 0], [0, 3, 0], [0, 0, 3]]},
        "sites": [],
    }
    for step in range(1, 6):
        entries = [{"composition": {"Li": 1.0}}] if step == 1 else []
        records = {"step_1_0": {"org": None, "opt": relaxed}} if step == 1 else {}
        (cse_root / f"step_{step}.json.bz2").write_bytes(
            bz2.compress(json.dumps({"entries": entries}).encode())
        )
        (structures_root / f"wbm-structures-step-{step}.json.bz2").write_bytes(
            bz2.compress(json.dumps(records).encode())
        )

    with pytest.raises(ValueError, match="missing WBM org initial structure"):
        MODULE["read_observable_candidates"](
            cse_root=cse_root,
            structures_root=structures_root,
            cleaned_ids={"wbm-1-1"},
        )
