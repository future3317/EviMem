from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tools.augment_matpes_structure_embeddings import run as augment_structure_embeddings
from tools.build_matpes_protocol_task import (
    ELEMENT_FRACTION_ORDER,
    run,
    source_descriptor,
)
from tools.run_matpes_protocol_closed_loop_exploratory import (
    ExperimentConfig,
    _requires_protocol_transport,
)
from tools.run_matpes_protocol_closed_loop_exploratory import (
    run as run_closed_loop,
)


def _row(*, identifier: str, functional: str, formation: float, parent: str) -> dict:
    return {
        "matpes_id": identifier,
        "functional": functional,
        "nsites": 2,
        "elements": ["Fe", "O"],
        "chemsys": "Fe-O",
        "composition": {"Fe": 1.0, "O": 1.0},
        "energy": -10.0 if functional == "PBE" else -11.0,
        "formation_energy_per_atom": formation,
        "cohesive_energy_per_atom": -3.0,
        "volume": 20.0,
        "density": 5.0,
        "bandgap": None,
        "forces": [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]],
        "stress": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "symmetry": {"number": 2},
        "provenance": {"original_mp_id": parent},
        "structure": {
            "lattice": {"matrix": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]},
            "properties": {},
            "sites": [
                {"species": [{"element": "Fe", "occu": 1}], "abc": [0.0, 0.0, 0.0]},
                {"species": [{"element": "O", "occu": 1}], "abc": [0.5, 0.5, 0.5]},
            ],
        },
    }


def test_delta_hull_runner_requires_frozen_protocol_transport() -> None:
    assert _requires_protocol_transport("delta_hull_active_search")
    assert _requires_protocol_transport("conformal_source_rollout_delta_hull")
    assert _requires_protocol_transport("protocol_hull_knowledge_gradient")
    assert not _requires_protocol_transport("source_margin")


def _release(root: Path, *, stem: str, functional: str, target: bool) -> None:
    root.mkdir(parents=True)
    for index, split in enumerate(("train", "valid", "test")):
        row = _row(
            identifier=f"matpes-{index}",
            functional=functional,
            formation=-0.6 - index / 100 if target else -0.5 - index / 100,
            parent=f"mp-{index}",
        )
        (root / f"{stem}-{split}.jsonl").write_text(json.dumps(row) + "\n")


def test_source_descriptor_preserves_stoichiometric_element_fractions() -> None:
    row = _row(identifier="weighted", functional="PBE", formation=-0.5, parent="mp-x")
    row["composition"] = {"Fe": 2.0, "O": 1.0}
    descriptor = source_descriptor(row)
    fractions = descriptor[-len(ELEMENT_FRACTION_ORDER) :]
    assert sum(fractions) == 1.0
    assert fractions[ELEMENT_FRACTION_ORDER.index("Fe")] == 2.0 / 3.0
    assert fractions[ELEMENT_FRACTION_ORDER.index("O")] == 1.0 / 3.0


def test_builder_keeps_target_values_only_in_oracle_vault(tmp_path: Path) -> None:
    pbe = tmp_path / "pbe"
    r2scan = tmp_path / "r2scan"
    _release(pbe, stem="MatPES-PBE-2025.2", functional="PBE", target=False)
    _release(r2scan, stem="MatPES-R2SCAN-2025.2", functional="r2SCAN", target=True)
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            {
                "decision": {"same_configuration_protocol_task_supported": True},
                "pairing": {"same_configuration_pair_id_set_sha256": "fixture"},
            }
        )
    )
    task_path = tmp_path / "task.json"
    vault_path = tmp_path / "vault.json"
    summary = run(
        pbe_root=pbe,
        r2scan_root=r2scan,
        audit_path=audit,
        task_output=task_path,
        vault_output=vault_path,
        max_systems=1,
        max_candidates_per_system=2,
        minimum_candidates_per_system=2,
        minimum_parents_per_system=1,
    )
    task = json.loads(task_path.read_text())
    vault = json.loads(vault_path.read_text())
    assert summary["selected_pair_count"] == 2
    assert len(task["development_pairs"]) == 2
    assert len(vault["target_outcomes"]) == 2
    assert sum(task["development_pairs"][0]["composition"].values()) == 2.0
    outcomes_by_id = {row["pair_id"]: row for row in vault["target_outcomes"]}
    assert set(outcomes_by_id) == {row["pair_id"] for row in task["development_pairs"]}
    for outcome in outcomes_by_id.values():
        atom_count = sum(outcome["composition"].values())
        assert outcome["target_corrected_total_energy_ev"] == (
            outcome["target_formation_energy_ev_per_atom"] * atom_count
        )
    assert all(
        "target_formation_energy_ev_per_atom" not in row for row in task["development_pairs"]
    )
    assert all(row["split"] == "development" for row in vault["target_outcomes"])
    assert task["descriptor"]["dimension"] == 133
    assert task["descriptor"]["element_fraction_order"] == list(ELEMENT_FRACTION_ORDER)
    element_fractions = task["development_pairs"][0]["source_environment_embedding"][
        -len(ELEMENT_FRACTION_ORDER) :
    ]
    assert sum(element_fractions) == 1.0
    assert element_fractions[ELEMENT_FRACTION_ORDER.index("Fe")] == 0.5
    assert element_fractions[ELEMENT_FRACTION_ORDER.index("O")] == 0.5
    assert {row["entry_id"] for row in task["development_initial_phase_entries"]["Fe-O"]} == {
        "reference-Fe",
        "reference-O",
    }

    class _FixtureEncoder:
        metadata = {
            "encoder": "fixture frozen crystal feature",
            "checkpoint_sha256": "sha256:fixture",
            "target_structure_used": False,
            "target_outcomes_used": False,
        }

        def encode(self, structures, *, batch_size: int) -> np.ndarray:
            assert batch_size == 2
            return np.asarray([[structure.num_sites, 1.0] for structure in structures])

    augmented_task_path = tmp_path / "task-with-structure-embedding.json"
    augmentation = augment_structure_embeddings(
        task_path=task_path,
        pbe_root=pbe,
        output_path=augmented_task_path,
        device="cpu",
        batch_size=2,
        encoder=_FixtureEncoder(),
    )
    augmented = json.loads(augmented_task_path.read_text())
    assert augmentation["embedding_dimension"] == 2
    assert augmented["local_environment_representation"]["uses_target_outcome"] is False
    assert all(
        row["source_local_environment_embedding"] == [2.0, 1.0]
        for row in augmented["development_pairs"]
    )
    assert all(
        row["source_local_environment_isolated_atom_count"] == 0
        for row in augmented["development_pairs"]
    )
    assert all(
        "target_formation_energy_ev_per_atom" not in row
        for row in augmented["development_pairs"]
    )

    experiment_output = tmp_path / "closed-loop.json"
    run_closed_loop(
        task_path=task_path,
        development_vault_path=vault_path,
        output_path=experiment_output,
        config=ExperimentConfig(
            max_systems=1,
            minimum_candidates=2,
            maximum_budget=1,
            seed=7,
            policies=("source_margin",),
        ),
    )
    experiment = json.loads(experiment_output.read_text())
    assert experiment["development_systems"] == ["Fe-O"]
    assert experiment["active_policies"] == ["source_margin"]
    assert set(experiment["code_provenance"]) == {
        "frozen_structure_encoder_sha256",
        "protocol_closed_loop_sha256",
        "protocol_knowledge_gradient_sha256",
        "protocol_policy_worker_sha256",
    }
    assert all(
        len(values["selected_pair_ids"]) == 1
        for values in experiment["systems"]["Fe-O"]["strategies"].values()
    )
    assert all(
        "oracle_pool_confirmed_discoveries" in values
        for values in experiment["systems"]["Fe-O"]["strategies"].values()
    )
