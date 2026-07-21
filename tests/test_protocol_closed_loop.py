from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pymatgen.entries.computed_entries import ComputedEntry

from matmem.identity import StructureArtifactIdentity
from matmem.protocol_closed_loop import (
    AppendOnlyProtocolEventLog,
    ObservableProtocolQuery,
    ProtocolCandidate,
    ProtocolCausalHull,
    ProtocolOracleOutcome,
    ProtocolOracleVault,
    ProtocolPolicyState,
    ProtocolPolicySubprocess,
    SecureProtocolQueryRunner,
)
from matmem.protocol_knowledge_gradient import fit_protocol_ridge_transport
from matmem.protocols import ProtocolCertificate


def _protocol(functional: str) -> ProtocolCertificate:
    return ProtocolCertificate(
        functional=functional,
        pseudopotential_set="PAW-PBE",
        correction_scheme="fixture-reference",
        relaxation_protocol=f"{functional}-relaxed",
        calculation_code="VASP",
    )


def _fixture(
    *,
    hidden_q1_target: float = -0.2,
) -> tuple[tuple[ProtocolCandidate, ...], tuple[ProtocolOracleOutcome, ...]]:
    source = _protocol("PBE")
    target = _protocol("r2SCAN")
    specifications = (
        ("q1", {"Fe": 0.5, "Zr": 0.5}, -0.10, hidden_q1_target),
        ("q2", {"Fe": 2 / 3, "Zr": 1 / 3}, -0.30, -0.12),
        ("q3", {"Fe": 1 / 3, "Zr": 2 / 3}, -0.05, -0.08),
    )
    candidates = tuple(
        ProtocolCandidate(
            pair_id=pair_id,
            source_structure_hash=f"hash-{pair_id}",
            source_structure_identity=StructureArtifactIdentity.low_fidelity_relaxed(
                pair_id, f"hash-{pair_id}"
            ),
            chemical_system=("Fe", "Zr"),
            composition=composition,
            source_formation_energy_ev_per_atom=source_energy,
            source_environment_embedding=(float(index), 1.0),
            source_local_environment_embedding=(float(index), -1.0),
            source_protocol=source,
            target_protocol=target,
        )
        for index, (pair_id, composition, source_energy, _) in enumerate(specifications)
    )
    outcomes = tuple(
        ProtocolOracleOutcome(
            pair_id=pair_id,
            source_structure_hash=f"hash-{pair_id}",
            chemical_system=("Fe", "Zr"),
            composition=composition,
            target_corrected_total_energy_ev=target_energy,
            target_formation_energy_ev_per_atom=target_energy,
            split="fixture",
        )
        for pair_id, composition, _, target_energy in specifications
    )
    return candidates, outcomes


def _run(tmp_path: Path, *, name: str, hidden_q1_target: float = -0.2, budget: float = 2):
    candidates, outcomes = _fixture(hidden_q1_target=hidden_q1_target)
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    event_log = AppendOnlyProtocolEventLog(tmp_path / name)
    runner = SecureProtocolQueryRunner(
        candidates=candidates,
        vault=vault,
        causal_hull=ProtocolCausalHull(
            (
                ComputedEntry("Fe", 0.0, entry_id="Fe"),
                ComputedEntry("Zr", 0.0, entry_id="Zr"),
            ),
            chemical_system=("Fe", "Zr"),
        ),
        policy=ProtocolPolicySubprocess("source_margin"),
        event_log=event_log,
    )
    result = runner.run(oracle_budget=budget)
    event_log.close()
    return result, vault


def test_selected_action_is_the_only_revealed_action(tmp_path: Path) -> None:
    result, vault = _run(tmp_path, name="selected-only.jsonl", budget=1)
    assert result.selected_pair_ids == ("q2",)
    assert result.revealed_pair_ids == result.selected_pair_ids
    assert vault.revealed_pair_ids == result.selected_pair_ids
    records = [json.loads(line) for line in (tmp_path / "selected-only.jsonl").read_text().splitlines()]
    assert [record["kind"] for record in records] == ["action", "reveal"]
    assert records[0]["selected_pair_id"] == records[1]["selected_pair_id"] == "q2"


def test_confirmatory_split_is_explicitly_supported() -> None:
    candidates, outcomes = _fixture()
    confirmatory = tuple(outcome.model_copy(update={"split": "confirmatory"}) for outcome in outcomes)
    vault = ProtocolOracleVault(confirmatory, expected_split="confirmatory")
    assert vault.expected_split == "confirmatory"
    assert vault.reveal_count == 0
    assert len(candidates) == len(confirmatory)


def test_unrevealed_outcome_cannot_change_first_action(tmp_path: Path) -> None:
    baseline, _ = _run(tmp_path, name="baseline.jsonl", hidden_q1_target=-0.2, budget=1)
    changed, _ = _run(tmp_path, name="changed.jsonl", hidden_q1_target=10.0, budget=1)
    assert baseline.selected_pair_ids == changed.selected_pair_ids == ("q2",)
    assert baseline.events[0].pre_reveal_state_checksum == changed.events[0].pre_reveal_state_checksum
    assert baseline.events[0].action_checksum == changed.events[0].action_checksum


def test_policy_payload_contains_no_unrevealed_target_fields() -> None:
    candidates, _ = _fixture()
    hull = ProtocolCausalHull(
        (ComputedEntry("Fe", 0.0), ComputedEntry("Zr", 0.0)),
        chemical_system=("Fe", "Zr"),
    )
    policy = ProtocolPolicySubprocess("source_margin")
    state = ProtocolPolicyState.create(
        round_index=1,
        remaining_budget=1.0,
        queries=(
            ObservableProtocolQuery(
                pair_id=candidate.pair_id,
                source_structure_hash=candidate.source_structure_hash,
                chemical_system=candidate.chemical_system,
                composition=candidate.composition,
                source_formation_energy_ev_per_atom=(
                    candidate.source_formation_energy_ev_per_atom
                ),
                source_environment_embedding=candidate.source_environment_embedding,
                source_local_environment_embedding=(
                    candidate.source_local_environment_embedding
                ),
                current_competing_hull_ev_per_atom=(
                    hull.competing_hull_formation_energy(candidate.composition)
                ),
                source_protocol_fingerprint=(
                    candidate.source_protocol.scientific_fingerprint
                ),
                target_protocol_fingerprint=(
                    candidate.target_protocol.scientific_fingerprint
                ),
                oracle_cost=candidate.oracle_cost,
            )
            for candidate in candidates
        ),
        causal_hull_phases=hull.observable_phases,
        revealed_history=(),
        policy_identity_checksum=policy.identity_checksum,
    )
    payload = json.loads(state.serialized_for_policy())
    forbidden = {
        "target_corrected_total_energy_ev",
        "target_formation_energy_ev_per_atom",
        "target_structure",
        "stable_label",
    }
    assert all(not (set(row) & forbidden) for row in payload["queries"])
    assert all("target_protocol_fingerprint" in row for row in payload["queries"])
    assert all(
        row["source_local_environment_embedding"] is not None
        for row in payload["queries"]
    )


def test_vault_rejects_reveal_not_bound_to_persisted_selection(tmp_path: Path) -> None:
    candidates, outcomes = _fixture()
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    log = AppendOnlyProtocolEventLog(tmp_path / "wrong-reveal.jsonl")
    authorization = log.append_action(
        round_index=1,
        selected_pair_id="q1",
        pre_reveal_state_checksum="sha256:" + "0" * 64,
    )
    with pytest.raises(RuntimeError, match="persisted action"):
        vault.reveal(candidates[1], authorization=authorization, event_log=log)
    log.close()


def test_policy_rejects_unknown_subprocess_action(tmp_path: Path) -> None:
    worker = tmp_path / "malicious_worker.py"
    worker.write_text("print('not-a-candidate')\n", encoding="utf-8")
    candidates, _ = _fixture()
    hull = ProtocolCausalHull(
        (ComputedEntry("Fe", 0.0), ComputedEntry("Zr", 0.0)),
        chemical_system=("Fe", "Zr"),
    )
    policy = ProtocolPolicySubprocess("source_margin", worker_path=worker)
    state = ProtocolPolicyState.create(
        round_index=1,
        remaining_budget=1,
        queries=(
            ObservableProtocolQuery(
                pair_id=candidate.pair_id,
                source_structure_hash=candidate.source_structure_hash,
                chemical_system=candidate.chemical_system,
                composition=candidate.composition,
                source_formation_energy_ev_per_atom=(
                    candidate.source_formation_energy_ev_per_atom
                ),
                source_environment_embedding=candidate.source_environment_embedding,
                source_local_environment_embedding=(
                    candidate.source_local_environment_embedding
                ),
                current_competing_hull_ev_per_atom=(
                    hull.competing_hull_formation_energy(candidate.composition)
                ),
                source_protocol_fingerprint=(
                    candidate.source_protocol.scientific_fingerprint
                ),
                target_protocol_fingerprint=(
                    candidate.target_protocol.scientific_fingerprint
                ),
                oracle_cost=candidate.oracle_cost,
            )
            for candidate in candidates
        ),
        causal_hull_phases=hull.observable_phases,
        revealed_history=(),
        policy_identity_checksum=policy.identity_checksum,
    )
    with pytest.raises(RuntimeError, match="unknown pair ID"):
        policy.select(state)


def test_chic_policy_drives_the_only_oracle_reveal(tmp_path: Path) -> None:
    candidates, outcomes = _fixture()
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    event_log = AppendOnlyProtocolEventLog(tmp_path / "chic-selected-only.jsonl")
    runner = SecureProtocolQueryRunner(
        candidates=candidates,
        vault=vault,
        causal_hull=ProtocolCausalHull(
            (
                ComputedEntry("Fe", 0.0, entry_id="Fe"),
                ComputedEntry("Zr", 0.0, entry_id="Zr"),
            ),
            chemical_system=("Fe", "Zr"),
        ),
        policy=ProtocolPolicySubprocess("chic_hull_influence"),
        event_log=event_log,
    )
    result = runner.run(oracle_budget=1)
    event_log.close()
    assert len(result.selected_pair_ids) == 1
    assert result.selected_pair_ids == result.revealed_pair_ids
    assert result.selected_pair_ids == vault.revealed_pair_ids


def test_predicted_final_policy_drives_the_only_oracle_reveal(tmp_path: Path) -> None:
    candidates, outcomes = _fixture()
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    event_log = AppendOnlyProtocolEventLog(tmp_path / "predicted-final-selected-only.jsonl")
    runner = SecureProtocolQueryRunner(
        candidates=candidates,
        vault=vault,
        causal_hull=ProtocolCausalHull(
            (
                ComputedEntry("Fe", 0.0, entry_id="Fe"),
                ComputedEntry("Zr", 0.0, entry_id="Zr"),
            ),
            chemical_system=("Fe", "Zr"),
        ),
        policy=ProtocolPolicySubprocess("ridge_predicted_final_margin"),
        event_log=event_log,
    )
    result = runner.run(oracle_budget=1)
    event_log.close()
    assert len(result.selected_pair_ids) == 1
    assert result.selected_pair_ids == result.revealed_pair_ids
    assert result.selected_pair_ids == vault.revealed_pair_ids


def _protocol_transport_fixture():
    return fit_protocol_ridge_transport(
        features=np.asarray([[0.0, 1.0], [1.0, 1.0], [0.2, 0.0], [0.8, 0.0]]),
        source_energies=np.asarray([-0.4, -0.2, -0.3, -0.1]),
        target_energies=np.asarray([-0.38, -0.17, -0.34, -0.13]),
        system_ids=("Fe-O", "Fe-O", "O-Zr", "O-Zr"),
    )


@pytest.mark.parametrize(
    "policy_name",
    (
        "delta_hull_active_search",
        "source_rollout_delta_hull",
        "conformal_source_rollout_delta_hull",
        "protocol_hull_knowledge_gradient",
    ),
)
def test_protocol_hull_policy_requires_disjoint_transport_model(policy_name: str) -> None:
    with pytest.raises(ValueError, match="frozen transport model"):
        ProtocolPolicySubprocess(policy_name)


def test_source_rollout_rejects_non_blockable_sample_count() -> None:
    with pytest.raises(ValueError, match="sixteen power-of-two Sobol blocks"):
        ProtocolPolicySubprocess(
            "source_rollout_delta_hull",
            transport_model=_protocol_transport_fixture(),
            posterior_sample_count=24,
        )


def test_conformal_source_rollout_requires_frozen_threshold() -> None:
    with pytest.raises(ValueError, match="finite non-negative threshold"):
        ProtocolPolicySubprocess(
            "conformal_source_rollout_delta_hull",
            transport_model=_protocol_transport_fixture(),
            posterior_sample_count=32,
        )


def test_conformal_source_rollout_high_threshold_is_source_fallback(
    tmp_path: Path,
) -> None:
    candidates, outcomes = _fixture()
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    event_log = AppendOnlyProtocolEventLog(tmp_path / "conformal-source.jsonl")
    runner = SecureProtocolQueryRunner(
        candidates=candidates,
        vault=vault,
        causal_hull=ProtocolCausalHull(
            (
                ComputedEntry("Fe", 0.0, entry_id="Fe"),
                ComputedEntry("Zr", 0.0, entry_id="Zr"),
            ),
            chemical_system=("Fe", "Zr"),
        ),
        policy=ProtocolPolicySubprocess(
            "conformal_source_rollout_delta_hull",
            transport_model=_protocol_transport_fixture(),
            conformal_threshold=1e9,
            posterior_sample_count=32,
            fantasy_count=1,
        ),
        event_log=event_log,
    )
    result = runner.run(oracle_budget=2)
    event_log.close()
    assert result.selected_pair_ids == ("q2", "q1")
    assert result.selected_pair_ids == result.revealed_pair_ids


@pytest.mark.parametrize(
    "policy_name",
    ("protocol_hull_knowledge_gradient", "protocol_hull_risk_reduction"),
)
def test_protocol_hull_policy_drives_only_authorized_reveals(
    tmp_path: Path,
    policy_name: str,
) -> None:
    candidates, outcomes = _fixture()
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    event_log = AppendOnlyProtocolEventLog(tmp_path / "protocol-hull-selected-only.jsonl")
    runner = SecureProtocolQueryRunner(
        candidates=candidates,
        vault=vault,
        causal_hull=ProtocolCausalHull(
            (
                ComputedEntry("Fe", 0.0, entry_id="Fe"),
                ComputedEntry("Zr", 0.0, entry_id="Zr"),
            ),
            chemical_system=("Fe", "Zr"),
        ),
        policy=ProtocolPolicySubprocess(
            policy_name,
            transport_model=_protocol_transport_fixture(),
            posterior_sample_count=8,
            fantasy_count=1,
        ),
        event_log=event_log,
    )
    result = runner.run(oracle_budget=2)
    event_log.close()
    assert len(result.selected_pair_ids) == 2
    assert result.selected_pair_ids == result.revealed_pair_ids
    assert result.selected_pair_ids == vault.revealed_pair_ids


def test_delta_hull_fixed_composition_backend_is_action_equivalent(
    tmp_path: Path,
) -> None:
    candidates, outcomes = _fixture()
    selected: dict[str, tuple[str, ...]] = {}
    for backend in ("pymatgen", "fixed_composition"):
        vault = ProtocolOracleVault(outcomes, expected_split="fixture")
        event_log = AppendOnlyProtocolEventLog(tmp_path / f"delta-{backend}.jsonl")
        runner = SecureProtocolQueryRunner(
            candidates=candidates,
            vault=vault,
            causal_hull=ProtocolCausalHull(
                (
                    ComputedEntry("Fe", 0.0, entry_id="Fe"),
                    ComputedEntry("Zr", 0.0, entry_id="Zr"),
                ),
                chemical_system=("Fe", "Zr"),
            ),
            policy=ProtocolPolicySubprocess(
                "delta_hull_active_search",
                transport_model=_protocol_transport_fixture(),
                posterior_sample_count=8,
                fantasy_count=1,
                hull_backend=backend,
            ),
            event_log=event_log,
        )
        result = runner.run(oracle_budget=2)
        event_log.close()
        selected[backend] = result.selected_pair_ids
        assert result.selected_pair_ids == result.revealed_pair_ids
    assert selected["fixed_composition"] == selected["pymatgen"]


def test_source_rollout_drives_only_authorized_reveals(tmp_path: Path) -> None:
    candidates, outcomes = _fixture()
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    event_log = AppendOnlyProtocolEventLog(tmp_path / "source-rollout.jsonl")
    runner = SecureProtocolQueryRunner(
        candidates=candidates,
        vault=vault,
        causal_hull=ProtocolCausalHull(
            (
                ComputedEntry("Fe", 0.0, entry_id="Fe"),
                ComputedEntry("Zr", 0.0, entry_id="Zr"),
            ),
            chemical_system=("Fe", "Zr"),
        ),
        policy=ProtocolPolicySubprocess(
            "source_rollout_delta_hull",
            transport_model=_protocol_transport_fixture(),
            posterior_sample_count=32,
            hull_backend="fixed_composition",
        ),
        event_log=event_log,
    )
    result = runner.run(oracle_budget=2)
    event_log.close()
    assert len(result.selected_pair_ids) == 2
    assert result.selected_pair_ids == result.revealed_pair_ids
    assert result.selected_pair_ids == vault.revealed_pair_ids


def test_protocol_hull_policy_falls_back_on_unseen_elements(tmp_path: Path) -> None:
    unsupported = fit_protocol_ridge_transport(
        features=np.asarray([[0.0, 1.0], [1.0, 1.0], [0.2, 0.0], [0.8, 0.0]]),
        source_energies=np.asarray([-0.4, -0.2, -0.3, -0.1]),
        target_energies=np.asarray([-0.38, -0.17, -0.34, -0.13]),
        system_ids=("A-B", "A-B", "C-D", "C-D"),
    )
    candidates, outcomes = _fixture()
    vault = ProtocolOracleVault(outcomes, expected_split="fixture")
    event_log = AppendOnlyProtocolEventLog(tmp_path / "unsupported-fallback.jsonl")
    runner = SecureProtocolQueryRunner(
        candidates=candidates,
        vault=vault,
        causal_hull=ProtocolCausalHull(
            (
                ComputedEntry("Fe", 0.0, entry_id="Fe"),
                ComputedEntry("Zr", 0.0, entry_id="Zr"),
            ),
            chemical_system=("Fe", "Zr"),
        ),
        policy=ProtocolPolicySubprocess(
            "delta_hull_active_search",
            transport_model=unsupported,
            posterior_sample_count=8,
            fantasy_count=1,
        ),
        event_log=event_log,
    )
    result = runner.run(oracle_budget=2)
    event_log.close()
    source, _ = _run(tmp_path, name="unsupported-source.jsonl", budget=2)
    assert result.selected_pair_ids == source.selected_pair_ids


def test_protocol_candidate_accepts_shared_prequery_configuration() -> None:
    source = _protocol("PBE")
    target = _protocol("r2SCAN")
    candidate = ProtocolCandidate(
        pair_id="matpes-1",
        source_structure_hash="geometry-hash",
        source_structure_identity=StructureArtifactIdentity.initial(
            "matpes-1", "geometry-hash"
        ),
        chemical_system=("Fe", "O"),
        composition={"Fe": 0.5, "O": 0.5},
        source_formation_energy_ev_per_atom=-0.5,
        source_environment_embedding=(1.0, 2.0),
        source_protocol=source,
        target_protocol=target,
    )
    assert candidate.source_structure_identity.causal_available_before_query is True


def test_protocol_candidate_rejects_target_relaxed_structure() -> None:
    source = _protocol("PBE")
    target = _protocol("r2SCAN")
    with pytest.raises(ValueError, match="causal pre-query structure"):
        ProtocolCandidate(
            pair_id="leaky",
            source_structure_hash="target-relaxed",
            source_structure_identity=StructureArtifactIdentity.relaxed(
                "leaky", "target-relaxed"
            ),
            chemical_system=("Fe", "O"),
            composition={"Fe": 0.5, "O": 0.5},
            source_formation_energy_ev_per_atom=-0.5,
            source_environment_embedding=(1.0, 2.0),
            source_protocol=source,
            target_protocol=target,
        )


def test_reveal_preserves_stoichiometry_of_corrected_total_energy() -> None:
    hull = ProtocolCausalHull(
        (
            ComputedEntry("Fe", -8.0, entry_id="Fe"),
            ComputedEntry("Zr", -6.0, entry_id="Zr"),
        ),
        chemical_system=("Fe", "Zr"),
    )
    outcome = ProtocolOracleOutcome(
        pair_id="Fe2Zr2",
        source_structure_hash="hash",
        chemical_system=("Fe", "Zr"),
        composition={"Fe": 2.0, "Zr": 2.0},
        target_corrected_total_energy_ev=-30.0,
        target_formation_energy_ev_per_atom=-0.5,
        split="fixture",
    )
    assert sum(outcome.composition.values()) == pytest.approx(4.0)
    hull.add_revealed(outcome)
    revealed = next(
        phase for phase in hull.observable_phases if phase.entry_id == "Fe2Zr2"
    )
    assert sum(revealed.composition.values()) == pytest.approx(1.0)
    assert revealed.formation_energy_ev_per_atom == pytest.approx(-0.5)
    assert hull.competing_hull_formation_energy({"Fe": 1.0, "Zr": 1.0}) == pytest.approx(
        -0.5
    )
