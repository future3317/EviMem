from __future__ import annotations

import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pytest

from .test_matmem import _card, _query

MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "run_wbm_calibration_engineering_pilot.py")
)


def test_runner_gate_binds_initial_structure_soap_and_exact_files(tmp_path: Path) -> None:
    paths = {
        name: tmp_path / name
        for name in ("pool.json", "parity.json", "soap.npz", "cleaned.txt", "ppd.pkl.gz")
    }
    for index, path in enumerate(paths.values()):
        path.write_bytes(f"artifact-{index}".encode())
    gate = {
        "execution_effect": {"engineering_runner_smoke": True},
        "p1": {
            "soap_structure_stage": "initial",
            "soap_causal_available_before_query": True,
            "soap_structure_source_field": "org",
        },
        "pool_manifest_sha256": MODULE["_sha256"](paths["pool.json"]),
        "parity_audit_sha256": MODULE["_sha256"](paths["parity.json"]),
        "soap_cache_sha256": MODULE["_sha256"](paths["soap.npz"]),
        "cleaned_ids_sha256": MODULE["_sha256"](paths["cleaned.txt"]),
        "ppd_sha256": MODULE["_sha256"](paths["ppd.pkl.gz"]),
    }
    kwargs = {
        "pool_manifest": paths["pool.json"],
        "parity_audit": paths["parity.json"],
        "soap_cache": paths["soap.npz"],
        "cleaned_ids": paths["cleaned.txt"],
        "ppd": paths["ppd.pkl.gz"],
    }
    MODULE["_validate_gate_bindings"](gate, **kwargs)

    gate["execution_effect"]["engineering_runner_smoke"] = False
    gate["p1"]["engineering_p1_passed"] = True
    MODULE["_validate_gate_bindings"](
        gate,
        compute_relevance_only=True,
        **kwargs,
    )
    gate["p1"]["engineering_p1_passed"] = False
    with pytest.raises(ValueError, match="P1 provenance"):
        MODULE["_validate_gate_bindings"](
            gate,
            compute_relevance_only=True,
            **kwargs,
        )
    gate["execution_effect"]["engineering_runner_smoke"] = True
    gate["p1"]["engineering_p1_passed"] = True

    paths["soap.npz"].write_bytes(b"different")
    with pytest.raises(ValueError, match="differ from the audited gate"):
        MODULE["_validate_gate_bindings"](gate, **kwargs)

    gate["p1"]["soap_structure_stage"] = "relaxed"
    with pytest.raises(ValueError, match="initial-structure-only"):
        MODULE["_validate_gate_bindings"](gate, **kwargs)

    gate["p1"]["soap_structure_stage"] = "initial"
    gate["p1"]["soap_structure_source_field"] = "opt"
    with pytest.raises(ValueError, match="org initial-structure field"):
        MODULE["_validate_gate_bindings"](gate, **kwargs)


def test_exact_gram_embedding_preserves_all_pool_soap_inner_products() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(8, 20))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    compact = MODULE["exact_gram_embedding"](vectors)
    assert compact.shape[0] == 8
    assert compact.shape[1] <= 8
    np.testing.assert_allclose(compact @ compact.T, vectors @ vectors.T, atol=1e-9)


def test_runner_rebuilds_oracle_entries_from_parity_not_modern_corrections() -> None:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    ppd = PhaseDiagram(
        (
            ComputedEntry("Li", -1.0, entry_id="Li-ref"),
            ComputedEntry("F", -2.0, entry_id="F-ref"),
        )
    )
    raw = ComputedEntry("LiF", -2.5, entry_id="wbm-parity")
    target = -0.30
    rebuilt = MODULE["_rebuild_pool_entries_from_parity"](
        {("F", "Li"): [raw]},
        required_ids_by_system={("F", "Li"): {"wbm-parity"}},
        parity_rows_by_id={
            "wbm-parity": {
                "parity_corrected_formation_energy_ev_per_atom": target,
            }
        },
        ppd=ppd,
    )

    entry = rebuilt[("F", "Li")][0]
    assert ppd.get_form_energy_per_atom(entry) == pytest.approx(target, abs=1e-12)
    assert entry.energy != pytest.approx(raw.energy)


def test_runner_hard_fails_when_pool_parity_energy_is_missing() -> None:
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.entries.computed_entries import ComputedEntry

    ppd = PhaseDiagram(
        (
            ComputedEntry("Li", -1.0, entry_id="Li-ref"),
            ComputedEntry("F", -2.0, entry_id="F-ref"),
        )
    )
    with pytest.raises(ValueError, match="missing_parity"):
        MODULE["_rebuild_pool_entries_from_parity"](
            {("F", "Li"): [ComputedEntry("LiF", -2.5, entry_id="wbm-parity")]},
            required_ids_by_system={("F", "Li"): {"wbm-parity"}},
            parity_rows_by_id={},
            ppd=ppd,
        )


def test_strategy_factorization_keeps_acquisition_matched() -> None:
    config = MODULE["FixedKernelGPConfig"](length_scale=0.35)
    fifo_policy, _ = MODULE["_strategy"]("fifo", 4, config)
    diversity_policy, _ = MODULE["_strategy"]("diversity", 4, config)
    variance_policy, _ = MODULE["_strategy"]("gp_variance_one_swap", 4, config)
    coreset_policy, _ = MODULE["_strategy"]("decision_coreset", 4, config)
    joint_policy, _ = MODULE["_strategy"]("joint_posterior_risk_one_swap", 4, config)
    p3c_policy, _ = MODULE["_strategy"]("p3c_twcrps", 4, config)
    p3c_safe_policy, p3c_safe_evidence = MODULE["_strategy"]("p3c_twcrps_decision_safe", 4, config)
    assert (
        fifo_policy.policy
        == diversity_policy.policy
        == variance_policy.policy
        == coreset_policy.policy
        == joint_policy.policy
        == p3c_policy.policy
        == p3c_safe_policy.policy
    )
    assert p3c_safe_evidence.memory.planner.scorer.max_decision_regret == 0.0
    assert fifo_policy.identity_checksum == diversity_policy.identity_checksum
    frozen_fifo, _ = MODULE["_strategy"]("fifo", 4, config, acquisition="frozen")
    frozen_coreset, _ = MODULE["_strategy"]("decision_coreset", 4, config, acquisition="frozen")
    frozen_joint, _ = MODULE["_strategy"](
        "joint_posterior_risk_one_swap", 4, config, acquisition="frozen"
    )
    assert (
        frozen_fifo.identity_checksum
        == frozen_coreset.identity_checksum
        == frozen_joint.identity_checksum
    )


def test_full_history_baseline_does_not_truncate_after_sixteen_witnesses() -> None:
    config = MODULE["FixedKernelGPConfig"](length_scale=0.35)
    _, evidence = MODULE["_strategy"]("full_history", 2, config)
    archive = tuple(range(32))
    assert evidence.active(archive) == archive


def test_matched_action_reference_supports_diagnostic_strategy_subsets() -> None:
    choose = MODULE["_matched_action_reference_strategy"]
    assert choose(["fifo", "p3c_log"]) == "fifo"
    assert choose(["gp_variance_one_swap", "p3c_log"]) == ("gp_variance_one_swap")
    with pytest.raises(ValueError, match="at least one"):
        choose([])


def test_runner_loads_and_hash_checks_frozen_gp_manifest(tmp_path: Path) -> None:
    gp_config = {
        "kernel": "matern52",
        "length_scale": 0.35,
        "signal_std_ev_per_atom": 0.08,
        "noise_std_ev_per_atom": 0.01,
        "jitter": 1e-10,
        "parameter_status": "engineering_smoke_only_must_refreeze_on_isolated_calibration_systems",
    }
    encoded = json.dumps(gp_config, sort_keys=True, separators=(",", ":"))
    systems = [f"B{index}-X" for index in range(4)] + [f"T{index}-X-Y" for index in range(4)]
    payload = {
        "schema_version": "wbm-gp-and-noninferiority-calibration-freeze-v1",
        "scope": "disjoint_calibration_only_no_evaluation_results_accessed",
        "evaluation_results_accessed": False,
        "gp_parameter_status": "frozen_on_disjoint_calibration_systems_v1",
        "full_history_prequential_sanity": {"passed": True},
        "calibration_system_ids": systems,
        "calibration_strata": {
            **{system: "binary" for system in systems[:4]},
            **{system: "ternary" for system in systems[4:]},
        },
        "gp_config": gp_config,
        "gp_config_sha256": "sha256:" + hashlib.sha256(encoded.encode()).hexdigest(),
    }
    registered = tmp_path / "registered.json"
    registered.write_text(json.dumps({"posterior": gp_config}), encoding="utf-8")
    payload["config_sha256"] = MODULE["_sha256"](registered)
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config, manifest_sha, loaded = MODULE["_load_calibration_freeze"](path, registered)
    assert config.length_scale == 0.35
    assert manifest_sha.startswith("sha256:")
    assert loaded["calibration_system_ids"] == systems

    payload["gp_config"]["length_scale"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        MODULE["_load_calibration_freeze"](path, registered)


def test_p3c_strategy_records_union_and_archive_projection_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MODULE["FixedKernelGPConfig"](length_scale=0.35)
    _, evidence = MODULE["_strategy"]("p3c_brier", 1, config, acquisition="frozen")
    queries = (_query("p3c-runner-q", embedding=(1.0, 0.0)),)
    first = _card("p3c-runner-a", embedding=(1.0, 0.0), formation_energy=-1.05)
    second = _card("p3c-runner-b", embedding=(0.7, 0.7), formation_energy=-0.95)
    evidence.active(())
    evidence.admit(first, queries)
    evidence.active((first,))
    captured_current_ids: list[tuple[str, ...]] = []
    archive_select = evidence.archive_planner.select

    def _capture_archive_select(*args, **kwargs):
        captured_current_ids.append(tuple(card.card_id for card in kwargs["current_cards"]))
        return archive_select(*args, **kwargs)

    monkeypatch.setattr(evidence.archive_planner, "select", _capture_archive_select)
    evidence.admit(second, queries)
    assert captured_current_ids == [(first.card_id,)]
    assert len(evidence.diagnostics) == 2
    final = evidence.diagnostics[-1]
    assert final["archive_exact_candidate_count"] == 3
    assert final["online_vs_archive_optimization_gap"] >= 0
    assert "retained_minus_archive_residual_mean" in final
    assert set(final["reference_search_factorial"]) == {
        "union_reference__online_search",
        "union_reference__archive_search",
        "archive_reference__online_search",
        "archive_reference__archive_search",
    }
    assert final["timing"]["online_retention_seconds"] >= 0
    assert (
        final["timing"]["union_reference_fit_seconds"]
        >= (final["online"]["reference_prediction_seconds"])
    )
    context = evidence.consume_factorial_context()
    assert tuple(card.card_id for card in context["union_reference_cards"]) == (
        first.card_id,
        second.card_id,
    )
