from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from matmem import (
    CalibrationUtilityBuilder,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    GaussianNLLShapleyAttribution,
    PrequentialCausalEvaluator,
    ProtocolCompatibilityResolver,
    aggregate_prequential_prefix,
    frozen_grid_cells,
    gaussian_nll_shapley_attribution,
    paired_system_bootstrap,
    paired_system_improvement_bootstrap,
    reference_headroom_recovery,
)

from .test_matmem import _card, _query

MANIFEST_MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "build_wbm_frozen_grid_manifest.py")
)
GRID_MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "run_wbm_frozen_grid.py"))


def test_frozen_grid_has_37_labels_and_15_physical_traces() -> None:
    cells = frozen_grid_cells()
    assert len(cells) == 37
    assert len({item.execution_key for item in cells}) == 15
    assert sum(item.physical_execution for item in cells) == 15
    assert all(item.capacity is None or item.capacity < item.budget for item in cells)
    joint = [item for item in cells if item.strategy == "joint_posterior_risk_one_swap"]
    assert [(item.budget, item.capacity) for item in joint] == [(8, 2), (12, 4)]
    assert not any(item.strategy == "survival_coreset" for item in cells)


def test_grid_requires_disjoint_calibration_freeze(tmp_path: Path) -> None:
    posterior = {
        "kernel": "matern52",
        "length_scale": 0.35,
    }
    registered = tmp_path / "config.json"
    registered.write_text(json.dumps({"posterior": posterior}), encoding="utf-8")
    freeze = {
        "scope": "disjoint_calibration_only_no_evaluation_results_accessed",
        "evaluation_results_accessed": False,
        "gp_parameter_status": "frozen_on_disjoint_calibration_systems_v1",
        "full_history_prequential_sanity": {"passed": True},
        "brier_margin": 0.0,
        "log_loss_margin": 0.02,
        "gp_config": posterior,
        "config_sha256": GRID_MODULE["_sha256"](registered),
    }
    path = tmp_path / "freeze.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    assert GRID_MODULE["_read_calibration_freeze"](path, registered) == {
        "boundary_weighted_causal_brier": 0.0,
        "boundary_weighted_causal_log_loss": 0.02,
    }

    freeze["evaluation_results_accessed"] = True
    path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(ValueError, match="evaluation results"):
        GRID_MODULE["_read_calibration_freeze"](path, registered)


def test_frozen_system_selection_keeps_all_candidates_and_never_mixes_systems() -> None:
    candidate_type = MANIFEST_MODULE["ObservableCandidate"]
    candidates = []
    systems = [
        *(tuple((f"B{index}", "X")) for index in range(10)),
        *(tuple((f"T{index}", "X", "Y")) for index in range(10)),
        *(tuple((f"Q{index}", "W", "X", "Y")) for index in range(3)),
    ]
    for system in systems:
        for index in range(16 + len(system)):
            candidates.append(
                candidate_type(
                    query_id=f"{'-'.join(system)}-{index}",
                    chemical_system=system,
                    composition=tuple((element, 1.0) for element in system),
                    exact_structure_sha256=f"sha256:{'-'.join(system)}:{index}",
                )
            )
    selection = MANIFEST_MODULE["select_frozen_grid_systems"](candidates, release_id="test-release")
    assert selection["selected_system_count"] <= 24
    for pool in selection["pools"].values():
        system = tuple(pool["chemical_system"])
        expected = sum(item.chemical_system == system for item in candidates)
        assert pool["candidate_count"] == expected
        assert len(pool["candidates"]) == expected
        assert all(tuple(item["chemical_system"]) == system for item in pool["candidates"])
    assert all(item["selected_system_count"] <= 8 for item in selection["strata"].values())

    next_panel = MANIFEST_MODULE["select_frozen_grid_systems"](
        candidates,
        release_id="test-release",
        systems_per_stratum=2,
        stratum_offset=2,
    )
    first_panel = MANIFEST_MODULE["select_frozen_grid_systems"](
        candidates,
        release_id="test-release",
        systems_per_stratum=2,
        stratum_offset=0,
    )
    assert set(next_panel["pools"]).isdisjoint(first_panel["pools"])
    assert next_panel["stratum_rank_offset"] == 2


def test_prequential_metrics_are_round_weighted_and_prefix_aggregated() -> None:
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(
            ProtocolCompatibilityResolver(),
            config=FixedKernelGPConfig(length_scale=0.35),
        )
    )
    queries = (
        _query("preq-a", embedding=(1.0, 0.0), base_energy=-1.02),
        _query("preq-b", embedding=(0.0, 1.0), base_energy=-0.96),
    )
    evaluator = PrequentialCausalEvaluator(
        builder,
        {"preq-a": -1.04, "preq-b": -0.92},
    )
    first = evaluator.evaluate(round_index=1, queries=queries, cards=())
    snapshot = evaluator.evaluate_snapshot(name="threshold-audit", queries=queries, cards=())
    thresholds = {
        item.query_id: item.residual_threshold_ev_per_atom
        for item in snapshot.query_evaluations
    }
    assert thresholds == pytest.approx(
        {
            query.query_id: (
                query.stability_threshold_ev_per_atom
                - query.base_hull_distance_ev_per_atom
            )
            for query in queries
        }
    )
    second = evaluator.evaluate(
        round_index=2,
        queries=queries,
        cards=(_card("preq-card", embedding=(1.0, 0.0), formation_energy=-1.04),),
        retention_seconds=0.25,
        parent_rss_bytes=100,
    )
    aggregate = aggregate_prequential_prefix((first, second), 2)
    assert aggregate["round_count"] == 2
    assert aggregate["boundary_weighted_causal_crps"] == pytest.approx(
        (first.boundary_weighted_causal_crps + second.boundary_weighted_causal_crps) / 2
    )
    assert aggregate["retention_seconds"] == pytest.approx(0.25)
    assert aggregate["peak_parent_rss_bytes"] == 100
    with pytest.raises(ValueError, match="exactly"):
        aggregate_prequential_prefix((first,), 2)


def test_system_clustered_bootstrap_is_paired_and_deterministic() -> None:
    dacc = {"A-B": 0.1, "C-D": 0.2, "E-F": 0.3}
    baseline = {"A-B": 0.2, "C-D": 0.1, "E-F": 0.4}
    left = paired_system_bootstrap(dacc, baseline, seed=17, iterations=1000)
    right = paired_system_bootstrap(dacc, baseline, seed=17, iterations=1000)
    assert left == right
    assert left["system_count"] == 3
    assert left["mean_paired_difference"] == pytest.approx(-1 / 30)
    with pytest.raises(ValueError, match="identical"):
        paired_system_bootstrap(dacc, {"A-B": 0.2}, seed=17, iterations=10)


def test_system_clustered_improvement_bootstrap_uses_positive_loss_reduction() -> None:
    fifo = {"A-B": 0.3, "C-D": 0.2, "E-F": 0.4}
    gpv = {"A-B": 0.2, "C-D": 0.1, "E-F": 0.3}
    result = paired_system_improvement_bootstrap(fifo, gpv, seed=19, iterations=1000)
    assert result["mean_improvement"] == pytest.approx(0.1)
    assert result == paired_system_improvement_bootstrap(fifo, gpv, seed=19, iterations=1000)


def test_reference_headroom_recovery_obeys_exact_loss_identity() -> None:
    result = reference_headroom_recovery(
        reference_loss=0.10,
        projected_loss=0.14,
        comparator_loss=0.20,
    )
    assert result["reference_headroom"] == pytest.approx(0.10)
    assert result["compression_loss"] == pytest.approx(0.04)
    assert result["projected_minus_comparator"] == pytest.approx(-0.06)
    assert result["projection_recovery"] == pytest.approx(0.60)
    no_headroom = reference_headroom_recovery(
        reference_loss=0.21,
        projected_loss=0.19,
        comparator_loss=0.20,
    )
    assert no_headroom["projection_recovery"] is None


def test_gaussian_nll_shapley_is_symmetric_and_exactly_additive() -> None:
    result: GaussianNLLShapleyAttribution = gaussian_nll_shapley_attribution(
        truth=0.08,
        p3c_mean=0.05,
        p3c_std=0.04,
        gpv_mean=-0.01,
        gpv_std=0.09,
    )
    assert result.mean_attribution + result.variance_attribution == pytest.approx(
        result.p3c_minus_gpv_nll,
        abs=1e-12,
    )
    assert result.p3c_squared_error < result.gpv_squared_error
    with pytest.raises(ValueError, match="positive"):
        gaussian_nll_shapley_attribution(
            truth=0.0,
            p3c_mean=0.0,
            p3c_std=0.0,
            gpv_mean=0.0,
            gpv_std=0.1,
        )


def test_selection_effect_audit_records_outcomes_geometry_and_influence() -> None:
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(
            ProtocolCompatibilityResolver(),
            config=FixedKernelGPConfig(length_scale=0.35),
        )
    )
    evaluator = PrequentialCausalEvaluator(
        builder,
        {"selection-q-a": -1.04, "selection-q-b": -0.92},
    )
    queries = (
        _query("selection-q-a", embedding=(1.0, 0.0), base_energy=-1.02),
        _query("selection-q-b", embedding=(0.0, 1.0), base_energy=-0.96),
    )
    cards = (
        _card(
            "selection-small",
            embedding=(1.0, 0.0),
            formation_energy=-1.02,
            base_energy=-1.03,
        ),
        _card(
            "selection-extreme",
            embedding=(0.0, 1.0),
            formation_energy=-0.83,
            base_energy=-1.03,
        ),
    )
    records = evaluator.selection_effect_records(
        queries=queries,
        union_cards=cards,
        retained_card_ids=("selection-extreme",),
    )
    by_id = {item.card_id: item for item in records}
    assert by_id["selection-extreme"].retained is True
    assert by_id["selection-small"].retained is False
    assert (
        by_id["selection-extreme"].absolute_residual_ev_per_atom
        > by_id["selection-small"].absolute_residual_ev_per_atom
    )
    assert all(item.reference_mean_influence >= 0 for item in records)
    with pytest.raises(ValueError, match="belong"):
        evaluator.selection_effect_records(
            queries=queries,
            union_cards=cards,
            retained_card_ids=("not-a-card",),
        )
