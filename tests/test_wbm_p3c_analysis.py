from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tools.analyze_wbm_p3c_p1 import (
    P3C_STRATEGIES,
    _paired_summary,
    _selection_regression,
    analyze,
)


def _summary_payload() -> dict[str, object]:
    strategies = ("gp_variance_one_swap", *P3C_STRATEGIES)
    runs = []
    for pool_index, pool in enumerate(("A-B", "C-D")):
        for strategy_index, strategy in enumerate(strategies):
            offset = 0.01 * pool_index + 0.001 * strategy_index
            run = {
                "pool": pool,
                "strategy": strategy,
                "selected_query_ids": [f"{pool}-query"],
                "prequential": {
                    "boundary_weighted_causal_crps": 0.1 + offset,
                    "boundary_weighted_causal_brier": 0.2 + offset,
                    "boundary_weighted_causal_log_loss": 0.3 + offset,
                    "residual_rmse_ev_per_atom": 0.4 + offset,
                    "residual_gaussian_nll": 0.5 + offset,
                },
            }
            if strategy in P3C_STRATEGIES:
                run["posterior_projection_rounds"] = [
                    {
                        "archive_exact_candidate_count": 2,
                        "online_vs_archive_optimization_gap": 0.0,
                        "retained_minus_archive_residual_mean": 0.01,
                        "online": {
                            "selected_proper_divergence": 0.0,
                            "selected_card_ids": [f"{pool}-{strategy}"],
                        },
                    }
                ]
            runs.append(run)
    return {
        "budget": 1,
        "runs": runs,
        "aggregates": {strategy: {} for strategy in strategies},
        "matched_frozen_acquisition_action_parity": True,
        "gp_parameter_status": "frozen_on_disjoint_calibration_systems_v1",
        "calibration_freeze_sha256": "sha256:test",
        "posterior_projection": {
            strategy: {"archive_reactivation_round_count": 0} for strategy in P3C_STRATEGIES
        },
    }


def _diagnostic_summary_payload() -> dict[str, object]:
    payload = _summary_payload()
    payload["reference_path_diagnostics_included"] = True

    def snapshot(name: str, *, brier: float, log_loss: float, mean: float) -> dict:
        return {
            "name": name,
            "witness_card_ids": [name],
            "remaining_candidate_count": 1,
            "boundary_weight_sum": 1.0,
            "boundary_weighted_causal_crps": 0.1 + abs(mean),
            "boundary_weighted_causal_brier": brier,
            "boundary_weighted_causal_log_loss": log_loss,
            "residual_rmse_ev_per_atom": abs(0.05 - mean),
            "residual_gaussian_nll": 0.5 + abs(0.05 - mean),
            "query_evaluations": [
                {
                    "query_id": "diagnostic-q",
                    "boundary_weight": 1.0,
                    "true_residual_ev_per_atom": 0.05,
                    "causal_stable_label": 1.0,
                    "posterior_mean_ev_per_atom": mean,
                    "posterior_std_ev_per_atom": 0.08,
                    "residual_threshold_ev_per_atom": 0.04,
                    "stable_probability": 0.7,
                    "gaussian_crps": 0.1,
                    "causal_brier": brier,
                    "causal_log_loss": log_loss,
                    "gaussian_nll": 0.5,
                    "squared_error": (0.05 - mean) ** 2,
                    "posterior_variance": 0.08**2,
                    "squared_standardized_error": ((0.05 - mean) / 0.08) ** 2,
                }
            ],
            "posterior_fit_seconds": 0.001,
            "prediction_seconds": 0.001,
        }

    for run in payload["runs"]:
        if run["strategy"] == "gp_variance_one_swap":
            run["prequential_posterior_snapshots"] = [
                snapshot("gpv", brier=0.20, log_loss=0.30, mean=0.00)
            ]
            continue
        record = run["posterior_projection_rounds"][0]
        evaluations = {
            "union_reference": snapshot("union_reference", brier=0.10, log_loss=0.20, mean=0.04),
            "archive_reference": snapshot(
                "archive_reference", brier=0.09, log_loss=0.19, mean=0.045
            ),
            "union_reference__online_search": snapshot(
                "union_reference__online_search",
                brier=0.15,
                log_loss=0.25,
                mean=0.03,
            ),
            "union_reference__archive_search": snapshot(
                "union_reference__archive_search",
                brier=0.14,
                log_loss=0.24,
                mean=0.035,
            ),
            "archive_reference__online_search": snapshot(
                "archive_reference__online_search",
                brier=0.13,
                log_loss=0.23,
                mean=0.035,
            ),
            "archive_reference__archive_search": snapshot(
                "archive_reference__archive_search",
                brier=0.12,
                log_loss=0.22,
                mean=0.04,
            ),
        }
        record["causal_evaluations"] = evaluations
        record["selection_effect_records"] = [
            {
                "absolute_residual_ev_per_atom": 0.10,
                "residual_sign": 1,
                "mean_kernel_similarity_to_queries": 0.5,
                "reference_mean_influence": 0.02,
                "reference_variance_influence": 0.001,
                "reference_stable_logit_influence": 0.03,
                "signed_residual_ev_per_atom": 0.10,
                "retained": True,
            },
            {
                "absolute_residual_ev_per_atom": 0.01,
                "residual_sign": -1,
                "mean_kernel_similarity_to_queries": 0.5,
                "reference_mean_influence": 0.01,
                "reference_variance_influence": 0.001,
                "reference_stable_logit_influence": 0.01,
                "signed_residual_ev_per_atom": -0.01,
                "retained": False,
            },
        ]
        record["timing"] = {
            "union_reference_fit_seconds": 0.001,
            "online_candidate_projection_seconds": 0.002,
            "archive_reference_fit_seconds": 0.003,
            "archive_subset_enumeration_seconds": 0.004,
            "prequential_evaluator_seconds": 0.005,
            "hull_update_seconds": 0.006,
        }
    return payload


def test_paired_summary_is_deterministic_and_uses_system_level_units() -> None:
    differences = np.asarray((-0.2, 0.0, 0.1), dtype=float)
    left = _paired_summary(
        differences,
        bootstrap_seed=7,
        bootstrap_iterations=1_000,
    )
    right = _paired_summary(
        differences,
        bootstrap_seed=7,
        bootstrap_iterations=1_000,
    )
    assert left == right
    assert left["mean_difference"] == pytest.approx(-1 / 30)
    assert (left["wins"], left["ties"], left["losses"]) == (1, 1, 1)
    assert left["exact_sign_flip"]["enumeration_count"] == 8
    assert left["exact_sign_flip"]["mode"] == "exact_enumeration"
    assert len(left["leave_one_system_out_mean_differences"]) == 3
    assert left["median_difference"] == pytest.approx(0.0)


def test_paired_summary_scales_beyond_exact_sign_flip_enumeration() -> None:
    differences = np.linspace(-0.03, 0.02, 32)
    left = _paired_summary(differences, bootstrap_seed=13, bootstrap_iterations=500)
    right = _paired_summary(differences, bootstrap_seed=13, bootstrap_iterations=500)
    assert left == right
    assert left["exact_sign_flip"]["mode"] == "deterministic_monte_carlo"
    assert left["exact_sign_flip"]["enumeration_count"] == 100_000


def test_selection_regression_detects_residual_dependent_retention() -> None:
    records = []
    for index in range(40):
        magnitude = 0.0025 * index
        records.append(
            {
                "absolute_residual_ev_per_atom": magnitude,
                "residual_sign": -1 if index % 2 else 1,
                "mean_kernel_similarity_to_queries": 0.5,
                "reference_mean_influence": 0.01,
                "reference_variance_influence": 0.001,
                "reference_stable_logit_influence": 0.02,
                "retained": index >= 20,
            }
        )
    result = _selection_regression(records)
    assert result["status"] == "fit"
    assert result["in_sample_roc_auc"] > 0.99
    assert result["standardized_coefficients"]["absolute_residual_ev_per_atom"] > 0


def test_p3c_analysis_passes_complete_result_and_detects_negative_gap(
    tmp_path: Path,
) -> None:
    path = tmp_path / "summary.json"
    payload = _summary_payload()
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = analyze(path, bootstrap_seed=11, bootstrap_iterations=100)
    assert result["quality"]["passed"] is True
    assert result["quality"]["pool_count"] == 2
    assert result["quality"]["strategy_count"] == 6
    assert result["bootstrap"]["statistical_unit"] == "exact_chemical_system"

    p3c_run = next(run for run in payload["runs"] if run["strategy"] == "p3c_brier")
    p3c_run["posterior_projection_rounds"][0]["online_vs_archive_optimization_gap"] = -0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    failed = analyze(path, bootstrap_seed=11, bootstrap_iterations=100)
    assert failed["quality"]["passed"] is False
    assert any("negative archive gap" in issue for issue in failed["quality"]["issues"])


def test_p3c_analysis_computes_reference_path_selection_decomposition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "diagnostic-summary.json"
    path.write_text(json.dumps(_diagnostic_summary_payload()), encoding="utf-8")
    result = analyze(path, bootstrap_seed=11, bootstrap_iterations=100)
    decomposition = result["reference_path_selection_decomposition"]
    assert result["quality"]["passed"] is True
    assert decomposition["status"] == "analyzed"
    assert decomposition["gates"]["A_reference"]["passed_by_any_diagnostic_variant"] is True
    log = decomposition["strategies"]["p3c_log"]
    assert log["nll_shapley"]["A-B"]["additivity_error"] == pytest.approx(0.0)
    assert log["selection_effect"]["regression"]["status"] == "fit"
    assert log["timing"]["archive_diagnostics_excluded_from_online_retention"] is True


def test_p3c_analysis_accepts_a_preregistered_strategy_subset(tmp_path: Path) -> None:
    payload = _diagnostic_summary_payload()
    keep = {"gp_variance_one_swap", "p3c_brier", "p3c_log"}
    payload["runs"] = [run for run in payload["runs"] if run["strategy"] in keep]
    payload["aggregates"] = {
        strategy: value
        for strategy, value in payload["aggregates"].items()
        if strategy in keep
    }
    payload["posterior_projection"] = {
        strategy: value
        for strategy, value in payload["posterior_projection"].items()
        if strategy in keep
    }
    path = tmp_path / "subset-summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = analyze(path, bootstrap_seed=7, bootstrap_iterations=100)

    assert result["quality"]["passed"]
    assert set(result["paired_p3c_minus_gp_variance"]) == {"p3c_brier", "p3c_log"}
    assert set(result["reference_path_selection_decomposition"]["strategies"]) == {
        "p3c_brier",
        "p3c_log",
    }
    assert set(result["p3c_selection_agreement"]) == {"p3c_brier__p3c_log"}
