"""Validate and summarize a matched-action P3C engineering P1 result."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matmem import gaussian_nll_shapley_attribution, reference_headroom_recovery

P3C_STRATEGIES = (
    "p3c_brier",
    "p3c_log",
    "p3c_gaussian_kl",
    "p3c_twcrps",
    "p3c_twcrps_decision_safe",
)
METRICS = {
    "crps": "boundary_weighted_causal_crps",
    "brier": "boundary_weighted_causal_brier",
    "log_loss": "boundary_weighted_causal_log_loss",
    "rmse": "residual_rmse_ev_per_atom",
    "gaussian_nll": "residual_gaussian_nll",
}


def _paired_summary(
    differences: np.ndarray,
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    if differences.ndim != 1 or not len(differences):
        raise ValueError("paired differences must be a nonempty vector")
    generator = np.random.default_rng(bootstrap_seed)
    indices = generator.integers(
        0,
        len(differences),
        size=(bootstrap_iterations, len(differences)),
    )
    bootstrap = differences[indices].mean(axis=1)
    if len(differences) <= 20:
        sign_flip_distribution = np.asarray(
            [
                float(np.mean(differences * np.asarray(signs, dtype=float)))
                for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
            ],
            dtype=float,
        )
        sign_flip_mode = "exact_enumeration"
    else:
        draw_count = max(100_000, bootstrap_iterations)
        signs = generator.choice((-1.0, 1.0), size=(draw_count, len(differences)))
        sign_flip_distribution = np.mean(signs * differences, axis=1)
        sign_flip_mode = "deterministic_monte_carlo"
    observed = float(np.mean(differences))
    exact_two_sided_p = float(np.mean(np.abs(sign_flip_distribution) >= abs(observed) - 1e-15))
    leave_one_out = [
        float(np.mean(np.delete(differences, index))) if len(differences) > 1 else None
        for index in range(len(differences))
    ]
    largest_improvement_index = int(np.argmin(differences))
    gross_improvements = np.sort(np.maximum(-differences, 0.0))[::-1]
    gross_improvement = float(np.sum(gross_improvements))
    return {
        "mean_difference": observed,
        "median_difference": float(np.median(differences)),
        "ci95": [float(value) for value in np.quantile(bootstrap, (0.025, 0.975))],
        "wins": int(np.sum(differences < -1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences > 1e-12)),
        "leave_one_system_out_mean_differences": leave_one_out,
        "largest_improvement_contributor_index": largest_improvement_index,
        "mean_without_largest_improvement_contributor": (leave_one_out[largest_improvement_index]),
        "exact_sign_flip": {
            "distribution": [float(value) for value in sign_flip_distribution],
            "two_sided_p_value": exact_two_sided_p,
            "enumeration_count": len(sign_flip_distribution),
            "mode": sign_flip_mode,
        },
        "improvement_concentration": {
            "gross_improvement": gross_improvement,
            "gross_degradation": float(np.sum(np.maximum(differences, 0.0))),
            "top_1_share": (
                float(np.sum(gross_improvements[:1]) / gross_improvement)
                if gross_improvement > 0
                else None
            ),
            "top_3_share": (
                float(np.sum(gross_improvements[:3]) / gross_improvement)
                if gross_improvement > 0
                else None
            ),
        },
    }


def _selection_regression(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive retention regression; not a selective-inference correction."""

    features = (
        "absolute_residual_ev_per_atom",
        "residual_sign",
        "mean_kernel_similarity_to_queries",
        "reference_mean_influence",
        "reference_variance_influence",
        "reference_stable_logit_influence",
    )
    if not records:
        return {"status": "no_records", "feature_names": features}
    x = np.asarray([[record[name] for name in features] for record in records], dtype=float)
    y = np.asarray([record["retained"] for record in records], dtype=int)
    if len(np.unique(y)) < 2:
        return {
            "status": "single_class",
            "feature_names": features,
            "record_count": len(records),
            "retained_fraction": float(np.mean(y)),
        }
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, solver="liblinear", random_state=0),
    )
    model.fit(x, y)
    probability = model.predict_proba(x)[:, 1]
    coefficients = model.named_steps["logisticregression"].coef_[0]
    return {
        "status": "fit",
        "scope": "descriptive_in_sample_not_formal_inference",
        "record_count": len(records),
        "retained_fraction": float(np.mean(y)),
        "feature_names": features,
        "standardized_coefficients": {
            name: float(value) for name, value in zip(features, coefficients, strict=True)
        },
        "in_sample_roc_auc": float(roc_auc_score(y, probability)),
    }


def _projection_decomposition(
    *,
    payload: dict[str, Any],
    by_key: dict[tuple[str, str], dict[str, Any]],
    pools: list[str],
    p3c_strategies: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Analyze the frozen reference/path/selection diagnostics."""

    if not payload.get("reference_path_diagnostics_included", False):
        return {"status": "not_recorded_in_source_result"}, []
    issues: list[str] = []
    comparator = "gp_variance_one_swap"
    result: dict[str, Any] = {"status": "analyzed", "strategies": {}}
    for strategy in p3c_strategies:
        strategy_result: dict[str, Any] = {
            "reference_headroom_and_recovery": {},
            "reference_search_factorial": {},
            "nll_shapley": {},
        }
        selection_records: list[dict[str, Any]] = []
        timing_records: list[dict[str, float]] = []
        run_records: dict[str, list[dict[str, Any]]] = {}
        gpv_snapshots: dict[str, list[dict[str, Any]]] = {}
        for pool in pools:
            run = by_key[(pool, strategy)]
            records = run.get("posterior_projection_rounds") or []
            snapshots = by_key[(pool, comparator)].get("prequential_posterior_snapshots") or []
            if len(records) != len(snapshots) or len(records) != int(payload["budget"]):
                issues.append(f"{pool}:{strategy} diagnostic round alignment failed")
                continue
            run_records[pool] = records
            gpv_snapshots[pool] = snapshots
            for record in records:
                selection_records.extend(record.get("selection_effect_records") or [])
                timing = record.get("timing")
                if timing is None:
                    issues.append(f"{pool}:{strategy} lacks phase timing")
                else:
                    timing_records.append(timing)

        if len(run_records) != len(pools):
            result["strategies"][strategy] = {"status": "incomplete"}
            continue

        for metric_label, metric_name in METRICS.items():
            reference_results: dict[str, Any] = {}
            for reference_name in ("union_reference", "archive_reference"):
                system_values: dict[str, dict[str, float | None]] = {}
                all_round_values: list[dict[str, float | None]] = []
                for pool in pools:
                    round_values = []
                    for record, gpv in zip(run_records[pool], gpv_snapshots[pool], strict=True):
                        evaluations = record.get("causal_evaluations") or {}
                        if reference_name not in evaluations:
                            issues.append(f"{pool}:{strategy} lacks {reference_name} evaluation")
                            continue
                        round_values.append(
                            reference_headroom_recovery(
                                reference_loss=float(evaluations[reference_name][metric_name]),
                                projected_loss=float(
                                    evaluations["union_reference__online_search"][metric_name]
                                ),
                                comparator_loss=float(gpv[metric_name]),
                            )
                        )
                    if not round_values:
                        continue
                    all_round_values.extend(round_values)
                    recoveries = [
                        item["projection_recovery"]
                        for item in round_values
                        if item["projection_recovery"] is not None
                    ]
                    system_values[pool] = {
                        "reference_headroom": float(
                            np.mean([item["reference_headroom"] for item in round_values])
                        ),
                        "compression_loss": float(
                            np.mean([item["compression_loss"] for item in round_values])
                        ),
                        "projected_minus_comparator": float(
                            np.mean([item["projected_minus_comparator"] for item in round_values])
                        ),
                        "mean_positive_headroom_recovery": (
                            float(np.mean(recoveries)) if recoveries else None
                        ),
                        "median_positive_headroom_recovery": (
                            float(np.median(recoveries)) if recoveries else None
                        ),
                        "headroom_weighted_recovery": (
                            float(
                                np.sum(
                                    [
                                        item["reference_headroom"] - item["compression_loss"]
                                        for item in round_values
                                        if item["reference_headroom"] > 0
                                    ]
                                )
                                / np.sum(
                                    [
                                        item["reference_headroom"]
                                        for item in round_values
                                        if item["reference_headroom"] > 0
                                    ]
                                )
                            )
                            if recoveries
                            else None
                        ),
                        "positive_headroom_round_fraction": float(
                            np.mean([item["reference_headroom"] > 0 for item in round_values])
                        ),
                    }
                headrooms = np.asarray(
                    [system_values[pool]["reference_headroom"] for pool in pools],
                    dtype=float,
                )
                positive_rounds = [
                    item for item in all_round_values if item["reference_headroom"] > 0
                ]
                pooled_recoveries = [item["projection_recovery"] for item in positive_rounds]
                reference_results[reference_name] = {
                    "system_macro": system_values,
                    "mean_reference_headroom": float(np.mean(headrooms)),
                    "median_reference_headroom": float(np.median(headrooms)),
                    "positive_headroom_system_count": int(np.sum(headrooms > 0)),
                    "positive_headroom_round_count": len(positive_rounds),
                    "median_positive_headroom_recovery": (
                        float(np.median(pooled_recoveries)) if pooled_recoveries else None
                    ),
                    "headroom_weighted_recovery": (
                        float(
                            np.sum(
                                [
                                    item["reference_headroom"] - item["compression_loss"]
                                    for item in positive_rounds
                                ]
                            )
                            / np.sum([item["reference_headroom"] for item in positive_rounds])
                        )
                        if positive_rounds
                        else None
                    ),
                    "cross_system_headroom_supported": bool(
                        np.mean(headrooms) > 0
                        and np.median(headrooms) > 0
                        and np.sum(headrooms > 0) > len(pools) / 2
                    ),
                }
            strategy_result["reference_headroom_and_recovery"][metric_label] = reference_results

            factorial_by_name: dict[str, Any] = {}
            for cell_name in (
                "union_reference__online_search",
                "union_reference__archive_search",
                "archive_reference__online_search",
                "archive_reference__archive_search",
            ):
                reference_name = (
                    "union_reference"
                    if cell_name.startswith("union_reference")
                    else "archive_reference"
                )
                by_system = {
                    pool: float(
                        np.mean(
                            [
                                record["causal_evaluations"][cell_name][metric_name]
                                for record in run_records[pool]
                            ]
                        )
                    )
                    for pool in pools
                }
                cell_round_diagnostics = {
                    pool: [
                        reference_headroom_recovery(
                            reference_loss=float(
                                record["causal_evaluations"][reference_name][metric_name]
                            ),
                            projected_loss=float(
                                record["causal_evaluations"][cell_name][metric_name]
                            ),
                            comparator_loss=float(gpv[metric_name]),
                        )
                        for record, gpv in zip(run_records[pool], gpv_snapshots[pool], strict=True)
                    ]
                    for pool in pools
                }
                positive_cell_rounds = [
                    item
                    for pool in pools
                    for item in cell_round_diagnostics[pool]
                    if item["reference_headroom"] > 0
                ]
                cell_recoveries = [item["projection_recovery"] for item in positive_cell_rounds]
                factorial_by_name[cell_name] = {
                    "system_macro_losses": by_system,
                    "mean_loss": float(np.mean(list(by_system.values()))),
                    "positive_headroom_round_count": len(positive_cell_rounds),
                    "median_positive_headroom_recovery": (
                        float(np.median(cell_recoveries)) if cell_recoveries else None
                    ),
                    "headroom_weighted_recovery": (
                        float(
                            np.sum(
                                [
                                    item["reference_headroom"] - item["compression_loss"]
                                    for item in positive_cell_rounds
                                ]
                            )
                            / np.sum([item["reference_headroom"] for item in positive_cell_rounds])
                        )
                        if positive_cell_rounds
                        else None
                    ),
                }
            online = factorial_by_name["union_reference__online_search"]["system_macro_losses"]
            archive_search = factorial_by_name["union_reference__archive_search"][
                "system_macro_losses"
            ]
            search_differences = np.asarray(
                [archive_search[pool] - online[pool] for pool in pools], dtype=float
            )
            factorial_by_name["union_reference_search_space_effect"] = {
                **_paired_summary(
                    search_differences,
                    bootstrap_seed=20270719,
                    bootstrap_iterations=10_000,
                ),
                "definition": "archive-search minus online-search loss under Q_U",
            }
            strategy_result["reference_search_factorial"][metric_label] = factorial_by_name

        for pool in pools:
            round_mean_parts: list[float] = []
            round_variance_parts: list[float] = []
            round_total_parts: list[float] = []
            for record, gpv in zip(run_records[pool], gpv_snapshots[pool], strict=True):
                p3c = record["causal_evaluations"]["union_reference__online_search"]
                p3c_by_id = {item["query_id"]: item for item in p3c["query_evaluations"]}
                gpv_by_id = {item["query_id"]: item for item in gpv["query_evaluations"]}
                if set(p3c_by_id) != set(gpv_by_id):
                    issues.append(f"{pool}:{strategy} NLL query alignment failed")
                    continue
                query_mean_parts: list[float] = []
                query_variance_parts: list[float] = []
                query_total_parts: list[float] = []
                for query_id in sorted(p3c_by_id):
                    p_item = p3c_by_id[query_id]
                    g_item = gpv_by_id[query_id]
                    attribution = gaussian_nll_shapley_attribution(
                        truth=float(p_item["true_residual_ev_per_atom"]),
                        p3c_mean=float(p_item["posterior_mean_ev_per_atom"]),
                        p3c_std=float(p_item["posterior_std_ev_per_atom"]),
                        gpv_mean=float(g_item["posterior_mean_ev_per_atom"]),
                        gpv_std=float(g_item["posterior_std_ev_per_atom"]),
                    )
                    query_mean_parts.append(attribution.mean_attribution)
                    query_variance_parts.append(attribution.variance_attribution)
                    query_total_parts.append(attribution.p3c_minus_gpv_nll)
                round_mean_parts.append(float(np.mean(query_mean_parts)))
                round_variance_parts.append(float(np.mean(query_variance_parts)))
                round_total_parts.append(float(np.mean(query_total_parts)))
            strategy_result["nll_shapley"][pool] = {
                "aggregation": "query_mean_then_round_mean_with_exact_system_as_statistical_unit",
                "mean_attribution": float(np.mean(round_mean_parts)),
                "variance_attribution": float(np.mean(round_variance_parts)),
                "total_p3c_minus_gpv_nll": float(np.mean(round_total_parts)),
                "additivity_error": float(
                    np.mean(round_total_parts)
                    - np.mean(round_mean_parts)
                    - np.mean(round_variance_parts)
                ),
            }

        strategy_result["selection_effect"] = {
            "regression": _selection_regression(selection_records),
            "record_count": len(selection_records),
            "retained_residual_summary": {
                "mean_signed": float(
                    np.mean(
                        [
                            item["signed_residual_ev_per_atom"]
                            for item in selection_records
                            if item["retained"]
                        ]
                    )
                ),
                "mean_absolute": float(
                    np.mean(
                        [
                            item["absolute_residual_ev_per_atom"]
                            for item in selection_records
                            if item["retained"]
                        ]
                    )
                ),
            },
            "evicted_residual_summary": {
                "mean_signed": float(
                    np.mean(
                        [
                            item["signed_residual_ev_per_atom"]
                            for item in selection_records
                            if not item["retained"]
                        ]
                    )
                ),
                "mean_absolute": float(
                    np.mean(
                        [
                            item["absolute_residual_ev_per_atom"]
                            for item in selection_records
                            if not item["retained"]
                        ]
                    )
                ),
            },
        }
        required_timing = (
            "union_reference_fit_seconds",
            "online_candidate_projection_seconds",
            "archive_reference_fit_seconds",
            "archive_subset_enumeration_seconds",
            "prequential_evaluator_seconds",
            "hull_update_seconds",
        )
        strategy_result["timing"] = {
            name: float(np.sum([item[name] for item in timing_records]))
            for name in required_timing
            if all(name in item for item in timing_records)
        }
        strategy_result["timing"]["archive_diagnostics_excluded_from_online_retention"] = True
        result["strategies"][strategy] = strategy_result

    gate_a = {}
    for strategy in ("p3c_brier", "p3c_log"):
        if strategy not in p3c_strategies:
            continue
        evidence = result["strategies"].get(strategy, {})
        gate_a[strategy] = {
            metric: any(
                reference["cross_system_headroom_supported"]
                for reference in evidence.get("reference_headroom_and_recovery", {})
                .get(metric, {})
                .values()
            )
            for metric in ("brier", "log_loss")
        }
    result["gates"] = {
        "A_reference": {
            "criterion": "cross-system positive reference headroom in both causal Brier and log loss",
            "by_strategy": gate_a,
            "passed_by_any_diagnostic_variant": any(
                all(metrics.values()) for metrics in gate_a.values()
            ),
        },
        "B_projection": {
            "status": "descriptive_only_threshold_not_retuned",
            "criterion": "positive-headroom recovery is reported without inventing a post-hoc cutoff",
        },
        "C_path_dependence": {
            "status": "descriptive_only",
            "criterion": "union-reference archive-search minus online-search causal losses",
        },
        "D_compute": {
            "passed": not issues
            and all(
                len(item.get("timing", {})) >= 7
                for item in result["strategies"].values()
                if item.get("status") != "incomplete"
            ),
            "archive_diagnostics_excluded_from_online_retention": True,
        },
    }
    return result, issues


def analyze(
    path: Path,
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    runs = payload["runs"]
    pools = sorted({run["pool"] for run in runs})
    strategies = sorted(payload["aggregates"])
    p3c_strategies = tuple(
        strategy for strategy in P3C_STRATEGIES if strategy in strategies
    )
    by_key = {(run["pool"], run["strategy"]): run for run in runs}
    issues: list[str] = []
    if len(by_key) != len(runs):
        issues.append("duplicate pool-strategy run keys")
    expected_keys = {(pool, strategy) for pool in pools for strategy in strategies}
    if set(by_key) != expected_keys:
        issues.append("pool-strategy Cartesian product is incomplete")
    if not payload["matched_frozen_acquisition_action_parity"]:
        issues.append("matched-action parity failed")
    budget = int(payload["budget"])
    for run in runs:
        key = f"{run['pool']}:{run['strategy']}"
        if len(run["selected_query_ids"]) != budget:
            issues.append(f"{key} has the wrong action count")
        numeric = [
            value for value in run["prequential"].values() if isinstance(value, (int, float))
        ]
        if not all(math.isfinite(value) for value in numeric):
            issues.append(f"{key} contains a non-finite prequential metric")
        records = run.get("posterior_projection_rounds") or []
        if not records:
            continue
        if len(records) != budget:
            issues.append(f"{key} has the wrong projection-round count")
        for round_index, record in enumerate(records, 1):
            expected_subsets = 1 + round_index + round_index * (round_index - 1) // 2
            if record["archive_exact_candidate_count"] != expected_subsets:
                issues.append(f"{key} archive subset count fails at round {round_index}")
            if record["online_vs_archive_optimization_gap"] < -1e-12:
                issues.append(f"{key} has a negative archive gap")
            if record["online"]["selected_proper_divergence"] < -1e-12:
                issues.append(f"{key} has a negative proper divergence")

    comparator = "gp_variance_one_swap"
    paired: dict[str, Any] = {}
    mechanism: dict[str, Any] = {}
    for strategy in p3c_strategies:
        paired[strategy] = {}
        for label, metric in METRICS.items():
            differences = np.asarray(
                [
                    by_key[(pool, strategy)]["prequential"][metric]
                    - by_key[(pool, comparator)]["prequential"][metric]
                    for pool in pools
                ],
                dtype=float,
            )
            paired[strategy][label] = {
                **(
                    summary := _paired_summary(
                        differences,
                        bootstrap_seed=bootstrap_seed,
                        bootstrap_iterations=bootstrap_iterations,
                    )
                ),
                "system_differences": {
                    pool: float(value) for pool, value in zip(pools, differences, strict=True)
                },
                "leave_one_system_out_mean_differences_by_system": {
                    pool: value
                    for pool, value in zip(
                        pools,
                        summary["leave_one_system_out_mean_differences"],
                        strict=True,
                    )
                },
                "largest_improvement_contributor_system": pools[
                    summary["largest_improvement_contributor_index"]
                ],
            }
        records = [
            record
            for pool in pools
            for record in by_key[(pool, strategy)]["posterior_projection_rounds"]
        ]
        residual_bias = np.asarray(
            [
                record["retained_minus_archive_residual_mean"]
                for record in records
                if record["retained_minus_archive_residual_mean"] is not None
            ],
            dtype=float,
        )
        gaps = np.asarray(
            [record["online_vs_archive_optimization_gap"] for record in records],
            dtype=float,
        )
        mechanism[strategy] = {
            **payload["posterior_projection"][strategy],
            "mean_absolute_retained_minus_archive_residual_mean": float(
                np.mean(np.abs(residual_bias))
            ),
            "positive_archive_gap_round_count": int(np.sum(gaps > 1e-12)),
            "maximum_archive_gap": float(np.max(gaps)),
        }

    agreement: dict[str, Any] = {}
    for left, right in (
        ("p3c_twcrps", "p3c_twcrps_decision_safe"),
        ("p3c_brier", "p3c_log"),
        ("p3c_gaussian_kl", "p3c_twcrps"),
    ):
        if left not in p3c_strategies or right not in p3c_strategies:
            continue
        comparisons = [
            set(left_round["online"]["selected_card_ids"])
            == set(right_round["online"]["selected_card_ids"])
            for pool in pools
            for left_round, right_round in zip(
                by_key[(pool, left)]["posterior_projection_rounds"],
                by_key[(pool, right)]["posterior_projection_rounds"],
                strict=True,
            )
        ]
        agreement[f"{left}__{right}"] = {
            "agreement_count": int(sum(comparisons)),
            "round_count": len(comparisons),
            "agreement_rate": float(np.mean(comparisons)),
        }

    decomposition, decomposition_issues = _projection_decomposition(
        payload=payload,
        by_key=by_key,
        pools=pools,
        p3c_strategies=p3c_strategies,
    )
    issues.extend(decomposition_issues)
    return {
        "schema_version": "wbm-p3c-p1-analysis-v2",
        "source_summary": str(path.resolve()),
        "source_summary_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "quality": {
            "passed": not issues,
            "issues": issues,
            "run_count": len(runs),
            "pool_count": len(pools),
            "strategy_count": len(strategies),
            "matched_action_parity": payload["matched_frozen_acquisition_action_parity"],
            "gp_parameter_status": payload["gp_parameter_status"],
            "calibration_freeze_sha256": payload["calibration_freeze_sha256"],
        },
        "aggregates": payload["aggregates"],
        "paired_p3c_minus_gp_variance": paired,
        "posterior_projection_mechanism": mechanism,
        "reference_path_selection_decomposition": decomposition,
        "p3c_selection_agreement": agreement,
        "bootstrap": {
            "seed": bootstrap_seed,
            "iterations": bootstrap_iterations,
            "statistical_unit": "exact_chemical_system",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20270719)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    args = parser.parse_args()
    result = analyze(
        args.summary,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        if args.output.exists():
            raise FileExistsError(f"analysis output already exists: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
