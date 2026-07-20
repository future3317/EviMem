"""Audit fixed-GP headroom, kernel alignment, and WBM compute topology.

This is an evaluator-only diagnostic over already frozen matched-action panels.
It does not run acquisition, alter a trace, or tune a kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process.kernels import RBF, Matern

from matmem.ceiling_diagnostics import (
    centered_kernel_target_alignment,
    deterministic_bootstrap_mean,
    kernel_moran_autocorrelation,
    nearest_neighbor_sign_agreement,
    regularized_effective_dimension,
    signal_kernel_matrix,
)
from matmem.residual_posterior import FixedKernelGPConfig

METRICS = ("crps", "brier", "log_loss", "rmse", "gaussian_nll")
SUMMARY_KEYS = {
    "crps": "boundary_weighted_causal_crps",
    "brier": "boundary_weighted_causal_brier",
    "log_loss": "boundary_weighted_causal_log_loss",
    "rmse": "residual_rmse_ev_per_atom",
    "gaussian_nll": "residual_gaussian_nll",
}


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _base_kernel(config: FixedKernelGPConfig):
    return (
        Matern(length_scale=config.length_scale, nu=2.5)
        if config.kernel == "matern52"
        else RBF(length_scale=config.length_scale)
    )


def _predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    config: FixedKernelGPConfig,
) -> tuple[np.ndarray, np.ndarray]:
    base = _base_kernel(config)
    signal_variance = config.signal_std_ev_per_atom**2
    predictive_variance = signal_variance + config.noise_std_ev_per_atom**2
    if not len(train_y):
        return (
            np.zeros(len(test_x), dtype=float),
            np.full(len(test_x), math.sqrt(predictive_variance), dtype=float),
        )
    covariance = signal_variance * np.asarray(base(train_x), dtype=float)
    covariance += np.eye(len(train_y)) * (
        config.noise_std_ev_per_atom**2 + config.jitter
    )
    cross = signal_variance * np.asarray(base(test_x, train_x), dtype=float)
    factor = np.linalg.cholesky(covariance)
    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, train_y))
    mean = cross @ alpha
    solved = np.linalg.solve(factor, cross.T)
    variance = np.maximum(predictive_variance - np.sum(solved**2, axis=0), config.jitter)
    return mean, np.sqrt(variance)


def _metrics(
    *,
    truth: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, float]:
    probability = np.clip(ndtr((thresholds - mean) / std), 1e-12, 1 - 1e-12)
    z = (truth - mean) / std
    normal_pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    normal_cdf = ndtr(z)
    crps = std * (
        z * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / math.sqrt(math.pi)
    )
    brier = (probability - labels) ** 2
    log_loss = -(labels * np.log(probability) + (1 - labels) * np.log(1 - probability))
    nll = 0.5 * np.log(2 * math.pi * std**2) + 0.5 * z**2
    normalized_weights = weights / np.sum(weights)
    return {
        "crps": float(normalized_weights @ crps),
        "brier": float(normalized_weights @ brier),
        "log_loss": float(normalized_weights @ log_loss),
        "rmse": float(np.sqrt(np.mean((truth - mean) ** 2))),
        "gaussian_nll": float(np.mean(nll)),
    }


def _snapshot_arrays(
    snapshot: dict[str, Any],
    embeddings: dict[str, np.ndarray],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    records = snapshot["query_evaluations"]
    ids = [item["query_id"] for item in records]
    truth = np.asarray([item["true_residual_ev_per_atom"] for item in records], dtype=float)
    labels = np.asarray([item["causal_stable_label"] for item in records], dtype=float)
    weights = np.asarray([item["boundary_weight"] for item in records], dtype=float)
    thresholds = np.asarray(
        [item["residual_threshold_ev_per_atom"] for item in records], dtype=float
    )
    x = np.vstack([embeddings[item] for item in ids])
    return ids, x, truth, labels, weights, thresholds


def _oracle_best_k(
    *,
    archive_ids: list[str],
    residual_by_id: dict[str, float],
    embeddings: dict[str, np.ndarray],
    test_x: np.ndarray,
    truth: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    thresholds: np.ndarray,
    capacity: int,
    config: FixedKernelGPConfig,
) -> tuple[dict[str, float], dict[str, list[str]], int]:
    candidates = [
        subset
        for size in range(min(capacity, len(archive_ids)) + 1)
        for subset in itertools.combinations(archive_ids, size)
    ]
    best = dict.fromkeys(METRICS, math.inf)
    selected: dict[str, list[str]] = {}
    for subset in candidates:
        train_x = (
            np.vstack([embeddings[item] for item in subset])
            if subset
            else np.empty((0, test_x.shape[1]), dtype=float)
        )
        train_y = np.asarray([residual_by_id[item] for item in subset], dtype=float)
        mean, std = _predict(train_x, train_y, test_x, config)
        values = _metrics(
            truth=truth,
            labels=labels,
            weights=weights,
            thresholds=thresholds,
            mean=mean,
            std=std,
        )
        for metric, value in values.items():
            key = list(subset)
            if value < best[metric] - 1e-15 or (
                math.isclose(value, best[metric], abs_tol=1e-15)
                and key < selected.get(metric, ["~"])
            ):
                best[metric] = value
                selected[metric] = key
    return {key: float(value) for key, value in best.items()}, selected, len(candidates)


def _mean_round_metrics(rounds: list[dict[str, float]]) -> dict[str, float]:
    return {metric: float(np.mean([item[metric] for item in rounds])) for metric in METRICS}


def _benchmark_prefix_factorizations(
    action_ids: list[str],
    embeddings: dict[str, np.ndarray],
    config: FixedKernelGPConfig,
    *,
    repetitions: int = 200,
) -> dict[str, float | int]:
    """Time only Cholesky factorizations for all full-history trace prefixes."""

    matrix = np.vstack([embeddings[item] for item in action_ids])
    kernel = signal_kernel_matrix(matrix, config)
    regularizer = config.noise_std_ev_per_atom**2 + config.jitter
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        for size in range(1, len(action_ids) + 1):
            np.linalg.cholesky(kernel[:size, :size] + np.eye(size) * regularizer)
        durations.append(time.perf_counter() - started)
    return {
        "budget": len(action_ids),
        "factorization_count_per_trace": len(action_ids),
        "repetitions": repetitions,
        "median_prefix_factorization_seconds": float(np.median(durations)),
        "p90_prefix_factorization_seconds": float(np.quantile(durations, 0.9)),
    }


def _analyze_panel(
    summary_path: Path,
    soap_path: Path,
    config: FixedKernelGPConfig,
    *,
    oracle_capacity: int,
) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not payload.get("matched_frozen_acquisition_action_parity"):
        raise ValueError("ceiling diagnostics require a matched-action source panel")
    with np.load(soap_path, allow_pickle=False) as cache:
        ids = [str(item) for item in cache["query_ids"].tolist()]
        vectors = np.asarray(cache["vectors"], dtype=float)
    raw_embeddings = {
        query_id: vector / np.linalg.norm(vector)
        for query_id, vector in zip(ids, vectors, strict=True)
    }
    by_key = {(run["pool"], run["strategy"]): run for run in payload["runs"]}
    pools = sorted({run["pool"] for run in payload["runs"]})
    diagnostic_strategy = "p3c_log" if "p3c_log" in payload["aggregates"] else "p3c_brier"
    methods_by_system: dict[str, dict[str, dict[str, float]]] = {}
    kernel_by_system: dict[str, dict[str, float | int]] = {}
    compute_by_system: dict[str, dict[str, float]] = {}
    oracle_subset_count = 0
    for pool in pools:
        diagnostic = by_key[(pool, diagnostic_strategy)]
        gpv = by_key[(pool, "gp_variance_one_swap")]
        residual_by_id: dict[str, float] = {}
        for record in diagnostic["posterior_projection_rounds"]:
            for item in record.get("selection_effect_records") or []:
                residual_by_id[item["card_id"].removeprefix("wbm-card:")] = float(
                    item["signed_residual_ev_per_atom"]
                )
            for item in record["causal_evaluations"]["archive_reference"]["query_evaluations"]:
                residual_by_id[item["query_id"]] = float(item["true_residual_ev_per_atom"])
        system_ids = sorted(residual_by_id)
        if not system_ids or not set(system_ids).issubset(raw_embeddings):
            raise ValueError(f"SOAP/residual identity coverage failed for {pool}")
        raw_system_x = np.vstack([raw_embeddings[item] for item in system_ids])
        gram = raw_system_x @ raw_system_x.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        keep = eigenvalues > 1e-12
        system_x = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
        if not np.allclose(system_x @ system_x.T, gram, atol=1e-9):
            raise ValueError(f"exact SOAP Gram factorization failed for {pool}")
        embeddings = dict(zip(system_ids, system_x, strict=True))
        system_y = np.asarray([residual_by_id[item] for item in system_ids], dtype=float)
        kernel = signal_kernel_matrix(system_x, config)
        loo_mean: list[float] = []
        loo_std: list[float] = []
        for index in range(len(system_ids)):
            mask = np.arange(len(system_ids)) != index
            mean, std = _predict(system_x[mask], system_y[mask], system_x[index : index + 1], config)
            loo_mean.append(float(mean[0]))
            loo_std.append(float(std[0]))
        loo_mean_array = np.asarray(loo_mean)
        loo_std_array = np.asarray(loo_std)
        loo_z = (system_y - loo_mean_array) / loo_std_array
        positive_fraction = float(np.mean(system_y > 0))
        chance_sign_agreement = positive_fraction**2 + (1 - positive_fraction) ** 2
        neighbor_sign_agreement = nearest_neighbor_sign_agreement(kernel, system_y)
        kernel_by_system[pool] = {
            "candidate_count": len(system_ids),
            "effective_dimension": regularized_effective_dimension(system_x, config),
            "effective_dimension_over_k2": regularized_effective_dimension(system_x, config) / 2,
            "centered_kernel_target_alignment": centered_kernel_target_alignment(kernel, system_y),
            "kernel_moran_autocorrelation": kernel_moran_autocorrelation(kernel, system_y),
            "positive_residual_fraction": positive_fraction,
            "chance_residual_sign_agreement": chance_sign_agreement,
            "nearest_neighbor_residual_sign_agreement": neighbor_sign_agreement,
            "nearest_neighbor_sign_agreement_above_chance": (
                neighbor_sign_agreement - chance_sign_agreement
            ),
            "residual_sample_variance": float(np.var(system_y, ddof=1)),
            "residual_variance_over_frozen_noise_variance": float(
                np.var(system_y, ddof=1) / config.noise_std_ev_per_atom**2
            ),
            "full_history_loo_rmse_ev_per_atom": float(
                np.sqrt(np.mean((system_y - loo_mean_array) ** 2))
            ),
            "full_history_loo_gaussian_nll": float(
                np.mean(0.5 * np.log(2 * math.pi * loo_std_array**2) + 0.5 * loo_z**2)
            ),
        }
        rounds = {name: [] for name in ("prior", "full_history", "gp_variance_k2", "oracle_best_k")}
        full_fit_seconds = 0.0
        full_prediction_seconds = 0.0
        for projection, gpv_snapshot in zip(
            diagnostic["posterior_projection_rounds"],
            gpv["prequential_posterior_snapshots"],
            strict=True,
        ):
            full = projection["causal_evaluations"]["archive_reference"]
            query_ids, test_x, truth, labels, weights, thresholds = _snapshot_arrays(
                full, embeddings
            )
            del query_ids
            prior_mean, prior_std = _predict(
                np.empty((0, test_x.shape[1])), np.empty(0), test_x, config
            )
            rounds["prior"].append(
                _metrics(
                    truth=truth,
                    labels=labels,
                    weights=weights,
                    thresholds=thresholds,
                    mean=prior_mean,
                    std=prior_std,
                )
            )
            rounds["full_history"].append(
                {metric: float(full[SUMMARY_KEYS[metric]]) for metric in METRICS}
            )
            rounds["gp_variance_k2"].append(
                {metric: float(gpv_snapshot[SUMMARY_KEYS[metric]]) for metric in METRICS}
            )
            archive_ids = [
                item.removeprefix("wbm-card:") for item in full["witness_card_ids"]
            ]
            oracle_values, _, candidate_count = _oracle_best_k(
                archive_ids=archive_ids,
                residual_by_id=residual_by_id,
                embeddings=embeddings,
                test_x=test_x,
                truth=truth,
                labels=labels,
                weights=weights,
                thresholds=thresholds,
                capacity=oracle_capacity,
                config=config,
            )
            oracle_subset_count += candidate_count
            rounds["oracle_best_k"].append(oracle_values)
            full_fit_seconds += float(full["posterior_fit_seconds"])
            full_prediction_seconds += float(full["prediction_seconds"])
        methods_by_system[pool] = {
            name: _mean_round_metrics(values) for name, values in rounds.items()
        }
        compute_by_system[pool] = {
            "budget": float(payload["budget"]),
            "gp_variance_fit_seconds": float(gpv["prequential"]["posterior_fit_seconds"]),
            "gp_variance_prediction_seconds": float(
                gpv["prequential"]["prediction_seconds"]
            ),
            "full_history_diagnostic_fit_seconds": full_fit_seconds,
            "full_history_diagnostic_prediction_seconds": full_prediction_seconds,
            "gp_variance_wall_seconds": float(gpv["wall_seconds"]),
            "diagnostic_trace_wall_seconds": float(diagnostic["wall_seconds"]),
            "gp_variance_posterior_share_of_wall": float(
                (
                    gpv["prequential"]["posterior_fit_seconds"]
                    + gpv["prequential"]["prediction_seconds"]
                )
                / gpv["wall_seconds"]
            ),
            "full_history_posterior_share_of_diagnostic_wall": float(
                (full_fit_seconds + full_prediction_seconds)
                / diagnostic["wall_seconds"]
            ),
            "timing_semantics": (
                "FixedKernelResidualGP is lazy: numerical factorization occurs inside "
                "prediction_seconds, not posterior_fit_seconds"
            ),
            "cholesky_microbenchmark": _benchmark_prefix_factorizations(
                [str(item) for item in diagnostic["selected_query_ids"]],
                embeddings,
                config,
            ),
        }
    return {
        "summary": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "soap_cache": str(soap_path.resolve()),
        "soap_cache_sha256": _sha256(soap_path),
        "pool_count": len(pools),
        "budget": payload["budget"],
        "oracle_best_k_capacity": oracle_capacity,
        "oracle_subset_evaluations": oracle_subset_count,
        "methods_by_system": methods_by_system,
        "kernel_by_system": kernel_by_system,
        "compute_by_system": compute_by_system,
    }


def analyze(
    panels: list[tuple[Path, Path]],
    config: FixedKernelGPConfig,
    *,
    oracle_capacity: int,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    results = [
        _analyze_panel(summary, soap, config, oracle_capacity=oracle_capacity)
        for summary, soap in panels
    ]
    systems: dict[str, dict[str, dict[str, float]]] = {}
    kernel: dict[str, dict[str, float | int]] = {}
    compute: dict[str, dict[str, float]] = {}
    for panel_index, panel in enumerate(results):
        for pool, values in panel["methods_by_system"].items():
            key = f"panel{panel_index + 1}:{pool}"
            systems[key] = values
            kernel[key] = panel["kernel_by_system"][pool]
            compute[key] = panel["compute_by_system"][pool]
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        comparisons[metric] = {}
        for left, right, label in (
            ("prior", "full_history", "full_history_minus_prior"),
            ("full_history", "gp_variance_k2", "gp_variance_k2_minus_full_history"),
            ("oracle_best_k", "full_history", "full_history_minus_oracle_best_k"),
        ):
            differences = [systems[item][right][metric] - systems[item][left][metric] for item in systems]
            comparisons[metric][label] = deterministic_bootstrap_mean(
                differences,
                seed=bootstrap_seed,
                iterations=bootstrap_iterations,
            )
    effective_dimensions = [float(item["effective_dimension"]) for item in kernel.values()]
    posterior_shares = [
        item["full_history_posterior_share_of_diagnostic_wall"]
        for item in compute.values()
    ]
    return {
        "schema_version": "wbm-fixed-gp-ceiling-v2",
        "scope": "evaluator_only_frozen_trace_diagnostic_no_policy_or_parameter_changes",
        "gp_config": config.__dict__,
        "panels": results,
        "combined": {
            "system_count": len(systems),
            "comparisons": comparisons,
            "effective_dimension": deterministic_bootstrap_mean(
                effective_dimensions,
                seed=bootstrap_seed,
                iterations=bootstrap_iterations,
            ),
            "systems_with_effective_dimension_above_k2": int(
                np.sum(np.asarray(effective_dimensions) > 2)
            ),
            "systems_with_effective_dimension_above_k4": int(
                np.sum(np.asarray(effective_dimensions) > 4)
            ),
            "kernel_alignment": {
                name: deterministic_bootstrap_mean(
                    [float(item[name]) for item in kernel.values()],
                    seed=bootstrap_seed,
                    iterations=bootstrap_iterations,
                )
                for name in (
                    "centered_kernel_target_alignment",
                    "kernel_moran_autocorrelation",
                    "nearest_neighbor_residual_sign_agreement",
                    "nearest_neighbor_sign_agreement_above_chance",
                    "residual_variance_over_frozen_noise_variance",
                    "full_history_loo_rmse_ev_per_atom",
                    "full_history_loo_gaussian_nll",
                )
            },
            "compute_ceiling": {
                "observed_budget": sorted({int(item["budget"]) for item in compute.values()}),
                "full_history_posterior_share_of_diagnostic_wall": deterministic_bootstrap_mean(
                    posterior_shares,
                    seed=bootstrap_seed,
                    iterations=bootstrap_iterations,
                ),
                "requested_budget_status": {
                    str(budget): (
                        "observed" if budget in {int(item["budget"]) for item in compute.values()} else "not_observed_in_these_short_frozen_traces"
                    )
                    for budget in (8, 12, 24, 40)
                },
            },
            "methods_by_system": systems,
            "kernel_by_system": kernel,
            "compute_by_system": compute,
        },
        "bootstrap": {
            "seed": bootstrap_seed,
            "iterations": bootstrap_iterations,
            "statistical_unit": "exact_chemical_system",
        },
        "probability_threshold_source": {
            "exact": True,
            "field": "residual_threshold_ev_per_atom",
            "guardrail": (
                "ceiling inputs without explicit residual thresholds fail closed; "
                "thresholds are never reconstructed from stored probabilities"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", nargs=2, action="append", metavar=("SUMMARY", "SOAP"), required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--oracle-capacity", type=int, default=2)
    parser.add_argument("--bootstrap-seed", type=int, default=20270720)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"diagnostic output already exists: {args.output}")
    freeze = json.loads(args.calibration_freeze.read_text(encoding="utf-8"))
    gp = freeze["gp_config"]
    config = FixedKernelGPConfig(
        kernel=gp["kernel"],
        length_scale=float(gp["length_scale"]),
        signal_std_ev_per_atom=float(gp["signal_std_ev_per_atom"]),
        noise_std_ev_per_atom=float(gp["noise_std_ev_per_atom"]),
        jitter=float(gp["jitter"]),
    )
    started = time.perf_counter()
    result = analyze(
        [(Path(summary), Path(soap)) for summary, soap in args.panel],
        config,
        oracle_capacity=args.oracle_capacity,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    result["wall_seconds"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["combined"], indent=2, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
