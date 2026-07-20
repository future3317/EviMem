"""Evaluate the preregistered WBM long-archive compute-relevance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.gaussian_process.kernels import RBF, Matern
from threadpoolctl import threadpool_info, threadpool_limits

from matmem.residual_posterior import FixedKernelGPConfig

BUDGETS = (8, 12, 24, 40)
AMDAHL_TARGET_SPEEDUP = 1.10
AMDAHL_MINIMUM_GP_FRACTION = 1.0 - 1.0 / AMDAHL_TARGET_SPEEDUP


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _timing_summary(values: list[float]) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=float)
    median = float(np.median(vector))
    return {
        "repetitions": len(values),
        "median_seconds": median,
        "mad_seconds": float(np.median(np.abs(vector - median))),
        "p95_seconds": float(np.quantile(vector, 0.95)),
    }


def _measure(operation: Callable[[], None], *, warmup: int, repetitions: int) -> dict:
    for _ in range(warmup):
        operation()
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        operation()
        durations.append(time.perf_counter() - started)
    return _timing_summary(durations)


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    norms = np.linalg.norm(matrix, axis=1)
    if matrix.ndim != 2 or np.any(norms <= 0) or not np.all(np.isfinite(matrix)):
        raise ValueError("compute benchmark embeddings must be finite and nonzero")
    return matrix / norms[:, None]


def _base_kernel(config: FixedKernelGPConfig):
    return (
        Matern(length_scale=config.length_scale, nu=2.5)
        if config.kernel == "matern52"
        else RBF(length_scale=config.length_scale)
    )


def _numerical_benchmark(
    train_x: np.ndarray,
    probe_x: np.ndarray,
    config: FixedKernelGPConfig,
    *,
    warmup: int,
    repetitions: int,
) -> dict[str, Any]:
    train = _normalized_rows(train_x)
    probe = _normalized_rows(probe_x)
    base = _base_kernel(config)
    signal_variance = config.signal_std_ev_per_atom**2
    regularizer = config.noise_std_ev_per_atom**2 + config.jitter
    train_y = np.linspace(-0.05, 0.05, len(train), dtype=float)

    def construct() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        train_kernel = signal_variance * np.asarray(base(train), dtype=float)
        cross = signal_variance * np.asarray(base(probe, train), dtype=float)
        probe_kernel = signal_variance * np.asarray(base(probe), dtype=float)
        return train_kernel, cross, probe_kernel

    train_kernel, cross, probe_kernel = construct()
    covariance = train_kernel + np.eye(len(train)) * regularizer

    def factorize() -> None:
        np.linalg.cholesky(covariance)

    def marginal_prediction() -> None:
        factor = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, train_y))
        mean = cross @ alpha
        solved = np.linalg.solve(factor, cross.T)
        variance = signal_variance + config.noise_std_ev_per_atom**2 - np.sum(
            solved**2, axis=0
        )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(variance)):
            raise FloatingPointError("non-finite marginal GP prediction")

    def full_covariance_prediction() -> None:
        factor = np.linalg.cholesky(covariance)
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, train_y))
        mean = cross @ alpha
        solved = np.linalg.solve(factor, cross.T)
        predictive_covariance = (
            probe_kernel
            + np.eye(len(probe)) * config.noise_std_ev_per_atom**2
            - solved.T @ solved
        )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(predictive_covariance)):
            raise FloatingPointError("non-finite full-covariance GP prediction")

    with threadpool_limits(limits=1, user_api="blas"):
        return {
            "kernel_construction": _measure(
                lambda: construct(), warmup=warmup, repetitions=repetitions
            ),
            "factorization_only": _measure(
                factorize, warmup=warmup, repetitions=repetitions
            ),
            "factorization_plus_marginal_prediction": _measure(
                marginal_prediction, warmup=warmup, repetitions=repetitions
            ),
            "factorization_plus_full_covariance_prediction": _measure(
                full_covariance_prediction,
                warmup=warmup,
                repetitions=repetitions,
            ),
            "archive_size": len(train),
            "probe_count": len(probe),
            "dense_train_kernel_bytes": int(len(train) ** 2 * np.dtype(float).itemsize),
        }


def _real_trace_prefix(run: dict[str, Any], budget: int) -> dict[str, float | int | bool]:
    rounds = run["prequential_rounds"][:budget]
    if len(rounds) != budget:
        raise ValueError(f"{run['pool']} does not contain a complete B{budget} prefix")
    gp_seconds = sum(
        float(item["posterior_fit_seconds"]) + float(item["prediction_seconds"])
        for item in rounds
    )
    pipeline_seconds = sum(float(item["round_pipeline_seconds"]) for item in rounds)
    hull_seconds = sum(
        float(item["hull_update_seconds"])
        for item in run["phase_timings"][:budget]
    )
    fraction = gp_seconds / pipeline_seconds
    return {
        "budget": budget,
        "gp_numerical_seconds": gp_seconds,
        "round_pipeline_seconds": pipeline_seconds,
        "hull_update_seconds": hull_seconds,
        "gp_fraction_of_round_pipeline": fraction,
        "ideal_amdahl_speedup": 1.0 / (1.0 - fraction),
        "passes_10pct_ideal_speedup_gate": fraction >= AMDAHL_MINIMUM_GP_FRACTION,
        "peak_parent_rss_bytes": int(run["peak_parent_rss_bytes"]),
        "dense_train_kernel_bytes": budget**2 * np.dtype(float).itemsize,
    }


def analyze(
    *,
    summary_path: Path,
    pool_manifest_path: Path,
    soap_cache_path: Path,
    calibration_freeze_path: Path,
    checkpoint_path: Path,
    warmup: int,
    repetitions: int,
    probe_count: int,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(pool_manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(calibration_freeze_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("scope") != "pre_physical_run_provenance_checkpoint_not_a_git_commit":
        raise ValueError("compute result requires the pre-run provenance checkpoint")
    changed_sources = [
        item["path"]
        for item in checkpoint["execution_sources"]
        if _sha256(Path(item["path"])) != item["sha256"]
    ]
    if changed_sources:
        raise ValueError(f"execution sources changed after checkpoint: {changed_sources}")
    if summary.get("scope") != (
        "wbm_long_archive_compute_relevance_only_no_policy_or_discovery_claim"
    ):
        raise ValueError("compute analysis requires the dedicated timing-only runner scope")
    if summary.get("budget") != 40 or summary.get("acquisition") != "frozen":
        raise ValueError("compute analysis requires a frozen B40 source trace")
    if {run["strategy"] for run in summary["runs"]} != {"full_history"}:
        raise ValueError("compute analysis requires full-history traces only")
    gp = freeze["gp_config"]
    config = FixedKernelGPConfig(
        kernel=gp["kernel"],
        length_scale=float(gp["length_scale"]),
        signal_std_ev_per_atom=float(gp["signal_std_ev_per_atom"]),
        noise_std_ev_per_atom=float(gp["noise_std_ev_per_atom"]),
        jitter=float(gp["jitter"]),
    )
    with np.load(soap_cache_path, allow_pickle=False) as cache:
        query_ids = [str(item) for item in cache["query_ids"]]
        vectors = np.asarray(cache["vectors"], dtype=float)
    embedding_by_id = dict(zip(query_ids, vectors, strict=True))
    runs = {run["pool"]: run for run in summary["runs"]}
    pools = manifest["selection"]["pools"]
    if set(runs) != set(pools):
        raise ValueError("compute summary and manifest exact systems differ")
    systems: dict[str, Any] = {}
    for name, pool in sorted(pools.items()):
        run = runs[name]
        selected = run["selected_query_ids"]
        if len(selected) != 40 or len(set(selected)) != 40:
            raise ValueError(f"{name} does not contain 40 unique frozen actions")
        fixed_probe_ids = [
            item["query_id"] for item in pool["candidates"][: min(probe_count, len(pool["candidates"]))]
        ]
        probe_x = np.vstack([embedding_by_id[item] for item in fixed_probe_ids])
        systems[name] = {
            "candidate_count": int(pool["candidate_count"]),
            "fixed_probe_ids_sha256": "sha256:"
            + hashlib.sha256(("\n".join(fixed_probe_ids) + "\n").encode()).hexdigest(),
            "real_trace": {
                str(budget): _real_trace_prefix(run, budget) for budget in BUDGETS
            },
            "fixed_probe": {
                str(budget): _numerical_benchmark(
                    np.vstack([embedding_by_id[item] for item in selected[:budget]]),
                    probe_x,
                    config,
                    warmup=warmup,
                    repetitions=repetitions,
                )
                for budget in BUDGETS
            },
            "b40_wall_seconds": float(run["wall_seconds"]),
        }
    b40_fractions = [
        float(item["real_trace"]["40"]["gp_fraction_of_round_pipeline"])
        for item in systems.values()
    ]
    max_fraction = max(b40_fractions)
    return {
        "schema_version": "wbm-long-archive-compute-relevance-result-v1",
        "scope": "timing_and_memory_only_no_policy_discovery_or_calibration_claim",
        "inputs": {
            "summary": str(summary_path.resolve()),
            "summary_sha256": _sha256(summary_path),
            "pool_manifest_sha256": _sha256(pool_manifest_path),
            "soap_cache_sha256": _sha256(soap_cache_path),
            "calibration_freeze_sha256": _sha256(calibration_freeze_path),
            "pre_run_checkpoint_sha256": _sha256(checkpoint_path),
            "pre_run_code_tree_sha256": checkpoint["current_code_tree_sha256"],
        },
        "budgets": list(BUDGETS),
        "probe_count": probe_count,
        "warmup": warmup,
        "repetitions": repetitions,
        "blas": {
            "requested_threads": 1,
            "threadpool_info": threadpool_info(),
            "environment": {
                name: os.environ.get(name)
                for name in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                )
            },
        },
        "platform": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
        },
        "amdahl_gate": {
            "target_ideal_speedup": AMDAHL_TARGET_SPEEDUP,
            "minimum_gp_fraction": AMDAHL_MINIMUM_GP_FRACTION,
            "maximum_observed_b40_gp_fraction": max_fraction,
            "maximum_observed_ideal_speedup": 1.0 / (1.0 - max_fraction),
            "passed": max_fraction >= AMDAHL_MINIMUM_GP_FRACTION,
            "decision_if_failed": (
                "stop WBM end-to-end compute-Pareto claims and do not authorize AKSC "
                "as this paper's WBM main method"
            ),
        },
        "systems": systems,
        "system_macro": {
            "b40_gp_fraction_mean": statistics.fmean(b40_fractions),
            "b40_gp_fraction_median": statistics.median(b40_fractions),
        },
        "guardrails": [
            "real traces preserve exact chemical systems and frozen action prefixes",
            "fixed probes use only initial-structure SOAP and are not causal metrics",
            "each timing repetition performs fresh dense linear algebra without GP cache reuse",
            "P1.5 discovery support is irrelevant to this timing-only gate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--soap-cache", type=Path, required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--probe-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.warmup, args.repetitions, args.probe_count) < 1:
        raise ValueError("warmup, repetitions, and probe count must be positive")
    if args.output.exists():
        raise FileExistsError(f"immutable compute result exists: {args.output}")
    result = analyze(
        summary_path=args.summary,
        pool_manifest_path=args.pool_manifest,
        soap_cache_path=args.soap_cache,
        calibration_freeze_path=args.calibration_freeze,
        checkpoint_path=args.checkpoint,
        warmup=args.warmup,
        repetitions=args.repetitions,
        probe_count=args.probe_count,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["amdahl_gate"], indent=2, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
