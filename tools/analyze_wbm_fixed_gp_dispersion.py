"""Measure fixed-GP LOO dispersion on frozen, structure-correct WBM panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from matmem.ceiling_diagnostics import (
    deterministic_bootstrap_mean,
    gaussian_dispersion_diagnostics,
)
from matmem.residual_posterior import FixedKernelGPConfig

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from analyze_wbm_model_ceiling import _predict, _sha256  # noqa: E402


def _panel_dispersion(
    summary_path: Path,
    soap_path: Path,
    config: FixedKernelGPConfig,
) -> dict[str, Any]:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("matched_frozen_acquisition_action_parity") is not True:
        raise ValueError("dispersion diagnostic requires a matched frozen-action panel")
    with np.load(soap_path, allow_pickle=False) as cache:
        ids = [str(item) for item in cache["query_ids"].tolist()]
        vectors = np.asarray(cache["vectors"], dtype=float)
    raw_embeddings = dict(zip(ids, vectors, strict=True))
    diagnostic_strategy = "p3c_log" if "p3c_log" in payload["aggregates"] else "p3c_brier"
    by_pool = {
        run["pool"]: run
        for run in payload["runs"]
        if run["strategy"] == diagnostic_strategy
    }
    systems: dict[str, Any] = {}
    for pool, run in sorted(by_pool.items()):
        residual_by_id: dict[str, float] = {}
        for record in run["posterior_projection_rounds"]:
            for item in record.get("selection_effect_records") or []:
                residual_by_id[item["card_id"].removeprefix("wbm-card:")] = float(
                    item["signed_residual_ev_per_atom"]
                )
            for item in record["causal_evaluations"]["archive_reference"][
                "query_evaluations"
            ]:
                residual_by_id[item["query_id"]] = float(
                    item["true_residual_ev_per_atom"]
                )
        system_ids = sorted(residual_by_id)
        if not system_ids or not set(system_ids).issubset(raw_embeddings):
            raise ValueError(f"SOAP/residual coverage failed for {pool}")
        raw_x = np.vstack([raw_embeddings[item] for item in system_ids])
        gram = raw_x @ raw_x.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        keep = eigenvalues > 1e-12
        x = eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])
        y = np.asarray([residual_by_id[item] for item in system_ids], dtype=float)
        loo_mean = []
        loo_std = []
        for index in range(len(system_ids)):
            mask = np.arange(len(system_ids)) != index
            mean, std = _predict(x[mask], y[mask], x[index : index + 1], config)
            loo_mean.append(float(mean[0]))
            loo_std.append(float(std[0]))
        systems[pool] = gaussian_dispersion_diagnostics(
            y,
            np.asarray(loo_mean),
            np.asarray(loo_std),
        )
    return {
        "summary": str(summary_path.resolve()),
        "summary_sha256": _sha256(summary_path),
        "soap_cache": str(soap_path.resolve()),
        "soap_cache_sha256": _sha256(soap_path),
        "systems": systems,
    }


def analyze(
    panels: list[tuple[Path, Path]],
    config: FixedKernelGPConfig,
    *,
    bootstrap_seed: int,
    bootstrap_iterations: int,
) -> dict[str, Any]:
    panel_results = [_panel_dispersion(summary, soap, config) for summary, soap in panels]
    systems = {
        f"panel{panel_index + 1}:{name}": values
        for panel_index, panel in enumerate(panel_results)
        for name, values in panel["systems"].items()
    }
    metrics = (
        "mean_squared_standardized_residual",
        "central_50_coverage",
        "central_80_coverage",
        "central_90_coverage",
    )
    return {
        "schema_version": "wbm-fixed-gp-loo-dispersion-v1",
        "scope": "evaluator_only_fixed_gp_loo_dispersion_no_policy_or_threshold_inversion",
        "gp_config": config.__dict__,
        "panels": panel_results,
        "system_count": len(systems),
        "system_macro": {
            metric: deterministic_bootstrap_mean(
                [float(values[metric]) for values in systems.values()],
                seed=bootstrap_seed,
                iterations=bootstrap_iterations,
            )
            for metric in metrics
        },
        "systems": systems,
        "bootstrap": {
            "statistical_unit": "exact_chemical_system",
            "seed": bootstrap_seed,
            "iterations": bootstrap_iterations,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", nargs=2, action="append", required=True)
    parser.add_argument("--calibration-freeze", type=Path, required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20270720)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable dispersion output exists: {args.output}")
    freeze = json.loads(args.calibration_freeze.read_text(encoding="utf-8"))
    gp = freeze["gp_config"]
    config = FixedKernelGPConfig(
        kernel=gp["kernel"],
        length_scale=float(gp["length_scale"]),
        signal_std_ev_per_atom=float(gp["signal_std_ev_per_atom"]),
        noise_std_ev_per_atom=float(gp["noise_std_ev_per_atom"]),
        jitter=float(gp["jitter"]),
    )
    result = analyze(
        [(Path(summary), Path(soap)) for summary, soap in args.panel],
        config,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["system_macro"], indent=2, sort_keys=True))
    print(f"output={args.output.resolve()}")


if __name__ == "__main__":
    main()
