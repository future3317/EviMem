"""Diagnose transport shift and representation loss in the frozen pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _basis_and_base(
    task: dict[str, Any], freeze: dict[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    rows = task["calibration_pairs"] + task["evaluation_pairs"]
    matrix = np.asarray([row["source_descriptor"] for row in rows], dtype=float)
    basis = freeze["observable_basis"]
    standardized = (matrix - np.asarray(basis["scaler_mean"])) / np.asarray(
        basis["scaler_scale"]
    )
    reduced = (standardized - np.asarray(basis["pca_mean"])) @ np.asarray(
        basis["pca_components"]
    ).T
    if basis["append_constant_feature"]:
        reduced = np.column_stack((reduced, np.ones(len(reduced))))
    by_pair = dict(zip((row["pair_id"] for row in rows), reduced, strict=True))
    coefficient = np.asarray(freeze["base_predictor"]["coefficient"])
    intercept = float(freeze["base_predictor"]["intercept"])
    base = {
        pair_id: float(feature @ coefficient + intercept)
        for pair_id, feature in by_pair.items()
    }
    return by_pair, base


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "mae": float(np.abs(array).mean()),
        "rmse": float(np.sqrt(np.mean(array**2))),
        "p90_absolute": float(np.quantile(np.abs(array), 0.9)),
        "maximum_absolute": float(np.abs(array).max()),
    }


def _stratum_map(task: dict[str, Any]) -> dict[str, str]:
    return {
        system: stratum
        for stratum, systems in task["selection"]["evaluation_systems"].items()
        for system in systems
    }


def analyze(args: argparse.Namespace) -> None:
    task = json.loads(args.task.read_text(encoding="utf-8"))
    vault = json.loads(args.vault.read_text(encoding="utf-8"))
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    if freeze["task_manifest_sha256"] != _sha256(args.task):
        raise ValueError("calibration freeze and task differ")
    if result["calibration_freeze_sha256"] != _sha256(args.freeze):
        raise ValueError("result and calibration freeze differ")
    outcomes = {row["pair_id"]: row for row in vault["target_outcomes"]}
    basis, base = _basis_and_base(task, freeze)
    transport = freeze["transport_map"]
    slope = float(transport["slope"])
    intercept = float(transport["intercept_ev_per_atom"])
    radius = float(transport["error_radius_ev_per_atom"])
    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task["evaluation_pairs"]:
        rows_by_system[row["chemical_system"]].append(row)
    stratum_by_system = _stratum_map(task)
    system_reports: dict[str, dict[str, Any]] = {}
    all_errors: list[float] = []
    all_deltas: list[float] = []
    repeated_system_means: list[float] = []
    projection_errors: list[float] = []
    paired_target_errors: list[float] = []
    structure_rms: list[float] = []
    absolute_transport_errors: list[float] = []
    state_config = json.loads(args.config.read_text(encoding="utf-8"))[
        "all_outcome_state"
    ]
    observation_variance = (
        state_config["observation_noise_std_ev_per_atom"] ** 2 + radius**2
    )
    prior_precision = 1 / state_config["prior_std_ev_per_atom"] ** 2
    for system, rows in sorted(rows_by_system.items()):
        source_residuals = np.asarray(
            [row["source_formation_energy_ev_per_atom"] - base[row["pair_id"]] for row in rows]
        )
        target_residuals = np.asarray(
            [
                outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"]
                - base[row["pair_id"]]
                for row in rows
            ]
        )
        transported = slope * source_residuals + intercept
        errors = target_residuals - transported
        deltas = np.asarray(
            [
                outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"]
                - row["source_formation_energy_ev_per_atom"]
                for row in rows
            ]
        )
        features = np.vstack([basis[row["pair_id"]] for row in rows])
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
        precision = prior_precision * np.eye(features.shape[1])
        precision += features.T @ features / observation_variance
        eta = features.T @ transported / observation_variance
        state_means = features @ np.linalg.solve(precision, eta)
        projection_error = state_means - transported
        paired_error = transported - target_residuals
        system_reports[system] = {
            "stratum": stratum_by_system[system],
            "pair_count": len(rows),
            "transport_signed_error": _summary(errors.tolist()),
            "target_minus_source_formation_energy": _summary(deltas.tolist()),
            "certificate_coverage": float(np.mean(np.abs(errors) <= radius)),
            "rank16_projection_error_to_transported_residual": _summary(
                projection_error.tolist()
            ),
            "paired_transport_error_to_target_residual": _summary(
                paired_error.tolist()
            ),
            "initial_stable_fraction": float(
                np.mean(
                    [
                        outcomes[row["pair_id"]]["initial_e_above_hull_ev_per_atom"]
                        <= 1e-8
                        for row in rows
                    ]
                )
            ),
        }
        all_errors.extend(errors.tolist())
        all_deltas.extend(deltas.tolist())
        repeated_system_means.extend([float(deltas.mean())] * len(deltas))
        projection_errors.extend(projection_error.tolist())
        paired_target_errors.extend(paired_error.tolist())
        structure_rms.extend(float(row["structure_match_rms"]) for row in rows)
        absolute_transport_errors.extend(np.abs(errors).tolist())
    total_delta_variance = float(np.var(all_deltas))
    between_system_variance = float(np.var(repeated_system_means))
    system_shift_fraction = (
        between_system_variance / total_delta_variance
        if total_delta_variance > 0
        else math.nan
    )
    rms_correlation = float(
        np.corrcoef(np.asarray(structure_rms), np.asarray(absolute_transport_errors))[0, 1]
    )
    report = {
        "schema_version": 1,
        "scope": "post-pilot_failure_attribution_no_parameter_refit",
        "task_manifest_sha256": _sha256(args.task),
        "calibration_freeze_sha256": _sha256(args.freeze),
        "pilot_result_sha256": _sha256(args.result),
        "implementation_gates_all_passed": bool(result["hard_gates_passed"]),
        "global_transport_radius_ev_per_atom": radius,
        "evaluation_transport_error": _summary(all_errors),
        "evaluation_certificate_coverage": float(
            np.mean(np.abs(np.asarray(all_errors)) <= radius)
        ),
        "target_minus_source_formation_energy": _summary(all_deltas),
        "between_exact_system_fraction_of_reference_shift_variance": system_shift_fraction,
        "structure_match_rms_vs_absolute_transport_error_correlation": rms_correlation,
        "rank16_projection_error_to_transported_residual": _summary(projection_errors),
        "paired_transport_error_to_target_residual": _summary(paired_target_errors),
        "system_reports": system_reports,
        "attribution": {
            "code_degeneracy_detected": False,
            "data_pairing_failure_detected": False,
            "protocol_model_failure": (
                "global affine transport omits composition/element-dependent reference shifts; "
                "the disjoint-system certificate does not generalize"
            ),
            "representation_failure": (
                "the rank-16 global basis smooths away useful same-item low-fidelity signal"
            ),
            "next_version_constraint": (
                "evaluation systems are now development-only; any composition-aware transport "
                "and hull-certificate method requires fresh hash-selected systems"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("diagnostic output already exists")
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"implementation_gates_all_passed={report['implementation_gates_all_passed']}")
    print(f"evaluation_certificate_coverage={report['evaluation_certificate_coverage']:.9f}")
    print(f"between_system_shift_fraction={system_shift_fraction:.9f}")
    print(
        "rank16_projection_rmse="
        f"{report['rank16_projection_error_to_transported_residual']['rmse']:.9f}"
    )
    print(f"output={args.output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    analyze(parser.parse_args())


if __name__ == "__main__":
    main()
