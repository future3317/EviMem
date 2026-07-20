"""Calibrate and evaluate the frozen JARVIS--MP protocol-activation pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from scipy.special import ndtr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from matmem import (
    AllOutcomeLinearGaussianState,
    CompositionAwareProtocolTransportMap,
    HullSnapshot,
    MatchedEnergyPair,
    MatchedResidualPair,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    ProtocolCertificate,
    ProtocolCompatibilityResolver,
    ProtocolTransportMap,
    SourceProvenance,
    StructureArtifactIdentity,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _require_external(path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if path.resolve().is_relative_to(repo_root):
        raise ValueError("calibration and experiment outputs must remain outside Git")


class EvaluationOracleVault:
    """Target outcomes with explicit calibration/reveal/evaluator boundaries."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._rows = {row["pair_id"]: row for row in payload["target_outcomes"]}
        if len(self._rows) != len(payload["target_outcomes"]):
            raise ValueError("oracle vault pair IDs are not unique")
        self._revealed: set[str] = set()

    def calibration_rows(self, systems: set[str]) -> dict[str, dict[str, Any]]:
        rows = {
            pair_id: row
            for pair_id, row in self._rows.items()
            if row["split"] == "calibration" and row["chemical_system"] in systems
        }
        if not rows:
            raise ValueError("requested calibration partition has no outcomes")
        return rows

    def reveal(self, pair_id: str) -> dict[str, Any]:
        row = self._rows[pair_id]
        if row["split"] != "evaluation":
            raise ValueError("reveal boundary accepts evaluation outcomes only")
        if pair_id in self._revealed:
            raise ValueError("evaluation outcome cannot be revealed twice")
        self._revealed.add(pair_id)
        return row

    def evaluate(self, pair_ids: tuple[str, ...]) -> dict[str, dict[str, Any]]:
        if any(self._rows[pair_id]["split"] != "evaluation" for pair_id in pair_ids):
            raise ValueError("evaluator accepts evaluation outcomes only")
        return {pair_id: self._rows[pair_id] for pair_id in pair_ids}


def _load_bound_inputs(
    task_path: Path,
    vault_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], EvaluationOracleVault, dict[str, Any]]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault_payload = json.loads(vault_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_hash = _sha256(task_path)
    if task_hash != config["task_manifest_sha256"]:
        raise ValueError("pilot config is not bound to this task manifest")
    if vault_payload["task_manifest_sha256"] != task_hash:
        raise ValueError("oracle vault is not bound to this task manifest")
    if task["release_id"] != config["task_release_id"]:
        raise ValueError("task release differs from frozen pilot config")
    calibration_systems = {
        system
        for values in task["selection"]["calibration_systems"].values()
        for system in values
    }
    evaluation_systems = {
        system
        for values in task["selection"]["evaluation_systems"].values()
        for system in values
    }
    configured_calibration = {
        system
        for values in config["calibration_partition"].values()
        for system in values
    }
    if configured_calibration != calibration_systems:
        raise ValueError("calibration subpartitions do not exactly cover frozen systems")
    if calibration_systems & evaluation_systems:
        raise ValueError("calibration and evaluation exact systems overlap")
    return task, EvaluationOracleVault(vault_payload), config


def _fit_observable_basis(
    task: dict[str, Any], config: dict[str, Any]
) -> tuple[StandardScaler, PCA, dict[str, np.ndarray]]:
    rows = task["calibration_pairs"] + task["evaluation_pairs"]
    pair_ids = [row["pair_id"] for row in rows]
    matrix = np.asarray([row["source_descriptor"] for row in rows], dtype=np.float64)
    expected_dimension = config["observable_basis"]["descriptor_dimension"]
    if matrix.shape != (len(rows), expected_dimension):
        raise ValueError("task descriptor matrix differs from frozen basis configuration")
    scaler = StandardScaler().fit(matrix)
    standardized = scaler.transform(matrix)
    pca = PCA(
        n_components=config["observable_basis"]["pca_rank"],
        svd_solver=config["observable_basis"]["pca_solver"],
    ).fit(standardized)
    reduced = pca.transform(standardized)
    if config["observable_basis"]["append_constant_feature"]:
        reduced = np.column_stack((reduced, np.ones(len(reduced), dtype=np.float64)))
    return scaler, pca, dict(zip(pair_ids, reduced, strict=True))


def _basis_from_freeze(
    task: dict[str, Any], freeze: dict[str, Any]
) -> dict[str, np.ndarray]:
    rows = task["calibration_pairs"] + task["evaluation_pairs"]
    matrix = np.asarray([row["source_descriptor"] for row in rows], dtype=np.float64)
    basis = freeze["observable_basis"]
    standardized = (matrix - np.asarray(basis["scaler_mean"])) / np.asarray(
        basis["scaler_scale"]
    )
    reduced = (standardized - np.asarray(basis["pca_mean"])) @ np.asarray(
        basis["pca_components"]
    ).T
    if basis["append_constant_feature"]:
        reduced = np.column_stack((reduced, np.ones(len(reduced), dtype=np.float64)))
    return dict(zip((row["pair_id"] for row in rows), reduced, strict=True))


def _base_prediction(
    basis_by_pair: dict[str, np.ndarray], coefficient: np.ndarray, intercept: float
) -> dict[str, float]:
    return {
        pair_id: float(feature @ coefficient + intercept)
        for pair_id, feature in basis_by_pair.items()
    }


def calibrate(
    task_path: Path,
    vault_path: Path,
    config_path: Path,
    output_path: Path,
) -> None:
    _require_external(output_path)
    if output_path.exists():
        raise FileExistsError("calibration freeze already exists; never overwrite it")
    task, vault, config = _load_bound_inputs(task_path, vault_path, config_path)
    scaler, pca, basis_by_pair = _fit_observable_basis(task, config)
    calibration_rows = {row["pair_id"]: row for row in task["calibration_pairs"]}
    partitions = config["calibration_partition"]
    base_systems = set(partitions["base_fit_exact_systems"])
    transport_fit_systems = set(partitions["transport_fit_exact_systems"])
    radius_systems = set(partitions["transport_radius_exact_systems"])
    all_calibration_systems = base_systems | transport_fit_systems | radius_systems
    if base_systems & transport_fit_systems or base_systems & radius_systems or transport_fit_systems & radius_systems:
        raise ValueError("base, transport-fit, and radius systems must be disjoint")
    target_outcomes = vault.calibration_rows(all_calibration_systems)
    base_ids = sorted(
        pair_id
        for pair_id, row in calibration_rows.items()
        if row["chemical_system"] in base_systems
    )
    ridge_config = config["frozen_base_predictor"]
    ridge = Ridge(
        alpha=ridge_config["alpha"],
        fit_intercept=ridge_config["fit_intercept"],
    ).fit(
        np.vstack([basis_by_pair[pair_id] for pair_id in base_ids]),
        np.asarray(
            [
                target_outcomes[pair_id]["target_formation_energy_ev_per_atom"]
                for pair_id in base_ids
            ]
        ),
    )
    base_by_pair = _base_prediction(
        basis_by_pair, np.asarray(ridge.coef_), float(ridge.intercept_)
    )

    def matched_pairs(systems: set[str]) -> list[MatchedResidualPair]:
        return [
            MatchedResidualPair(
                exact_calculation_id=(
                    f"{row['jarvis_id']}->{row['mp_entry_id']}"
                ),
                canonical_structure_id=row["canonical_structure_id"],
                source_residual_ev_per_atom=(
                    row["source_formation_energy_ev_per_atom"]
                    - base_by_pair[pair_id]
                ),
                target_residual_ev_per_atom=(
                    target_outcomes[pair_id]["target_formation_energy_ev_per_atom"]
                    - base_by_pair[pair_id]
                ),
            )
            for pair_id, row in sorted(calibration_rows.items())
            if row["chemical_system"] in systems
        ]

    source_protocol = ProtocolCertificate.model_validate(task["source_protocol"])
    target_protocol = ProtocolCertificate.model_validate(task["target_protocol"])
    evaluation_groups = tuple(
        row["canonical_structure_id"] for row in task["evaluation_pairs"]
    )
    transport_config = config["transport"]
    transport = ProtocolTransportMap.fit_same_structure_split(
        source_protocol,
        target_protocol,
        matched_pairs(transport_fit_systems),
        matched_pairs(radius_systems),
        calibration_id="jarvis-mp-three-way-disjoint-transport-v1",
        alpha=transport_config["alpha"],
        held_out_canonical_structure_ids=evaluation_groups,
    )
    radius_pairs = matched_pairs(radius_systems)
    radius_errors = [
        abs(
            pair.target_residual_ev_per_atom
            - transport.transport(pair.source_residual_ev_per_atom)
        )
        for pair in radius_pairs
    ]
    radius_coverage = float(
        np.mean(np.asarray(radius_errors) <= transport.error_radius_ev_per_atom)
    )
    certificate_pass = (
        math.isfinite(transport.error_radius_ev_per_atom)
        and transport.error_radius_ev_per_atom
        <= transport_config["maximum_certifiable_radius_ev_per_atom"]
    )
    freeze = {
        "schema_version": 1,
        "status": (
            "certificate_passed_evaluation_still_unopened"
            if certificate_pass
            else "certificate_failed_evaluation_forbidden"
        ),
        "task_manifest_sha256": _sha256(task_path),
        "config_sha256": _sha256(config_path),
        "evaluation_results_accessed": False,
        "observable_basis": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "pca_components": pca.components_.tolist(),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "append_constant_feature": config["observable_basis"][
                "append_constant_feature"
            ],
            "output_dimension": len(next(iter(basis_by_pair.values()))),
        },
        "base_predictor": {
            "coefficient": np.asarray(ridge.coef_).tolist(),
            "intercept": float(ridge.intercept_),
            "alpha": ridge_config["alpha"],
            "fit_pair_count": len(base_ids),
            "fit_exact_systems": sorted(base_systems),
        },
        "transport_map": transport.model_dump(mode="json"),
        "transport_fit_exact_systems": sorted(transport_fit_systems),
        "transport_radius_exact_systems": sorted(radius_systems),
        "radius_calibration_empirical_coverage": radius_coverage,
        "maximum_certifiable_radius_ev_per_atom": transport_config[
            "maximum_certifiable_radius_ev_per_atom"
        ],
        "certificate_passed": certificate_pass,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"certificate_passed={certificate_pass}")
    print(f"transport_slope={transport.slope:.9f}")
    print(f"transport_intercept={transport.intercept_ev_per_atom:.9f}")
    print(f"transport_radius={transport.error_radius_ev_per_atom:.9f}")
    print(f"radius_calibration_coverage={radius_coverage:.9f}")
    print(f"calibration_freeze={output_path.resolve()}")


def _element_fractions(row: dict[str, Any]) -> dict[str, float]:
    composition = {element: float(amount) for element, amount in row["composition"].items()}
    total = sum(composition.values())
    return {element: amount / total for element, amount in composition.items()}


def calibrate_composition(
    task_path: Path,
    vault_path: Path,
    config_path: Path,
    output_path: Path,
) -> None:
    """Freeze composition-aware transport without opening fresh evaluation outcomes."""

    _require_external(output_path)
    if output_path.exists():
        raise FileExistsError("composition transport freeze already exists")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault_payload = json.loads(vault_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_hash = _sha256(task_path)
    if task_hash != config["task_manifest_sha256"]:
        raise ValueError("composition config is not bound to the task")
    if vault_payload["task_manifest_sha256"] != task_hash:
        raise ValueError("oracle vault is not bound to the task")
    vault = EvaluationOracleVault(vault_payload)
    transport_config = config["composition_transport"]
    fit_systems = set(transport_config["fit_exact_systems"])
    radius_systems = set(transport_config["radius_exact_systems"])
    task_calibration_systems = {
        system
        for systems in task["selection"]["calibration_systems"].values()
        for system in systems
    }
    task_evaluation_systems = {
        system
        for systems in task["selection"]["evaluation_systems"].values()
        for system in systems
    }
    if fit_systems & radius_systems or fit_systems | radius_systems != task_calibration_systems:
        raise ValueError("composition fit/radius systems must exactly partition calibration")
    forbidden = set(config["forbidden_prior_evaluation_exact_systems"])
    if task_evaluation_systems & forbidden:
        raise ValueError("fresh evaluation reuses a prior opened evaluation system")
    calibration_outcomes = vault.calibration_rows(task_calibration_systems)
    rows = {row["pair_id"]: row for row in task["calibration_pairs"]}

    def pairs(systems: set[str]) -> list[MatchedEnergyPair]:
        return [
            MatchedEnergyPair(
                exact_calculation_id=f"{row['jarvis_id']}->{row['mp_entry_id']}",
                canonical_structure_id=row["canonical_structure_id"],
                chemical_system=row["chemical_system"],
                element_fractions=_element_fractions(row),
                source_energy_ev_per_atom=row[
                    "source_formation_energy_ev_per_atom"
                ],
                target_energy_ev_per_atom=calibration_outcomes[pair_id][
                    "target_formation_energy_ev_per_atom"
                ],
            )
            for pair_id, row in sorted(rows.items())
            if row["chemical_system"] in systems
        ]

    source_protocol, target_protocol = _protocols(task)
    transport = CompositionAwareProtocolTransportMap.fit_same_structure_system_split(
        source_protocol,
        target_protocol,
        pairs(fit_systems),
        pairs(radius_systems),
        calibration_id="jarvis-mp-composition-aware-clustered-v2",
        alpha=transport_config["clustered_conformal_alpha"],
        ridge_penalty=transport_config["ridge_penalty"],
        held_out_canonical_structure_ids=tuple(
            row["canonical_structure_id"] for row in task["evaluation_pairs"]
        ),
    )
    radius_errors_by_system: dict[str, list[float]] = defaultdict(list)
    for pair in pairs(radius_systems):
        prediction = transport.transport(
            pair.source_energy_ev_per_atom, pair.element_fractions
        )
        if prediction is None:
            raise ValueError("radius composition unexpectedly lacks fit support")
        radius_errors_by_system[pair.chemical_system].append(
            abs(pair.target_energy_ev_per_atom - prediction)
        )
    cluster_scores = {
        system: max(errors) for system, errors in radius_errors_by_system.items()
    }
    cluster_coverage = float(
        np.mean(
            np.asarray(list(cluster_scores.values()))
            <= transport.error_radius_ev_per_atom
        )
    )
    scaler, pca, basis_by_pair = _fit_observable_basis(task, config)
    vocabulary = set(transport.element_offset_ev_per_atom)
    supported_evaluation = sorted(
        system
        for system in task_evaluation_systems
        if set(system.split("-")) <= vocabulary
    )
    unsupported_evaluation = sorted(task_evaluation_systems - set(supported_evaluation))
    certificate_passed = (
        transport.error_radius_ev_per_atom
        <= transport_config["maximum_certifiable_radius_ev_per_atom"]
    )
    freeze = {
        "schema_version": 2,
        "status": (
            "composition_certificate_passed_fresh_evaluation_unopened"
            if certificate_passed
            else "composition_certificate_failed_evaluation_forbidden"
        ),
        "task_manifest_sha256": task_hash,
        "config_sha256": _sha256(config_path),
        "evaluation_results_accessed": False,
        "composition_transport_map": transport.model_dump(mode="json"),
        "radius_cluster_scores_ev_per_atom": cluster_scores,
        "radius_cluster_empirical_coverage": cluster_coverage,
        "supported_evaluation_exact_systems": supported_evaluation,
        "unsupported_evaluation_exact_systems": unsupported_evaluation,
        "certificate_passed": certificate_passed,
        "observable_basis": {
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "pca_mean": pca.mean_.tolist(),
            "pca_components": pca.components_.tolist(),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "append_constant_feature": config["observable_basis"][
                "append_constant_feature"
            ],
            "output_dimension": len(next(iter(basis_by_pair.values()))),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"certificate_passed={certificate_passed}")
    print(f"source_slope={transport.source_slope:.9f}")
    print(f"clustered_radius={transport.error_radius_ev_per_atom:.9f}")
    print(f"radius_cluster_coverage={cluster_coverage:.9f}")
    print(f"supported_evaluation_systems={len(supported_evaluation)}")
    print(f"unsupported_evaluation_systems={len(unsupported_evaluation)}")
    print(f"calibration_freeze={output_path.resolve()}")


def _protocols(task: dict[str, Any]) -> tuple[ProtocolCertificate, ProtocolCertificate]:
    return (
        ProtocolCertificate.model_validate(task["source_protocol"]),
        ProtocolCertificate.model_validate(task["target_protocol"]),
    )


def _hull_formation_energy(diagram: PhaseDiagram, composition: Composition) -> float:
    total_per_atom = float(diagram.get_hull_energy_per_atom(composition))
    fake = ComputedEntry(composition, total_per_atom * composition.num_atoms)
    return float(diagram.get_form_energy_per_atom(fake))


def _phase_checksum(entries: list[ComputedEntry]) -> str:
    payload = "\n".join(sorted(str(entry.entry_id) for entry in entries)) + "\n"
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _snapshot(
    system: str,
    composition: Composition,
    diagram: PhaseDiagram,
    entries: list[ComputedEntry],
    round_index: int,
) -> HullSnapshot:
    timestamp = datetime(2023, 2, 8, tzinfo=UTC)
    return HullSnapshot(
        snapshot_id=f"jarvis-mp-{system}-round-{round_index}",
        chemical_system=tuple(system.split("-")),
        reference_hull_energy_ev_per_atom=_hull_formation_energy(diagram, composition),
        phase_set_checksum=_phase_checksum(entries),
        known_through=datetime(2023, 2, 7, tzinfo=UTC),
        built_at=timestamp,
        source_version="mp-cse-2023-02-07-causal-exclusion-v1",
    )


def _query(
    row: dict[str, Any],
    feature: np.ndarray,
    base_prediction: float,
    target_protocol: ProtocolCertificate,
    snapshot: HullSnapshot,
) -> MaterialQuery:
    pair_id = row["pair_id"]
    structure_hash = row["source_structure_sha256"]
    return MaterialQuery(
        query_id=pair_id,
        structure_hash=structure_hash,
        structure_identity=StructureArtifactIdentity.low_fidelity_relaxed(
            pair_id, structure_hash
        ),
        identity=MaterialIdentity(
            exact_calculation_id=row["mp_entry_id"],
            canonical_structure_id=row["canonical_structure_id"],
            composition_family=row["chemical_system"],
        ),
        composition=Composition(row["composition"]).reduced_formula,
        embedding=tuple(float(value) for value in feature),
        protocol=target_protocol,
        hull_snapshot=snapshot,
        base_predicted_formation_energy_ev_per_atom=base_prediction,
    )


def _card(
    row: dict[str, Any],
    feature: np.ndarray,
    base_prediction: float,
    protocol: ProtocolCertificate,
    formation_energy: float,
    snapshot: HullSnapshot,
    *,
    source: bool,
) -> MaterialMemoryCard:
    material_id = row["jarvis_id"] if source else row["pair_id"]
    calculation_id = (
        row["jarvis_id"] if source else row["mp_entry_id"]
    )
    structure_hash = row["source_structure_sha256"]
    return MaterialMemoryCard(
        card_id=("source:" if source else "target:") + row["pair_id"],
        material_id=material_id,
        structure_hash=structure_hash,
        structure_identity=StructureArtifactIdentity.low_fidelity_relaxed(
            material_id, structure_hash
        ),
        identity=MaterialIdentity(
            exact_calculation_id=calculation_id,
            canonical_structure_id=row["canonical_structure_id"],
            composition_family=row["chemical_system"],
        ),
        composition=Composition(row["composition"]).reduced_formula,
        embedding=tuple(float(value) for value in feature),
        protocol=protocol,
        provenance=SourceProvenance(
            source_name="JARVIS-DFT" if source else "Materials Project",
            source_version="2022-12-12" if source else "2023-02-07",
            record_locator=calculation_id,
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
        ),
        formation_energy_ev_per_atom=formation_energy,
        base_predicted_formation_energy_ev_per_atom=base_prediction,
        oracle_residual_ev_per_atom=formation_energy - base_prediction,
        hull_snapshot=snapshot,
        observed_at=datetime(2026, 7, 20, tzinfo=UTC),
    )


def _new_state(
    resolver: ProtocolCompatibilityResolver,
    target_protocol: ProtocolCertificate,
    config: dict[str, Any],
    dimension: int,
) -> AllOutcomeLinearGaussianState:
    state_config = config["all_outcome_state"]
    return AllOutcomeLinearGaussianState(
        resolver,
        target_protocol,
        feature_dimension=dimension,
        prior_std_ev_per_atom=state_config["prior_std_ev_per_atom"],
        observation_noise_std_ev_per_atom=state_config[
            "observation_noise_std_ev_per_atom"
        ],
    )


def _expected_positive_part(mean: float, std: float, margin: float) -> float:
    z = (margin - mean) / std
    return float((margin - mean) * ndtr(z) + std * math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi))


def _gaussian_crps(mean: float, std: float, target: float) -> float:
    z = (target - mean) / std
    return float(
        std
        * (
            z * (2 * ndtr(z) - 1)
            + 2 * math.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
            - 1 / math.sqrt(math.pi)
        )
    )


def _prediction_rows(
    queries: tuple[MaterialQuery, ...],
    state: AllOutcomeLinearGaussianState,
) -> dict[str, dict[str, float]]:
    prediction = state.predict(queries)
    return {
        query.query_id: {
            "residual_mean": prediction.mean_ev_per_atom[index],
            "residual_std": prediction.std_ev_per_atom[index],
            "stable_probability": prediction.stable_probability[index],
        }
        for index, query in enumerate(queries)
    }


def _prediction_equal(
    left: dict[str, dict[str, float]], right: dict[str, dict[str, float]]
) -> bool:
    return left == right


def _metric_summary(values: dict[str, list[float]]) -> dict[str, float]:
    return {
        key: float(np.mean(items)) if items else math.nan
        for key, items in sorted(values.items())
    }


def _bootstrap_difference(
    system_metrics: dict[str, dict[str, dict[str, float]]],
    left: str,
    right: str,
    metric: str,
    *,
    seed: int,
    replicates: int,
) -> dict[str, float]:
    systems = sorted(system_metrics)
    differences = np.asarray(
        [
            system_metrics[system][left][metric]
            - system_metrics[system][right][metric]
            for system in systems
        ]
    )
    generator = np.random.default_rng(seed)
    draws = generator.choice(differences, size=(replicates, len(differences)), replace=True).mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def evaluate(
    task_path: Path,
    vault_path: Path,
    config_path: Path,
    freeze_path: Path,
    output_path: Path,
) -> None:
    _require_external(output_path)
    if output_path.exists():
        raise FileExistsError("pilot result already exists; never overwrite it")
    task, vault, config = _load_bound_inputs(task_path, vault_path, config_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["task_manifest_sha256"] != _sha256(task_path):
        raise ValueError("calibration freeze is not bound to the task")
    if freeze["config_sha256"] != _sha256(config_path):
        raise ValueError("calibration freeze is not bound to the config")
    if not freeze["certificate_passed"]:
        raise ValueError("transport certificate failed; evaluation is forbidden")
    basis_by_pair = _basis_from_freeze(task, freeze)
    coefficient = np.asarray(freeze["base_predictor"]["coefficient"])
    base_by_pair = _base_prediction(
        basis_by_pair, coefficient, float(freeze["base_predictor"]["intercept"])
    )
    source_protocol, target_protocol = _protocols(task)
    transport = ProtocolTransportMap.model_validate(freeze["transport_map"])
    certified_resolver = ProtocolCompatibilityResolver([transport])
    rejecting_resolver = ProtocolCompatibilityResolver()
    evaluation_rows = {row["pair_id"]: row for row in task["evaluation_pairs"]}
    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in task["evaluation_pairs"]:
        rows_by_system[row["chemical_system"]].append(row)
    evaluation_config = config["evaluation"]
    methods = (
        "target_only",
        "protocol_rejection",
        "naive_full_history_pooling",
        "paired_affine_multifidelity",
        "certified_all_outcome_state",
    )
    system_metrics: dict[str, dict[str, dict[str, float]]] = {}
    parity = {
        "target_rejection_exact": True,
        "certified_persistent_replay_exact": True,
        "homogeneous_null_persistent_replay_exact": True,
        "all_outcome_count_exact": True,
    }
    runtime: dict[str, float] = defaultdict(float)
    action_records: list[dict[str, Any]] = []

    for system in sorted(rows_by_system):
        rows = sorted(
            rows_by_system[system],
            key=lambda row: _stable_hash(
                task["release_id"], "reveal", row["pair_id"]
            ),
        )
        row_by_id = {row["pair_id"]: row for row in rows}
        initial_rows = task["evaluation_initial_phase_entries"][system]
        phase_entries = [
            ComputedEntry(
                row["composition"],
                row["corrected_total_energy_ev"],
                entry_id=row["entry_id"],
            )
            for row in initial_rows
        ]
        diagram = PhaseDiagram(phase_entries)
        initial_snapshots = {
            row["pair_id"]: _snapshot(
                system,
                Composition(row["composition"]),
                diagram,
                phase_entries,
                0,
            )
            for row in rows
        }
        source_cards = tuple(
            _card(
                row,
                basis_by_pair[row["pair_id"]],
                base_by_pair[row["pair_id"]],
                source_protocol,
                row["source_formation_energy_ev_per_atom"],
                initial_snapshots[row["pair_id"]],
                source=True,
            )
            for row in rows
        )
        relabeled_source_cards = tuple(
            card.model_copy(update={"protocol": target_protocol}) for card in source_cards
        )
        dimension = len(next(iter(basis_by_pair.values())))
        target_state = _new_state(
            rejecting_resolver, target_protocol, config, dimension
        )
        rejection_state = _new_state(
            rejecting_resolver, target_protocol, config, dimension
        )
        rejection_state.update_many(source_cards)
        naive_state = _new_state(
            rejecting_resolver, target_protocol, config, dimension
        )
        naive_state.update_many(relabeled_source_cards)
        certified_state = _new_state(
            certified_resolver, target_protocol, config, dimension
        )
        certified_state.update_many(source_cards)
        target_cards: list[MaterialMemoryCard] = []
        revealed_ids: list[str] = []
        metric_lists: dict[str, dict[str, list[float]]] = {
            method: defaultdict(list) for method in methods
        }
        budget = min(evaluation_config["oracle_budget"], len(rows) - 1)

        for round_index in range(budget):
            remaining = tuple(
                row["pair_id"] for row in rows if row["pair_id"] not in revealed_ids
            )
            snapshots = {
                pair_id: _snapshot(
                    system,
                    Composition(row_by_id[pair_id]["composition"]),
                    diagram,
                    phase_entries,
                    round_index,
                )
                for pair_id in remaining
            }
            queries = tuple(
                _query(
                    row_by_id[pair_id],
                    basis_by_pair[pair_id],
                    base_by_pair[pair_id],
                    target_protocol,
                    snapshots[pair_id],
                )
                for pair_id in remaining
            )
            predictions: dict[str, dict[str, dict[str, float]]] = {}
            started = time.perf_counter()
            predictions["target_only"] = _prediction_rows(queries, target_state)
            runtime["target_only_prediction_seconds"] += time.perf_counter() - started
            predictions["protocol_rejection"] = _prediction_rows(
                queries, rejection_state
            )
            predictions["naive_full_history_pooling"] = _prediction_rows(
                queries, naive_state
            )
            started = time.perf_counter()
            predictions["certified_all_outcome_state"] = _prediction_rows(
                queries, certified_state
            )
            runtime["certified_persistent_prediction_seconds"] += (
                time.perf_counter() - started
            )
            source_by_pair = {
                card.card_id.removeprefix("source:"): card for card in source_cards
            }
            paired: dict[str, dict[str, float]] = {}
            paired_std = math.sqrt(
                config["all_outcome_state"]["observation_noise_std_ev_per_atom"] ** 2
                + transport.error_radius_ev_per_atom**2
            )
            for query in queries:
                card = source_by_pair[query.query_id]
                mean = transport.transport(card.oracle_residual_ev_per_atom)
                probability = float(
                    ndtr(
                        (
                            query.stability_threshold_ev_per_atom
                            - query.base_hull_distance_ev_per_atom
                            - mean
                        )
                        / paired_std
                    )
                )
                paired[query.query_id] = {
                    "residual_mean": mean,
                    "residual_std": paired_std,
                    "stable_probability": probability,
                }
            predictions["paired_affine_multifidelity"] = paired

            replay_started = time.perf_counter()
            certified_replay = _new_state(
                certified_resolver, target_protocol, config, dimension
            )
            certified_replay.update_many((*source_cards, *target_cards))
            replay_prediction = _prediction_rows(queries, certified_replay)
            runtime["certified_full_history_replay_seconds"] += (
                time.perf_counter() - replay_started
            )
            homogeneous_replay = _new_state(
                rejecting_resolver, target_protocol, config, dimension
            )
            homogeneous_replay.update_many((*relabeled_source_cards, *target_cards))
            parity["target_rejection_exact"] &= _prediction_equal(
                predictions["target_only"], predictions["protocol_rejection"]
            )
            parity["certified_persistent_replay_exact"] &= (
                certified_state.state_checksum() == certified_replay.state_checksum()
                and _prediction_equal(
                    predictions["certified_all_outcome_state"], replay_prediction
                )
            )
            parity["homogeneous_null_persistent_replay_exact"] &= (
                naive_state.state_checksum() == homogeneous_replay.state_checksum()
                and _prediction_equal(
                    predictions["naive_full_history_pooling"],
                    _prediction_rows(queries, homogeneous_replay),
                )
            )
            parity["all_outcome_count_exact"] &= (
                target_state.accepted_outcome_count == len(target_cards)
                and rejection_state.accepted_outcome_count == len(target_cards)
                and certified_state.accepted_outcome_count
                == len(source_cards) + len(target_cards)
                and naive_state.accepted_outcome_count
                == len(source_cards) + len(target_cards)
            )

            oracle_rows = vault.evaluate(remaining)
            true_utility = {
                pair_id: max(
                    0.0,
                    evaluation_config["discovery_margin_ev_per_atom"]
                    - (
                        oracle_rows[pair_id]["target_formation_energy_ev_per_atom"]
                        - snapshots[pair_id].reference_hull_energy_ev_per_atom
                    ),
                )
                for pair_id in remaining
            }
            optimal_utility = max(true_utility.values())
            actions: dict[str, str] = {}
            for method in methods:
                action = min(
                    remaining,
                    key=lambda pair_id: (
                        -_expected_positive_part(
                            (
                                base_by_pair[pair_id]
                                + predictions[method][pair_id]["residual_mean"]
                                - snapshots[pair_id].reference_hull_energy_ev_per_atom
                            ),
                            predictions[method][pair_id]["residual_std"],
                            evaluation_config["discovery_margin_ev_per_atom"],
                        ),
                        pair_id,
                    ),
                )
                actions[method] = action
                metric_lists[method]["one_step_regret"].append(
                    optimal_utility - true_utility[action]
                )
                metric_lists[method]["epsilon_optimal_action_coverage"].append(
                    float(
                        true_utility[action]
                        >= optimal_utility
                        - evaluation_config["epsilon_optimal_action_ev_per_atom"]
                    )
                )
            if actions["target_only"] != actions["protocol_rejection"]:
                parity["target_rejection_exact"] = False
            if actions["certified_all_outcome_state"] != min(
                remaining,
                key=lambda pair_id: (
                    -_expected_positive_part(
                        base_by_pair[pair_id]
                        + replay_prediction[pair_id]["residual_mean"]
                        - snapshots[pair_id].reference_hull_energy_ev_per_atom,
                        replay_prediction[pair_id]["residual_std"],
                        evaluation_config["discovery_margin_ev_per_atom"],
                    ),
                    pair_id,
                ),
            ):
                parity["certified_persistent_replay_exact"] = False

            decision_threshold = evaluation_config["false_stable_cost"] / (
                evaluation_config["false_stable_cost"]
                + evaluation_config["false_unstable_cost"]
            )
            for pair_id in remaining:
                target_formation = oracle_rows[pair_id][
                    "target_formation_energy_ev_per_atom"
                ]
                true_e_above = (
                    target_formation
                    - snapshots[pair_id].reference_hull_energy_ev_per_atom
                )
                stable = true_e_above <= evaluation_config[
                    "stable_tolerance_ev_per_atom"
                ]
                target_residual = target_formation - base_by_pair[pair_id]
                for method in methods:
                    item = predictions[method][pair_id]
                    residual_error = item["residual_mean"] - target_residual
                    probability = min(
                        1 - 1e-12, max(1e-12, item["stable_probability"])
                    )
                    stable_decision = probability >= decision_threshold
                    cost = (
                        evaluation_config["false_stable_cost"]
                        if stable_decision and not stable
                        else evaluation_config["false_unstable_cost"]
                        if not stable_decision and stable
                        else 0.0
                    )
                    metric_lists[method]["hull_decision_cost"].append(cost)
                    metric_lists[method]["hull_decision_agreement"].append(
                        float(stable_decision == stable)
                    )
                    metric_lists[method]["brier"].append(
                        (probability - float(stable)) ** 2
                    )
                    metric_lists[method]["log_loss"].append(
                        -math.log(probability if stable else 1 - probability)
                    )
                    metric_lists[method]["residual_rmse_component"].append(
                        residual_error**2
                    )
                    metric_lists[method]["gaussian_nll"].append(
                        0.5
                        * (
                            math.log(2 * math.pi * item["residual_std"] ** 2)
                            + residual_error**2 / item["residual_std"] ** 2
                        )
                    )
                    metric_lists[method]["crps"].append(
                        _gaussian_crps(
                            item["residual_mean"],
                            item["residual_std"],
                            target_residual,
                        )
                    )
            action_records.append(
                {
                    "chemical_system": system,
                    "round": round_index,
                    "fixed_reveal_pair_id": remaining[0],
                    "method_actions": actions,
                    "remaining_count": len(remaining),
                }
            )
            reveal_id = remaining[0]
            reveal = vault.reveal(reveal_id)
            target_card = _card(
                row_by_id[reveal_id],
                basis_by_pair[reveal_id],
                base_by_pair[reveal_id],
                target_protocol,
                reveal["target_formation_energy_ev_per_atom"],
                snapshots[reveal_id],
                source=False,
            )
            target_cards.append(target_card)
            target_state.update(target_card)
            rejection_state.update(target_card)
            naive_state.update(target_card)
            certified_state.update(target_card)
            phase_entries.append(
                ComputedEntry(
                    reveal["composition"],
                    reveal["target_corrected_total_energy_ev"],
                    entry_id=reveal["mp_entry_id"],
                )
            )
            diagram = PhaseDiagram(phase_entries)
            revealed_ids.append(reveal_id)

        system_metrics[system] = {}
        for method in methods:
            summary = _metric_summary(metric_lists[method])
            summary["residual_rmse"] = math.sqrt(
                summary.pop("residual_rmse_component")
            )
            system_metrics[system][method] = summary

    all_eval_ids = tuple(sorted(evaluation_rows))
    eval_oracles = vault.evaluate(all_eval_ids)
    transport_errors = []
    for pair_id in all_eval_ids:
        source_residual = (
            evaluation_rows[pair_id]["source_formation_energy_ev_per_atom"]
            - base_by_pair[pair_id]
        )
        target_residual = (
            eval_oracles[pair_id]["target_formation_energy_ev_per_atom"]
            - base_by_pair[pair_id]
        )
        transport_errors.append(
            abs(target_residual - transport.transport(source_residual))
        )
    violation_rate = float(
        np.mean(np.asarray(transport_errors) > transport.error_radius_ev_per_atom)
    )
    macro = {
        method: {
            metric: float(
                np.mean(
                    [system_metrics[system][method][metric] for system in system_metrics]
                )
            )
            for metric in next(iter(system_metrics.values()))[method]
        }
        for method in methods
    }
    comparisons = {
        baseline: {
            metric: _bootstrap_difference(
                system_metrics,
                "certified_all_outcome_state",
                baseline,
                metric,
                seed=evaluation_config["bootstrap_seed"],
                replicates=evaluation_config["bootstrap_replicates"],
            )
            for metric in ("hull_decision_cost", "one_step_regret", "crps", "brier")
        }
        for baseline in ("target_only", "naive_full_history_pooling", "paired_affine_multifidelity")
    }
    stratum_improvements: dict[str, dict[str, float]] = {}
    for stratum, systems in task["selection"]["evaluation_systems"].items():
        stratum_improvements[stratum] = {
            baseline: float(
                np.mean(
                    [
                        system_metrics[system]["certified_all_outcome_state"][
                            "hull_decision_cost"
                        ]
                        - system_metrics[system][baseline]["hull_decision_cost"]
                        for system in systems
                    ]
                )
            )
            for baseline in ("target_only", "naive_full_history_pooling")
        }
    hard_gates_passed = all(parity.values())
    improving_strata = sum(
        all(value < 0 for value in differences.values())
        for differences in stratum_improvements.values()
    )
    pilot_go = (
        hard_gates_passed
        and macro["certified_all_outcome_state"]["hull_decision_cost"]
        < macro["target_only"]["hull_decision_cost"]
        and macro["certified_all_outcome_state"]["hull_decision_cost"]
        < macro["naive_full_history_pooling"]["hull_decision_cost"]
        and macro["certified_all_outcome_state"]["one_step_regret"]
        < macro["target_only"]["one_step_regret"]
        and macro["certified_all_outcome_state"]["one_step_regret"]
        < macro["naive_full_history_pooling"]["one_step_regret"]
        and violation_rate <= 0.15
        and improving_strata >= 2
    )
    paired_only_warning = (
        macro["paired_affine_multifidelity"]["hull_decision_cost"]
        <= macro["certified_all_outcome_state"]["hull_decision_cost"]
        and macro["paired_affine_multifidelity"]["one_step_regret"]
        <= macro["certified_all_outcome_state"]["one_step_regret"]
    )
    report = {
        "schema_version": 1,
        "scope": "exploratory_frozen_multi_protocol_mechanism_pilot_not_paper_claim",
        "task_manifest_sha256": _sha256(task_path),
        "config_sha256": _sha256(config_path),
        "calibration_freeze_sha256": _sha256(freeze_path),
        "evaluation_system_count": len(system_metrics),
        "evaluation_candidate_count": len(evaluation_rows),
        "transport_certificate": {
            "error_radius_ev_per_atom": transport.error_radius_ev_per_atom,
            "evaluation_violation_rate": violation_rate,
            "certificate_violation_gate": 0.15,
        },
        "hard_parity_gates": parity,
        "hard_gates_passed": hard_gates_passed,
        "system_metrics": system_metrics,
        "system_macro_metrics": macro,
        "paired_system_bootstrap_differences": comparisons,
        "stratum_hull_decision_cost_differences": stratum_improvements,
        "runtime_diagnostics": dict(runtime),
        "fixed_state_size_scalars": (
            len(next(iter(basis_by_pair.values()))) ** 2
            + len(next(iter(basis_by_pair.values())))
            + 4
        ),
        "action_records": action_records,
        "paired_only_warning": paired_only_warning,
        "pilot_go": pilot_go,
        "paper_claim_authorized": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"hard_gates_passed={hard_gates_passed}")
    print(f"evaluation_transport_violation_rate={violation_rate:.9f}")
    print(f"paired_only_warning={paired_only_warning}")
    print(f"pilot_go={pilot_go}")
    print(f"result={output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibrate_parser = subparsers.add_parser("calibrate")
    composition_parser = subparsers.add_parser("calibrate-composition")
    evaluate_parser = subparsers.add_parser("evaluate")
    for subparser in (calibrate_parser, composition_parser, evaluate_parser):
        subparser.add_argument("--task", type=Path, required=True)
        subparser.add_argument("--vault", type=Path, required=True)
        subparser.add_argument("--config", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    composition_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--freeze", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.task, args.vault, args.config, args.output)
    elif args.command == "calibrate-composition":
        calibrate_composition(args.task, args.vault, args.config, args.output)
    else:
        evaluate(args.task, args.vault, args.config, args.freeze, args.output)


if __name__ == "__main__":
    main()
