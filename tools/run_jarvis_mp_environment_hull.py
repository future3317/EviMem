"""Calibrate the environment-conditional all-outcome robust-hull method.

The calibration command accepts only a calibration-only oracle vault.  It has
no evaluation-vault argument, so a failed gate cannot accidentally open fresh
evaluation outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneGroupOut

from matmem import (
    ActionValueInterval,
    AllOutcomeTargetCorrectionState,
    EnvironmentConditionalProtocolTransportMap,
    EnvironmentTransportStatus,
    MatchedEnvironmentEnergyPair,
    PhaseEnergyInterval,
    ProtocolCertificate,
    RobustHullDecisionCertifier,
    RobustHullDecisionKind,
    certify_epsilon_optimal_actions,
    clustered_conformal_quantile,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _element_fractions(composition: dict[str, float]) -> dict[str, float]:
    values = {element: float(amount) for element, amount in composition.items()}
    total = sum(values.values())
    if total <= 0:
        raise ValueError("composition must have positive mass")
    return {element: amount / total for element, amount in values.items()}


def _partition_calibration_systems(
    systems: set[str], release_id: str
) -> tuple[set[str], set[str], set[str]]:
    partitions = (set(), set(), set())
    for system in systems:
        residue = int(
            _stable_hash(release_id, "environment-hull-v4", system), 16
        ) % 3
        partitions[residue].add(system)
    if any(not partition for partition in partitions):
        raise ValueError("environment transport calibration partition is empty")
    if set.union(*partitions) != systems or any(
        left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise ValueError("environment transport calibration partitions are not exact")
    return partitions


class CalibrationOnlyVault:
    """A vault type that structurally rejects evaluation rows."""

    def __init__(self, payload: dict[str, Any]) -> None:
        rows = payload["target_outcomes"]
        if any(row["split"] != "calibration" for row in rows):
            raise ValueError("calibration runner received an evaluation oracle row")
        self.rows = {row["pair_id"]: row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("calibration-only oracle pair IDs are not unique")


def _pair(
    row: dict[str, Any], outcome: dict[str, Any]
) -> MatchedEnvironmentEnergyPair:
    return MatchedEnvironmentEnergyPair(
        exact_calculation_id=f"{row['jarvis_id']}->{row['mp_entry_id']}",
        canonical_structure_id=row["canonical_structure_id"],
        chemical_system=row["chemical_system"],
        element_fractions=_element_fractions(row["composition"]),
        source_descriptor=tuple(row["source_environment_embedding"]),
        source_energy_ev_per_atom=row["source_formation_energy_ev_per_atom"],
        target_energy_ev_per_atom=outcome["target_formation_energy_ev_per_atom"],
    )


def _new_state(
    transport: EnvironmentConditionalProtocolTransportMap,
) -> AllOutcomeTargetCorrectionState:
    return AllOutcomeTargetCorrectionState(
        feature_mean=transport.feature_mean[1:],
        feature_scale=transport.feature_scale[1:],
        ridge_penalty=transport.ridge_penalty,
        residual_variance_ev2_per_atom2=transport.fit_residual_variance_ev2_per_atom2,
    )


def _base_and_scale(
    row: dict[str, Any],
    transport: EnvironmentConditionalProtocolTransportMap,
    state: AllOutcomeTargetCorrectionState,
) -> tuple[float, float, bool]:
    embedding = tuple(row["source_environment_embedding"])
    transported = transport.predict(
        row["source_formation_energy_ev_per_atom"],
        embedding,
        _element_fractions(row["composition"]),
    )
    correction_mean, correction_std = state.predict(embedding)
    correction_leverage = correction_std / math.sqrt(
        transport.fit_residual_variance_ev2_per_atom2
    )
    if transported.status is not EnvironmentTransportStatus.CERTIFIED:
        return correction_mean, correction_leverage, False
    assert transported.target_energy_ev_per_atom is not None
    assert transported.leverage_scale is not None
    combined_scale = math.sqrt(
        transported.leverage_scale**2 + correction_leverage**2
    )
    return transported.target_energy_ev_per_atom + correction_mean, combined_scale, True


def _fixed_transport_base(
    row: dict[str, Any], transport: EnvironmentConditionalProtocolTransportMap
) -> tuple[float, bool]:
    transported = transport.predict(
        row["source_formation_energy_ev_per_atom"],
        tuple(row["source_environment_embedding"]),
        _element_fractions(row["composition"]),
    )
    if transported.status is not EnvironmentTransportStatus.CERTIFIED:
        return 0.0, False
    assert transported.target_energy_ev_per_atom is not None
    return transported.target_energy_ev_per_atom, True


def _ordered_rows(rows: list[dict[str, Any]], release_id: str) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: _stable_hash(release_id, "environment-hull-reveal", row["pair_id"]),
    )


def _trajectory_score(
    rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    transport: EnvironmentConditionalProtocolTransportMap,
    release_id: str,
) -> tuple[float | None, dict[str, Any]]:
    ordered = _ordered_rows(rows, release_id)
    budget = len(ordered) // 2
    state = _new_state(transport)
    history: list[tuple[tuple[float, ...], float]] = []
    normalized_errors: list[float] = []
    parity_rounds = 0
    for round_index in range(budget):
        remaining = ordered[round_index:]
        for row in remaining:
            point, scale, supported = _base_and_scale(row, transport, state)
            if not supported:
                continue
            target = outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"]
            normalized_errors.append(abs(target - point) / scale)
        revealed = ordered[round_index]
        base, _ = _fixed_transport_base(revealed, transport)
        target = outcomes[revealed["pair_id"]]["target_formation_energy_ev_per_atom"]
        embedding = tuple(revealed["source_environment_embedding"])
        residual = target - base
        state.update(embedding, residual)
        history.append((embedding, residual))
        replay = _new_state(transport)
        for old_embedding, old_residual in history:
            replay.update(old_embedding, old_residual)
        if state.state_checksum() != replay.state_checksum():
            raise AssertionError("all-outcome streaming/replay checksum mismatch")
        for row in remaining[1:]:
            if _base_and_scale(row, transport, state) != _base_and_scale(
                row, transport, replay
            ):
                raise AssertionError("all-outcome streaming/replay prediction mismatch")
        parity_rounds += 1
    score = max(normalized_errors) if normalized_errors else None
    return score, {
        "rounds": budget,
        "accepted_target_outcomes": state.accepted_outcome_count,
        "parity_rounds": parity_rounds,
        "supported_prediction_count": len(normalized_errors),
    }


def _initial_phase_state(
    rows: list[dict[str, Any]],
) -> tuple[list[ComputedEntry], list[PhaseEnergyInterval]]:
    entries = [
        ComputedEntry(
            row["composition"],
            row["corrected_total_energy_ev"],
            entry_id=row["entry_id"],
        )
        for row in rows
    ]
    diagram = PhaseDiagram(entries)
    intervals = [
        PhaseEnergyInterval(
            phase_id=str(entry.entry_id),
            element_fractions=_element_fractions(entry.composition.as_dict()),
            lower_energy_ev_per_atom=float(diagram.get_form_energy_per_atom(entry)),
            upper_energy_ev_per_atom=float(diagram.get_form_energy_per_atom(entry)),
        )
        for entry in entries
    ]
    return entries, intervals


def _certificate_metrics(
    rows: list[dict[str, Any]],
    initial_phase_rows: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
    transport: EnvironmentConditionalProtocolTransportMap,
    release_id: str,
    radius: float,
) -> dict[str, Any]:
    ordered = _ordered_rows(rows, release_id)
    budget = len(ordered) // 2
    state = _new_state(transport)
    exact_entries, reference_intervals = _initial_phase_state(initial_phase_rows)
    certifier = RobustHullDecisionCertifier(stability_tolerance_ev_per_atom=0.0)
    certified = 0
    supported_count = 0
    errors = 0
    for round_index in range(budget):
        diagram = PhaseDiagram(exact_entries)
        for row in ordered[round_index:]:
            point, scale, supported = _base_and_scale(row, transport, state)
            if not supported:
                continue
            supported_count += 1
            candidate = PhaseEnergyInterval(
                phase_id=row["pair_id"],
                element_fractions=_element_fractions(row["composition"]),
                lower_energy_ev_per_atom=point - radius * scale,
                upper_energy_ev_per_atom=point + radius * scale,
            )
            decision = certifier.certify(candidate, tuple(reference_intervals))
            if decision.kind is RobustHullDecisionKind.ABSTAIN:
                continue
            certified += 1
            outcome = outcomes[row["pair_id"]]
            actual_entry = ComputedEntry(
                outcome["composition"],
                outcome["target_corrected_total_energy_ev"],
                entry_id=row["pair_id"],
            )
            actual_stable = (
                diagram.get_e_above_hull(actual_entry, allow_negative=True) <= 1e-8
            )
            predicted_stable = decision.kind is RobustHullDecisionKind.STABLE
            errors += int(actual_stable != predicted_stable)
        revealed = ordered[round_index]
        base, _ = _fixed_transport_base(revealed, transport)
        outcome = outcomes[revealed["pair_id"]]
        embedding = tuple(revealed["source_environment_embedding"])
        state.update(
            embedding,
            outcome["target_formation_energy_ev_per_atom"] - base,
        )
        revealed_entry = ComputedEntry(
            outcome["composition"],
            outcome["target_corrected_total_energy_ev"],
            entry_id=revealed["pair_id"],
        )
        exact_entries.append(revealed_entry)
        updated_diagram = PhaseDiagram(exact_entries)
        formation = float(updated_diagram.get_form_energy_per_atom(revealed_entry))
        reference_intervals.append(
            PhaseEnergyInterval(
                phase_id=revealed["pair_id"],
                element_fractions=_element_fractions(outcome["composition"]),
                lower_energy_ev_per_atom=formation,
                upper_energy_ev_per_atom=formation,
            )
        )
    return {
        "supported_decision_count": supported_count,
        "certified_decision_count": certified,
        "certified_decision_error_count": errors,
        "certified_coverage": certified / supported_count if supported_count else 0.0,
        "selective_error": errors / certified if certified else 0.0,
    }


def _bootstrap_lower_95(values: list[float], task_sha: str) -> float:
    if not values:
        raise ValueError("bootstrap requires system-level values")
    seed = int(task_sha[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    vector = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(vector), size=(5000, len(vector)))
    means = vector[indices].mean(axis=1)
    return float(np.quantile(means, 0.025, method="linear"))


def _fit_direct_target_baseline(
    rows: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    matrix = np.asarray(
        [row["source_environment_embedding"] for row in rows], dtype=float
    )
    target = np.asarray(
        [outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"] for row in rows],
        dtype=float,
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    standardized = (matrix - mean) / scale
    groups = np.asarray([row["chemical_system"] for row in rows])
    logo = LeaveOneGroupOut()
    model = RidgeCV(
        alphas=np.logspace(-6, 6, 25),
        fit_intercept=True,
        cv=logo.split(standardized, target, groups),
        scoring="neg_mean_squared_error",
    ).fit(standardized, target)
    residual = target - model.predict(standardized)
    return {
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "coefficient": np.asarray(model.coef_).tolist(),
        "intercept_ev_per_atom": float(model.intercept_),
        "ridge_penalty": float(model.alpha_),
        "fit_residual_variance_ev2_per_atom2": max(
            float(np.mean(residual**2)), np.finfo(float).eps
        ),
        "selection_rule": "leave_one_exact_system_out_log_grid_1e-6_to_1e6",
    }


def _fit_global_paired_delta_baseline(
    rows: list[dict[str, Any]], outcomes: dict[str, dict[str, Any]]
) -> dict[str, float]:
    source = np.asarray(
        [row["source_formation_energy_ev_per_atom"] for row in rows], dtype=float
    )
    target = np.asarray(
        [outcomes[row["pair_id"]]["target_formation_energy_ev_per_atom"] for row in rows],
        dtype=float,
    )
    design = np.column_stack((source, np.ones(len(source), dtype=float)))
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    return {
        "slope": float(slope),
        "intercept_ev_per_atom": float(intercept),
    }


class EvaluationOnlyVault:
    """Fresh evaluation outcomes with one-time reveal semantics."""

    def __init__(self, payload: dict[str, Any]) -> None:
        rows = payload["target_outcomes"]
        if any(row["split"] != "evaluation" for row in rows):
            raise ValueError("evaluation runner received a calibration oracle row")
        self.rows = {row["pair_id"]: row for row in rows}
        if len(self.rows) != len(rows):
            raise ValueError("evaluation oracle pair IDs are not unique")
        self.revealed: set[str] = set()

    def reveal(self, pair_id: str) -> dict[str, Any]:
        if pair_id in self.revealed:
            raise ValueError("evaluation outcome cannot be revealed twice")
        self.revealed.add(pair_id)
        return self.rows[pair_id]


def _direct_base(row: dict[str, Any], model: dict[str, Any]) -> float:
    embedding = np.asarray(row["source_environment_embedding"], dtype=float)
    standardized = (embedding - np.asarray(model["feature_mean"])) / np.asarray(
        model["feature_scale"]
    )
    return float(
        standardized @ np.asarray(model["coefficient"])
        + model["intercept_ev_per_atom"]
    )


def _new_direct_state(model: dict[str, Any]) -> AllOutcomeTargetCorrectionState:
    return AllOutcomeTargetCorrectionState(
        feature_mean=tuple(model["feature_mean"]),
        feature_scale=tuple(model["feature_scale"]),
        ridge_penalty=model["ridge_penalty"],
        residual_variance_ev2_per_atom2=model[
            "fit_residual_variance_ev2_per_atom2"
        ],
    )


def _hull_formation_energy(diagram: PhaseDiagram, composition: dict[str, float]) -> float:
    parsed = Composition(composition)
    total_per_atom = float(diagram.get_hull_energy_per_atom(parsed))
    fake = ComputedEntry(parsed, total_per_atom * parsed.num_atoms)
    return float(diagram.get_form_energy_per_atom(fake))


def _paired_bootstrap_difference(
    system_metrics: dict[str, dict[str, Any]],
    method: str,
    baseline: str,
    metric: str,
    task_sha: str,
) -> dict[str, float]:
    systems = sorted(system_metrics)
    differences = np.asarray(
        [
            system_metrics[system]["methods"][method][metric]
            - system_metrics[system]["methods"][baseline][metric]
            for system in systems
        ],
        dtype=float,
    )
    seed = int(_stable_hash(task_sha, method, baseline, metric)[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(5000, len(differences)))
    draws = differences[indices].mean(axis=1)
    return {
        "mean_difference": float(differences.mean()),
        "lower_95": float(np.quantile(draws, 0.025, method="linear")),
        "upper_95": float(np.quantile(draws, 0.975, method="linear")),
    }


def _evaluate_system(
    rows: list[dict[str, Any]],
    initial_phase_rows: list[dict[str, Any]],
    vault: EvaluationOnlyVault,
    transport: EnvironmentConditionalProtocolTransportMap,
    direct_model: dict[str, Any],
    global_model: dict[str, float],
    release_id: str,
    simultaneous_radius: float,
) -> dict[str, Any]:
    ordered = _ordered_rows(rows, release_id)
    budget = len(ordered) // 2
    target_state = _new_direct_state(direct_model)
    method_state = _new_state(transport)
    target_history: list[tuple[tuple[float, ...], float]] = []
    method_history: list[tuple[tuple[float, ...], float]] = []
    exact_entries, reference_intervals = _initial_phase_state(initial_phase_rows)
    certifier = RobustHullDecisionCertifier(stability_tolerance_ev_per_atom=0.0)
    method_names = (
        "target_only",
        "naive_source_as_target",
        "global_paired_delta",
        "environment_transport_only",
        "environment_all_outcome",
    )
    absolute_errors: dict[str, list[float]] = defaultdict(list)
    hull_errors: dict[str, list[float]] = defaultdict(list)
    action_regrets: dict[str, list[float]] = defaultdict(list)
    normalized_method_errors: list[float] = []
    certified = 0
    certified_errors = 0
    supported_decisions = 0
    possible_action_hits = 0
    possible_action_rounds = 0
    replay_rounds = 0
    system_supported = set(ordered[0]["chemical_system"].split("-")) <= set(
        transport.supported_elements
    )
    for round_index in range(budget):
        diagram = PhaseDiagram(exact_entries)
        remaining = ordered[round_index:]
        predictions: dict[str, dict[str, float]] = {name: {} for name in method_names}
        actual_margin: dict[str, float] = {}
        action_intervals: list[ActionValueInterval] = []
        for row in remaining:
            pair_id = row["pair_id"]
            outcome = vault.rows[pair_id]
            embedding = tuple(row["source_environment_embedding"])
            direct = _direct_base(row, direct_model)
            target_correction, _ = target_state.predict(embedding)
            target_point = direct + target_correction
            transport_point, combined_scale, supported = _base_and_scale(
                row, transport, method_state
            )
            environment_base, environment_supported = _fixed_transport_base(row, transport)
            if not supported or not system_supported:
                transport_point = target_point
                environment_base = direct
                environment_supported = False
            points = {
                "target_only": target_point,
                "naive_source_as_target": row["source_formation_energy_ev_per_atom"],
                "global_paired_delta": (
                    global_model["slope"] * row["source_formation_energy_ev_per_atom"]
                    + global_model["intercept_ev_per_atom"]
                ),
                "environment_transport_only": environment_base,
                "environment_all_outcome": transport_point,
            }
            target = outcome["target_formation_energy_ev_per_atom"]
            hull_energy = _hull_formation_energy(diagram, outcome["composition"])
            actual_entry = ComputedEntry(
                outcome["composition"],
                outcome["target_corrected_total_energy_ev"],
                entry_id=pair_id,
            )
            margin = float(diagram.get_e_above_hull(actual_entry, allow_negative=True))
            actual_margin[pair_id] = margin
            actual_stable = margin <= 1e-8
            for name, point in points.items():
                predictions[name][pair_id] = point
                absolute_errors[name].append(abs(point - target))
                predicted_stable = point - hull_energy <= 0
                hull_errors[name].append(float(predicted_stable != actual_stable))
            if environment_supported:
                supported_decisions += 1
                normalized_method_errors.append(abs(target - transport_point) / combined_scale)
                lower = transport_point - simultaneous_radius * combined_scale
                upper = transport_point + simultaneous_radius * combined_scale
                candidate = PhaseEnergyInterval(
                    phase_id=pair_id,
                    element_fractions=_element_fractions(outcome["composition"]),
                    lower_energy_ev_per_atom=lower,
                    upper_energy_ev_per_atom=upper,
                )
                decision = certifier.certify(candidate, tuple(reference_intervals))
                if decision.kind is not RobustHullDecisionKind.ABSTAIN:
                    certified += 1
                    certified_errors += int(
                        (decision.kind is RobustHullDecisionKind.STABLE) != actual_stable
                    )
                action_intervals.append(
                    ActionValueInterval(
                        action_id=pair_id,
                        lower_value=-(upper - hull_energy),
                        upper_value=-(lower - hull_energy),
                    )
                )
        best_actual = min(actual_margin.values())
        best_actual_id = min(
            actual_margin,
            key=lambda pair_id: (actual_margin[pair_id], pair_id),
        )
        for name in method_names:
            selected = min(
                predictions[name],
                key=lambda pair_id: (
                    predictions[name][pair_id]
                    - _hull_formation_energy(diagram, vault.rows[pair_id]["composition"]),
                    pair_id,
                ),
            )
            action_regrets[name].append(actual_margin[selected] - best_actual)
        if action_intervals:
            possible_action_rounds += 1
            action_set = certify_epsilon_optimal_actions(
                tuple(action_intervals), epsilon=0.0
            )
            possible_action_hits += int(
                best_actual_id in action_set.possible_epsilon_optimal_action_ids
            )

        revealed = remaining[0]
        outcome = vault.reveal(revealed["pair_id"])
        embedding = tuple(revealed["source_environment_embedding"])
        direct = _direct_base(revealed, direct_model)
        environment_base, environment_supported = _fixed_transport_base(
            revealed, transport
        )
        if not environment_supported or not system_supported:
            environment_base = direct
        target_residual = outcome["target_formation_energy_ev_per_atom"] - direct
        method_residual = (
            outcome["target_formation_energy_ev_per_atom"] - environment_base
        )
        target_state.update(embedding, target_residual)
        method_state.update(embedding, method_residual)
        target_history.append((embedding, target_residual))
        method_history.append((embedding, method_residual))
        target_replay = _new_direct_state(direct_model)
        method_replay = _new_state(transport)
        for old_embedding, residual in target_history:
            target_replay.update(old_embedding, residual)
        for old_embedding, residual in method_history:
            method_replay.update(old_embedding, residual)
        if target_state.state_checksum() != target_replay.state_checksum():
            raise AssertionError("target-only streaming/replay mismatch")
        if method_state.state_checksum() != method_replay.state_checksum():
            raise AssertionError("environment-state streaming/replay mismatch")
        replay_rounds += 1
        revealed_entry = ComputedEntry(
            outcome["composition"],
            outcome["target_corrected_total_energy_ev"],
            entry_id=revealed["pair_id"],
        )
        exact_entries.append(revealed_entry)
        updated_diagram = PhaseDiagram(exact_entries)
        formation = float(updated_diagram.get_form_energy_per_atom(revealed_entry))
        reference_intervals.append(
            PhaseEnergyInterval(
                phase_id=revealed["pair_id"],
                element_fractions=_element_fractions(outcome["composition"]),
                lower_energy_ev_per_atom=formation,
                upper_energy_ev_per_atom=formation,
            )
        )
    methods = {
        name: {
            "mae_ev_per_atom": float(np.mean(absolute_errors[name])),
            "hull_misclassification": float(np.mean(hull_errors[name])),
            "one_step_regret_ev_per_atom": float(np.mean(action_regrets[name])),
        }
        for name in method_names
    }
    interval_event = bool(
        normalized_method_errors
        and max(normalized_method_errors) <= simultaneous_radius + 1e-12
    )
    return {
        "chemical_system": ordered[0]["chemical_system"],
        "candidate_count": len(ordered),
        "oracle_budget": budget,
        "transport_supported": system_supported,
        "methods": methods,
        "certificate": {
            "supported_decision_count": supported_decisions,
            "certified_decision_count": certified,
            "certified_decision_error_count": certified_errors,
            "certified_coverage": certified / supported_decisions if supported_decisions else 0.0,
            "selective_error": certified_errors / certified if certified else 0.0,
            "simultaneous_interval_event": interval_event,
            "maximum_normalized_error": (
                max(normalized_method_errors) if normalized_method_errors else None
            ),
            "possible_zero_optimal_action_coverage": (
                possible_action_hits / possible_action_rounds
                if possible_action_rounds
                else 0.0
            ),
        },
        "all_outcome_count": method_state.accepted_outcome_count,
        "replay_rounds": replay_rounds,
        "replay_gate_passed": replay_rounds == budget,
        "state_size_scalars": method_state.state_size_scalars,
    }


def evaluate(
    task_path: Path,
    evaluation_vault_path: Path,
    config_path: Path,
    calibration_freeze_path: Path,
    output_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("evaluation output must remain outside Git")
    if output_path.exists():
        raise FileExistsError("environment-hull evaluation result already exists")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze = json.loads(calibration_freeze_path.read_text(encoding="utf-8"))
    task_sha = _sha256(task_path)
    if not freeze["certificate_passed"] or freeze["evaluation_results_accessed"]:
        raise ValueError("calibration freeze does not authorize fresh evaluation")
    if freeze["task_manifest_sha256"] != task_sha:
        raise ValueError("calibration freeze is not bound to the evaluation task")
    if freeze["config_sha256"] != _sha256(config_path):
        raise ValueError("evaluation config changed after calibration freeze")
    if freeze["runner_sha256"] != _sha256(Path(__file__)):
        raise ValueError("environment-hull runner changed after calibration freeze")
    if _sha256(evaluation_vault_path) != config["sealed_evaluation_vault_sha256"]:
        raise ValueError("sealed evaluation vault checksum differs from freeze")
    vault_payload = json.loads(evaluation_vault_path.read_text(encoding="utf-8"))
    if vault_payload["task_manifest_sha256"] != task_sha:
        raise ValueError("sealed evaluation vault is not bound to the task")
    vault = EvaluationOnlyVault(vault_payload)
    rows = {row["pair_id"]: row for row in task["evaluation_pairs"]}
    if set(rows) != set(vault.rows):
        raise ValueError("evaluation task/vault pair join is not exact")
    transport = EnvironmentConditionalProtocolTransportMap.model_validate(
        freeze["transport_map"]
    )
    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        rows_by_system[row["chemical_system"]].append(row)
    radius = freeze["simultaneous_interval_calibration"]["radius"]
    system_metrics = {
        system: _evaluate_system(
            rows_by_system[system],
            task["evaluation_initial_phase_entries"][system],
            vault,
            transport,
            freeze["direct_target_baseline"],
            freeze["global_paired_delta_baseline"],
            task["release_id"],
            radius,
        )
        for system in sorted(rows_by_system)
    }
    method_names = tuple(next(iter(system_metrics.values()))["methods"])
    macro = {
        method: {
            metric: float(
                np.mean(
                    [
                        values["methods"][method][metric]
                        for values in system_metrics.values()
                    ]
                )
            )
            for metric in (
                "mae_ev_per_atom",
                "hull_misclassification",
                "one_step_regret_ev_per_atom",
            )
        }
        for method in method_names
    }
    comparisons = {
        baseline: {
            metric: _paired_bootstrap_difference(
                system_metrics,
                "environment_all_outcome",
                baseline,
                metric,
                task_sha,
            )
            for metric in ("hull_misclassification", "one_step_regret_ev_per_atom")
        }
        for baseline in ("target_only", "naive_source_as_target")
    }
    supported_systems = [
        values for values in system_metrics.values() if values["transport_supported"]
    ]
    interval_event_rate = float(
        np.mean(
            [values["certificate"]["simultaneous_interval_event"] for values in supported_systems]
        )
    )
    coverage_values = [
        values["certificate"]["certified_coverage"]
        for values in system_metrics.values()
    ]
    coverage_lower = _bootstrap_lower_95(coverage_values, task_sha)
    inlier_errors = sum(
        values["certificate"]["certified_decision_error_count"]
        for values in supported_systems
        if values["certificate"]["simultaneous_interval_event"]
    )
    strata_with_improvement: set[str] = set()
    for stratum, size_rule in {
        "binary": lambda n: n == 2,
        "ternary": lambda n: n == 3,
        "quaternary_plus": lambda n: n >= 4,
    }.items():
        systems = [
            values
            for system, values in system_metrics.items()
            if size_rule(len(system.split("-")))
        ]
        if systems and all(
            np.mean(
                [
                    values["methods"]["environment_all_outcome"][metric]
                    - values["methods"]["target_only"][metric]
                    for values in systems
                ]
            )
            < 0
            for metric in ("hull_misclassification", "one_step_regret_ev_per_atom")
        ):
            strata_with_improvement.add(stratum)
    replay_passed = all(
        values["replay_gate_passed"]
        and values["all_outcome_count"] == values["oracle_budget"]
        for values in system_metrics.values()
    )
    strict_comparison_passed = all(
        comparisons[baseline][metric]["upper_95"] < 0
        for baseline in comparisons
        for metric in comparisons[baseline]
    )
    paper_go = bool(
        replay_passed
        and interval_event_rate >= 0.9
        and coverage_lower > 0
        and inlier_errors == 0
        and strict_comparison_passed
        and len(strata_with_improvement) >= 2
    )
    result = {
        "schema_version": 1,
        "status": "paper_level_go" if paper_go else "evaluation_no_go",
        "task_manifest_sha256": task_sha,
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__)),
        "calibration_freeze_sha256": _sha256(calibration_freeze_path),
        "sealed_evaluation_vault_sha256": _sha256(evaluation_vault_path),
        "evaluation_results_accessed": True,
        "evaluation_exact_system_count": len(system_metrics),
        "evaluation_pair_count": len(rows),
        "system_metrics": system_metrics,
        "system_macro_metrics": macro,
        "paired_comparisons": comparisons,
        "supported_exact_system_count": len(supported_systems),
        "simultaneous_interval_event_rate": interval_event_rate,
        "system_bootstrap_lower_95_certified_coverage": coverage_lower,
        "inlier_certified_decision_error_count": inlier_errors,
        "improving_complexity_strata": sorted(strata_with_improvement),
        "all_outcome_replay_gate_passed": replay_passed,
        "strict_comparison_gate_passed": strict_comparison_passed,
        "paper_go": paper_go,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"paper_go={paper_go}")
    print(f"evaluation_systems={len(system_metrics)}")
    print(f"supported_systems={len(supported_systems)}")
    print(f"simultaneous_interval_event_rate={interval_event_rate:.9f}")
    print(f"certified_coverage_lower_95={coverage_lower:.9f}")
    print(f"inlier_decision_errors={inlier_errors}")
    print(f"strict_comparison_gate_passed={strict_comparison_passed}")
    print(f"evaluation_result={output_path.resolve()}")


def calibrate(
    task_path: Path,
    calibration_vault_path: Path,
    config_path: Path,
    output_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("calibration output must remain outside Git")
    if output_path.exists():
        raise FileExistsError("environment-hull calibration freeze already exists")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    vault_payload = json.loads(calibration_vault_path.read_text(encoding="utf-8"))
    task_sha = _sha256(task_path)
    if task_sha != config["task_manifest_sha256"]:
        raise ValueError("environment-hull config is not bound to the task")
    if _sha256(calibration_vault_path) != config["calibration_only_vault_sha256"]:
        raise ValueError("calibration-only vault checksum differs from freeze")
    if vault_payload["task_manifest_sha256"] != task_sha:
        raise ValueError("calibration-only vault is not bound to the task")
    vault = CalibrationOnlyVault(vault_payload)
    rows = {row["pair_id"]: row for row in task["calibration_pairs"]}
    if set(rows) != set(vault.rows):
        raise ValueError("calibration task/vault pair join is not exact")
    task_systems = {
        system
        for values in task["selection"]["calibration_systems"].values()
        for system in values
    }
    fit_systems, radius_systems, decision_systems = _partition_calibration_systems(
        task_systems, task["release_id"]
    )

    def pairs(systems: set[str]) -> list[MatchedEnvironmentEnergyPair]:
        return [
            _pair(row, vault.rows[pair_id])
            for pair_id, row in sorted(rows.items())
            if row["chemical_system"] in systems
        ]

    source_protocol = ProtocolCertificate.model_validate(task["source_protocol"])
    target_protocol = ProtocolCertificate.model_validate(task["target_protocol"])
    transport = EnvironmentConditionalProtocolTransportMap.fit_same_candidate_system_split(
        source_protocol,
        target_protocol,
        pairs(fit_systems),
        pairs(radius_systems),
        calibration_id="jarvis-mp-chgnet-environment-transport-v4",
        alpha=config["environment_transport"]["alpha"],
        ridge_penalty=None,
        held_out_canonical_structure_ids=tuple(
            row["canonical_structure_id"] for row in task["evaluation_pairs"]
        ),
    )
    fit_rows = [row for row in rows.values() if row["chemical_system"] in fit_systems]
    direct_target_baseline = _fit_direct_target_baseline(fit_rows, vault.rows)
    global_delta_baseline = _fit_global_paired_delta_baseline(fit_rows, vault.rows)
    rows_by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        rows_by_system[row["chemical_system"]].append(row)
    trajectory_scores: dict[str, float] = {}
    trajectory_audit: dict[str, dict[str, Any]] = {}
    for system in sorted(decision_systems):
        score, audit = _trajectory_score(
            rows_by_system[system], vault.rows, transport, task["release_id"]
        )
        if score is not None:
            trajectory_scores[system] = score
        trajectory_audit[system] = audit
    calibration = clustered_conformal_quantile(
        tuple(trajectory_scores.values()),
        alpha=config["simultaneous_decision_interval"]["clustered_conformal_alpha"],
    )
    phase_rows = task["calibration_initial_phase_entries"]
    system_metrics: dict[str, dict[str, Any]] = {}
    for system in sorted(decision_systems):
        system_metrics[system] = _certificate_metrics(
            rows_by_system[system],
            phase_rows[system],
            vault.rows,
            transport,
            task["release_id"],
            calibration.radius,
        )
    inlier_systems = {
        system
        for system, score in trajectory_scores.items()
        if score <= calibration.radius + 1e-12
    }
    inlier_errors = sum(
        system_metrics[system]["certified_decision_error_count"]
        for system in inlier_systems
    )
    coverage_values = [
        system_metrics[system]["certified_coverage"]
        for system in sorted(decision_systems)
    ]
    coverage_lower = _bootstrap_lower_95(coverage_values, task_sha)
    all_outcomes = all(
        audit["accepted_target_outcomes"] == audit["rounds"]
        and audit["parity_rounds"] == audit["rounds"]
        for audit in trajectory_audit.values()
    )
    certificate_passed = bool(
        math.isfinite(calibration.radius)
        and all_outcomes
        and inlier_errors == 0
        and coverage_lower > 0
    )
    freeze = {
        "schema_version": 1,
        "status": (
            "decision_certificate_calibration_passed_evaluation_still_sealed"
            if certificate_passed
            else "decision_certificate_calibration_failed_evaluation_forbidden"
        ),
        "task_manifest_sha256": task_sha,
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__)),
        "calibration_only_vault_sha256": _sha256(calibration_vault_path),
        "sealed_evaluation_vault_sha256": config["sealed_evaluation_vault_sha256"],
        "evaluation_results_accessed": False,
        "partitions": {
            "transport_fit_exact_systems": sorted(fit_systems),
            "transport_radius_exact_systems": sorted(radius_systems),
            "decision_trajectory_calibration_exact_systems": sorted(decision_systems),
        },
        "transport_map": transport.model_dump(mode="json"),
        "direct_target_baseline": direct_target_baseline,
        "global_paired_delta_baseline": global_delta_baseline,
        "trajectory_cluster_scores": trajectory_scores,
        "trajectory_audit": trajectory_audit,
        "simultaneous_interval_calibration": calibration.model_dump(mode="json"),
        "system_certificate_metrics": system_metrics,
        "inlier_exact_system_count": len(inlier_systems),
        "unsupported_decision_calibration_exact_systems": sorted(
            decision_systems - set(trajectory_scores)
        ),
        "inlier_certified_decision_error_count": inlier_errors,
        "system_macro_certified_coverage": float(np.mean(coverage_values)),
        "system_bootstrap_lower_95_certified_coverage": coverage_lower,
        "all_outcome_replay_gate_passed": all_outcomes,
        "certificate_passed": certificate_passed,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"certificate_passed={certificate_passed}")
    print(f"transport_fit_systems={len(fit_systems)}")
    print(f"transport_radius_systems={len(radius_systems)}")
    print(f"decision_calibration_systems={len(decision_systems)}")
    print(f"simultaneous_normalized_radius={calibration.radius:.9f}")
    print(f"macro_certified_coverage={np.mean(coverage_values):.9f}")
    print(f"coverage_lower_95={coverage_lower:.9f}")
    print(f"inlier_decision_errors={inlier_errors}")
    print(f"calibration_freeze={output_path.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    calibration = subparsers.add_parser("calibrate")
    calibration.add_argument("--task", type=Path, required=True)
    calibration.add_argument("--calibration-vault", type=Path, required=True)
    calibration.add_argument("--config", type=Path, required=True)
    calibration.add_argument("--output", type=Path, required=True)
    evaluation = subparsers.add_parser("evaluate")
    evaluation.add_argument("--task", type=Path, required=True)
    evaluation.add_argument("--evaluation-vault", type=Path, required=True)
    evaluation.add_argument("--config", type=Path, required=True)
    evaluation.add_argument("--calibration-freeze", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "calibrate":
        calibrate(args.task, args.calibration_vault, args.config, args.output)
    elif args.command == "evaluate":
        evaluate(
            args.task,
            args.evaluation_vault,
            args.config,
            args.calibration_freeze,
            args.output,
        )


if __name__ == "__main__":
    main()
