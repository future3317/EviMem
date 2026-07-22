"""Run an oracle-isolated MatPES protocol discovery experiment.

This is an exploratory development-system experiment, not a confirmatory gate.
Each policy's selected action is persisted and is the only target outcome that
the secure runner reveals.  Disjoint transport-fit outcomes are compiled into
a frozen model before query-system execution; query-system outcomes remain
available only to the reveal vault and the post-trace development evaluator.
"""

# NumPy/SciPy imports intentionally follow the pre-import BLAS thread limits.
# ruff: noqa: E402

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

# The phase-diagram Monte Carlo uses many small dense operations.  Allowing a
# BLAS backend to fan each one out over the whole host is slower and disruptive
# on a shared server, and can make timing irreproducible.  These variables must
# be set before importing NumPy/SciPy and are inherited by policy subprocesses.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from scipy.stats import norm

from matmem.identity import StructureArtifactIdentity
from matmem.protocol_closed_loop import (
    AppendOnlyProtocolEventLog,
    ProtocolCandidate,
    ProtocolCausalHull,
    ProtocolOracleOutcome,
    ProtocolOracleVault,
    ProtocolPolicySubprocess,
    SecureProtocolQueryRunner,
    requires_protocol_transport,
)
from matmem.protocol_knowledge_gradient import (
    FrozenProtocolRidgeTransport,
    fit_protocol_kernel_transport,
    fit_protocol_ridge_transport,
    protocol_hull_posterior_summary,
    protocol_target_energy_posterior,
)
from matmem.protocols import ProtocolCertificate

POLICIES = (
    "random",
    "source_margin",
    "source_online_offset",
    "source_online_affine",
    "ridge_margin",
    "ridge_uncertainty",
    "chic_hull_influence",
    "ridge_predicted_final_margin",
    "delta_hull_active_search",
    "source_rollout_delta_hull",
    "constrained_dual_horizon_source_rollout",
    "independent_confirmation_source_rollout",
    "conformal_source_rollout_delta_hull",
    "protocol_hull_knowledge_gradient",
    "protocol_hull_risk_reduction",
)


def _requires_protocol_transport(policy_name: str) -> bool:
    """Return whether a policy consumes the frozen cross-protocol posterior."""

    return requires_protocol_transport(policy_name)


@dataclass(frozen=True)
class ExperimentConfig:
    max_systems: int = 8
    minimum_candidates: int = 12
    maximum_budget: int = 6
    seed: int = 20270720
    ridge_penalty: float = 1.0
    prior_standard_deviation: float = 0.1
    boundary_temperature_ev_per_atom: float = 0.05
    posterior_sample_count: int = 1024
    posterior_diagnostic_sample_count: int = 0
    fantasy_count: int = 3
    conformal_threshold: float | None = None
    hull_backend: Literal["pymatgen", "fixed_composition"] = "pymatgen"
    transport_family: Literal[
        "ridge_random_intercept",
        "hierarchical_matern52_frozen_structure",
    ] = "ridge_random_intercept"
    split: Literal["development", "confirmatory"] = "development"
    transport_model_path: Path | None = None
    policies: tuple[str, ...] = POLICIES
    query_systems: tuple[str, ...] | None = None
    fit_systems: tuple[str, ...] | None = None
    crossfit_manifest_sha256: str | None = None
    crossfit_fold_index: int | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def _initial_entries(rows: list[dict[str, Any]]) -> list[ComputedEntry]:
    return [
        ComputedEntry(
            row["composition"],
            row["corrected_total_energy_ev"],
            entry_id=row["entry_id"],
        )
        for row in rows
    ]


def _candidate(
    row: dict[str, Any],
    *,
    source_protocol: ProtocolCertificate,
    target_protocol: ProtocolCertificate,
) -> ProtocolCandidate:
    pair_id = row["pair_id"]
    structure_hash = row["source_structure_sha256"]
    return ProtocolCandidate(
        pair_id=pair_id,
        source_structure_hash=structure_hash,
        source_structure_identity=StructureArtifactIdentity.initial(pair_id, structure_hash),
        chemical_system=tuple(row["chemical_system"].split("-")),
        composition=row["composition"],
        source_formation_energy_ev_per_atom=row["source_formation_energy_ev_per_atom"],
        source_environment_embedding=tuple(row["source_environment_embedding"]),
        source_local_environment_embedding=(
            None
            if row.get("source_local_environment_embedding") is None
            else tuple(row["source_local_environment_embedding"])
        ),
        source_protocol=source_protocol,
        target_protocol=target_protocol,
        oracle_cost=1.0,
    )


def _outcome(task_row: dict[str, Any], oracle_row: dict[str, Any]) -> ProtocolOracleOutcome:
    return ProtocolOracleOutcome(
        pair_id=task_row["pair_id"],
        source_structure_hash=task_row["source_structure_sha256"],
        chemical_system=tuple(task_row["chemical_system"].split("-")),
        composition=oracle_row["composition"],
        target_corrected_total_energy_ev=oracle_row["target_corrected_total_energy_ev"],
        target_formation_energy_ev_per_atom=oracle_row["target_formation_energy_ev_per_atom"],
        split=oracle_row["split"],
    )


def _evaluate_action_trace(
    *,
    selected_pair_ids: tuple[str, ...],
    candidate_rows: list[dict[str, Any]],
    outcome_rows: dict[str, dict[str, Any]],
    initial_rows: list[dict[str, Any]],
    transport_model: FrozenProtocolRidgeTransport | None,
    posterior_diagnostic_sample_count: int,
    seed: int,
) -> dict[str, Any]:
    """Open calibration outcomes only after a policy trace is complete."""

    remaining = {row["pair_id"]: row for row in candidate_rows}
    entries = _initial_entries(initial_rows)
    oracle_pool_entries = [*_initial_entries(initial_rows)]
    oracle_pool_entries.extend(
        ComputedEntry(
            outcome_rows[row["pair_id"]]["composition"],
            outcome_rows[row["pair_id"]]["target_corrected_total_energy_ev"],
            entry_id=row["pair_id"],
        )
        for row in candidate_rows
    )
    oracle_pool_diagram = PhaseDiagram(oracle_pool_entries)
    candidate_ids = {row["pair_id"] for row in candidate_rows}
    oracle_pool_stable_candidate_ids = {
        str(entry.entry_id)
        for entry in oracle_pool_diagram.stable_entries
        if str(entry.entry_id) in candidate_ids
    }
    rounds: list[dict[str, Any]] = []
    stable_discoveries = 0
    causal_discovery_ids: list[str] = []
    selected_ids: list[str] = []
    selected_rows: list[dict[str, Any]] = []
    for round_index, selected_id in enumerate(selected_pair_ids, start=1):
        diagram = PhaseDiagram(entries)
        margins: dict[str, float] = {}
        for pair_id in remaining:
            outcome = outcome_rows[pair_id]
            entry = ComputedEntry(
                outcome["composition"],
                outcome["target_corrected_total_energy_ev"],
                entry_id=pair_id,
            )
            margins[pair_id] = float(diagram.get_e_above_hull(entry, allow_negative=True))
        selected_margin = margins[selected_id]
        best_margin = min(margins.values())
        stable = selected_margin <= 1e-8
        stable_discoveries += int(stable)
        if stable:
            causal_discovery_ids.append(selected_id)
        selected_ids.append(selected_id)
        selected_rows.append(remaining[selected_id])
        rounds.append(
            {
                "round_index": round_index,
                "selected_pair_id": selected_id,
                "selected_causal_margin_ev_per_atom": selected_margin,
                "best_available_causal_margin_ev_per_atom": best_margin,
                "action_regret_ev_per_atom": selected_margin - best_margin,
                "stable_discovery": stable,
                "remaining_candidate_count": len(remaining),
            }
        )
        outcome = outcome_rows[selected_id]
        entries.append(
            ComputedEntry(
                outcome["composition"],
                outcome["target_corrected_total_energy_ev"],
                entry_id=selected_id,
            )
        )
        del remaining[selected_id]
        if transport_model is not None and remaining:
            query_rows = sorted(remaining.values(), key=lambda row: row["pair_id"])
            descriptor_dimension = len(query_rows[0]["source_environment_embedding"])
            posterior = protocol_target_energy_posterior(
                transport_model,
                query_features=np.asarray(
                    [row["source_environment_embedding"] for row in query_rows]
                ),
                query_source_energies=np.asarray(
                    [row["source_formation_energy_ev_per_atom"] for row in query_rows]
                ),
                history_features=np.asarray(
                    [row["source_environment_embedding"] for row in selected_rows]
                ).reshape(len(selected_rows), descriptor_dimension),
                history_source_energies=np.asarray(
                    [row["source_formation_energy_ev_per_atom"] for row in selected_rows]
                ),
                history_target_energies=np.asarray(
                    [
                        outcome_rows[row["pair_id"]]["target_formation_energy_ev_per_atom"]
                        for row in selected_rows
                    ]
                ),
                query_kernel_features=(
                    None
                    if transport_model.local_kernel == "independent"
                    else np.asarray(
                        [row["source_local_environment_embedding"] for row in query_rows]
                    )
                ),
                history_kernel_features=(
                    None
                    if transport_model.local_kernel == "independent"
                    else np.asarray(
                        [row["source_local_environment_embedding"] for row in selected_rows]
                    ).reshape(len(selected_rows), len(transport_model.kernel_feature_mean))
                ),
            )
            posterior_mean = np.asarray(posterior.mean, dtype=float)
            posterior_variance = np.maximum(
                np.diag(np.asarray(posterior.covariance, dtype=float)),
                1e-12,
            )
            true_target_energies = np.asarray(
                [
                    outcome_rows[row["pair_id"]]["target_formation_energy_ev_per_atom"]
                    for row in query_rows
                ],
                dtype=float,
            )
            energy_errors = posterior_mean - true_target_energies
            posterior_standard_deviation = np.sqrt(posterior_variance)
            rounds[-1]["posterior_energy_mae_ev_per_atom"] = float(np.mean(np.abs(energy_errors)))
            rounds[-1]["posterior_energy_rmse_ev_per_atom"] = float(
                np.sqrt(np.mean(energy_errors**2))
            )
            rounds[-1]["posterior_energy_gaussian_nll"] = float(
                np.mean(
                    0.5 * np.log(2.0 * math.pi * posterior_variance)
                    + 0.5 * energy_errors**2 / posterior_variance
                )
            )
            rounds[-1]["posterior_energy_90pct_coverage"] = float(
                np.mean(np.abs(energy_errors) <= norm.ppf(0.95) * posterior_standard_deviation)
            )
            if posterior_diagnostic_sample_count:
                causal_diagram = PhaseDiagram(entries)
                summary = protocol_hull_posterior_summary(
                    posterior,
                    query_compositions=tuple(row["composition"] for row in query_rows),
                    reference_compositions=tuple(entry.composition.as_dict() for entry in entries),
                    reference_energies=np.asarray(
                        [causal_diagram.get_form_energy_per_atom(entry) for entry in entries]
                    ),
                    posterior_sample_count=posterior_diagnostic_sample_count,
                    seed=seed + 1009 * round_index,
                )
                true_hull_formation_energies: list[float] = []
                for composition in summary.evaluation_compositions:
                    parsed = Composition(composition)
                    total_per_atom = oracle_pool_diagram.get_hull_energy_per_atom(parsed)
                    hull_entry = ComputedEntry(parsed, total_per_atom * parsed.num_atoms)
                    true_hull_formation_energies.append(
                        float(oracle_pool_diagram.get_form_energy_per_atom(hull_entry))
                    )
                errors = np.asarray(summary.mean_hull_energies) - np.asarray(
                    true_hull_formation_energies
                )
                rounds[-1]["posterior_hull_bayes_risk"] = summary.bayes_risk
                rounds[-1]["posterior_mean_hull_mae_ev_per_atom"] = float(np.mean(np.abs(errors)))
                rounds[-1]["posterior_mean_hull_rmse_ev_per_atom"] = float(
                    np.sqrt(np.mean(errors**2))
                )
    final_causal_diagram = PhaseDiagram(entries)
    final_causal_confirmed_ids: list[str] = []
    oracle_pool_confirmed_ids: list[str] = []
    for pair_id in selected_ids:
        outcome = outcome_rows[pair_id]
        entry = ComputedEntry(
            outcome["composition"],
            outcome["target_corrected_total_energy_ev"],
            entry_id=pair_id,
        )
        if final_causal_diagram.get_e_above_hull(entry, allow_negative=True) <= 1e-8:
            final_causal_confirmed_ids.append(pair_id)
        if oracle_pool_diagram.get_e_above_hull(entry, allow_negative=True) <= 1e-8:
            oracle_pool_confirmed_ids.append(pair_id)
    causal_set = set(causal_discovery_ids)
    final_causal_set = set(final_causal_confirmed_ids)
    oracle_pool_set = set(oracle_pool_confirmed_ids)
    if not oracle_pool_set <= final_causal_set <= causal_set:
        raise RuntimeError(
            "terminal hull metrics violated oracle-final <= final-causal <= causal-time order"
        )
    causal_count = len(causal_set)
    final_causal_count = len(final_causal_set)
    oracle_pool_count = len(oracle_pool_set)
    regrets = [row["action_regret_ev_per_atom"] for row in rounds]
    result = {
        "rounds": rounds,
        "mean_action_regret_ev_per_atom": float(np.mean(regrets)),
        "cumulative_action_regret_ev_per_atom": float(np.sum(regrets)),
        "stable_discoveries": stable_discoveries,
        "causal_discoveries": stable_discoveries,
        "causal_discovery_ids": causal_discovery_ids,
        "final_causal_confirmed_discoveries": final_causal_count,
        "final_causal_confirmed_ids": final_causal_confirmed_ids,
        "oracle_pool_confirmed_discoveries": oracle_pool_count,
        "oracle_pool_confirmed_ids": oracle_pool_confirmed_ids,
        # D, F and T are distinct estimands: an online causal declaration may
        # be revoked by a later selected phase, and a final-causal survivor may
        # be invalidated by an unqueried competitor in the complete pool.
        "within_campaign_revocations": causal_count - final_causal_count,
        "unqueried_competitor_invalidations": final_causal_count - oracle_pool_count,
        "causal_retention": (
            None if causal_count == 0 else final_causal_count / causal_count
        ),
        "oracle_validity": (
            None if final_causal_count == 0 else oracle_pool_count / final_causal_count
        ),
        "oracle_pool_available_discoveries": len(oracle_pool_stable_candidate_ids),
        "oracle_pool_discovery_ceiling": min(
            len(selected_pair_ids), len(oracle_pool_stable_candidate_ids)
        ),
        "oracle_pool_discovery_gap_to_ceiling": min(
            len(selected_pair_ids), len(oracle_pool_stable_candidate_ids)
        )
        - len(oracle_pool_confirmed_ids),
        "invalidated_causal_discoveries_by_final_causal_hull": len(
            causal_set - set(final_causal_confirmed_ids)
        ),
        "invalidated_causal_discoveries_by_oracle_pool_hull": len(
            causal_set - set(oracle_pool_confirmed_ids)
        ),
    }
    posterior_rounds = [row for row in rounds if "posterior_energy_mae_ev_per_atom" in row]
    if posterior_rounds:
        result.update(
            {
                "prequential_posterior_energy_mae_ev_per_atom": float(
                    np.mean([row["posterior_energy_mae_ev_per_atom"] for row in posterior_rounds])
                ),
                "prequential_posterior_energy_rmse_ev_per_atom": float(
                    np.mean([row["posterior_energy_rmse_ev_per_atom"] for row in posterior_rounds])
                ),
                "prequential_posterior_energy_gaussian_nll": float(
                    np.mean([row["posterior_energy_gaussian_nll"] for row in posterior_rounds])
                ),
                "prequential_posterior_energy_90pct_coverage": float(
                    np.mean([row["posterior_energy_90pct_coverage"] for row in posterior_rounds])
                ),
            }
        )
    hull_rounds = [row for row in rounds if "posterior_mean_hull_mae_ev_per_atom" in row]
    if hull_rounds:
        result.update(
            {
                "prequential_posterior_mean_hull_mae_ev_per_atom": float(
                    np.mean([row["posterior_mean_hull_mae_ev_per_atom"] for row in hull_rounds])
                ),
                "prequential_posterior_mean_hull_rmse_ev_per_atom": float(
                    np.mean([row["posterior_mean_hull_rmse_ev_per_atom"] for row in hull_rounds])
                ),
                "final_posterior_mean_hull_mae_ev_per_atom": hull_rounds[-1][
                    "posterior_mean_hull_mae_ev_per_atom"
                ],
                "final_posterior_mean_hull_rmse_ev_per_atom": hull_rounds[-1][
                    "posterior_mean_hull_rmse_ev_per_atom"
                ],
            }
        )
    return result


def fit_transport_model_for_task(
    *,
    task: dict[str, Any],
    outcome_rows: dict[str, dict[str, Any]],
    fit_systems: tuple[str, ...],
    ridge_penalty: float,
    transport_family: Literal[
        "ridge_random_intercept",
        "hierarchical_matern52_frozen_structure",
    ],
    pairs_key: str = "development_pairs",
) -> FrozenProtocolRidgeTransport:
    """Fit the frozen disjoint transport artifact used by development runs."""

    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in task[pairs_key]:
        by_system.setdefault(row["chemical_system"], []).append(row)
    fit_rows = [row for system in fit_systems for row in by_system[system]]
    fit_arguments = {
        "features": np.asarray(
            [row["source_environment_embedding"] for row in fit_rows], dtype=float
        ),
        "source_energies": np.asarray(
            [row["source_formation_energy_ev_per_atom"] for row in fit_rows]
        ),
        "target_energies": np.asarray(
            [
                outcome_rows[row["pair_id"]]["target_formation_energy_ev_per_atom"]
                for row in fit_rows
            ]
        ),
        "system_ids": [row["chemical_system"] for row in fit_rows],
        "ridge_penalty": ridge_penalty,
    }
    if transport_family == "hierarchical_matern52_frozen_structure":
        representation = task.get("local_environment_representation")
        if not isinstance(representation, dict):
            raise ValueError(
                "hierarchical frozen-structure transport requires representation metadata"
            )
        if any(row.get("source_local_environment_embedding") is None for row in fit_rows):
            raise ValueError(
                "hierarchical frozen-structure transport requires every source embedding"
            )
        return fit_protocol_kernel_transport(
            **fit_arguments,
            kernel_features=np.asarray(
                [row["source_local_environment_embedding"] for row in fit_rows], dtype=float
            ),
            kernel_feature_encoder=str(representation["encoder"]),
            kernel_feature_encoder_checksum=str(representation["checkpoint_sha256"]),
        )
    return fit_protocol_ridge_transport(**fit_arguments)


def run(
    *,
    task_path: Path,
    development_vault_path: Path,
    output_path: Path,
    config: ExperimentConfig,
) -> None:
    if output_path.exists():
        raise FileExistsError("CHIC closed-loop output already exists")
    repo_root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(repo_root):
        raise ValueError("CHIC exploratory output must remain outside Git")
    trace_dir = output_path.with_suffix("")
    if trace_dir.exists():
        raise FileExistsError("CHIC closed-loop trace directory already exists")
    trace_dir.mkdir(parents=True)

    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault_payload = json.loads(development_vault_path.read_text(encoding="utf-8"))
    expected_split = config.split
    if any(row["split"] != expected_split for row in vault_payload["target_outcomes"]):
        raise ValueError(f"runner accepts only {expected_split} outcomes")
    task_rows = task[f"{expected_split}_pairs"]
    outcome_rows = {row["pair_id"]: row for row in vault_payload["target_outcomes"]}
    if set(outcome_rows) != {row["pair_id"] for row in task_rows}:
        raise ValueError("calibration task/vault join is not exact")
    release_id = task["release_id"]
    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in task_rows:
        by_system.setdefault(row["chemical_system"], []).append(row)
    if config.query_systems is None:
        query_systems = sorted(
            (
                system
                for system, rows in by_system.items()
                if len(rows) >= config.minimum_candidates
            ),
            key=lambda system: _stable_hash(release_id, "chic-closed-loop-v1", system),
        )[: config.max_systems]
    else:
        query_systems = list(config.query_systems)
        if len(set(query_systems)) != len(query_systems):
            raise ValueError("explicit query systems must be unique")
        missing = set(query_systems) - set(by_system)
        undersized = {
            system
            for system in query_systems
            if system in by_system and len(by_system[system]) < config.minimum_candidates
        }
        if missing or undersized:
            raise ValueError(
                f"explicit query systems are unavailable: missing={sorted(missing)}, "
                f"undersized={sorted(undersized)}"
            )
    if not query_systems:
        raise ValueError(f"no eligible {expected_split} systems")

    fit_systems = (
        tuple(sorted(set(by_system) - set(query_systems)))
        if config.fit_systems is None
        else tuple(config.fit_systems)
    )
    if (
        len(set(fit_systems)) != len(fit_systems)
        or set(fit_systems) - set(by_system)
        or set(fit_systems) & set(query_systems)
    ):
        raise ValueError("explicit transport fit systems are invalid or overlap queries")
    transport_model = None
    if config.transport_model_path is not None:
        if not config.transport_model_path.exists():
            raise FileNotFoundError(config.transport_model_path)
        frozen_payload = json.loads(config.transport_model_path.read_text(encoding="utf-8"))
        transport_model = FrozenProtocolRidgeTransport.model_validate(
            frozen_payload.get("model", frozen_payload)
        )
        if set(transport_model.fit_system_ids) & set(query_systems):
            raise AssertionError("frozen transport fit and query systems overlap")
        fit_systems = transport_model.fit_system_ids
    elif expected_split == "development" and len(fit_systems) >= 2:
        transport_model = fit_transport_model_for_task(
            task=task,
            outcome_rows=outcome_rows,
            fit_systems=fit_systems,
            ridge_penalty=config.ridge_penalty,
            transport_family=config.transport_family,
            pairs_key=f"{expected_split}_pairs",
        )
        if set(transport_model.fit_system_ids) & set(query_systems):
            raise AssertionError("transport fit and query systems overlap")
    elif any(_requires_protocol_transport(policy) for policy in config.policies):
        raise ValueError("confirmatory transport policies require --transport-model")
    active_policies = tuple(
        policy
        for policy in config.policies
        if transport_model is not None or not _requires_protocol_transport(policy)
    )

    source_protocol = ProtocolCertificate.model_validate(task["source_protocol"])
    target_protocol = ProtocolCertificate.model_validate(task["target_protocol"])
    system_results: dict[str, Any] = {}
    for system in query_systems:
        rows = sorted(by_system[system], key=lambda row: row["pair_id"])
        budget = min(config.maximum_budget, len(rows) // 2)
        strategy_results: dict[str, Any] = {}
        for policy_name in active_policies:
            candidates = [
                _candidate(
                    row,
                    source_protocol=source_protocol,
                    target_protocol=target_protocol,
                )
                for row in rows
            ]
            outcomes = [_outcome(row, outcome_rows[row["pair_id"]]) for row in rows]
            policy = ProtocolPolicySubprocess(
                policy_name,
                seed=config.seed,
                ridge_penalty=config.ridge_penalty,
                prior_standard_deviation=config.prior_standard_deviation,
                boundary_temperature=config.boundary_temperature_ev_per_atom,
                transport_model=(
                    transport_model
                    if transport_model is not None and _requires_protocol_transport(policy_name)
                    else None
                ),
                posterior_sample_count=config.posterior_sample_count,
                fantasy_count=config.fantasy_count,
                conformal_threshold=config.conformal_threshold,
                hull_backend=config.hull_backend,
                selection_timeout_seconds=(
                    300.0
                    if policy_name
                    in {
                        "source_rollout_delta_hull",
                        "constrained_dual_horizon_source_rollout",
                        "independent_confirmation_source_rollout",
                        "conformal_source_rollout_delta_hull",
                        "protocol_hull_knowledge_gradient",
                        "protocol_hull_risk_reduction",
                    }
                    else 30.0
                ),
            )
            log_path = trace_dir / f"{system}__{policy_name}.jsonl"
            started = time.perf_counter()
            with AppendOnlyProtocolEventLog(log_path) as event_log:
                runner = SecureProtocolQueryRunner(
                    candidates=candidates,
                    vault=ProtocolOracleVault(outcomes, expected_split=expected_split),
                    causal_hull=ProtocolCausalHull(
                        _initial_entries(task[f"{expected_split}_initial_phase_entries"][system]),
                        chemical_system=tuple(system.split("-")),
                    ),
                    policy=policy,
                    event_log=event_log,
                )
                result = runner.run(oracle_budget=float(budget))
            elapsed = time.perf_counter() - started
            evaluation = _evaluate_action_trace(
                selected_pair_ids=result.selected_pair_ids,
                candidate_rows=rows,
                outcome_rows=outcome_rows,
                initial_rows=task[f"{expected_split}_initial_phase_entries"][system],
                transport_model=(
                    transport_model
                    if transport_model is not None
                    and set(system.split("-")) <= set(transport_model.fit_element_ids)
                    else None
                ),
                posterior_diagnostic_sample_count=(config.posterior_diagnostic_sample_count),
                seed=config.seed,
            )
            strategy_results[policy_name] = {
                "selected_pair_ids": result.selected_pair_ids,
                "policy_decision_rounds": [
                    event.model_dump(mode="json") for event in result.events
                ],
                "trace_checksum": result.trace_checksum,
                "event_log_sha256": _sha256(log_path),
                "wall_seconds": elapsed,
                **evaluation,
            }
        system_results[system] = {
            "candidate_count": len(rows),
            "budget": budget,
            "transport_element_support": (
                None
                if transport_model is None
                else set(system.split("-")) <= set(transport_model.fit_element_ids)
            ),
            "strategies": strategy_results,
        }

    aggregates: dict[str, Any] = {}
    for policy_name in active_policies:
        results = [system_results[system]["strategies"][policy_name] for system in query_systems]
        aggregates[policy_name] = {
            "system_macro_mean_action_regret_ev_per_atom": float(
                np.mean([row["mean_action_regret_ev_per_atom"] for row in results])
            ),
            "system_macro_cumulative_action_regret_ev_per_atom": float(
                np.mean([row["cumulative_action_regret_ev_per_atom"] for row in results])
            ),
            "system_macro_stable_discoveries": float(
                np.mean([row["stable_discoveries"] for row in results])
            ),
            "system_macro_final_causal_confirmed_discoveries": float(
                np.mean([row["final_causal_confirmed_discoveries"] for row in results])
            ),
            "system_macro_oracle_pool_confirmed_discoveries": float(
                np.mean([row["oracle_pool_confirmed_discoveries"] for row in results])
            ),
            "system_macro_oracle_pool_discovery_ceiling": float(
                np.mean([row["oracle_pool_discovery_ceiling"] for row in results])
            ),
            "system_macro_oracle_pool_discovery_gap_to_ceiling": float(
                np.mean([row["oracle_pool_discovery_gap_to_ceiling"] for row in results])
            ),
            "system_macro_invalidated_by_oracle_pool_hull": float(
                np.mean(
                    [row["invalidated_causal_discoveries_by_oracle_pool_hull"] for row in results]
                )
            ),
            "system_macro_wall_seconds": float(np.mean([row["wall_seconds"] for row in results])),
        }
        posterior_results = [
            row for row in results if "prequential_posterior_energy_mae_ev_per_atom" in row
        ]
        if posterior_results:
            aggregates[policy_name].update(
                {
                    "supported_system_count_for_posterior_metrics": len(posterior_results),
                    "system_macro_prequential_posterior_energy_mae_ev_per_atom": float(
                        np.mean(
                            [
                                row["prequential_posterior_energy_mae_ev_per_atom"]
                                for row in posterior_results
                            ]
                        )
                    ),
                    "system_macro_prequential_posterior_energy_rmse_ev_per_atom": float(
                        np.mean(
                            [
                                row["prequential_posterior_energy_rmse_ev_per_atom"]
                                for row in posterior_results
                            ]
                        )
                    ),
                    "system_macro_prequential_posterior_energy_gaussian_nll": float(
                        np.mean(
                            [
                                row["prequential_posterior_energy_gaussian_nll"]
                                for row in posterior_results
                            ]
                        )
                    ),
                    "system_macro_prequential_posterior_energy_90pct_coverage": float(
                        np.mean(
                            [
                                row["prequential_posterior_energy_90pct_coverage"]
                                for row in posterior_results
                            ]
                        )
                    ),
                }
            )
        hull_results = [
            row for row in results if "prequential_posterior_mean_hull_mae_ev_per_atom" in row
        ]
        if hull_results:
            aggregates[policy_name].update(
                {
                    "supported_system_count_for_hull_metrics": len(hull_results),
                    "system_macro_prequential_posterior_mean_hull_mae_ev_per_atom": float(
                        np.mean(
                            [
                                row["prequential_posterior_mean_hull_mae_ev_per_atom"]
                                for row in hull_results
                            ]
                        )
                    ),
                    "system_macro_final_posterior_mean_hull_mae_ev_per_atom": float(
                        np.mean(
                            [
                                row["final_posterior_mean_hull_mae_ev_per_atom"]
                                for row in hull_results
                            ]
                        )
                    ),
                }
            )
    config_payload = asdict(config)
    if config.transport_model_path is not None:
        config_payload["transport_model_path"] = str(config.transport_model_path)
    output = {
        "schema_version": 1,
        "status": (
            "exploratory_development_systems_only_not_confirmatory"
            if expected_split == "development"
            else "confirmatory_fresh_systems_frozen_transport"
        ),
        "estimand": "action_driven_closed_loop_query_selection",
        "task_sha256": _sha256(task_path),
        "oracle_vault_sha256": _sha256(development_vault_path),
        "development_vault_sha256": (
            _sha256(development_vault_path) if expected_split == "development" else None
        ),
        "script_sha256": _sha256(Path(__file__)),
        "code_provenance": {
            "protocol_policy_worker_sha256": _sha256(
                repo_root / "src" / "matmem" / "protocol_policy_worker.py"
            ),
            "protocol_closed_loop_sha256": _sha256(
                repo_root / "src" / "matmem" / "protocol_closed_loop.py"
            ),
            "protocol_knowledge_gradient_sha256": _sha256(
                repo_root / "src" / "matmem" / "protocol_knowledge_gradient.py"
            ),
            "frozen_structure_encoder_sha256": _sha256(
                repo_root / "src" / "matmem" / "frozen_structure_encoder.py"
            ),
        },
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "evaluation_systems_accessed": expected_split == "confirmatory",
        "transport_fit_systems": fit_systems,
        "transport_fit_system_count": len(fit_systems),
        "transport_fit_row_count": (
            0 if transport_model is None else transport_model.fit_row_count
        ),
        "transport_model_checksum": (
            None if transport_model is None else transport_model.identity_checksum
        ),
        "transport_model": (
            None if transport_model is None else transport_model.model_dump(mode="json")
        ),
        "transport_fit_and_query_systems_disjoint": transport_model is not None,
        "active_policies": active_policies,
        "selected_action_is_only_reveal": True,
        "posterior_sampler": "nested_scrambled_sobol_gaussian_v1",
        "training_archive_policy": "full_history",
        "config": config_payload,
        "development_systems": query_systems,
        "query_systems": query_systems,
        "split": expected_split,
        "transport_model_path": (
            None if config.transport_model_path is None else str(config.transport_model_path)
        ),
        "aggregates": aggregates,
        "systems": system_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output={output_path.resolve()}")
    for policy_name in active_policies:
        values = aggregates[policy_name]
        print(
            policy_name,
            f"regret={values['system_macro_mean_action_regret_ev_per_atom']:.6f}",
            f"causal={values['system_macro_stable_discoveries']:.3f}",
            f"final={values['system_macro_final_causal_confirmed_discoveries']:.3f}",
            f"oracle={values['system_macro_oracle_pool_confirmed_discoveries']:.3f}",
            f"seconds={values['system_macro_wall_seconds']:.3f}",
            (
                "hull_mae="
                f"{values['system_macro_prequential_posterior_mean_hull_mae_ev_per_atom']:.6f}"
                if "system_macro_prequential_posterior_mean_hull_mae_ev_per_atom" in values
                else "hull_mae=unsupported"
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--development-vault", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-systems", type=int, default=8)
    parser.add_argument("--minimum-candidates", type=int, default=12)
    parser.add_argument("--maximum-budget", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20270720)
    parser.add_argument("--ridge-penalty", type=float, default=1.0)
    parser.add_argument("--prior-standard-deviation", type=float, default=0.1)
    parser.add_argument("--boundary-temperature", type=float, default=0.05)
    parser.add_argument("--posterior-sample-count", type=int, default=1024)
    parser.add_argument("--posterior-diagnostic-sample-count", type=int, default=0)
    parser.add_argument("--fantasy-count", type=int, default=3)
    parser.add_argument("--conformal-threshold", type=float, default=None)
    parser.add_argument("--split", choices=("development", "confirmatory"), default="development")
    parser.add_argument("--transport-model", type=Path, default=None)
    parser.add_argument(
        "--hull-backend",
        choices=("pymatgen", "fixed_composition"),
        default="pymatgen",
    )
    parser.add_argument(
        "--transport-family",
        choices=(
            "ridge_random_intercept",
            "hierarchical_matern52_frozen_structure",
        ),
        default="ridge_random_intercept",
    )
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=POLICIES)
    parser.add_argument("--crossfit-manifest", type=Path, default=None)
    parser.add_argument("--fold-index", type=int, default=None)
    args = parser.parse_args()
    query_systems = None
    fit_systems = None
    crossfit_manifest_sha256 = None
    if (args.crossfit_manifest is None) != (args.fold_index is None):
        raise ValueError("--crossfit-manifest and --fold-index must be provided together")
    if args.crossfit_manifest is not None:
        manifest = json.loads(args.crossfit_manifest.read_text(encoding="utf-8"))
        if manifest.get("task_sha256") != _sha256(args.task):
            raise ValueError("cross-fit manifest does not match the task")
        folds = list(manifest["folds"])
        if args.fold_index < 0 or args.fold_index >= len(folds):
            raise ValueError("cross-fit fold index is out of range")
        query_systems = tuple(folds[args.fold_index]["query_systems"])
        eligible_systems = set(manifest["eligible_systems"])
        fit_systems = tuple(sorted(eligible_systems - set(query_systems)))
        crossfit_manifest_sha256 = _sha256(args.crossfit_manifest)
    config = ExperimentConfig(
        max_systems=(len(query_systems) if query_systems is not None else args.max_systems),
        minimum_candidates=args.minimum_candidates,
        maximum_budget=args.maximum_budget,
        seed=args.seed,
        ridge_penalty=args.ridge_penalty,
        prior_standard_deviation=args.prior_standard_deviation,
        boundary_temperature_ev_per_atom=args.boundary_temperature,
        posterior_sample_count=args.posterior_sample_count,
        posterior_diagnostic_sample_count=args.posterior_diagnostic_sample_count,
        fantasy_count=args.fantasy_count,
        conformal_threshold=args.conformal_threshold,
        hull_backend=args.hull_backend,
        transport_family=args.transport_family,
        split=args.split,
        transport_model_path=args.transport_model,
        policies=tuple(args.policies),
        query_systems=query_systems,
        fit_systems=fit_systems,
        crossfit_manifest_sha256=crossfit_manifest_sha256,
        crossfit_fold_index=args.fold_index,
    )
    if (
        config.max_systems < 1
        or config.minimum_candidates < 2
        or config.maximum_budget < 1
        or config.ridge_penalty <= 0
        or config.prior_standard_deviation <= 0
        or config.boundary_temperature_ev_per_atom <= 0
        or config.posterior_sample_count < 4
        or (
            (
                "source_rollout_delta_hull" in config.policies
                or "constrained_dual_horizon_source_rollout" in config.policies
                or "independent_confirmation_source_rollout" in config.policies
                or "conformal_source_rollout_delta_hull" in config.policies
            )
            and (
                config.posterior_sample_count % 16
                or config.posterior_sample_count // 16 < 2
                or config.posterior_sample_count // 16 & (config.posterior_sample_count // 16 - 1)
            )
        )
        or (
            config.posterior_diagnostic_sample_count != 0
            and config.posterior_diagnostic_sample_count < 4
        )
        or config.fantasy_count < 1
        or (
            "conformal_source_rollout_delta_hull" in config.policies
            and (
                config.conformal_threshold is None
                or not math.isfinite(config.conformal_threshold)
                or config.conformal_threshold < 0
            )
        )
        or not config.policies
        or len(set(config.policies)) != len(config.policies)
    ):
        raise ValueError("CHIC closed-loop exploratory configuration is invalid")
    run(
        task_path=args.task,
        development_vault_path=args.development_vault,
        output_path=args.output,
        config=config,
    )


if __name__ == "__main__":
    main()
