"""Offline attribution of Dual-Horizon SARR failure modes.

This evaluator is deliberately outside the policy subprocess.  It reconstructs
states from a development source trace, then uses the oracle vault only after
the trace is complete to compare posterior counterfactual rollout advantages
with exact target-energy continuation advantages.  It is diagnostic evidence,
not a policy runner or a holdout evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.phase_diagram import PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from scipy.stats import t as student_t

from matmem.protocol_knowledge_gradient import (
    FixedCompositionHullTemplate,
    FrozenProtocolRidgeTransport,
    _final_hull_membership,
    _source_rollout_rewards,
    constrained_dual_horizon_source_rollout,
    protocol_target_energy_posterior,
    source_margin_action_indices,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _formation_energy(composition: dict[str, float], total_energy: float) -> float:
    return float(total_energy) / float(Composition(composition).num_atoms)


def _reference_state(
    *,
    task: dict[str, Any],
    system: str,
    history_ids: list[str],
    rows_by_id: dict[str, dict[str, Any]],
    outcomes_by_id: dict[str, dict[str, Any]],
) -> tuple[tuple[dict[str, float], ...], np.ndarray]:
    initial = task["development_initial_phase_entries"][system]
    compositions = [dict(entry["composition"]) for entry in initial]
    energies = [
        _formation_energy(entry["composition"], float(entry["corrected_total_energy_ev"]))
        for entry in initial
    ]
    for pair_id in history_ids:
        row = rows_by_id[pair_id]
        outcome = outcomes_by_id[pair_id]
        compositions.append(dict(row["composition"]))
        energies.append(float(outcome["target_formation_energy_ev_per_atom"]))
    return tuple(compositions), np.asarray(energies, dtype=float)


def _current_hull_energies(
    *,
    query_compositions: tuple[dict[str, float], ...],
    reference_compositions: tuple[dict[str, float], ...],
    reference_energies: np.ndarray,
) -> np.ndarray:
    entries = [
        ComputedEntry(
            Composition(composition),
            float(energy) * Composition(composition).num_atoms,
            entry_id=f"reference:{index}",
        )
        for index, (composition, energy) in enumerate(
            zip(reference_compositions, reference_energies, strict=True)
        )
    ]
    diagram = PhaseDiagram(entries)
    values = []
    for composition in query_compositions:
        parsed = Composition(composition)
        hull = float(diagram.get_hull_energy_per_atom(parsed))
        fake = ComputedEntry(parsed, hull * parsed.num_atoms)
        values.append(float(diagram.get_form_energy_per_atom(fake)))
    return np.asarray(values, dtype=float)


def _upper_bounds(
    block_differences: np.ndarray,
    *,
    confidence: float,
    comparison_count: int,
) -> np.ndarray:
    values = np.asarray(block_differences, dtype=float)
    alpha = (1.0 - confidence) / float(comparison_count)
    critical = float(student_t.ppf(1.0 - alpha, len(values) - 1))
    means = values.mean(axis=0)
    errors = values.std(axis=0, ddof=1) / math.sqrt(len(values))
    return means + critical * errors


def _system_trace(summary: dict[str, Any], system: str) -> list[str]:
    return list(summary["systems"][system]["strategies"]["source_margin"]["selected_pair_ids"])


def run(
    *,
    task_path: Path,
    vault_path: Path,
    transport_path: Path,
    source_trace_path: Path,
    output_path: Path,
    max_systems: int,
    posterior_sample_count: int,
    integration_confidence: float = 0.95,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(output_path)
    if posterior_sample_count < 32 or posterior_sample_count % 16:
        raise ValueError("attribution posterior samples must be divisible by 16 and >=32")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    vault = json.loads(vault_path.read_text(encoding="utf-8"))
    transport_payload = json.loads(transport_path.read_text(encoding="utf-8"))
    transport = FrozenProtocolRidgeTransport.model_validate(
        transport_payload.get("model", transport_payload)
    )
    trace = json.loads(source_trace_path.read_text(encoding="utf-8"))
    rows_by_id = {row["pair_id"]: row for row in task["development_pairs"]}
    outcomes_by_id = {row["pair_id"]: row for row in vault["target_outcomes"]}
    systems = list(trace["query_systems"][:max_systems])
    state_records: list[dict[str, Any]] = []
    for system in systems:
        system_rows = [
            row for row in task["development_pairs"] if row["chemical_system"] == system
        ]
        system_rows.sort(key=lambda row: row["pair_id"])
        trace_ids = _system_trace(trace, system)
        if not trace_ids:
            continue
        for round_index in range(len(trace_ids)):
            history_ids = trace_ids[:round_index]
            query_rows = [row for row in system_rows if row["pair_id"] not in history_ids]
            query_rows.sort(key=lambda row: row["pair_id"])
            query_ids = tuple(row["pair_id"] for row in query_rows)
            query_compositions = tuple(dict(row["composition"]) for row in query_rows)
            query_source = np.asarray(
                [row["source_formation_energy_ev_per_atom"] for row in query_rows], dtype=float
            )
            references, reference_energies = _reference_state(
                task=task,
                system=system,
                history_ids=history_ids,
                rows_by_id=rows_by_id,
                outcomes_by_id=outcomes_by_id,
            )
            current_hull = _current_hull_energies(
                query_compositions=query_compositions,
                reference_compositions=references,
                reference_energies=reference_energies,
            )
            source_index = int(
                source_margin_action_indices(
                    source_energies=query_source,
                    competing_hull_energies=current_hull,
                    query_ids=query_ids,
                )[0]
            )
            fixed_template = FixedCompositionHullTemplate.from_compositions(
                query_compositions=query_compositions,
                reference_compositions=references,
            )
            query_features = np.asarray(
                [row["source_environment_embedding"] for row in query_rows], dtype=float
            )
            history_rows = [rows_by_id[pair_id] for pair_id in history_ids]
            history_features = np.asarray(
                [row["source_environment_embedding"] for row in history_rows], dtype=float
            ).reshape(len(history_rows), query_features.shape[1])
            history_source = np.asarray(
                [row["source_formation_energy_ev_per_atom"] for row in history_rows], dtype=float
            )
            history_target = np.asarray(
                [outcomes_by_id[pair_id]["target_formation_energy_ev_per_atom"] for pair_id in history_ids],
                dtype=float,
            )
            kernel_dim = len(transport.kernel_feature_mean)
            query_kernel = np.asarray(
                [row["source_local_environment_embedding"] for row in query_rows], dtype=float
            )
            history_kernel = np.asarray(
                [row["source_local_environment_embedding"] for row in history_rows], dtype=float
            ).reshape(len(history_rows), kernel_dim)
            posterior = protocol_target_energy_posterior(
                transport,
                query_features=query_features,
                query_source_energies=query_source,
                history_features=history_features,
                history_source_energies=history_source,
                history_target_energies=history_target,
                query_kernel_features=query_kernel,
                history_kernel_features=history_kernel,
            )
            estimated = constrained_dual_horizon_source_rollout(
                posterior,
                query_compositions=query_compositions,
                query_source_energies=query_source,
                query_ids=query_ids,
                reference_compositions=references,
                reference_energies=reference_energies,
                current_competing_hull_energies=current_hull,
                costs=np.ones(len(query_rows)),
                remaining_budget=float(len(trace_ids) - round_index),
                posterior_sample_count=posterior_sample_count,
                seed=20270720 + 1009 * round_index,
                fixed_template=fixed_template,
            )
            true_query_energies = np.asarray(
                [outcomes_by_id[pair_id]["target_formation_energy_ev_per_atom"] for pair_id in query_ids],
                dtype=float,
            )
            true_samples = true_query_energies[None, :]
            final_labels = _final_hull_membership(
                query_compositions=query_compositions,
                sampled_query_energies=true_samples,
                reference_compositions=references,
                reference_energies=reference_energies,
                fixed_template=fixed_template,
            )
            causal_rewards = np.empty((1, len(query_rows)), dtype=float)
            true_rewards = _source_rollout_rewards(
                sampled_query_energies=true_samples,
                final_hull_membership=final_labels,
                query_compositions=query_compositions,
                query_source_energies=query_source,
                query_ids=query_ids,
                reference_compositions=references,
                reference_energies=reference_energies,
                horizon=estimated.horizon,
                causal_rewards_output=causal_rewards,
            )
            oracle_t = true_rewards[0] - true_rewards[0, source_index]
            oracle_f = causal_rewards[0] - causal_rewards[0, source_index]
            point_t = np.asarray(estimated.terminal_paired_advantages)
            point_f = np.asarray(estimated.causal_paired_advantages)
            terminal_blocks = np.asarray(estimated.terminal_block_scores)
            causal_blocks = np.asarray(estimated.causal_block_scores)
            terminal_diff = terminal_blocks - terminal_blocks[:, [source_index]]
            causal_diff = causal_blocks - causal_blocks[:, [source_index]]
            count = 2 * max(len(query_rows) - 1, 1)
            upper_t = _upper_bounds(
                terminal_diff, confidence=integration_confidence, comparison_count=count
            )
            upper_f = _upper_bounds(
                causal_diff, confidence=integration_confidence, comparison_count=count
            )
            point_feasible = (point_t > 0) & (point_f >= 0)
            gate_feasible = np.asarray(estimated.feasible_mask, dtype=bool)
            oracle_feasible = (oracle_t > 0) & (oracle_f >= 0)
            point_choices = np.flatnonzero(point_feasible)
            point_selected = int(
                max(point_choices, key=lambda index: (point_t[index], -index))
                if len(point_choices)
                else source_index
            )
            oracle_choices = np.flatnonzero(oracle_feasible)
            oracle_best = int(
                max(oracle_choices, key=lambda index: (oracle_t[index], -index))
                if len(oracle_choices)
                else source_index
            )
            state_records.append(
                {
                    "system": system,
                    "round_index": round_index + 1,
                    "candidate_count": len(query_rows),
                    "history_count": len(history_ids),
                    "query_ids": query_ids,
                    "source_action_id": query_ids[source_index],
                    "trace_action_id": trace_ids[round_index],
                    "oracle_terminal_advantages": oracle_t.tolist(),
                    "oracle_selected_history_advantages": oracle_f.tolist(),
                    "posterior_terminal_advantages": point_t.tolist(),
                    "posterior_selected_history_advantages": point_f.tolist(),
                    "posterior_terminal_lower_bounds": list(estimated.terminal_lower_bounds),
                    "posterior_selected_history_lower_bounds": list(estimated.causal_lower_bounds),
                    "posterior_terminal_upper_bounds": upper_t.tolist(),
                    "posterior_selected_history_upper_bounds": upper_f.tolist(),
                    "oracle_feasible": oracle_feasible.tolist(),
                    "posterior_point_feasible": point_feasible.tolist(),
                    "posterior_gate_feasible": gate_feasible.tolist(),
                    "oracle_feasible_exists": bool(len(oracle_choices)),
                    "oracle_best_action_id": query_ids[oracle_best],
                    "posterior_point_action_id": query_ids[point_selected],
                    "dual_gate_action_id": query_ids[estimated.selected_action_index],
                }
            )

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    oracle_exists = [record["oracle_feasible_exists"] for record in state_records]
    oracle_action_count = sum(sum(record["oracle_feasible"]) for record in state_records)
    oracle_feasible_records = [record for record in state_records if record["oracle_feasible_exists"]]
    recall_values = [
        sum(bool(p) and bool(o) for p, o in zip(record["posterior_point_feasible"], record["oracle_feasible"]))
        / max(sum(record["oracle_feasible"]), 1)
        for record in oracle_feasible_records
    ]
    point_gate_rejection = [
        sum(bool(p) and not bool(g) for p, g in zip(record["posterior_point_feasible"], record["posterior_gate_feasible"]))
        / max(sum(record["posterior_point_feasible"]), 1)
        for record in state_records
        if sum(record["posterior_point_feasible"])
    ]
    t_sign = []
    f_sign = []
    coverage_t = []
    coverage_f = []
    point_regret = []
    for record in state_records:
        ot = np.asarray(record["oracle_terminal_advantages"])
        of = np.asarray(record["oracle_selected_history_advantages"])
        pt = np.asarray(record["posterior_terminal_advantages"])
        pf = np.asarray(record["posterior_selected_history_advantages"])
        t_sign.extend((np.sign(ot) == np.sign(pt)).tolist())
        f_sign.extend((np.sign(of) == np.sign(pf)).tolist())
        coverage_t.extend(((ot >= np.asarray(record["posterior_terminal_lower_bounds"])) & (ot <= np.asarray(record["posterior_terminal_upper_bounds"]))).tolist())
        coverage_f.extend(((of >= np.asarray(record["posterior_selected_history_lower_bounds"])) & (of <= np.asarray(record["posterior_selected_history_upper_bounds"]))).tolist())
        best = record["oracle_best_action_id"]
        ids = record["query_ids"]
        best_value = ot[ids.index(best)]
        point_regret.append(float(best_value - ot[ids.index(record["posterior_point_action_id"])]))
    result = {
        "schema_version": 1,
        "status": "development_attribution_only",
        "task_sha256": _sha256(task_path),
        "vault_sha256": _sha256(vault_path),
        "transport_sha256": _sha256(transport_path),
        "source_trace_sha256": _sha256(source_trace_path),
        "posterior_sample_count": posterior_sample_count,
        "systems": systems,
        "state_count": len(state_records),
        "states": state_records,
        "summary": {
            "oracle_feasible_action_existence_rate": mean([float(value) for value in oracle_exists]),
            "oracle_feasible_action_count": oracle_action_count,
            "posterior_recall_of_oracle_feasible_actions": mean(recall_values),
            "posterior_point_feasible_action_gate_rejection_rate": mean(point_gate_rejection),
            "terminal_sign_accuracy": mean([float(value) for value in t_sign]),
            "selected_history_sign_accuracy": mean([float(value) for value in f_sign]),
            "terminal_joint_advantage_interval_coverage": mean([float(value) for value in coverage_t]),
            "selected_history_joint_advantage_interval_coverage": mean([float(value) for value in coverage_f]),
            "posterior_point_action_oracle_terminal_regret": mean(point_regret),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--transport-model", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-systems", type=int, default=8)
    parser.add_argument("--posterior-sample-count", type=int, default=128)
    args = parser.parse_args()
    result = run(
        task_path=args.task,
        vault_path=args.vault,
        transport_path=args.transport_model,
        source_trace_path=args.source_trace,
        output_path=args.output,
        max_systems=args.max_systems,
        posterior_sample_count=args.posterior_sample_count,
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
