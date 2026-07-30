"""Post-hoc calibration audit for frozen IC-SARR decision diagnostics.

This tool opens target energies only after complete closed-loop traces already
exist.  It replays each recorded IC-SARR decision state with the true complete
pool and the already-frozen source continuation; it never feeds these values
back to an acquisition policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

_SRC_ROOT = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from matmem.hull_geometry import FixedCompositionHullTemplate, _final_hull_membership  # noqa: E402
from matmem.protocol_acquisition import _source_rollout_rewards  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_lookup(task: dict[str, Any], vault: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    candidates = defaultdict(list)
    for row in task["development_pairs"]:
        candidates[str(row["chemical_system"])].append(row)
    outcomes = {str(row["pair_id"]): row for row in vault["target_outcomes"]}
    return candidates, outcomes


def _record_state(
    *,
    system: str,
    round_index: int,
    trace_ids: tuple[str, ...],
    diagnostic: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    outcome_rows: dict[str, dict[str, Any]],
    initial_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    query_ids = tuple(str(value) for value in diagnostic.get("candidate_pair_ids", ()))
    source_id = diagnostic.get("source_pair_id")
    if not query_ids or not isinstance(source_id, str) or source_id not in query_ids:
        return None
    source_index = query_ids.index(source_id)
    advantages = diagnostic.get("stage_one_mean_advantages_over_source", {})
    if not isinstance(advantages, dict):
        return None
    selected_id = str(diagnostic["selected_pair_id"])
    screened_id = diagnostic.get("screened_pair_id")
    stage_one_id = diagnostic.get("stage_one_selected_pair_id")
    proposal = (
        selected_id
        if selected_id != source_id
        else screened_id
        if isinstance(screened_id, str) and screened_id != source_id
        else stage_one_id
        if isinstance(stage_one_id, str) and stage_one_id != source_id
        else max(
            (candidate for candidate in query_ids if candidate != source_id),
            key=lambda candidate: (float(advantages.get(candidate, float("-inf"))), candidate),
            default=source_id,
        )
    )
    if proposal == source_id:
        return None
    rows_by_id = {str(row["pair_id"]): row for row in candidate_rows}
    query_rows = [rows_by_id[pair_id] for pair_id in query_ids]
    query_compositions = tuple(dict(row["composition"]) for row in query_rows)
    references = tuple(dict(row["composition"]) for row in initial_rows)
    reference_energies = np.asarray(
        [float(row["formation_energy_ev_per_atom"]) for row in initial_rows], dtype=float
    )
    template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions, reference_compositions=references
    )
    true_energies = np.asarray(
        [float(outcome_rows[pair_id]["target_formation_energy_ev_per_atom"]) for pair_id in query_ids],
        dtype=float,
    )[None, :]
    labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=true_energies,
        reference_compositions=references,
        reference_energies=reference_energies,
        fixed_template=template,
    )
    causal = np.empty((1, len(query_ids)), dtype=float)
    rewards = _source_rollout_rewards(
        sampled_query_energies=true_energies,
        final_hull_membership=labels,
        query_compositions=query_compositions,
        query_source_energies=np.asarray(
            [float(row["source_formation_energy_ev_per_atom"]) for row in query_rows], dtype=float
        ),
        query_ids=query_ids,
        reference_compositions=references,
        reference_energies=reference_energies,
        horizon=len(trace_ids) - round_index,
        causal_rewards_output=causal,
    )
    proposal_index = query_ids.index(proposal)
    return {
        "system": system,
        "round_index": round_index + 1,
        "source_pair_id": source_id,
        "proposed_pair_id": proposal,
        "selected_pair_id": selected_id,
        "accepted_deviation": selected_id != source_id,
        "stage_two_used": bool(diagnostic.get("stage_two_used", False)),
        "predicted_advantage": float(advantages[proposal]),
        "actual_complete_pool_t_advantage": float(rewards[0, proposal_index] - rewards[0, source_index]),
        "actual_selected_history_f_advantage": float(causal[0, proposal_index] - causal[0, source_index]),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("no IC-SARR diagnostics with a non-source proposal were found")
    predicted = np.asarray([row["predicted_advantage"] for row in records], dtype=float)
    actual_t = np.asarray([row["actual_complete_pool_t_advantage"] for row in records], dtype=float)
    actual_f = np.asarray([row["actual_selected_history_f_advantage"] for row in records], dtype=float)
    accepted = np.asarray([row["accepted_deviation"] for row in records], dtype=bool)
    quantiles = np.quantile(predicted, np.linspace(0.0, 1.0, 11))
    deciles = []
    for index in range(10):
        left, right = quantiles[index], quantiles[index + 1]
        mask = (predicted >= left) & ((predicted <= right) if index == 9 else (predicted < right))
        if np.any(mask):
            deciles.append(
                {
                    "decile": index + 1,
                    "count": int(mask.sum()),
                    "predicted_mean": float(predicted[mask].mean()),
                    "actual_t_mean": float(actual_t[mask].mean()),
                    "actual_f_mean": float(actual_f[mask].mean()),
                }
            )
    strata = {}
    for name, mask in (("accepted", accepted), ("rejected_or_screened", ~accepted)):
        strata[name] = {
            "count": int(mask.sum()),
            "predicted_mean": float(predicted[mask].mean()) if np.any(mask) else None,
            "actual_t_mean": float(actual_t[mask].mean()) if np.any(mask) else None,
            "actual_f_mean": float(actual_f[mask].mean()) if np.any(mask) else None,
        }
    return {
        "decision_state_count": len(records),
        "spearman_predicted_vs_actual_t": float(spearmanr(predicted, actual_t).statistic),
        "spearman_predicted_vs_actual_f": float(spearmanr(predicted, actual_f).statistic),
        "accepted_deviation_count": int(accepted.sum()),
        "false_positive_accepted_deviation_rate": (
            None if not np.any(accepted) else float(np.mean(actual_t[accepted] <= 0.0))
        ),
        "deciles": deciles,
        "strata": strata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(args.task.read_text(encoding="utf-8"))
    vault = json.loads(args.vault.read_text(encoding="utf-8"))
    candidate_by_system, outcomes = _row_lookup(task, vault)
    records: list[dict[str, Any]] = []
    for input_path in args.inputs:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        for system, result in payload["systems"].items():
            strategy = result["strategies"].get("independent_confirmation_source_rollout")
            if strategy is None:
                continue
            trace_ids = tuple(str(value) for value in strategy["selected_pair_ids"])
            initial_rows = task["development_initial_phase_entries"][system]
            for round_index, event in enumerate(strategy["policy_decision_rounds"]):
                diagnostic = event.get("selection_diagnostics")
                if isinstance(diagnostic, dict):
                    record = _record_state(
                        system=system,
                        round_index=round_index,
                        trace_ids=trace_ids,
                        diagnostic=diagnostic,
                        candidate_rows=candidate_by_system[system],
                        outcome_rows=outcomes,
                        initial_rows=initial_rows,
                    )
                    if record is not None:
                        records.append(record)
    output = {
        "schema_version": 1,
        "status": "post_hoc_development_calibration_not_policy_feedback",
        "task_sha256": _sha256(args.task),
        "vault_sha256": _sha256(args.vault),
        "input_sha256": {str(path): _sha256(path) for path in args.inputs},
        "records": records,
        "summary": _summary(records),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"output={args.output.resolve()}")
    print(json.dumps(output["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
