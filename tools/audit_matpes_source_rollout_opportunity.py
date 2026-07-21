"""Replay frozen SARR states at high RQMC fidelity without new oracle reveals."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from matmem.protocol_closed_loop import (
    ObservableProtocolQuery,
    ProtocolCausalHull,
    ProtocolPolicyState,
    ProtocolPolicySubprocess,
    RevealedProtocolObservation,
)
from matmem.protocol_knowledge_gradient import FrozenProtocolRidgeTransport
from matmem.protocols import ProtocolCertificate
from tools.run_matpes_protocol_closed_loop_exploratory import _candidate, _initial_entries, _outcome


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _observable(candidate, hull: ProtocolCausalHull) -> ObservableProtocolQuery:
    return ObservableProtocolQuery(
        pair_id=candidate.pair_id, source_structure_hash=candidate.source_structure_hash,
        chemical_system=candidate.chemical_system, composition=candidate.composition,
        source_formation_energy_ev_per_atom=candidate.source_formation_energy_ev_per_atom,
        source_environment_embedding=candidate.source_environment_embedding,
        source_local_environment_embedding=candidate.source_local_environment_embedding,
        current_competing_hull_ev_per_atom=hull.competing_hull_formation_energy(candidate.composition),
        source_protocol_fingerprint=candidate.source_protocol.scientific_fingerprint,
        target_protocol_fingerprint=candidate.target_protocol.scientific_fingerprint,
        oracle_cost=candidate.oracle_cost,
    )


def run(*, task_path: Path, vault_path: Path, sarr_path: Path, plan_path: Path,
        transport_path: Path, output_path: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output_path.resolve().is_relative_to(root):
        raise ValueError("audit output must remain outside Git")
    if output_path.exists():
        raise FileExistsError(output_path)
    task = json.loads(task_path.read_text())
    vault = json.loads(vault_path.read_text())
    sarr = json.loads(sarr_path.read_text())
    plan = json.loads(plan_path.read_text())
    if plan["task_sha256"] != _sha256(task_path) or plan["sarr_result_sha256"] != _sha256(sarr_path):
        raise ValueError("frozen audit plan inputs disagree")
    if sarr["evaluation_systems_accessed"] or vault["target_outcomes"][0]["split"] != "development":
        raise ValueError("audit is development-only")
    model_payload = json.loads(transport_path.read_text())
    model = FrozenProtocolRidgeTransport.model_validate(model_payload.get("model", model_payload))
    outcomes = {row["pair_id"]: row for row in vault["target_outcomes"]}
    rows_by_system: dict[str, list[dict[str, Any]]] = {}
    for row in task["development_pairs"]:
        rows_by_system.setdefault(row["chemical_system"], []).append(row)
    requested = {(row["chemical_system"], int(row["round_index"])): row["reasons"] for row in plan["states"]}
    source_protocol = ProtocolCertificate.model_validate(task["source_protocol"])
    target_protocol = ProtocolCertificate.model_validate(task["target_protocol"])
    results: list[dict[str, Any]] = []
    for system in sorted({key[0] for key in requested}):
        rows = sorted(rows_by_system[system], key=lambda row: row["pair_id"])
        recorded = sarr["systems"][system]["strategies"]["source_rollout_delta_hull"]
        events = recorded["policy_decision_rounds"]
        candidates = {_candidate(row, source_protocol=source_protocol, target_protocol=target_protocol).pair_id:
                      _candidate(row, source_protocol=source_protocol, target_protocol=target_protocol) for row in rows}
        row_by_id = {row["pair_id"]: row for row in rows}
        hull = ProtocolCausalHull(_initial_entries(task["development_initial_phase_entries"][system]), chemical_system=tuple(system.split("-")))
        history: list[RevealedProtocolObservation] = []
        recorded_worker = ProtocolPolicySubprocess("source_rollout_delta_hull", seed=int(sarr["config"]["seed"]),
            transport_model=model, posterior_sample_count=int(sarr["config"]["posterior_sample_count"]),
            fantasy_count=int(sarr["config"]["fantasy_count"]), hull_backend=sarr["config"]["hull_backend"], selection_timeout_seconds=900.0)
        worker = ProtocolPolicySubprocess("source_rollout_delta_hull", seed=int(sarr["config"]["seed"]),
            transport_model=model, posterior_sample_count=int(plan["high_precision_posterior_sample_count"]),
            fantasy_count=int(sarr["config"]["fantasy_count"]), hull_backend=sarr["config"]["hull_backend"], selection_timeout_seconds=900.0)
        try:
            for index, selected_id in enumerate(recorded["selected_pair_ids"], start=1):
                event = events[index - 1]
                if selected_id != event["selected_pair_id"]:
                    raise ValueError("recorded selected action disagrees with event")
                state = ProtocolPolicyState.create(round_index=index, remaining_budget=float(6 - index + 1),
                    queries=(_observable(item, hull) for item in candidates.values()), causal_hull_phases=hull.observable_phases,
                    revealed_history=history, policy_identity_checksum=recorded_worker.identity_checksum)
                if state.state_checksum != event["pre_reveal_state_checksum"]:
                    raise AssertionError("replayed observable state checksum differs")
                key = (system, index)
                if key in requested:
                    worker.select(state)
                    diagnostic = worker.last_selection_diagnostics
                    if diagnostic is None:
                        raise AssertionError("high-precision worker omitted diagnostics")
                    ids = diagnostic["candidate_pair_ids"]
                    scores = [sum(block[col] for block in diagnostic["block_scores"]) / 16 for col in range(len(ids))]
                    best = min(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
                    chosen = ids.index(selected_id)
                    source = ids.index(diagnostic["source_pair_id"])
                    ordered = sorted(scores, reverse=True)
                    results.append({"chemical_system": system, "round_index": index, "reasons": requested[key],
                        "selected_pair_id": selected_id, "source_pair_id": ids[source], "reference_best_pair_id": ids[best],
                        "selected_opportunity_cost": scores[best] - scores[chosen], "source_opportunity_cost": scores[best] - scores[source],
                        "best_second_gap": ordered[0] - ordered[1] if len(ordered) > 1 else 0.0,
                        "selected_high_precision_advantage": diagnostic["mean_advantages_over_source"][selected_id],
                        "selected_high_precision_lower_bound": diagnostic["simultaneous_lower_bounds"][selected_id]})
                chosen = candidates.pop(selected_id)
                outcome = _outcome(row_by_id[selected_id], outcomes[selected_id])
                history.append(RevealedProtocolObservation(pair_id=selected_id, source_formation_energy_ev_per_atom=chosen.source_formation_energy_ev_per_atom,
                    revealed_target_formation_energy_ev_per_atom=outcome.target_formation_energy_ev_per_atom,
                    source_environment_embedding=chosen.source_environment_embedding, source_local_environment_embedding=chosen.source_local_environment_embedding))
                hull.add_revealed(outcome)
        finally:
            recorded_worker.close()
            worker.close()
    output = {"schema_version": 1, "status": "development_high_precision_replay", "plan_sha256": _sha256(plan_path),
              "task_sha256": _sha256(task_path), "sarr_sha256": _sha256(sarr_path), "evaluation_systems_accessed": False,
              "state_count": len(results), "states": results}
    if len(results) != len(requested):
        raise AssertionError("audit did not replay every frozen state")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("task", "vault", "sarr", "plan", "transport", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    a = p.parse_args()
    result = run(task_path=a.task, vault_path=a.vault, sarr_path=a.sarr, plan_path=a.plan, transport_path=a.transport, output_path=a.output)
    print(f"output={a.output} states={result['state_count']}")


if __name__ == "__main__":
    main()
