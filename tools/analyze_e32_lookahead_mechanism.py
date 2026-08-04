"""Audit the registered E32-A traces for direct lookahead mechanisms.

This is evaluator-only post-processing.  It does not rerun a policy, read an
unrevealed outcome, or modify the registered E32 estimand.  The frozen trace
contains candidate-level rank-switch probabilities and common-rollout Q
scores.  It does not contain the signed conditional hull probabilities needed
to identify Proposition 4's exact ``I_h(x)``; the output records that boundary
instead of silently substituting a different quantity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None
    rx = _rankdata(np.asarray(x, dtype=float))
    ry = _rankdata(np.asarray(y, dtype=float))
    if np.all(rx == rx[0]) or np.all(ry == ry[0]):
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _paired(values: list[float], *, seed: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return {"system_count": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(20_000, len(array)))
    means = array[indices].mean(axis=1)
    return {
        "system_count": int(len(array)),
        "mean": float(array.mean()),
        "bootstrap_95ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "wins": int(np.sum(array > 0)),
        "ties": int(np.sum(array == 0)),
        "losses": int(np.sum(array < 0)),
        "bootstrap_seed": seed,
        "bootstrap_replicates": 20_000,
    }


def _load_b6(input_root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    payloads: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for fold in range(1, 6):
        path = input_root / f"e32-fold{fold}-b6-main.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payloads.append(json.loads(path.read_text(encoding="utf-8")))
        hashes[path.name] = _sha256(path)
    return payloads, hashes


def _quantile_group(values: list[float], *, high: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    threshold = float(np.quantile(array, 0.75))
    if high:
        return array >= threshold
    return array <= float(np.quantile(array, 0.25))


def analyze(*, input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    payloads, hashes = _load_b6(input_root)
    states: list[dict[str, Any]] = []
    systems: dict[str, dict[str, Any]] = {}
    missing_diagnostics: list[str] = []

    for payload in payloads:
        for system, result in payload["systems"].items():
            anchored = result["strategies"]["delta_hull_anchored_rollout"]
            delta = result["strategies"]["delta_hull_active_search"]
            events: list[dict[str, Any]] = []
            for event in anchored.get("policy_decision_rounds", []):
                diagnostic = event.get("selection_diagnostics")
                if not isinstance(diagnostic, dict):
                    continue
                ids = list(diagnostic["candidate_pair_ids"])
                selected_id = str(diagnostic["selected_pair_id"])
                delta_id = str(diagnostic["delta_hull_action_id"])
                selected_index = ids.index(selected_id)
                delta_index = ids.index(delta_id)
                rollout_scores = np.asarray(diagnostic["rollout_scores"], dtype=float)
                rank_switch_by_candidate = np.asarray(
                    diagnostic["rank_switch_by_candidate"], dtype=float
                )
                state = {
                    "system": system,
                    "round": int(event["round_index"]),
                    "selected_pair_id": selected_id,
                    "delta_hull_pair_id": delta_id,
                    "action_disagreement": int(selected_id != delta_id),
                    "rank_switch_probability": float(diagnostic["rank_switch_probability"]),
                    "rank_switch_selected_action": float(rank_switch_by_candidate[selected_index]),
                    "rank_switch_delta_action": float(rank_switch_by_candidate[delta_index]),
                    "rank_switch_max_candidate": float(rank_switch_by_candidate.max()),
                    "rank_margin": float(diagnostic["rank_margin"]),
                    "coupling_score_normalized": float(
                        diagnostic["coupling_score_normalized"]
                    ),
                    # Q_gap is the model-relative value of replacing the
                    # Delta-Hull first action under the same anchored-rollout
                    # continuation evaluator.
                    "q_gap_over_delta_action": float(
                        rollout_scores.max() - rollout_scores[delta_index]
                    ),
                    "q_score_selected_action": float(rollout_scores[selected_index]),
                    "q_score_delta_action": float(rollout_scores[delta_index]),
                    "q_score_max": float(rollout_scores.max()),
                }
                states.append(state)
                events.append(state)
            if not events:
                missing_diagnostics.append(system)
                continue
            systems[system] = {
                "state_count": len(events),
                "mean_rank_switch_selected_action": float(
                    np.mean([row["rank_switch_selected_action"] for row in events])
                ),
                "max_rank_switch_selected_action": float(
                    np.max([row["rank_switch_selected_action"] for row in events])
                ),
                "mean_rank_switch_probability": float(
                    np.mean([row["rank_switch_probability"] for row in events])
                ),
                "max_rank_switch_probability": float(
                    np.max([row["rank_switch_probability"] for row in events])
                ),
                "mean_q_gap_over_delta_action": float(
                    np.mean([row["q_gap_over_delta_action"] for row in events])
                ),
                "max_q_gap_over_delta_action": float(
                    np.max([row["q_gap_over_delta_action"] for row in events])
                ),
                "anchored_minus_delta_T": float(
                    anchored["oracle_pool_confirmed_discoveries"]
                    - delta["oracle_pool_confirmed_discoveries"]
                ),
                "anchored_minus_delta_F": float(
                    anchored["final_causal_confirmed_discoveries"]
                    - delta["final_causal_confirmed_discoveries"]
                ),
                "anchored_minus_delta_D": float(
                    anchored["causal_discoveries"] - delta["causal_discoveries"]
                ),
            }

    state_disagreement = [float(row["action_disagreement"]) for row in states]
    state_rank = [float(row["rank_switch_selected_action"]) for row in states]
    state_rank_mean = [float(row["rank_switch_probability"]) for row in states]
    state_q_gap = [float(row["q_gap_over_delta_action"]) for row in states]
    system_names = sorted(systems)
    system_rank = [systems[name]["max_rank_switch_selected_action"] for name in system_names]
    system_rank_mean = [systems[name]["mean_rank_switch_selected_action"] for name in system_names]
    system_q_gap = [systems[name]["max_q_gap_over_delta_action"] for name in system_names]
    system_t = [systems[name]["anchored_minus_delta_T"] for name in system_names]

    def strata(values: list[float], label: str) -> dict[str, Any]:
        array = np.asarray(values, dtype=float)
        q25, q75 = np.quantile(array, [0.25, 0.75])
        result: dict[str, Any] = {"q25": float(q25), "q75": float(q75)}
        for name, mask in {
            "low": array <= q25,
            "middle": (array > q25) & (array < q75),
            "high": array >= q75,
        }.items():
            result[name] = {"metric": label, "state_count": int(mask.sum())}
            if name == "low":
                selected = [system_t[i] for i in np.flatnonzero(mask)]
            elif name == "high":
                selected = [system_t[i] for i in np.flatnonzero(mask)]
            else:
                selected = [system_t[i] for i in np.flatnonzero(mask)]
            result[name].update(_paired(selected, seed=20260807))
        return result

    result = {
        "schema_version": 1,
        "status": "complete_e32_b6_direct_lookahead_mechanism_audit",
        "input_root": str(input_root),
        "input_file_sha256": hashes,
        "system_count": len(system_names),
        "state_count": len(states),
        "diagnostic_coverage": {
            "systems_with_complete_anchored_states": len(system_names),
            "systems_without_anchored_states": len(missing_diagnostics),
            "missing_systems": sorted(missing_diagnostics),
        },
        "availability": {
            "exact_I_h": False,
            "reason": (
                "The frozen state trace stores absolute cross-candidate influence and "
                "rank-switch frequencies, but not the signed conditional hull "
                "probabilities p_{h,x,O_x}(y); Proposition 4's I_h(x) is therefore "
                "not identifiable without a new instrumented posterior replay."
            ),
            "directly_audited": [
                "candidate-level rank-switch probability r_h(x)",
                "model-relative anchored-rollout Q-gap over the Delta-Hull action",
                "realized system-level Delta-Hull contrast",
            ],
        },
        "state_level_correlations": {
            "action_disagreement_vs_selected_action_rank_switch": _spearman(
                state_disagreement, state_rank
            ),
            "action_disagreement_vs_mean_rank_switch": _spearman(
                state_disagreement, state_rank_mean
            ),
            "action_disagreement_vs_q_gap": _spearman(state_disagreement, state_q_gap),
            "selected_action_rank_switch_vs_q_gap": _spearman(state_rank, state_q_gap),
        },
        "system_level_correlations": {
            "realized_T_gain_vs_max_selected_action_rank_switch": _spearman(
                system_t, system_rank
            ),
            "realized_T_gain_vs_mean_selected_action_rank_switch": _spearman(
                system_t, system_rank_mean
            ),
            "realized_T_gain_vs_max_q_gap": _spearman(system_t, system_q_gap),
        },
        "system_strata_by_max_selected_action_rank_switch": strata(
            system_rank, "max_selected_action_rank_switch"
        ),
        "system_strata_by_max_q_gap": strata(system_q_gap, "max_q_gap_over_delta_action"),
        "system_rows": {name: systems[name] for name in system_names},
        "state_rows": states,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(input_root=args.input_root, output=args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "system_count": result["system_count"],
                "state_count": result["state_count"],
                "availability": result["availability"],
                "state_level_correlations": result["state_level_correlations"],
                "system_level_correlations": result["system_level_correlations"],
                "rank_strata": result["system_strata_by_max_selected_action_rank_switch"],
                "q_gap_strata": result["system_strata_by_max_q_gap"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
