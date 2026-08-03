"""Summarize the registered E32-A delayed-label mechanism suite.

The input root is external raw output.  This utility refuses incomplete
fold/budget schedules and reports direct paired exact-system contrasts rather
than inferring attribution from a common source baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

POLICIES = (
    "source_margin",
    "posterior_mean_target_margin",
    "posterior_current_hull_probability",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "delta_hull_anchored_rollout",
)
METRICS = {
    "D": "causal_discoveries",
    "F": "final_causal_confirmed_discoveries",
    "T": "oracle_pool_confirmed_discoveries",
    "D_minus_F": "within_campaign_revocations",
    "F_minus_T": "unqueried_competitor_invalidations",
    "wall_seconds": "wall_seconds",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _paired(values: np.ndarray, *, bootstrap_seed: int = 20260803) -> dict[str, Any]:
    values = np.asarray(values, dtype=float).reshape(-1)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("paired values must be nonempty and finite")
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(values), size=(20_000, len(values)))
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(20260804)
    signs = sign_rng.choice((-1.0, 1.0), size=(100_000, len(values)))
    observed = abs(float(values.sum()))
    randomized = np.abs(np.sum(signs * values[None, :], axis=1))
    return {
        "system_count": int(len(values)),
        "paired_mean_difference": float(values.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "losses": int(np.sum(values < 0)),
        "two_sided_sign_flip_p": float(
            (np.sum(randomized >= observed) + 1) / (len(randomized) + 1)
        ),
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": 20_000,
        "sign_flip_seed": 20260804,
        "sign_flip_draws": 100_000,
    }


def _load(path: Path, *, budget: int) -> dict[str, Any]:
    if not path.exists():
        failure = path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"registered E32 output failed: {failure}")
        raise FileNotFoundError(f"registered E32 output is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload.get("active_policies", ())) != POLICIES:
        raise ValueError(f"unexpected E32 policy roster in {path}")
    if payload.get("config", {}).get("query_budget") != budget:
        raise ValueError(f"wrong budget in {path}")
    for system, result in payload.get("systems", {}).items():
        if int(result["budget"]) != budget:
            raise ValueError(f"wrong system budget for {system} in {path}")
        if set(result["strategies"]) != set(POLICIES):
            raise ValueError(f"incomplete strategy roster for {system} in {path}")
    return payload


def _rows(payloads: list[dict[str, Any]], policy: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for system, result in payload["systems"].items():
            if system in rows:
                raise ValueError(f"chemical system occurs twice: {system}")
            rows[system] = result["strategies"][policy]
    if not rows:
        raise ValueError("no E32 system rows")
    return rows


def _metric_rows(
    policy_rows: dict[str, dict[str, Any]],
    baseline_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    systems = sorted(policy_rows)
    if set(systems) != set(baseline_rows):
        raise ValueError("policy and baseline systems differ")
    metrics: dict[str, Any] = {}
    for name, field in METRICS.items():
        policy_values = np.asarray([float(policy_rows[s][field]) for s in systems])
        baseline_values = np.asarray([float(baseline_rows[s][field]) for s in systems])
        metrics[name] = {
            "policy_mean": float(policy_values.mean()),
            "baseline_mean": float(baseline_values.mean()),
            "direct_paired": _paired(policy_values - baseline_values),
        }
    disagreement = [
        float(
            np.mean(
                np.asarray(policy_rows[s]["selected_pair_ids"], dtype=object)
                != np.asarray(baseline_rows[s]["selected_pair_ids"], dtype=object)
            )
        )
        for s in systems
    ]
    source_headroom = np.asarray(
        [
            float(baseline_rows[s]["oracle_pool_discovery_ceiling"])
            - float(baseline_rows[s]["oracle_pool_confirmed_discoveries"])
            for s in systems
        ]
    )
    recovered = np.asarray(
        [
            float(policy_rows[s]["oracle_pool_confirmed_discoveries"])
            - float(baseline_rows[s]["oracle_pool_confirmed_discoveries"])
            for s in systems
        ]
    )
    return {
        "systems": systems,
        "metrics": metrics,
        "action_disagreement_vs_baseline": {
            "mean_per_system_round_fraction": float(np.mean(disagreement)),
            "systems_with_any_disagreement": int(np.sum(np.asarray(disagreement) > 0)),
        },
        "headroom": {
            "mean_source_finite_pool_headroom": float(source_headroom.mean()),
            "mean_recovered_headroom": float(recovered.mean()),
            "mean_fraction_recovered_where_source_has_headroom": (
                None
                if not np.any(source_headroom > 0)
                else float(np.mean(recovered[source_headroom > 0] / source_headroom[source_headroom > 0]))
            ),
        },
    }


def _diagnostic_strata(
    anchored_rows: dict[str, dict[str, Any]],
    delta_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    state_rows: list[dict[str, Any]] = []
    system_values: dict[str, dict[str, list[float]]] = {}
    for system, row in anchored_rows.items():
        values = {"rank_margin": [], "coupling": [], "disagreement": []}
        for event in row.get("policy_decision_rounds", []):
            diagnostic = event.get("selection_diagnostics")
            if not isinstance(diagnostic, dict) or diagnostic.get("kind") != "delta_hull_anchored_rollout":
                continue
            rank_margin = float(diagnostic["rank_margin"])
            coupling = float(diagnostic["coupling_score_normalized"])
            disagreement = int(
                diagnostic["selected_pair_id"] != diagnostic["delta_hull_action_id"]
            )
            values["rank_margin"].append(rank_margin)
            values["coupling"].append(coupling)
            values["disagreement"].append(float(disagreement))
            state_rows.append(
                {
                    "system": system,
                    "rank_margin": rank_margin,
                    "coupling": coupling,
                    "action_disagreement": disagreement,
                }
            )
        if values["rank_margin"]:
            system_values[system] = values
    if not state_rows:
        return {"status": "no_diagnostics"}
    rank_values = np.asarray([row["rank_margin"] for row in state_rows])
    coupling_values = np.asarray([row["coupling"] for row in state_rows])
    rank_q = np.quantile(rank_values, [0.25, 0.75])
    coupling_q = np.quantile(coupling_values, [0.25, 0.75])
    state_summary = {
        "state_count": len(state_rows),
        "rank_margin_quartiles": [float(x) for x in rank_q],
        "coupling_quartiles": [float(x) for x in coupling_q],
        "action_disagreement_by_coupling_quartile": {},
        "action_disagreement_by_rank_margin_quartile": {},
    }
    for label, mask in {
        "low": coupling_values <= coupling_q[0],
        "middle": (coupling_values > coupling_q[0]) & (coupling_values < coupling_q[1]),
        "high": coupling_values >= coupling_q[1],
    }.items():
        state_summary["action_disagreement_by_coupling_quartile"][label] = {
            "state_count": int(mask.sum()),
            "mean_action_disagreement": float(
                np.mean([state_rows[i]["action_disagreement"] for i in np.flatnonzero(mask)])
            ),
        }
    for label, mask in {
        "low": rank_values <= rank_q[0],
        "middle": (rank_values > rank_q[0]) & (rank_values < rank_q[1]),
        "high": rank_values >= rank_q[1],
    }.items():
        state_summary["action_disagreement_by_rank_margin_quartile"][label] = {
            "state_count": int(mask.sum()),
            "mean_action_disagreement": float(
                np.mean([state_rows[i]["action_disagreement"] for i in np.flatnonzero(mask)])
            ),
        }

    def system_effect(mask_name: str, stratum: str) -> dict[str, Any]:
        selected_systems = []
        effects = []
        for system, values in system_values.items():
            statistic = float(np.median(values[mask_name]))
            quartiles = coupling_q if mask_name == "coupling" else rank_q
            threshold = quartiles[0] if stratum == "low" else quartiles[1]
            selected = statistic <= threshold if stratum == "low" else statistic >= threshold
            if selected:
                selected_systems.append(system)
                effects.append(
                    float(anchored_rows[system]["oracle_pool_confirmed_discoveries"])
                    - float(delta_rows[system]["oracle_pool_confirmed_discoveries"])
                )
        return {
            "system_count": len(effects),
            "systems": sorted(selected_systems),
            "anchored_minus_delta_hull_T": (
                None if not effects else _paired(np.asarray(effects), bootstrap_seed=20260805)
            ),
        }

    return {
        "status": "complete_posterior_only_strata",
        "state_summary": state_summary,
        "system_level_T_effect": {
            "low_rank_margin": system_effect("rank_margin", "low"),
            "high_rank_margin": system_effect("rank_margin", "high"),
            "low_coupling": system_effect("coupling", "low"),
            "high_coupling": system_effect("coupling", "high"),
        },
    }


def summarize(*, input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("E32 summaries must remain outside Git")
    payloads_by_budget = {
        budget: [
            _load(input_root / f"e32-fold{fold + 1}-b{budget}-main.json", budget=budget)
            for fold in range(5)
        ]
        for budget in range(1, 7)
    }
    source_by_budget = {
        budget: _rows(payloads, "source_margin") for budget, payloads in payloads_by_budget.items()
    }
    curve = {
        str(budget): {
            policy: _metric_rows(_rows(payloads, policy), source_by_budget[budget])
            for policy in POLICIES
        }
        for budget, payloads in payloads_by_budget.items()
    }
    direct = {
        str(budget): {
            f"{policy}_vs_{baseline}": _metric_rows(
                _rows(payloads, policy), _rows(payloads, baseline)
            )
            for policy in POLICIES
            for baseline in POLICIES
            if policy != baseline
        }
        for budget, payloads in payloads_by_budget.items()
    }
    b6_payloads = payloads_by_budget[6]
    diagnostic = _diagnostic_strata(
        _rows(b6_payloads, "delta_hull_anchored_rollout"),
        _rows(b6_payloads, "delta_hull_active_search"),
    )
    protocol_identity = json.loads((input_root / "e32_protocol_identity.json").read_text())
    result = {
        "schema_version": 1,
        "status": "complete_e32_a_retrospective_crossfitted_development_summary",
        "input_root": str(input_root),
        "input_sha256": {
            str(path): _sha256(path) for path in sorted(input_root.glob("e32-*.json"))
        },
        "protocol_identity": protocol_identity,
        "policies": list(POLICIES),
        "budgets": list(range(1, 7)),
        "system_count": len(source_by_budget[6]),
        "curve_vs_source": curve,
        "direct_pairwise_contrasts": direct,
        "b6_posterior_only_rank_coupling_strata": diagnostic,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(input_root=args.input_root, output=args.output)
    print(json.dumps({"status": result["status"], "system_count": result["system_count"]}))


if __name__ == "__main__":
    main()
