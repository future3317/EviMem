"""Summarize the fully registered, retrospective P0-v2 MatPES suite.

This utility deliberately refuses partial schedules.  It operates on the
external raw-result directory and emits a single external JSON summary at the
exact chemical-system unit; it is not a policy or experiment runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

CORE_POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "independent_confirmation_source_rollout",
)
ABLATION_POLICIES = (
    "source_margin",
    "posterior_mean_target_margin",
    "ridge_margin",
    "ridge_uncertainty",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "diagonal_ic_sarr",
    "independent_mc_ic_sarr",
    "independent_confirmation_source_rollout",
)
RANDOM_SEEDS = tuple(range(20270721, 20270726))
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


def _paired_summary(
    differences: np.ndarray,
    *,
    bootstrap_seed: int = 20260730,
    bootstrap_replicates: int = 20_000,
    sign_flip_seed: int = 20260731,
    sign_flip_draws: int = 100_000,
) -> dict[str, Any]:
    """Return registered paired-system inference without pseudo-replication."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("paired summaries require one nonempty finite value per system")
    bootstrap_rng = np.random.default_rng(bootstrap_seed)
    indices = bootstrap_rng.integers(0, len(values), size=(bootstrap_replicates, len(values)))
    bootstrap = values[indices].mean(axis=1)
    sign_rng = np.random.default_rng(sign_flip_seed)
    signs = sign_rng.choice((-1.0, 1.0), size=(sign_flip_draws, len(values)))
    observed = abs(float(values.sum()))
    randomized = np.abs(np.sum(signs * values[None, :], axis=1))
    return {
        "system_count": int(len(values)),
        "paired_mean_difference": float(values.mean()),
        "paired_bootstrap_95ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "wins": int(np.sum(values > 0.0)),
        "ties": int(np.sum(values == 0.0)),
        "losses": int(np.sum(values < 0.0)),
        "two_sided_sign_flip_p": float((np.sum(randomized >= observed) + 1) / (sign_flip_draws + 1)),
        "sign_flip_method": "deterministic_monte_carlo",
        "sign_flip_seed": sign_flip_seed,
        "sign_flip_draws": sign_flip_draws,
    }


def _expected_path(root: Path, tier: str, fold: int, budget: int, suffix: str) -> Path:
    return root / f"matpes-p0v2-{tier}-fold{fold}-b{budget}-{suffix}.json"


def _load(path: Path, *, expected_policies: tuple[str, ...], budget: int) -> dict[str, Any]:
    if not path.exists():
        failure = path.with_suffix(".failure.json")
        if failure.exists():
            raise RuntimeError(f"registered output failed: {failure}")
        raise FileNotFoundError(f"registered output is incomplete: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload.get("active_policies", ())) != expected_policies:
        raise ValueError(f"unexpected policy roster in {path}")
    for system, result in payload.get("systems", {}).items():
        if int(result["budget"]) != budget:
            raise ValueError(f"wrong budget for {system} in {path}")
        if set(result["strategies"]) != set(expected_policies) or len(result["strategies"]) != len(expected_policies):
            raise ValueError(f"unexpected strategies for {system} in {path}")
    return payload


def _system_rows(payloads: list[dict[str, Any]], policy: str) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        for system, result in payload["systems"].items():
            if system in rows:
                raise ValueError(f"exact chemical system occurs twice: {system}")
            rows[system] = result["strategies"][policy]
    if not rows:
        raise ValueError("no system rows")
    return rows


def _policy_summary(
    *,
    policy_rows: dict[str, dict[str, Any]],
    source_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if set(policy_rows) != set(source_rows):
        raise ValueError("policy and source systems differ")
    systems = sorted(policy_rows)
    metric_summary = {
        name: {
            "policy_mean": float(np.mean([float(policy_rows[system][field]) for system in systems])),
            "source_mean": float(np.mean([float(source_rows[system][field]) for system in systems])),
            "paired_vs_source": _paired_summary(
                np.asarray(
                    [float(policy_rows[system][field]) - float(source_rows[system][field]) for system in systems]
                )
            ),
        }
        for name, field in METRICS.items()
    }
    action_disagreement = [
        float(
            np.mean(
                np.asarray(policy_rows[system]["selected_pair_ids"], dtype=object)
                != np.asarray(source_rows[system]["selected_pair_ids"], dtype=object)
            )
        )
        for system in systems
    ]
    source_headroom = np.asarray(
        [
            float(source_rows[system]["oracle_pool_discovery_ceiling"])
            - float(source_rows[system]["oracle_pool_confirmed_discoveries"])
            for system in systems
        ]
    )
    recovered = np.asarray(
        [
            float(policy_rows[system]["oracle_pool_confirmed_discoveries"])
            - float(source_rows[system]["oracle_pool_confirmed_discoveries"])
            for system in systems
        ]
    )
    eligible = source_headroom > 0.0
    diagnostics: Counter[str] = Counter()
    accepted = 0
    rounds = 0
    for row in policy_rows.values():
        for event in row.get("policy_decision_rounds", []):
            diagnostic = event.get("selection_diagnostics")
            if not isinstance(diagnostic, dict):
                continue
            rounds += 1
            diagnostics[str(diagnostic.get("fallback_reason", "missing"))] += 1
            accepted += int(diagnostic.get("selected_pair_id") != diagnostic.get("source_pair_id"))
    return {
        "systems": systems,
        "metrics": metric_summary,
        "action_disagreement_vs_source": {
            "mean_per_system_round_fraction": float(np.mean(action_disagreement)),
            "systems_with_any_disagreement": int(np.sum(np.asarray(action_disagreement) > 0.0)),
        },
        "headroom": {
            "mean_source_finite_pool_headroom": float(source_headroom.mean()),
            "mean_recovered_headroom": float(recovered.mean()),
            "mean_fraction_recovered_where_source_has_headroom": (
                None if not np.any(eligible) else float(np.mean(recovered[eligible] / source_headroom[eligible]))
            ),
        },
        "gate_diagnostics": {
            "decision_round_count": rounds,
            "accepted_non_source_deviations": accepted,
            "fallback_reason_counts": dict(sorted(diagnostics.items())),
        },
    }


def _random_mean_rows(rows_by_seed: list[dict[str, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    all_systems = [set(rows) for rows in rows_by_seed]
    if any(systems != all_systems[0] for systems in all_systems[1:]):
        raise ValueError("random seeds do not contain identical systems")
    rows: dict[str, dict[str, Any]] = {}
    for system in sorted(all_systems[0]):
        strategies = [rows[system] for rows in rows_by_seed]
        averaged = {field: float(np.mean([float(row[field]) for row in strategies])) for field in METRICS.values()}
        averaged["selected_pair_ids"] = strategies[0]["selected_pair_ids"]
        averaged["policy_decision_rounds"] = []
        averaged["oracle_pool_discovery_ceiling"] = strategies[0]["oracle_pool_discovery_ceiling"]
        rows[system] = averaged
    return rows


def summarize(*, input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("P0 summaries must remain outside Git")
    core: dict[int, list[dict[str, Any]]] = {}
    for budget in range(1, 7):
        core[budget] = [
            _load(
                _expected_path(input_root, "core", fold, budget, "main"),
                expected_policies=CORE_POLICIES,
                budget=budget,
            )
            for fold in range(1, 6)
        ]
    ablation = [
        _load(
            _expected_path(input_root, "ablation", fold, 6, "main"),
            expected_policies=ABLATION_POLICIES,
            budget=6,
        )
        for fold in range(1, 6)
    ]
    random = {
        seed: [
            _load(
                _expected_path(input_root, "ablation", fold, 6, f"random-seed{seed}"),
                expected_policies=("random",),
                budget=6,
            )
            for fold in range(1, 6)
        ]
        for seed in RANDOM_SEEDS
    }
    source_by_budget = {budget: _system_rows(payloads, "source_margin") for budget, payloads in core.items()}
    curve = {
        str(budget): {
            policy: _policy_summary(
                policy_rows=_system_rows(payloads, policy), source_rows=source_by_budget[budget]
            )
            for policy in CORE_POLICIES
        }
        for budget, payloads in core.items()
    }
    ablation_source = _system_rows(ablation, "source_margin")
    ablation_summary = {
        policy: _policy_summary(policy_rows=_system_rows(ablation, policy), source_rows=ablation_source)
        for policy in ABLATION_POLICIES
    }
    random_rows = _random_mean_rows([_system_rows(random[seed], "random") for seed in RANDOM_SEEDS])
    ablation_summary["random_mean_over_five_registered_seeds"] = _policy_summary(
        policy_rows=random_rows, source_rows=ablation_source
    )
    output_payload = {
        "schema_version": 1,
        "status": "complete_retrospective_crossfitted_development_mechanism_summary",
        "input_root": str(input_root),
        "input_sha256": {
            str(path): _sha256(path)
            for path in sorted(input_root.glob("matpes-p0v2-*.json"))
        },
        "bootstrap": {"replicates": 20_000, "seed": 20260730},
        "sign_flip": {"draws": 100_000, "seed": 20260731},
        "curve": curve,
        "b6_baseline_component_audit": ablation_summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(input_root=args.input_root, output=args.output)
    print(f"output={args.output.resolve()}")
    print(json.dumps({"status": result["status"], "curve_budgets": sorted(result["curve"])}, indent=2))


if __name__ == "__main__":
    main()
