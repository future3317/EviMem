"""Summarize direct paired mechanism comparisons on the frozen MatPES roster.

The reduced P0-v3 output stores all six policies for each exact chemical system
in the same fold file.  This tool compares IC-SARR directly with the registered
mechanism variants instead of inferring component importance from separate
comparisons to source margin.  It reads external raw outputs and writes only an
external derived summary; it never runs a policy and refuses repository-local
outputs or overwrites.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from summarize_p0_matpes_mechanism_suite import _paired_summary

EXPECTED_POLICIES = (
    "source_margin",
    "delta_hull_active_search",
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    "diagonal_ic_sarr",
    "independent_confirmation_source_rollout",
)
IC_SARR = "independent_confirmation_source_rollout"
METRICS = {
    "D": "causal_discoveries",
    "F": "final_causal_confirmed_discoveries",
    "T": "oracle_pool_confirmed_discoveries",
    "wall_seconds": "wall_seconds",
}
COMPARISONS = {
    "ic_sarr_vs_delta_hull": (IC_SARR, "delta_hull_active_search"),
    "ic_sarr_vs_ungated_sarr": (IC_SARR, "ungated_source_rollout"),
    "ic_sarr_vs_diagonal_covariance": (IC_SARR, "diagonal_ic_sarr"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payloads(root: Path) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(root.glob("matpes-p0v3-reduced-ablation-fold*-b6.json"))
    expected_names = [f"matpes-p0v3-reduced-ablation-fold{fold}-b6.json" for fold in range(1, 6)]
    if [path.name for path in paths] != expected_names:
        raise ValueError(f"expected exactly five reduced fold outputs, found {[path.name for path in paths]}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, payload in zip(paths, payloads, strict=True):
        if tuple(payload.get("active_policies", ())) != EXPECTED_POLICIES:
            raise ValueError(f"unexpected policy roster in {path}")
        if payload.get("status") != "exploratory_development_systems_only_not_confirmatory":
            raise ValueError(f"unexpected completion status in {path}: {payload.get('status')}")
        if payload.get("evaluation_systems_accessed") is not False:
            raise ValueError(f"evaluation-system access is not false in {path}")
        for system, result in payload.get("systems", {}).items():
            if int(result.get("budget", -1)) != 6:
                raise ValueError(f"wrong budget for {system} in {path}")
            strategies = result.get("strategies", {})
            if set(strategies) != set(EXPECTED_POLICIES) or len(strategies) != len(EXPECTED_POLICIES):
                raise ValueError(f"unexpected strategies for {system} in {path}")
    return paths, payloads


def _system_rows(payloads: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    rows: dict[str, dict[str, dict[str, Any]]] = {}
    for payload in payloads:
        for system, result in payload["systems"].items():
            if system in rows:
                raise ValueError(f"exact chemical system occurs in multiple folds: {system}")
            rows[system] = result["strategies"]
    if len(rows) != 230:
        raise ValueError(f"expected 230 exact chemical systems, found {len(rows)}")
    return rows


def _direct_comparison(
    rows: dict[str, dict[str, dict[str, Any]]], policy: str, baseline: str
) -> dict[str, Any]:
    systems = sorted(rows)
    metric_deltas: dict[str, np.ndarray] = {
        name: np.asarray(
            [float(rows[system][policy][field]) - float(rows[system][baseline][field]) for system in systems],
            dtype=float,
        )
        for name, field in METRICS.items()
    }
    metric_result: dict[str, Any] = {}
    for name, values in metric_deltas.items():
        policy_values = np.asarray([float(rows[system][policy][METRICS[name]]) for system in systems])
        baseline_values = np.asarray([float(rows[system][baseline][METRICS[name]]) for system in systems])
        metric_result[name] = {
            "ic_sarr_mean": float(policy_values.mean()),
            "baseline_mean": float(baseline_values.mean()),
            "direct_paired": _paired_summary(values),
        }
    delta_t = metric_deltas["T"].mean()
    delta_wall = metric_deltas["wall_seconds"].mean()
    metric_result["incremental_efficiency"] = {
        "definition": "mean(delta_T) / mean(delta_wall_seconds)",
        "delta_T_per_extra_wall_second": None if delta_wall <= 0.0 else float(delta_t / delta_wall),
        "delta_T_per_system": float(delta_t),
        "delta_wall_seconds_per_system": float(delta_wall),
        "units": "complete-pool confirmations per additional wall-second per system",
    }
    return {
        "policy": policy,
        "baseline": baseline,
        "system_count": len(systems),
        "metrics": metric_result,
        "system_differences": {
            system: {name: float(values[index]) for name, values in metric_deltas.items()}
            for index, system in enumerate(systems)
        },
    }


def summarize(*, input_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("direct mechanism summaries must remain outside Git")
    paths, payloads = _load_payloads(input_root)
    rows = _system_rows(payloads)
    first = payloads[0]
    shared_identity = {
        key: first.get(key)
        for key in (
            "task_sha256",
            "oracle_vault_sha256",
            "development_vault_sha256",
            "script_sha256",
            "posterior_sampler",
            "selected_action_is_only_reveal",
        )
    }
    for payload in payloads[1:]:
        if any(payload.get(key) != value for key, value in shared_identity.items()):
            raise ValueError("frozen protocol identity differs between fold outputs")
    result = {
        "schema_version": 1,
        "status": "complete_direct_paired_mechanism_comparisons",
        "input_root": str(input_root),
        "input_sha256": {str(path): _sha256(path) for path in paths},
        "system_count": len(rows),
        "fold_count": len(payloads),
        "budget": 6,
        "policy_roster": list(EXPECTED_POLICIES),
        "frozen_protocol_identity": {
            **shared_identity,
            "transport_model_checksums_by_fold": {
                path.name: payload["transport_model_checksum"]
                for path, payload in zip(paths, payloads, strict=True)
            },
        },
        "inference": {
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20260730,
            "sign_flip_draws": 100_000,
            "sign_flip_seed": 20260731,
            "sign_flip_method": "deterministic_monte_carlo",
            "statistical_unit": "exact chemical system",
        },
        "comparisons": {
            name: _direct_comparison(rows, policy, baseline)
            for name, (policy, baseline) in COMPARISONS.items()
        },
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
    print(f"output={args.output.resolve()}")
    print(json.dumps({"status": result["status"], "comparisons": list(result["comparisons"])}, indent=2))


if __name__ == "__main__":
    main()
