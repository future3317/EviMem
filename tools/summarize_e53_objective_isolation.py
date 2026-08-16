"""Summarize E53 matched-adjudicator objective-isolation trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from matmem.paired_randomization import paired_sign_randomization

POLICIES = (
    "posterior_mean_target_margin",
    "matched_local_hull_probability",
    "delta_hull_active_search",
)
CONTRASTS = {
    "delta_minus_local": (
        "delta_hull_active_search",
        "matched_local_hull_probability",
    ),
    "delta_minus_target_margin": (
        "delta_hull_active_search",
        "posterior_mean_target_margin",
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prefix_utility(strategy: dict[str, Any], budget: int) -> float:
    selected = tuple(str(value) for value in strategy["selected_pair_ids"])
    labels = {
        str(pair_id): bool(label)
        for pair_id, label in strategy["oracle_pool_final_labels_by_pair_id"].items()
    }
    if len(selected) != 6 or len(set(selected)) != 6:
        raise ValueError("E53 strategy must contain six unique selected IDs")
    if any(pair_id not in labels for pair_id in selected):
        raise ValueError("E53 evaluator labels do not cover every selected ID")
    return float(sum(labels[pair_id] for pair_id in selected[:budget]))


def _load_panel(
    paths: list[Path],
    *,
    expected_system_count: int,
    posterior_sample_count: int,
) -> dict[str, Any]:
    rows: dict[str, dict[str, np.ndarray]] = {}
    task_hashes: set[str] = set()
    vault_hashes: set[str] = set()
    input_hashes: dict[str, str] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing E53 output: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tuple(payload.get("active_policies", ())) != POLICIES:
            raise ValueError(f"unexpected E53 policy roster in {path}")
        if int(payload.get("config", {}).get("query_budget", -1)) != 6:
            raise ValueError(f"E53 unit is not a B=6 trajectory: {path}")
        if int(payload.get("config", {}).get("posterior_sample_count", -1)) != posterior_sample_count:
            raise ValueError(
                f"E53 unit has the wrong posterior sample count "
                f"(expected {posterior_sample_count}): {path}"
            )
        task_hashes.add(str(payload.get("task_sha256")))
        vault_hashes.add(str(payload.get("oracle_vault_sha256")))
        input_hashes[str(path.resolve())] = _sha256(path)
        for system, system_payload in payload.get("systems", {}).items():
            if system in rows:
                raise ValueError(f"chemical system occurs twice: {system}")
            if int(system_payload.get("budget", -1)) != 6:
                raise ValueError(f"wrong system budget for {system}")
            strategies = system_payload.get("strategies", {})
            if set(strategies) != set(POLICIES):
                raise ValueError(f"system has the wrong E53 policy roster: {system}")
            rows[str(system)] = {
                policy: np.asarray(
                    [_prefix_utility(strategies[policy], budget) for budget in range(1, 7)],
                    dtype=np.float64,
                )
                for policy in POLICIES
            }
    if len(rows) != expected_system_count:
        raise ValueError(
            f"expected {expected_system_count} E53 systems, found {len(rows)}"
        )
    if len(task_hashes) != 1 or len(vault_hashes) != 1:
        raise ValueError("E53 task/vault identity changes within a panel")
    return {
        "rows": rows,
        "task_sha256": next(iter(task_hashes)),
        "vault_sha256": next(iter(vault_hashes)),
        "input_sha256": input_hashes,
    }


def _auc(values: np.ndarray) -> np.ndarray:
    if values.shape[0] != 6:
        raise ValueError("E53 AUC expects B=1..6")
    return 0.5 * values[0] + values[1:-1].sum(axis=0) + 0.5 * values[-1]


def _summarize_panel(
    panel: dict[str, Any],
    *,
    randomization_draws: int,
    seed: int,
) -> dict[str, Any]:
    systems = sorted(panel["rows"])
    arrays = {
        policy: np.column_stack([panel["rows"][system][policy] for system in systems])
        for policy in POLICIES
    }
    budgets: dict[str, Any] = {}
    for budget_index, budget in enumerate(range(1, 7)):
        contrasts = {}
        for name, (left, right) in CONTRASTS.items():
            inference = paired_sign_randomization(
                arrays[left][budget_index] - arrays[right][budget_index],
                draws=randomization_draws,
                seed=seed + budget,
            )
            contrasts[name] = inference.model_dump(mode="json")
        budgets[str(budget)] = {
            "absolute_mean_T": {
                policy: float(np.mean(values[budget_index]))
                for policy, values in arrays.items()
            },
            "contrasts": contrasts,
        }
    integrated = {}
    for name, (left, right) in CONTRASTS.items():
        inference = paired_sign_randomization(
            _auc(arrays[left]) - _auc(arrays[right]),
            draws=randomization_draws,
            seed=seed + 100,
        )
        integrated[name] = inference.model_dump(mode="json")
    return {
        "system_count": len(systems),
        "system_set_sha256": hashlib.sha256("\n".join(systems).encode()).hexdigest(),
        "task_sha256": panel["task_sha256"],
        "vault_sha256": panel["vault_sha256"],
        "input_sha256": panel["input_sha256"],
        "budgets": budgets,
        "integrated_budget_effects": integrated,
    }


def summarize(
    *,
    development_root: Path,
    secondary_path: Path | None,
    output: Path,
    expected_development_system_count: int,
    expected_secondary_system_count: int | None,
    randomization_draws: int = 100_000,
    posterior_sample_count: int = 1024,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("E53 summaries must remain outside Git")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E53 summary: {output}")
    development_paths = [
        development_root / f"fold{fold + 1}-b6.json" for fold in range(5)
    ]
    development = _load_panel(
        development_paths,
        expected_system_count=expected_development_system_count,
        posterior_sample_count=posterior_sample_count,
    )
    panels = {
        "development": _summarize_panel(
            development,
            randomization_draws=randomization_draws,
            seed=20260810,
        )
    }
    if secondary_path is not None:
        if expected_secondary_system_count is None:
            raise ValueError("secondary E53 summary needs its expected system count")
        secondary = _load_panel(
            [secondary_path],
            expected_system_count=expected_secondary_system_count,
            posterior_sample_count=posterior_sample_count,
        )
        panels["secondary"] = _summarize_panel(
            secondary,
            randomization_draws=randomization_draws,
            seed=20260811,
        )
    result = {
        "schema_version": 1,
        "status": "e53_objective_isolation_complete",
        "policies": POLICIES,
        "primary_identification_contrast": "delta_minus_local",
        "analysis_unit": "exact_chemical_system",
        "inference": "paired sign randomization with interval inversion",
        "panels": panels,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-development-systems", type=int, default=230)
    parser.add_argument("--expected-secondary-systems", type=int, default=94)
    parser.add_argument("--randomization-draws", type=int, default=100_000)
    parser.add_argument("--posterior-sample-count", type=int, default=1024)
    args = parser.parse_args()
    result = summarize(
        development_root=args.development_root,
        secondary_path=args.secondary,
        output=args.output,
        expected_development_system_count=args.expected_development_systems,
        expected_secondary_system_count=(
            args.expected_secondary_systems if args.secondary is not None else None
        ),
        randomization_draws=args.randomization_draws,
        posterior_sample_count=args.posterior_sample_count,
    )
    print(json.dumps({"status": result["status"], "panels": list(result["panels"])}))


if __name__ == "__main__":
    main()
