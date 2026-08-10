"""Summarize the E54 CAL-style hull-entropy campaign.

The summary treats an exact chemical system as the resampling unit and refuses
to report incomplete trajectories, input-identity changes, or missing CAL
diagnostics. Raw campaign JSON remains outside Git.
"""

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
    "delta_hull_active_search",
    "cal_style_hull_entropy",
)
CONTRASTS = {
    "delta_minus_cal": ("delta_hull_active_search", "cal_style_hull_entropy"),
    "delta_minus_target_margin": (
        "delta_hull_active_search",
        "posterior_mean_target_margin",
    ),
}
CAL_KIND = "cal_style_hull_entropy"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prefix_utility(strategy: dict[str, Any], budget: int) -> float:
    selected = tuple(str(value) for value in strategy["selected_pair_ids"])
    labels = {
        str(pair_id): bool(label)
        for pair_id, label in strategy["oracle_pool_final_labels_by_pair_id"].items()
    }
    if len(selected) != 6 or len(set(selected)) != 6:
        raise ValueError("E54 strategy must contain six unique selected IDs")
    if any(pair_id not in labels for pair_id in selected):
        raise ValueError("E54 evaluator labels do not cover every selected ID")
    return float(sum(labels[pair_id] for pair_id in selected[:budget]))


def _cal_rounds(strategy: dict[str, Any], *, system: str) -> list[dict[str, Any]]:
    rounds = strategy.get("policy_decision_rounds", [])
    diagnostics = [
        event.get("selection_diagnostics")
        for event in rounds
        if event.get("selection_diagnostics", {}).get("kind") == CAL_KIND
    ]
    if len(diagnostics) != 6 or any(item is None for item in diagnostics):
        raise ValueError(f"missing CAL diagnostics for {system}")
    required = (
        "wall_time_seconds",
        "state_candidate_count",
        "evaluation_composition_count",
        "posterior_sample_count",
        "fantasy_count",
        "relative_ridge",
    )
    for diagnostic in diagnostics:
        if any(key not in diagnostic for key in required):
            raise ValueError(f"incomplete CAL diagnostics for {system}")
        if not all(np.isfinite(float(diagnostic[key])) for key in required):
            raise ValueError(f"non-finite CAL diagnostics for {system}")
    return [dict(item) for item in diagnostics]


def _load_panel(
    paths: list[Path],
    *,
    expected_system_count: int,
    expected_fold_indices: set[int],
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    task_hashes: set[str] = set()
    vault_hashes: set[str] = set()
    crossfit_hashes: set[str] = set()
    fold_indices: set[int] = set()
    input_hashes: dict[str, str] = {}
    query_systems_seen: set[str] = set()

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing E54 output: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tuple(payload.get("active_policies", ())) != POLICIES:
            raise ValueError(f"unexpected E54 policy roster in {path}")
        config = payload.get("config", {})
        expected_config = {
            "query_budget": 6,
            "posterior_sample_count": 200,
            "fantasy_count": 10,
            "hull_backend": "fixed_composition",
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise ValueError(f"E54 unit has wrong {key}: {path}")
        fold_index = int(config.get("crossfit_fold_index", -1))
        fold_indices.add(fold_index)
        if fold_index not in expected_fold_indices:
            raise ValueError(f"unexpected E54 fold index in {path}")
        task_hashes.add(str(payload.get("task_sha256")))
        vault_hashes.add(str(payload.get("oracle_vault_sha256")))
        crossfit_hashes.add(str(config.get("crossfit_manifest_sha256")))
        input_hashes[str(path.resolve())] = _sha256(path)
        query_systems = {str(system) for system in payload.get("query_systems", ())}
        if query_systems_seen & query_systems:
            raise ValueError("E54 query systems overlap across folds")
        query_systems_seen.update(query_systems)
        for system, system_payload in payload.get("systems", {}).items():
            if system in rows:
                raise ValueError(f"chemical system occurs twice: {system}")
            if int(system_payload.get("budget", -1)) != 6:
                raise ValueError(f"wrong system budget for {system}")
            strategies = system_payload.get("strategies", {})
            if set(strategies) != set(POLICIES):
                raise ValueError(f"system has the wrong E54 policy roster: {system}")
            rows[str(system)] = {
                policy: np.asarray(
                    [_prefix_utility(strategies[policy], budget) for budget in range(1, 7)],
                    dtype=np.float64,
                )
                for policy in POLICIES
            }
            rows[str(system)]["transport_element_support"] = bool(
                system_payload.get("transport_element_support", True)
            )
            rows[str(system)]["cal_diagnostics"] = _cal_rounds(
                strategies["cal_style_hull_entropy"], system=str(system)
            )

        if payload.get("transport_fit_and_query_systems_disjoint") is not True:
            raise ValueError(f"E54 unit does not attest fit/query disjointness: {path}")
        if set(payload.get("transport_fit_systems", ())) & query_systems:
            raise ValueError(f"E54 fit/query overlap in {path}")

    if len(rows) != expected_system_count:
        raise ValueError(f"expected {expected_system_count} E54 systems, found {len(rows)}")
    if fold_indices != expected_fold_indices:
        raise ValueError(f"E54 fold roster is incomplete: {sorted(fold_indices)}")
    if len(task_hashes) != 1 or len(vault_hashes) != 1 or len(crossfit_hashes) != 1:
        raise ValueError("E54 task/vault/cross-fit identity changes within a panel")
    return {
        "rows": rows,
        "task_sha256": next(iter(task_hashes)),
        "vault_sha256": next(iter(vault_hashes)),
        "crossfit_manifest_sha256": next(iter(crossfit_hashes)),
        "input_sha256": input_hashes,
    }


def _auc(values: np.ndarray) -> np.ndarray:
    if values.shape[0] != 6:
        raise ValueError("E54 AUC expects B=1..6")
    return 0.5 * values[0] + values[1:-1].sum(axis=0) + 0.5 * values[-1]


def _runtime_summary(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    def values(key: str) -> np.ndarray:
        return np.asarray([float(item[key]) for item in diagnostics], dtype=np.float64)

    def describe(key: str) -> dict[str, float]:
        array = values(key)
        return {
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "max": float(np.max(array)),
        }

    return {
        "state_count": len(diagnostics),
        "wall_time_seconds": describe("wall_time_seconds"),
        "state_candidate_count": describe("state_candidate_count"),
        "evaluation_composition_count": describe("evaluation_composition_count"),
        "grid_sizes": sorted({int(item["evaluation_composition_count"]) for item in diagnostics}),
        "posterior_sample_counts": sorted({int(item["posterior_sample_count"]) for item in diagnostics}),
        "fantasy_counts": sorted({int(item["fantasy_count"]) for item in diagnostics}),
        "relative_ridges": sorted({float(item["relative_ridge"]) for item in diagnostics}),
    }


def _summarize_panel(
    panel: dict[str, Any], *, randomization_draws: int, seed: int
) -> dict[str, Any]:
    systems = sorted(panel["rows"])
    arrays = {
        policy: np.column_stack([panel["rows"][system][policy] for system in systems])
        for policy in POLICIES
    }
    diagnostics = [
        diagnostic
        for system in systems
        for diagnostic in panel["rows"][system]["cal_diagnostics"]
    ]
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
        "transport_element_supported_system_count": sum(
            bool(panel["rows"][system]["transport_element_support"])
            for system in systems
        ),
        "system_set_sha256": hashlib.sha256("\n".join(systems).encode()).hexdigest(),
        "task_sha256": panel["task_sha256"],
        "vault_sha256": panel["vault_sha256"],
        "crossfit_manifest_sha256": panel["crossfit_manifest_sha256"],
        "input_sha256": panel["input_sha256"],
        "budgets": budgets,
        "integrated_budget_effects": integrated,
        "cal_diagnostics": _runtime_summary(diagnostics),
    }


def _assert_no_failures(root: Path) -> None:
    failures = sorted(root.glob("**/*.failure.json"))
    if failures:
        raise ValueError("E54 failure markers present: " + ", ".join(map(str, failures)))


def summarize(
    *,
    development_root: Path,
    secondary_path: Path | None,
    output: Path,
    expected_development_system_count: int,
    expected_secondary_system_count: int | None,
    randomization_draws: int = 100_000,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("E54 summaries must remain outside Git")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite E54 summary: {output}")
    _assert_no_failures(development_root)
    development_paths = [
        development_root / f"fold{fold + 1}-b6.json" for fold in range(5)
    ]
    development = _load_panel(
        development_paths,
        expected_system_count=expected_development_system_count,
        expected_fold_indices=set(range(5)),
    )
    panels = {
        "development": _summarize_panel(
            development, randomization_draws=randomization_draws, seed=20260810
        )
    }
    if secondary_path is not None:
        if expected_secondary_system_count is None:
            raise ValueError("secondary E54 summary needs its expected system count")
        _assert_no_failures(secondary_path.parent)
        secondary = _load_panel(
            [secondary_path],
            expected_system_count=expected_secondary_system_count,
            expected_fold_indices={0},
        )
        panels["secondary"] = _summarize_panel(
            secondary, randomization_draws=randomization_draws, seed=20260811
        )
    result = {
        "schema_version": 1,
        "status": "e54_cal_style_hull_entropy_complete",
        "protocol": "E54-cal-style-hull-entropy-v1",
        "policies": POLICIES,
        "analysis_unit": "exact_chemical_system",
        "primary_contrast": "delta_minus_cal",
        "inference": "paired sign randomization with interval inversion",
        "trajectory_design": "one B=6 trajectory with B=1..6 prefixes",
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
    )
    print(json.dumps({"status": result["status"], "panels": list(result["panels"])}))


if __name__ == "__main__":
    main()
