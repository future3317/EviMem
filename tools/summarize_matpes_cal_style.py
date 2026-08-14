"""Summarize matched MatPES policy panels.

The summary treats an exact chemical system as the resampling unit and refuses
to report incomplete trajectories, input-identity changes, or missing
policy-specific diagnostics. It supports the E54 three-policy CAL comparison
and the E55 single-policy complete-pool mean-hull-margin ablation. Raw campaign
JSON remains outside Git.
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
BASELINE_POLICIES = ("complete_pool_posterior_mean_hull_margin",)
E54_PROTOCOL = "E54-cal-style-hull-entropy-v1"
BASELINE_PROTOCOL = "E55-complete-pool-mean-hull-margin-v1"
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


def _baseline_rounds(strategy: dict[str, Any], *, system: str) -> list[dict[str, Any]]:
    rounds = strategy.get("policy_decision_rounds", [])
    diagnostics: list[dict[str, Any]] = []
    for event in rounds:
        diagnostic = event.get("selection_diagnostics")
        if isinstance(diagnostic, dict) and diagnostic.get("kind") == BASELINE_POLICIES[0]:
            diagnostics.append(diagnostic)
    if len(diagnostics) != 6:
        raise ValueError(f"missing complete-pool margin diagnostics for {system}")
    margin_key = "complete_pool_posterior_mean_hull_margins"
    for diagnostic in diagnostics:
        if diagnostic.get("diagnostic_schema_version") != 1:
            raise ValueError(f"wrong baseline diagnostic schema for {system}")
        candidate_ids = tuple(str(item) for item in diagnostic.get("candidate_pair_ids", ()))
        margins = diagnostic.get(margin_key)
        if not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(f"invalid baseline candidate roster for {system}")
        if not isinstance(margins, dict) or set(margins) != set(candidate_ids):
            raise ValueError(f"incomplete baseline margins for {system}")
        if not all(np.isfinite(float(margins[pair_id])) for pair_id in candidate_ids):
            raise ValueError(f"non-finite baseline margins for {system}")
        if str(diagnostic.get("selected_pair_id")) not in candidate_ids:
            raise ValueError(f"baseline selected ID is outside candidate roster for {system}")
    return [dict(item) for item in diagnostics]


def _baseline_fallback_rounds(strategy: dict[str, Any], *, system: str) -> None:
    rounds = strategy.get("policy_decision_rounds", [])
    if len(rounds) != 6 or any(event.get("selection_diagnostics") is not None for event in rounds):
        raise ValueError(f"invalid complete-pool deterministic fallback for {system}")


def _is_common_fallback(
    strategies: dict[str, dict[str, Any]], *, system: str
) -> bool:
    """Recognize the deterministic unsupported-element fallback exactly.

    Unsupported transport elements are deliberately routed to the shared
    source-margin fallback before any posterior/CAL computation.  It is safe
    to retain those systems in utility summaries only when every policy has
    made the same six decisions and therefore has identical utility prefixes.
    """
    selected: list[tuple[str, ...]] = []
    utilities: list[tuple[float, ...]] = []
    for policy in POLICIES:
        strategy = strategies[policy]
        rounds = strategy.get("policy_decision_rounds", [])
        if len(rounds) != 6:
            return False
        if policy == CAL_KIND and any(
            event.get("selection_diagnostics", {}).get("kind") == CAL_KIND
            for event in rounds
        ):
            return False
        try:
            selected.append(
                tuple(str(pair_id) for pair_id in strategy["selected_pair_ids"])
            )
            utilities.append(
                tuple(_prefix_utility(strategy, budget) for budget in range(1, 7))
            )
        except (KeyError, TypeError, ValueError):
            return False
    return selected[0] == selected[1] == selected[2] and (
        utilities[0] == utilities[1] == utilities[2]
    )


def _load_panel(
    paths: list[Path],
    *,
    expected_system_count: int,
    expected_fold_indices: set[int],
    policies: tuple[str, ...],
) -> dict[str, Any]:
    if policies not in (POLICIES, BASELINE_POLICIES):
        raise ValueError(f"unsupported MatPES summary policy contract: {policies}")
    is_e54 = policies == POLICIES
    rows: dict[str, dict[str, Any]] = {}
    task_hashes: set[str] = set()
    vault_hashes: set[str] = set()
    crossfit_hashes: set[str] = set()
    fold_indices: set[int] = set()
    input_hashes: dict[str, str] = {}
    query_systems_seen: set[str] = set()

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing MatPES output: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if tuple(payload.get("active_policies", ())) != policies:
            raise ValueError(f"unexpected MatPES policy roster in {path}")
        config = payload.get("config", {})
        expected_config = {
            "query_budget": 6,
            "posterior_sample_count": 200,
            "fantasy_count": 10,
            "hull_backend": "fixed_composition",
        }
        for key, expected in expected_config.items():
            if config.get(key) != expected:
                raise ValueError(f"MatPES unit has wrong {key}: {path}")
        fold_index = int(config.get("crossfit_fold_index", -1))
        fold_indices.add(fold_index)
        if fold_index not in expected_fold_indices:
            raise ValueError(f"unexpected MatPES fold index in {path}")
        task_hashes.add(str(payload.get("task_sha256")))
        vault_hashes.add(str(payload.get("oracle_vault_sha256")))
        crossfit_hashes.add(str(config.get("crossfit_manifest_sha256")))
        input_hashes[str(path.resolve())] = _sha256(path)
        query_systems = {str(system) for system in payload.get("query_systems", ())}
        if query_systems_seen & query_systems:
            raise ValueError("MatPES query systems overlap across folds")
        query_systems_seen.update(query_systems)
        for system, system_payload in payload.get("systems", {}).items():
            if system in rows:
                raise ValueError(f"chemical system occurs twice: {system}")
            if int(system_payload.get("budget", -1)) != 6:
                raise ValueError(f"wrong system budget for {system}")
            strategies = system_payload.get("strategies", {})
            if set(strategies) != set(policies):
                raise ValueError(f"system has the wrong MatPES policy roster: {system}")
            transport_element_support = bool(
                system_payload.get("transport_element_support", True)
            )
            baseline_diagnostics: list[dict[str, Any]] = []
            deterministic_fallback = False
            if not is_e54:
                if transport_element_support:
                    baseline_diagnostics = _baseline_rounds(
                        strategies[BASELINE_POLICIES[0]], system=str(system)
                    )
                else:
                    _baseline_fallback_rounds(
                        strategies[BASELINE_POLICIES[0]], system=str(system)
                    )
                    deterministic_fallback = True
                cal_diagnostics = []
                common_fallback = False
            elif transport_element_support:
                cal_diagnostics = _cal_rounds(
                    strategies["cal_style_hull_entropy"], system=str(system)
                )
                common_fallback = False
            else:
                if not _is_common_fallback(strategies, system=str(system)):
                    raise ValueError(
                        f"unsupported system is not a common fallback: {system}"
                    )
                cal_diagnostics = []
                common_fallback = True
                deterministic_fallback = True
            rows[str(system)] = {
                policy: np.asarray(
                    [_prefix_utility(strategies[policy], budget) for budget in range(1, 7)],
                    dtype=np.float64,
                )
                for policy in policies
            }
            rows[str(system)]["transport_element_support"] = transport_element_support
            rows[str(system)]["common_fallback"] = common_fallback
            rows[str(system)]["deterministic_fallback"] = deterministic_fallback
            rows[str(system)]["cal_diagnostics"] = cal_diagnostics
            rows[str(system)]["baseline_diagnostics"] = baseline_diagnostics

        if payload.get("transport_fit_and_query_systems_disjoint") is not True:
            raise ValueError(f"MatPES unit does not attest fit/query disjointness: {path}")
        if set(payload.get("transport_fit_systems", ())) & query_systems:
            raise ValueError(f"MatPES fit/query overlap in {path}")

    if len(rows) != expected_system_count:
        raise ValueError(f"expected {expected_system_count} MatPES systems, found {len(rows)}")
    if fold_indices != expected_fold_indices:
        raise ValueError(f"MatPES fold roster is incomplete: {sorted(fold_indices)}")
    if len(task_hashes) != 1 or len(vault_hashes) != 1 or len(crossfit_hashes) != 1:
        raise ValueError("MatPES task/vault/cross-fit identity changes within a panel")
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


def _baseline_diagnostic_summary(
    diagnostics: list[dict[str, Any]],
    *,
    system_count: int,
    panel_system_count: int,
) -> dict[str, Any]:
    return {
        "state_count": len(diagnostics),
        "system_count": system_count,
        "panel_system_count": panel_system_count,
        "coverage_fraction": system_count / panel_system_count,
        "candidate_counts": sorted(
            {len(item["candidate_pair_ids"]) for item in diagnostics}
        ),
        "schema_versions": sorted(
            {int(item["diagnostic_schema_version"]) for item in diagnostics}
        ),
    }


def _integrated_policy_metrics(
    arrays: dict[str, np.ndarray],
) -> dict[str, dict[str, float]]:
    return {
        policy: {
            "mean_auc": float(np.mean(_auc(values))),
            "system_count": int(values.shape[1]),
        }
        for policy, values in arrays.items()
    }


def _summarize_effects(
    arrays: dict[str, np.ndarray], *, randomization_draws: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
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
    return budgets, integrated


def _summarize_panel(
    panel: dict[str, Any],
    *,
    policies: tuple[str, ...],
    randomization_draws: int,
    seed: int,
) -> dict[str, Any]:
    systems = sorted(panel["rows"])
    arrays = {
        policy: np.column_stack([panel["rows"][system][policy] for system in systems])
        for policy in policies
    }
    if policies == POLICIES:
        diagnostics = [
            diagnostic
            for system in systems
            for diagnostic in panel["rows"][system]["cal_diagnostics"]
        ]
        budgets, integrated = _summarize_effects(
            arrays, randomization_draws=randomization_draws, seed=seed
        )
    else:
        diagnostics = [
            diagnostic
            for system in systems
            for diagnostic in panel["rows"][system]["baseline_diagnostics"]
        ]
        budgets = {
            str(budget): {
                "absolute_mean_T": {
                    policy: float(np.mean(values[budget - 1]))
                    for policy, values in arrays.items()
                },
                "contrasts": {},
            }
            for budget in range(1, 7)
        }
        integrated = {}
    supported_systems = [
        system
        for system in systems
        if bool(panel["rows"][system]["transport_element_support"])
    ]
    if not supported_systems:
        raise ValueError("E54 panel has no transport-supported CAL systems")
    supported_arrays = {
        policy: np.column_stack(
            [panel["rows"][system][policy] for system in supported_systems]
        )
        for policy in policies
    }
    if policies == POLICIES:
        supported_budgets, supported_integrated = _summarize_effects(
            supported_arrays, randomization_draws=randomization_draws, seed=seed
        )
    else:
        supported_budgets = {}
        supported_integrated = {}
    common_fallback_systems = [
        system for system in systems if panel["rows"][system]["common_fallback"]
    ]
    deterministic_fallback_systems = [
        system for system in systems if panel["rows"][system]["deterministic_fallback"]
    ]
    baseline_diagnostic_systems = [
        system for system in systems if panel["rows"][system]["baseline_diagnostics"]
    ]
    return {
        "system_count": len(systems),
        "transport_element_supported_system_count": sum(
            bool(panel["rows"][system]["transport_element_support"])
            for system in systems
        ),
        "cal_executed_transport_supported_system_count": len(supported_systems),
        "common_fallback_system_count": len(common_fallback_systems),
        "deterministic_fallback_system_count": len(deterministic_fallback_systems),
        "common_fallback_system_set_sha256": hashlib.sha256(
            "\n".join(common_fallback_systems).encode()
        ).hexdigest(),
        "system_set_sha256": hashlib.sha256("\n".join(systems).encode()).hexdigest(),
        "task_sha256": panel["task_sha256"],
        "vault_sha256": panel["vault_sha256"],
        "crossfit_manifest_sha256": panel["crossfit_manifest_sha256"],
        "input_sha256": panel["input_sha256"],
        "budgets": budgets,
        "integrated_budget_effects": integrated,
        "integrated_policy_metrics": _integrated_policy_metrics(arrays),
        "transport_supported_sensitivity": {
            "system_count": len(supported_systems),
            "system_set_sha256": hashlib.sha256(
                "\n".join(supported_systems).encode()
            ).hexdigest(),
            "budgets": supported_budgets,
            "integrated_budget_effects": supported_integrated,
        },
        "cal_diagnostics": _runtime_summary(diagnostics) if policies == POLICIES else {},
        "baseline_diagnostics": (
            _baseline_diagnostic_summary(
                diagnostics,
                system_count=len(baseline_diagnostic_systems),
                panel_system_count=len(systems),
            )
            if policies == BASELINE_POLICIES
            else {}
        ),
    }


def _assert_no_failures(root: Path) -> None:
    failures = sorted(root.glob("**/*.failure.json"))
    if failures:
        raise ValueError("failure markers present: " + ", ".join(map(str, failures)))


def summarize(
    *,
    development_root: Path,
    secondary_path: Path | None,
    output: Path,
    expected_development_system_count: int,
    expected_secondary_system_count: int | None,
    expected_policies: tuple[str, ...] = POLICIES,
    protocol: str = E54_PROTOCOL,
    randomization_draws: int = 100_000,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(repo_root):
        raise ValueError("MatPES summaries must remain outside Git")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite MatPES summary: {output}")
    _assert_no_failures(development_root)
    development_paths = [
        development_root / f"fold{fold + 1}-b6.json" for fold in range(5)
    ]
    development = _load_panel(
        development_paths,
        expected_system_count=expected_development_system_count,
        expected_fold_indices=set(range(5)),
        policies=expected_policies,
    )
    is_e54 = expected_policies == POLICIES
    panels = {
        "development": _summarize_panel(
            development,
            policies=expected_policies,
            randomization_draws=randomization_draws,
            seed=20260810,
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
            policies=expected_policies,
        )
        panels["secondary"] = _summarize_panel(
            secondary,
            policies=expected_policies,
            randomization_draws=randomization_draws,
            seed=20260811,
        )
    result = {
        "schema_version": 1,
        "status": (
            "e54_cal_style_hull_entropy_complete"
            if is_e54
            else "e55_complete_pool_mean_hull_margin_baseline_complete"
        ),
        "protocol": protocol,
        "policies": list(expected_policies),
        "analysis_unit": "exact_chemical_system",
        "primary_contrast": "delta_minus_cal" if is_e54 else None,
        "inference": (
            "paired sign randomization with interval inversion"
            if is_e54
            else "descriptive paired-system policy summary; no within-baseline contrast"
        ),
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
    parser.add_argument("--policies", nargs="+", default=list(POLICIES))
    parser.add_argument("--protocol", default=E54_PROTOCOL)
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
        expected_policies=tuple(args.policies),
        protocol=args.protocol,
        randomization_draws=args.randomization_draws,
    )
    print(json.dumps({"status": result["status"], "panels": list(result["panels"])}))


if __name__ == "__main__":
    main()
