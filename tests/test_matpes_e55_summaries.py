from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

E55_POLICIES = {
    "delta": "delta_hull_active_search",
    "cal": "cal_style_hull_entropy",
}
E32_DELTA = "delta_hull_active_search"
E32_ROLLOUT = "delta_hull_anchored_rollout"
E32_POLICIES = (
    "source_margin",
    "posterior_mean_target_margin",
    "posterior_current_hull_probability",
    E32_DELTA,
    "ungated_source_rollout",
    "source_rollout_delta_hull",
    E32_ROLLOUT,
)


def _load_tool(name: str):
    path = Path(__file__).parents[1] / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _decision(round_index: int, pair_id: str, score_offset: float, score_field: str) -> dict[str, object]:
    return {
        "round_index": round_index,
        "selected_pair_id": pair_id,
        "selection_diagnostics": {
            "candidate_pair_ids": ["a", "b", "c"],
            score_field: {"a": score_offset, "b": score_offset + 0.1, "c": score_offset - 0.1},
        },
    }


def _e55_identity(
    *,
    stage: str,
    fold: int,
    samples: int,
    fantasies: int,
    systems: list[str],
) -> dict[str, object]:
    return {
        "protocol": "e55-numerical-convergence-v1",
        "stage": stage,
        "task_sha256": "task-hash",
        "vault_sha256": "vault-hash",
        "runner_path": "runner.py",
        "runner_sha256": "runner-hash",
        "crossfit_manifest_sha256": "crossfit-hash",
        "e55_manifest_sha256": "cal-manifest-hash",
        "runner_crossfit_manifest_sha256": "runner-crossfit-hash",
        "fold_index": fold,
        "query_systems": systems,
        "query_system_count": len(systems),
        "fit_systems": [f"Fit-{index}" for index in range(184)],
        "fit_system_count": 184,
        "budget": 6,
        "maximum_budget": 6,
        "minimum_candidates": 12,
        "seed": 20260810,
        "posterior_sample_count": samples,
        "fantasy_count": fantasies,
        "hull_candidate_workers": 1 if stage == "delta" else 8,
        "hull_backend": "fixed_composition",
        "transport_family": "hierarchical_matern52_frozen_structure",
        "selection_timeout_seconds": 7200.0 if stage == "delta" else 21600.0,
        "policy": E55_POLICIES[stage],
    }


def _e55_payload(identity: dict[str, object], *, selected: str, score_offset: float) -> dict[str, object]:
    policy = str(identity["policy"])
    score_field = "final_stability_probabilities" if policy == E55_POLICIES["delta"] else "cal_scores"
    systems = {}
    for index, system in enumerate(identity["query_systems"]):
        systems[system] = {
            "budget": 6,
            "transport_element_support": index != 0,
            "strategies": {
                policy: {
                    "selected_pair_ids": [selected, "b", "c", "d", "e", "f"],
                    "policy_decision_rounds": [
                        _decision(round_index, selected, score_offset, score_field)
                        for round_index in range(1, 7)
                    ],
                    "wall_seconds": float(index + 1 + score_offset),
                    "oracle_pool_confirmed_discoveries": float(index + samples_to_t(identity)),
                }
            },
        }
    return {
        "status": "exploratory_development_systems_only_not_confirmatory",
        "script_sha256": identity["runner_sha256"],
        "task_sha256": identity["task_sha256"],
        "oracle_vault_sha256": identity["vault_sha256"],
        "active_policies": [policy],
        "query_systems": identity["query_systems"],
        "transport_fit_systems": identity["fit_systems"],
        "transport_fit_system_count": identity["fit_system_count"],
        "transport_fit_and_query_systems_disjoint": True,
        "config": {
            "query_budget": identity["budget"],
            "maximum_budget": identity["maximum_budget"],
            "minimum_candidates": identity["minimum_candidates"],
            "seed": identity["seed"],
            "posterior_sample_count": identity["posterior_sample_count"],
            "fantasy_count": identity["fantasy_count"],
            "hull_candidate_workers": identity["hull_candidate_workers"],
            "hull_backend": identity["hull_backend"],
            "transport_family": identity["transport_family"],
            "rollout_selection_timeout_seconds": identity["selection_timeout_seconds"],
            "crossfit_manifest_sha256": identity["runner_crossfit_manifest_sha256"],
        },
        "systems": systems,
    }


def samples_to_t(identity: dict[str, object]) -> float:
    return float(int(identity["posterior_sample_count"]) / 1000)


def _write_e55_root(tmp_path: Path) -> Path:
    root = tmp_path / "e55"
    units: list[dict[str, object]] = []
    grids = {
        "delta": [(samples, 10) for samples in (64, 128, 256, 512, 1024)],
        "cal": [(samples, fantasies) for samples in (100, 200, 400) for fantasies in (5, 10, 20)],
    }
    for stage, settings in grids.items():
        for fold in range(5):
            systems = [f"{stage.upper()}-{fold}-A", f"{stage.upper()}-{fold}-B"]
            for samples, fantasies in settings:
                path = root / stage / f"{stage}-fold{fold + 1}-m{samples}-k{fantasies}-b6.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                identity = _e55_identity(
                    stage=stage,
                    fold=fold,
                    samples=samples,
                    fantasies=fantasies,
                    systems=systems,
                )
                path.write_text(
                    json.dumps(
                        _e55_payload(
                            identity,
                            selected="a" if samples in (1024, 400) and fantasies in (20, 10) else "b",
                            score_offset=float(samples) / 1000,
                        )
                    ),
                    encoding="utf-8",
                )
                units.append({"output": str(path), "identity": identity})
    (root / "e55-unit-manifest.json").write_text(
        json.dumps({"protocol": "e55-numerical-convergence-v1", "unit_count": len(units), "units": units}),
        encoding="utf-8",
    )
    return root


def _e32_strategy(
    *, budget: int, system_index: int, policy: str, common_fallback: bool
) -> dict[str, object]:
    is_rollout = policy == "delta_hull_anchored_rollout"
    is_delta = policy == "delta_hull_active_search"
    return {
        "selected_pair_ids": [
            "fallback" if common_fallback else "rollout" if is_rollout else "delta"
        ] * budget,
        "policy_decision_rounds": [{"round_index": round_index} for round_index in range(1, budget + 1)],
        "wall_seconds": float(budget + system_index % 3 + (2 if is_rollout else 1)),
        "oracle_pool_confirmed_discoveries": float(
            budget
            if common_fallback
            else budget + (1 if is_rollout else 0 if is_delta else -1)
        ),
    }


def _e32_payload(*, budget: int, fold: int, systems: list[str]) -> dict[str, object]:
    return {
        "status": "exploratory_development_systems_only_not_confirmatory",
        "task_sha256": "e32-task",
        "oracle_vault_sha256": "e32-vault",
        "script_sha256": "e32-runner",
        "active_policies": list(E32_POLICIES),
        "query_systems": systems,
        "transport_fit_systems": [f"Fit-{index}" for index in range(184)],
        "transport_fit_system_count": 184,
        "transport_fit_and_query_systems_disjoint": True,
        "config": {
            "query_budget": budget,
            "maximum_budget": budget,
            "seed": 20270720,
            "posterior_sample_count": 128,
            "fantasy_count": 16,
            "hull_backend": "fixed_composition",
            "transport_family": "hierarchical_matern52_frozen_structure",
            "crossfit_manifest_sha256": "e32-crossfit",
            "crossfit_fold_index": fold,
        },
        "systems": {
            system: {
                "budget": budget,
                "transport_element_support": system_index % 7 != 0,
                "strategies": {
                    policy: _e32_strategy(
                        budget=budget,
                        system_index=system_index,
                        policy=policy,
                        common_fallback=system_index % 7 == 0,
                    )
                    for policy in E32_POLICIES
                },
            }
            for system_index, system in enumerate(systems)
        },
    }


def _write_e32_root(tmp_path: Path) -> Path:
    root = tmp_path / "e32"
    root.mkdir()
    all_systems = [f"S{index:03d}-X" for index in range(230)]
    for budget in range(1, 7):
        for fold in range(5):
            systems = all_systems[fold * 46 : (fold + 1) * 46]
            (root / f"e32-fold{fold + 1}-b{budget}-main.json").write_text(
                json.dumps(_e32_payload(budget=budget, fold=fold, systems=systems)),
                encoding="utf-8",
            )
    return root


def test_e55_summary_audits_complete_grid_and_reference_diagnostics(tmp_path: Path) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    result = tool.summarize_e55(_write_e55_root(tmp_path), tmp_path / "e55-summary.json")

    assert result["unit_counts"] == {"delta": 25, "cal": 45}
    delta = result["stages"]["delta"]["configurations"]["m64-k10"]
    assert delta["reference"] == "m1024-k10"
    assert delta["action_agreement"]["common_state_count"] == 10
    assert delta["score_rank_diagnostics"]["available"] is True
    assert delta["terminal_T"]["reference_paired"]["system_count"] == 10
    assert delta["runtime"]["population"] == "per_system_policy_wall_seconds_from_traces"
    assert result["stages"]["cal"]["configurations"]["m100-k5"]["reference"] == "m400-k20"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("remove", "missing E55 output"),
        ("duplicate", "duplicate E55 unit"),
        ("hash", "task_sha256"),
        ("config", "posterior_sample_count"),
        ("policy", "policy roster"),
        ("rounds", "decision rounds"),
    ],
)
def test_e55_summary_rejects_incomplete_or_identity_mismatched_units(
    tmp_path: Path, change: str, message: str
) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    manifest = json.loads((root / "e55-unit-manifest.json").read_text(encoding="utf-8"))
    unit = manifest["units"][0]
    if change == "remove":
        Path(unit["output"]).unlink()
    elif change == "duplicate":
        manifest["units"].append(dict(unit))
        manifest["unit_count"] += 1
        (root / "e55-unit-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    else:
        payload = json.loads(Path(unit["output"]).read_text(encoding="utf-8"))
        if change == "hash":
            payload["task_sha256"] = "wrong"
        elif change == "config":
            payload["config"]["posterior_sample_count"] = 7
        elif change == "policy":
            payload["active_policies"] = ["wrong"]
        else:
            strategy = next(iter(payload["systems"].values()))["strategies"][E55_POLICIES["delta"]]
            strategy["policy_decision_rounds"] = strategy["policy_decision_rounds"][:-1]
        Path(unit["output"]).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        tool.summarize_e55(root, tmp_path / "summary.json")


def test_e55_summary_refuses_output_overwrite(tmp_path: Path) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    output = tmp_path / "summary.json"
    output.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        tool.summarize_e55(_write_e55_root(tmp_path), output)


def test_e55_summary_rejects_a_manifest_identity_that_drifts_with_its_output(tmp_path: Path) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    manifest_path = root / "e55-unit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    unit = manifest["units"][0]
    unit["identity"]["task_sha256"] = "forged-task"
    payload_path = Path(unit["output"])
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["task_sha256"] = "forged-task"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen E55 identity"):
        tool.summarize_e55(root, tmp_path / "summary.json")


def test_e32_summary_reads_independent_budget_artifacts_and_integrates_before_inference(
    tmp_path: Path,
) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    result = tool.summarize_e32(_write_e32_root(tmp_path), tmp_path / "e32-summary.json")

    assert sorted(result["budgets"]) == ["2", "3", "4", "5", "6"]
    assert result["system_count"] == 230
    assert result["inference"] == {
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": 20260803,
        "sign_flip_draws": 100_000,
        "sign_flip_seed": 20260804,
        "statistical_unit": "exact_chemical_system",
    }
    budget_two = result["budgets"]["2"]
    assert budget_two["paired_terminal_T"]["paired_mean_difference"] == pytest.approx(195 / 230)
    assert budget_two["action_disagreement"]["systems_with_any_disagreement"] == 195
    assert budget_two["runtime"]["population"] == "per_system_policy_wall_seconds_from_traces"
    assert result["integrated_b0_to_b6_terminal_T"]["paired_mean_difference"] == pytest.approx(5 * 195 / 230)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("missing", "missing"),
        ("duplicate", "occurs twice"),
        ("identity", "frozen identity"),
        ("policy", "policy roster"),
    ],
)
def test_e32_summary_rejects_incomplete_or_nonfrozen_artifacts(
    tmp_path: Path, change: str, message: str
) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    path = root / "e32-fold1-b2-main.json"
    if change == "missing":
        path.unlink()
    else:
        payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        if change == "duplicate":
            duplicate = next(iter(payload["systems"].values()))
            payload["systems"]["S046-X"] = duplicate
        elif change == "identity":
            payload["config"]["seed"] = 0
        else:
            payload["active_policies"] = list(E32_POLICIES[:-1])
        path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError), match=message):
        tool.summarize_e32(root, tmp_path / "summary.json")


def test_e32_summary_refuses_output_overwrite(tmp_path: Path) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    output = tmp_path / "summary.json"
    output.write_text("already here", encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        tool.summarize_e32(_write_e32_root(tmp_path), output)


def test_e32_summary_rejects_unsupported_noncommon_policy_actions(tmp_path: Path) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    path = root / "e32-fold1-b2-main.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    system = payload["systems"]["S000-X"]
    system["transport_element_support"] = False
    system["strategies"][E32_ROLLOUT]["selected_pair_ids"] = ["different"] * 2
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported fallback"):
        tool.summarize_e32(root, tmp_path / "summary.json")
