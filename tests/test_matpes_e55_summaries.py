from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

E32_TASK_SHA256 = "f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df"
E32_VAULT_SHA256 = "a272d3a2ce6286443ae6fce35726a688751a37284e3df362c5d1f70e2fcb9952"
E32_CROSSFIT_SHA256 = "a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa"

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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision(
    round_index: int,
    pair_id: str,
    score_offset: float,
    score_field: str,
    *,
    state_checksum: str,
) -> dict[str, object]:
    return {
        "round_index": round_index,
        "selected_pair_id": pair_id,
        "pre_reveal_state_checksum": state_checksum,
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
        "runner_fold_index": 0 if stage == "cal" else fold,
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
    selected_ids = [selected, *("a" if selected == "b" else "b", "c", "d", "e", "f")]
    systems = {}
    for index, system in enumerate(identity["query_systems"]):
        systems[system] = {
            "budget": 6,
            "transport_element_support": index != 0,
            "strategies": {
                policy: {
                    "selected_pair_ids": selected_ids,
                    "policy_decision_rounds": [
                        _decision(
                            round_index,
                            selected_id,
                            score_offset,
                            score_field,
                            state_checksum=f"state-{system}-{round_index}",
                        )
                        | {
                            "pre_reveal_state_checksum": (
                                f"state-{system}-{round_index}"
                                if selected == "a"
                                else f"state-{system}-{round_index if round_index == 1 else 'diverged'}"
                            )
                        }
                        for round_index, selected_id in enumerate(selected_ids, 1)
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
    source_root = root / "source-inputs"
    source_root.mkdir(parents=True)
    task_path = source_root / "matpes-e52-pool-100-task.json"
    vault_path = source_root / "matpes-e52-pool-100-vault.json"
    crossfit_path = source_root / "matpes-e52-pool-100-crossfit.json"
    cal_manifest_path = root / "e55-cal-manifest.json"
    runner_path = source_root / "run_matpes_protocol_closed_loop_exploratory.py"
    task_path.write_text(json.dumps({"release_id": "fixture"}), encoding="utf-8")
    vault_path.write_text(json.dumps({"vault": "fixture"}), encoding="utf-8")
    runner_path.write_text("# fixture runner\n", encoding="utf-8")
    delta_systems = [f"DELTA-{fold}-{index:02d}" for fold in range(5) for index in range(46)]
    crossfit_folds = []
    for fold in range(5):
        query = delta_systems[fold * 46 : (fold + 1) * 46]
        crossfit_folds.append(
            {
                "fold_index": fold,
                "query_systems": query,
                "fit_systems": [system for system in delta_systems if system not in query],
            }
        )
    crossfit_path.write_text(
        json.dumps(
            {
                "task_sha256": _sha256(task_path),
                "eligible_systems": delta_systems,
                "fold_count": 5,
                "folds": crossfit_folds,
            }
        ),
        encoding="utf-8",
    )
    cal_folds = [
        {
            "fold_index": fold,
            "query_systems": [f"CAL-{fold}-{index}" for index in range(3)],
            "fit_systems": crossfit_folds[fold]["fit_systems"],
        }
        for fold in range(5)
    ]
    cal_manifest_path.write_text(
        json.dumps(
            {
                "status": "e55_cal_convergence_roster_frozen",
                "task_sha256": _sha256(task_path),
                "development_crossfit_sha256": _sha256(crossfit_path),
                "fold_count": 5,
                "folds": cal_folds,
            }
        ),
        encoding="utf-8",
    )
    cal_runner_paths = []
    for fold, cal_fold in enumerate(cal_folds):
        runner_manifest = root / "cal-runner-manifests" / f"fold{fold + 1}.json"
        runner_manifest.parent.mkdir(parents=True, exist_ok=True)
        runner_manifest.write_text(
            json.dumps(
                {
                    "task_sha256": _sha256(task_path),
                    "source_e55_manifest_sha256": _sha256(cal_manifest_path),
                    "eligible_systems": sorted(
                        (*cal_fold["query_systems"], *cal_fold["fit_systems"])
                    ),
                    "fold_count": 1,
                    "folds": [
                        {"fold_index": 0, "query_systems": cal_fold["query_systems"]}
                    ],
                }
            ),
            encoding="utf-8",
        )
        cal_runner_paths.append(runner_manifest)
    units: list[dict[str, object]] = []
    grids = {
        "delta": [(samples, 10) for samples in (64, 128, 256, 512, 1024)],
        "cal": [(samples, fantasies) for samples in (100, 200, 400) for fantasies in (5, 10, 20)],
    }
    for stage, settings in grids.items():
        for fold in range(5):
            systems = (
                crossfit_folds[fold]["query_systems"]
                if stage == "delta"
                else cal_folds[fold]["query_systems"]
            )
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
                identity.update(
                    {
                        "task_sha256": _sha256(task_path),
                        "vault_sha256": _sha256(vault_path),
                        "runner_path": str(runner_path.resolve()),
                        "runner_sha256": _sha256(runner_path),
                        "crossfit_manifest_sha256": _sha256(crossfit_path),
                        "e55_manifest_sha256": _sha256(cal_manifest_path),
                        "runner_crossfit_manifest_sha256": (
                            _sha256(crossfit_path)
                            if stage == "delta"
                            else _sha256(cal_runner_paths[fold])
                        ),
                        "fit_systems": (
                            crossfit_folds[fold]["fit_systems"]
                            if stage == "delta"
                            else cal_folds[fold]["fit_systems"]
                        ),
                    }
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
                command = [
                    sys.executable,
                    str(runner_path.resolve()),
                    "--task",
                    str(task_path.resolve()),
                    "--development-vault",
                    str(vault_path.resolve()),
                    "--output",
                    str(path),
                    "--query-budget",
                    "6",
                    "--maximum-budget",
                    "6",
                    "--minimum-candidates",
                    "12",
                    "--seed",
                    "20260810",
                    "--posterior-sample-count",
                    str(samples),
                    "--fantasy-count",
                    str(fantasies),
                    "--hull-candidate-workers",
                    str(identity["hull_candidate_workers"]),
                    "--hull-backend",
                    "fixed_composition",
                    "--transport-family",
                    "hierarchical_matern52_frozen_structure",
                    "--rollout-selection-timeout-seconds",
                    str(identity["selection_timeout_seconds"]),
                    "--crossfit-manifest",
                    str(
                        crossfit_path.resolve()
                        if stage == "delta"
                        else cal_runner_paths[fold].resolve()
                    ),
                    "--fold-index",
                    str(fold if stage == "delta" else 0),
                    "--policies",
                    str(identity["policy"]),
                ]
                units.append({"output": str(path), "identity": identity, "command": command})
    (root / "e55-unit-manifest.json").write_text(
        json.dumps({"protocol": "e55-numerical-convergence-v1", "unit_count": len(units), "units": units}),
        encoding="utf-8",
    )
    return root


def _refresh_e55_source_hashes(root: Path) -> None:
    crossfit_path = root / "source-inputs" / "matpes-e52-pool-100-crossfit.json"
    cal_path = root / "e55-cal-manifest.json"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    cal["development_crossfit_sha256"] = _sha256(crossfit_path)
    cal_path.write_text(json.dumps(cal), encoding="utf-8")
    runner_hashes = {}
    for fold in range(5):
        runner_path = root / "cal-runner-manifests" / f"fold{fold + 1}.json"
        runner = json.loads(runner_path.read_text(encoding="utf-8"))
        runner["source_e55_manifest_sha256"] = _sha256(cal_path)
        runner_path.write_text(json.dumps(runner), encoding="utf-8")
        runner_hashes[fold] = _sha256(runner_path)
    manifest_path = root / "e55-unit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for unit in manifest["units"]:
        identity = unit["identity"]
        stage = identity["stage"]
        identity["crossfit_manifest_sha256"] = _sha256(crossfit_path)
        identity["e55_manifest_sha256"] = _sha256(cal_path)
        identity["runner_crossfit_manifest_sha256"] = (
            _sha256(crossfit_path)
            if stage == "delta"
            else runner_hashes[identity["fold_index"]]
        )
        payload_path = Path(unit["output"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["config"]["crossfit_manifest_sha256"] = identity[
            "runner_crossfit_manifest_sha256"
        ]
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _e32_strategy(
    *, budget: int, system_index: int, policy: str, common_fallback: bool
) -> dict[str, object]:
    is_rollout = policy == "delta_hull_anchored_rollout"
    is_delta = policy == "delta_hull_active_search"
    return {
        "selected_pair_ids": [
            "fallback" if common_fallback else "rollout" if is_rollout else "delta"
        ] * budget,
        "policy_decision_rounds": [
            {
                "round_index": round_index,
                "selected_pair_id": (
                    "fallback" if common_fallback else "rollout" if is_rollout else "delta"
                ),
                "pre_reveal_state_checksum": f"e32-state-{system_index}-{round_index}",
            }
            for round_index in range(1, budget + 1)
        ],
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
        "task_sha256": E32_TASK_SHA256,
        "oracle_vault_sha256": E32_VAULT_SHA256,
        "script_sha256": "e32-runner",
        "active_policies": list(E32_POLICIES),
        "query_systems": systems,
        "transport_fit_systems": [f"Fit-{index}" for index in range(230)],
        "transport_fit_system_count": 230,
        "transport_fit_and_query_systems_disjoint": True,
        "config": {
            "query_budget": budget,
            "maximum_budget": 6,
            "minimum_candidates": 12,
            "seed": 20270720,
            "posterior_sample_count": 128,
            "fantasy_count": 3,
            "rollout_selection_timeout_seconds": 900.0,
            "hull_backend": "fixed_composition",
            "transport_family": "hierarchical_matern52_frozen_structure",
            "crossfit_manifest_sha256": E32_CROSSFIT_SHA256,
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
    protocol_identity = {
        "protocol": "docs/DELAYED_LABEL_FOLLOWUP_PROTOCOL_E32.md",
        "task_sha256": E32_TASK_SHA256,
        "vault_sha256": E32_VAULT_SHA256,
        "crossfit_manifest_sha256": E32_CROSSFIT_SHA256,
        "runner_sha256": "e32-runner",
        "policy_worker_sha256": "worker-sha",
        "acquisition_sha256": "acquisition-sha",
        "seed": 20270720,
        "policies": list(E32_POLICIES),
        "budgets": list(range(1, 7)),
        "folds": list(range(5)),
        "max_systems": 1000,
        "posterior_sample_count": 128,
        "hull_backend": "fixed_composition",
        "transport_family": "hierarchical_matern52_frozen_structure",
    }
    executor_identity = {
        **protocol_identity,
        "executor": "parallel_unit_scheduler_v1",
        "rollout_selection_timeout_seconds": 900.0,
        "max_workers": 16,
        "blas_threads_per_unit": 1,
    }
    (root / "e32_protocol_identity.json").write_text(
        json.dumps(protocol_identity), encoding="utf-8"
    )
    (root / "e32_parallel_executor_identity.json").write_text(
        json.dumps(executor_identity), encoding="utf-8"
    )
    for budget in range(1, 7):
        for fold in range(5):
            systems = all_systems[fold * 46 : (fold + 1) * 46]
            fit_systems = [f"Fit-{fold}-{index}" for index in range(230)]
            (root / f"e32-fold{fold + 1}-b{budget}-main.json").write_text(
                json.dumps(
                    _e32_payload(budget=budget, fold=fold, systems=systems)
                    | {
                        "transport_fit_systems": fit_systems,
                        "transport_fit_system_count": len(fit_systems),
                    }
                ),
                encoding="utf-8",
            )
    return root


def test_e55_summary_audits_complete_grid_and_reference_diagnostics(tmp_path: Path) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    result = tool.summarize_e55(_write_e55_root(tmp_path), tmp_path / "e55-summary.json")

    assert result["unit_counts"] == {"delta": 25, "cal": 45}
    delta = result["stages"]["delta"]["configurations"]["m64-k10"]
    assert delta["reference"] == "m1024-k10"
    assert delta["action_agreement"]["common_state_count"] == 230
    assert delta["score_rank_diagnostics"]["score_available"] is True
    assert delta["score_rank_diagnostics"]["rank_available"] is True
    assert delta["terminal_T"]["reference_paired"]["system_count"] == 230
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

    with pytest.raises(ValueError, match="(frozen E55 identity|task path/hash)"):
        tool.summarize_e55(root, tmp_path / "summary.json")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("task", "task"),
        ("vault", "vault"),
        ("runner", "runner"),
        ("crossfit", "cross-fit"),
        ("cal", "CAL manifest"),
        ("command", "command"),
    ],
)
def test_e55_summary_independently_audits_source_paths_hashes_and_commands(
    tmp_path: Path, source: str, message: str
) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    if source == "task":
        path = root / "source-inputs" / "matpes-e52-pool-100-task.json"
        path.write_text("changed", encoding="utf-8")
    elif source == "vault":
        path = root / "source-inputs" / "matpes-e52-pool-100-vault.json"
        path.write_text("changed", encoding="utf-8")
    elif source == "runner":
        path = root / "source-inputs" / "run_matpes_protocol_closed_loop_exploratory.py"
        path.write_text("changed", encoding="utf-8")
    elif source == "crossfit":
        path = root / "source-inputs" / "matpes-e52-pool-100-crossfit.json"
        path.write_text("changed", encoding="utf-8")
    elif source == "cal":
        path = root / "e55-cal-manifest.json"
        path.write_text("changed", encoding="utf-8")
    else:
        manifest_path = root / "e55-unit-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        command = manifest["units"][0]["command"]
        command[command.index("--seed") + 1] = "0"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        tool.summarize_e55(root, tmp_path / "summary.json")


def test_e55_summary_accepts_count_only_source_fit_rosters(tmp_path: Path) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    for path in (
        root / "source-inputs" / "matpes-e52-pool-100-crossfit.json",
        root / "e55-cal-manifest.json",
    ):
        source = json.loads(path.read_text(encoding="utf-8"))
        for fold in source["folds"]:
            fold["fit_system_count"] = len(fold.pop("fit_systems"))
        path.write_text(json.dumps(source), encoding="utf-8")
    _refresh_e55_source_hashes(root)

    result = tool.summarize_e55(root, tmp_path / "summary.json")

    assert result["unit_counts"] == {"delta": 25, "cal": 45}


def test_e55_summary_independently_audits_cal_runner_manifest_content(
    tmp_path: Path,
) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    path = root / "cal-runner-manifests" / "fold1.json"
    runner = json.loads(path.read_text(encoding="utf-8"))
    runner["source_e55_manifest_sha256"] = "forged"
    path.write_text(json.dumps(runner), encoding="utf-8")
    manifest_path = root / "e55-unit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for unit in manifest["units"]:
        identity = unit["identity"]
        if identity["stage"] == "cal" and identity["fold_index"] == 0:
            identity["runner_crossfit_manifest_sha256"] = _sha256(path)
            payload_path = Path(unit["output"])
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["config"]["crossfit_manifest_sha256"] = _sha256(path)
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="CAL runner manifest"):
        tool.summarize_e55(root, tmp_path / "summary.json")


@pytest.mark.parametrize("field", ["budget", "cal_workers"])
def test_e55_summary_rejects_mutually_consistent_nonfrozen_campaign_config(
    tmp_path: Path, field: str
) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    manifest_path = root / "e55-unit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for unit in manifest["units"]:
        if field == "cal_workers" and unit["identity"]["stage"] != "cal":
            continue
        identity_key = "budget" if field == "budget" else "hull_candidate_workers"
        config_key = "query_budget" if field == "budget" else "hull_candidate_workers"
        option = "--query-budget" if field == "budget" else "--hull-candidate-workers"
        value = 5 if field == "budget" else 1
        unit["identity"][identity_key] = value
        command = unit["command"]
        command[command.index(option) + 1] = str(value)
        payload_path = Path(unit["output"])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        payload["config"][config_key] = value
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen E55 config"):
        tool.summarize_e55(root, tmp_path / "summary.json")


@pytest.mark.parametrize("stage", ["delta", "cal"])
def test_e55_summary_rejects_wrong_fold_roster_cardinality_or_overlap(
    tmp_path: Path, stage: str
) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    manifest_path = root / "e55-unit-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = [unit for unit in manifest["units"] if unit["identity"]["stage"] == stage]
    changed_systems = units[0]["identity"]["query_systems"][:-1]
    for unit in units:
        if unit["identity"]["fold_index"] == 0:
            unit["identity"]["query_systems"] = changed_systems
            unit["identity"]["query_system_count"] = len(changed_systems)
            payload_path = Path(unit["output"])
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            payload["query_systems"] = changed_systems
            payload["systems"] = {
                system: payload["systems"][system] for system in changed_systems
            }
            payload_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="query systems"):
        tool.summarize_e55(root, tmp_path / "summary.json")


def test_e55_common_states_require_checksum_and_reconcile_event_actions(tmp_path: Path) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    manifest = json.loads((root / "e55-unit-manifest.json").read_text(encoding="utf-8"))
    unit = next(
        unit
        for unit in manifest["units"]
        if unit["identity"]["stage"] == "delta"
        and unit["identity"]["posterior_sample_count"] == 64
    )
    path = Path(unit["output"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    strategy = next(iter(payload["systems"].values()))["strategies"][E55_POLICIES["delta"]]
    strategy["policy_decision_rounds"][0]["selected_pair_id"] = "not-selected"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="selected_pair_ids"):
        tool.summarize_e55(root, tmp_path / "summary.json")


def test_e55_score_and_rank_availability_counts_are_separate(tmp_path: Path) -> None:
    tool = _load_tool("summarize_matpes_e55_convergence")
    root = _write_e55_root(tmp_path)
    manifest = json.loads((root / "e55-unit-manifest.json").read_text(encoding="utf-8"))
    unit = next(
        unit
        for unit in manifest["units"]
        if unit["identity"]["stage"] == "delta"
        and unit["identity"]["posterior_sample_count"] == 64
    )
    path = Path(unit["output"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    event = next(iter(payload["systems"].values()))["strategies"][E55_POLICIES["delta"]][
        "policy_decision_rounds"
    ][0]
    event["selection_diagnostics"]["candidate_pair_ids"] = ["a", "b", "c"]
    event["selection_diagnostics"]["final_stability_probabilities"] = {
        "a": 0.5,
        "b": 0.5,
        "c": 0.5,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = tool.summarize_e55(root, tmp_path / "summary.json")
    diagnostics = result["stages"]["delta"]["configurations"]["m64-k10"][
        "score_rank_diagnostics"
    ]
    assert diagnostics["score_bearing_state_count"] > diagnostics["rank_comparable_state_count"]
    assert diagnostics["score_available"] is True
    assert diagnostics["rank_available"] is True


def test_e32_summary_reads_only_independent_b2_to_b6_artifacts(
    tmp_path: Path,
) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    for fold in range(1, 6):
        (root / f"e32-fold{fold}-b1-main.json").unlink()
    result = tool.summarize_e32(root, tmp_path / "e32-summary.json")

    assert sorted(result["budgets"]) == ["2", "3", "4", "5", "6"]
    assert all("-b1-" not in path for path in result["input_sha256"])
    assert len(result["input_sha256"]) == 27
    assert "integrated_b0_to_b6_terminal_T" not in result
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


def test_e32_summary_rejects_b2_identity_mismatch(tmp_path: Path) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    path = root / "e32-fold1-b2-main.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["seed"] = 0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="wrong E32 frozen identity: seed"):
        tool.summarize_e32(root, tmp_path / "e32-summary.json")


@pytest.mark.parametrize("change", ["count", "overlap"])
def test_e32_summary_rejects_wrong_fit_count_or_query_overlap(
    tmp_path: Path, change: str
) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    path = root / "e32-fold1-b2-main.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if change == "count":
        payload["transport_fit_systems"] = payload["transport_fit_systems"][:-1]
        payload["transport_fit_system_count"] = 229
    else:
        payload["transport_fit_systems"][0] = payload["query_systems"][0]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="wrong E32 fit roster"):
        tool.summarize_e32(root, tmp_path / "e32-summary.json")


@pytest.mark.parametrize("fantasy_count", [1, 16])
def test_e32_summary_rejects_nondefault_fantasy_count(
    tmp_path: Path, fantasy_count: int
) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    path = root / "e32-fold1-b2-main.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    payload["config"]["fantasy_count"] = fantasy_count
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="wrong E32 frozen identity: fantasy_count"):
        tool.summarize_e32(root, tmp_path / "e32-summary.json")


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
            for budget in range(1, 7):
                budget_path = root / f"e32-fold1-b{budget}-main.json"
                budget_payload = json.loads(budget_path.read_text(encoding="utf-8"))
                replaced = budget_payload["query_systems"][-1]
                budget_payload["query_systems"][-1] = "S046-X"
                budget_payload["systems"]["S046-X"] = budget_payload["systems"].pop(
                    replaced
                )
                budget_path.write_text(json.dumps(budget_payload), encoding="utf-8")
            payload = None
        elif change == "identity":
            payload["config"]["seed"] = 0
        else:
            payload["active_policies"] = list(E32_POLICIES[:-1])
        if payload is not None:
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
    rollout = system["strategies"][E32_ROLLOUT]
    rollout["selected_pair_ids"] = ["different"] * 2
    for event in rollout["policy_decision_rounds"]:
        event["selected_pair_id"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported fallback"):
        tool.summarize_e32(root, tmp_path / "summary.json")


@pytest.mark.parametrize(
    ("identity_file", "field", "message"),
    [
        ("e32_protocol_identity.json", "seed", "protocol identity"),
        ("e32_protocol_identity.json", "task_sha256", "protocol identity"),
        ("e32_protocol_identity.json", "runner_sha256", "protocol identity"),
        ("e32_parallel_executor_identity.json", "posterior_sample_count", "executor identity"),
        ("e32_parallel_executor_identity.json", "acquisition_sha256", "executor identity"),
        ("e32_parallel_executor_identity.json", "rollout_selection_timeout_seconds", "executor identity"),
    ],
)
def test_e32_summary_audits_protocol_and_executor_identities(
    tmp_path: Path, identity_file: str, field: str, message: str
) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    path = root / identity_file
    identity = json.loads(path.read_text(encoding="utf-8"))
    identity[field] = "wrong"
    path.write_text(json.dumps(identity), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        tool.summarize_e32(root, tmp_path / "summary.json")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ("status", "status"),
        ("fold", "fold"),
        ("roster", "query roster"),
        ("fit_roster", "fit roster"),
        ("events", "event count"),
        ("action_count", "action count"),
        ("actions", "selected_pair_ids"),
        ("support", "unsupported fallback"),
    ],
)
def test_e32_summary_audits_unit_status_folds_rosters_events_and_fallback(
    tmp_path: Path, change: str, message: str
) -> None:
    tool = _load_tool("summarize_e32_rollout_curve")
    root = _write_e32_root(tmp_path)
    path = root / "e32-fold1-b2-main.json"
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    system = payload["systems"]["S000-X"]
    if change == "status":
        payload["status"] = "wrong"
    elif change == "fold":
        payload["config"]["crossfit_fold_index"] = 1
    elif change == "roster":
        payload["query_systems"] = payload["query_systems"][:-1]
    elif change == "fit_roster":
        payload["transport_fit_systems"][0] = "not-the-fold-complement"
    elif change == "events":
        system["strategies"][E32_DELTA]["policy_decision_rounds"] = []
    elif change == "action_count":
        system["strategies"][E32_DELTA]["selected_pair_ids"] = []
    elif change == "actions":
        system["strategies"][E32_DELTA]["policy_decision_rounds"][0][
            "selected_pair_id"
        ] = "wrong"
    else:
        system["transport_element_support"] = None
        rollout = system["strategies"][E32_ROLLOUT]
        rollout["selected_pair_ids"] = ["different"] * 2
        for event in rollout["policy_decision_rounds"]:
            event["selected_pair_id"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        tool.summarize_e32(root, tmp_path / "summary.json")


@pytest.mark.parametrize(
    ("tool_name", "writer_name"),
    [
        ("summarize_matpes_e55_convergence", "summarize_e55"),
        ("summarize_e32_rollout_curve", "summarize_e32"),
    ],
)
def test_summary_writes_are_atomic_and_exclusive_under_contention(
    tmp_path: Path, tool_name: str, writer_name: str
) -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    tool = _load_tool(tool_name)
    input_root = _write_e55_root(tmp_path) if "e55" in tool_name else _write_e32_root(tmp_path)
    output = tmp_path / f"{tool_name}.json"
    barrier = Barrier(2)

    def write() -> str:
        barrier.wait()
        try:
            getattr(tool, writer_name)(input_root, output)
        except FileExistsError:
            return "exists"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: write(), range(2)))

    assert sorted(results) == ["created", "exists"]
    assert json.loads(output.read_text(encoding="utf-8"))["status"].startswith("complete")
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
