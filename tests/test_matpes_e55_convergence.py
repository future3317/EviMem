from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from tools import run_matpes_e55_convergence as launcher
from tools.run_matpes_e55_convergence import (
    CAL_FANTASY_COUNTS,
    CAL_POSTERIOR_SAMPLE_COUNTS,
    CAL_SELECTION_TIMEOUT_SECONDS,
    CAL_WORKERS,
    DELTA_FANTASY_COUNT,
    DELTA_POSTERIOR_SAMPLE_COUNTS,
    DELTA_SELECTION_TIMEOUT_SECONDS,
    DELTA_WORKERS,
    SEED,
    _run_unit,
    _runner_manifest,
    _write_failure,
    _write_json_exclusive,
    _write_top_level_manifest,
    build_units,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    full_pool_root = tmp_path / "e52"
    task = full_pool_root / "matpes-e52-pool-100-task.json"
    vault = full_pool_root / "matpes-e52-pool-100-vault.json"
    task.parent.mkdir(parents=True)
    task.write_text('{"release_id": "fixture"}\n', encoding="utf-8")
    vault.write_text('{"oracle": "fixture"}\n', encoding="utf-8")

    systems = [f"S{index:03d}-X" for index in range(230)]
    folds = []
    for fold_index in range(5):
        query_systems = systems[46 * fold_index : 46 * (fold_index + 1)]
        folds.append(
            {
                "fold_index": fold_index,
                "query_systems": query_systems,
                "fit_systems": [system for system in systems if system not in query_systems],
            }
        )
    crossfit = full_pool_root / "matpes-e52-pool-100-crossfit.json"
    crossfit.write_text(
        json.dumps(
            {
                "task_sha256": _sha256(task),
                "eligible_systems": systems,
                "fold_count": 5,
                "folds": folds,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cal_folds = []
    for fold in folds:
        cal_folds.append(
            {
                "fold_index": fold["fold_index"],
                "query_systems": fold["query_systems"][:3],
                "fit_systems": fold["fit_systems"],
            }
        )
    cal_manifest = tmp_path / "e55-cal-manifest.json"
    cal_manifest.write_text(
        json.dumps(
            {
                "task_sha256": _sha256(task),
                "development_crossfit_sha256": _sha256(crossfit),
                "fold_count": 5,
                "folds": cal_folds,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return full_pool_root, cal_manifest, task, vault


def _units(tmp_path: Path, *, stages: tuple[str, ...] = ("delta", "cal")):
    full_pool_root, cal_manifest, _, _ = _write_inputs(tmp_path)
    return build_units(
        full_pool_root=full_pool_root,
        cal_manifest=cal_manifest,
        output_root=tmp_path / "outputs",
        runner=Path(launcher.__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
        stages=stages,
    )


def test_e55_builds_exact_delta_and_cal_grids_with_frozen_identity(tmp_path: Path) -> None:
    full_pool_root, cal_manifest, task, vault = _write_inputs(tmp_path)

    units = build_units(
        full_pool_root=full_pool_root,
        cal_manifest=cal_manifest,
        output_root=tmp_path / "outputs",
        runner=Path(launcher.__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
        stages=("delta", "cal"),
    )

    delta_units = [unit for unit in units if unit.identity["stage"] == "delta"]
    cal_units = [unit for unit in units if unit.identity["stage"] == "cal"]
    assert len(delta_units) == 25
    assert len(cal_units) == 45
    assert {
        (unit.identity["fold_index"], unit.identity["posterior_sample_count"])
        for unit in delta_units
    } == {(fold, samples) for fold in range(5) for samples in DELTA_POSTERIOR_SAMPLE_COUNTS}
    assert {
        (
            unit.identity["fold_index"],
            unit.identity["posterior_sample_count"],
            unit.identity["fantasy_count"],
        )
        for unit in cal_units
    } == {
        (fold, samples, fantasies)
        for fold in range(5)
        for samples in CAL_POSTERIOR_SAMPLE_COUNTS
        for fantasies in CAL_FANTASY_COUNTS
    }
    assert all(unit.identity["policy"] == "delta_hull_active_search" for unit in delta_units)
    assert all(unit.identity["policy"] == "cal_style_hull_entropy" for unit in cal_units)
    assert all(unit.identity["budget"] == 6 for unit in units)
    assert all(unit.identity["maximum_budget"] == 6 for unit in units)
    assert all(unit.identity["minimum_candidates"] == 12 for unit in units)
    assert all(unit.identity["seed"] == SEED for unit in units)
    assert all(unit.identity["hull_backend"] == "fixed_composition" for unit in units)
    assert all(unit.identity["transport_family"] == "hierarchical_matern52_frozen_structure" for unit in units)
    assert all(unit.identity["task_sha256"] == _sha256(task) for unit in units)
    assert all(unit.identity["vault_sha256"] == _sha256(vault) for unit in units)
    canonical_runner = Path(launcher.__file__).with_name(
        "run_matpes_protocol_closed_loop_exploratory.py"
    )
    assert all(unit.identity["runner_path"] == str(canonical_runner.resolve()) for unit in units)
    assert all(unit.identity["runner_sha256"] == _sha256(canonical_runner) for unit in units)
    assert all(unit.identity["query_system_count"] == 46 for unit in delta_units)
    assert all(unit.identity["fit_system_count"] == 184 for unit in delta_units)
    assert all(unit.identity["query_system_count"] == 3 for unit in cal_units)
    assert all(unit.identity["fit_system_count"] == 184 for unit in cal_units)
    assert all(unit.identity["e55_manifest_sha256"] == _sha256(cal_manifest) for unit in cal_units)
    assert all(unit.identity["fantasy_count"] == DELTA_FANTASY_COUNT for unit in delta_units)
    assert all(unit.identity["hull_candidate_workers"] == DELTA_WORKERS for unit in delta_units)
    assert all(unit.identity["selection_timeout_seconds"] == DELTA_SELECTION_TIMEOUT_SECONDS for unit in delta_units)
    assert all(unit.identity["hull_candidate_workers"] == CAL_WORKERS for unit in cal_units)
    assert all(unit.identity["selection_timeout_seconds"] == CAL_SELECTION_TIMEOUT_SECONDS for unit in cal_units)
    assert all("delta-fold" in unit.output.name and "-m" in unit.output.name for unit in delta_units)
    assert all("cal-fold" in unit.output.name and "-m" in unit.output.name and "-k" in unit.output.name for unit in cal_units)


def test_e55_stage_filtering_deduplicates_and_rejects_unknown_stages(tmp_path: Path) -> None:
    assert len(_units(tmp_path / "delta", stages=("delta", "delta"))) == 25
    assert len(_units(tmp_path / "cal", stages=("cal", "cal"))) == 45
    with pytest.raises(ValueError, match="stage"):
        _units(tmp_path / "bad", stages=("delta", "unknown"))


def test_e55_rejects_output_under_a_git_worktree(tmp_path: Path) -> None:
    full_pool_root, cal_manifest, _, _ = _write_inputs(tmp_path)
    git_root = tmp_path / "other-worktree"
    (git_root / ".git").mkdir(parents=True)

    with pytest.raises(ValueError, match="outside Git"):
        build_units(
            full_pool_root=full_pool_root,
            cal_manifest=cal_manifest,
            output_root=git_root / "outputs",
            runner=Path(launcher.__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
            stages=("delta",),
        )


def _valid_output(unit) -> dict[str, object]:
    systems = {}
    for system in unit.identity["query_systems"]:
        selected = [f"{system}-pair-{round_index}" for round_index in range(1, 7)]
        systems[system] = {
            "budget": 6,
            "transport_element_support": True,
            "strategies": {
                unit.identity["policy"]: {
                    "selected_pair_ids": selected,
                    "policy_decision_rounds": [
                        {"round_index": round_index} for round_index in range(1, 7)
                    ],
                    "trace_checksum": f"trace-{system}",
                    "event_log_sha256": f"event-{system}",
                    "wall_seconds": 1.0,
                    "final_causal_confirmed_discoveries": 2,
                    "oracle_pool_confirmed_discoveries": 2,
                    "oracle_pool_discovery_ceiling": 3,
                    "oracle_pool_discovery_gap_to_ceiling": 1,
                    "invalidated_causal_discoveries_by_oracle_pool_hull": 0,
                    "oracle_pool_final_labels_by_pair_id": {
                        pair_id: True for pair_id in selected
                    },
                }
            },
        }
    return {
        "status": "exploratory_development_systems_only_not_confirmatory",
        "script_sha256": unit.identity["runner_sha256"],
        "task_sha256": unit.identity["task_sha256"],
        "oracle_vault_sha256": unit.identity["vault_sha256"],
        "active_policies": [unit.identity["policy"]],
        "query_systems": unit.identity["query_systems"],
        "transport_fit_systems": unit.identity["fit_systems"],
        "transport_fit_system_count": unit.identity["fit_system_count"],
        "transport_fit_and_query_systems_disjoint": True,
        "config": {
            "query_budget": 6,
            "maximum_budget": 6,
            "minimum_candidates": 12,
            "seed": SEED,
            "posterior_sample_count": unit.identity["posterior_sample_count"],
            "fantasy_count": unit.identity["fantasy_count"],
            "hull_candidate_workers": unit.identity["hull_candidate_workers"],
            "hull_backend": "fixed_composition",
            "transport_family": "hierarchical_matern52_frozen_structure",
            "rollout_selection_timeout_seconds": unit.identity["selection_timeout_seconds"],
            "crossfit_manifest_sha256": unit.identity["runner_crossfit_manifest_sha256"],
            "crossfit_fold_index": 0,
        },
        "systems": systems,
    }


def test_e55_resume_requires_complete_existing_output_identity(tmp_path: Path) -> None:
    unit = _units(tmp_path, stages=("cal",))[0]
    unit.output.parent.mkdir(parents=True)
    unit.output.write_text(json.dumps(_valid_output(unit)), encoding="utf-8")

    assert _run_unit(unit) == f"resume-skip={unit.output}"

    invalid = _valid_output(unit)
    invalid["config"]["posterior_sample_count"] = 999
    unit.output.write_text(json.dumps(invalid), encoding="utf-8")
    with pytest.raises(ValueError, match="posterior_sample_count"):
        _run_unit(unit)
    failure = unit.output.with_suffix(".failure.json")
    assert failure.is_file()
    assert "posterior_sample_count" in json.loads(failure.read_text(encoding="utf-8"))["reason"]


def test_e55_resume_rejects_output_with_altered_runner_hash(tmp_path: Path) -> None:
    unit = _units(tmp_path, stages=("delta",))[0]
    unit.output.parent.mkdir(parents=True)
    payload = _valid_output(unit)
    payload["script_sha256"] = "not-the-canonical-runner"
    unit.output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="script_sha256"):
        _run_unit(unit)

    assert unit.output.with_suffix(".failure.json").is_file()


def test_e55_refuses_terminal_failures_and_reuses_only_exact_top_level_manifest(
    tmp_path: Path,
) -> None:
    units = _units(tmp_path, stages=("delta",))
    unit = units[0]
    unit.output.parent.mkdir(parents=True)
    unit.output.with_suffix(".failure.json").write_text(
        json.dumps({"status": "failed_incomplete", "identity": unit.identity}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="already failed"):
        _run_unit(unit)

    manifest = tmp_path / "outputs" / "e55-unit-manifest.json"
    _write_top_level_manifest(manifest, units)
    _write_top_level_manifest(manifest, units)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["units"][0]["identity"]["seed"] = 0
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong identity"):
        _write_top_level_manifest(manifest, units)


def test_e55_main_resumes_only_the_exact_existing_top_level_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    full_pool_root, cal_manifest, _, _ = _write_inputs(tmp_path)
    output_root = tmp_path / "outputs"
    units = build_units(
        full_pool_root=full_pool_root,
        cal_manifest=cal_manifest,
        output_root=output_root,
        runner=Path(launcher.__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
        stages=("delta",),
    )
    _write_top_level_manifest(output_root / "e55-unit-manifest.json", units)
    seen = []
    monkeypatch.setattr(launcher, "_run_unit", lambda unit: seen.append(unit) or "skipped")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_matpes_e55_convergence.py",
            "--full-pool-root",
            str(full_pool_root),
            "--cal-manifest",
            str(cal_manifest),
            "--output-root",
            str(output_root),
            "--stages",
            "delta",
        ],
    )

    launcher.main()

    assert len(seen) == 25


def test_e55_rejects_altered_runner_and_malformed_delta_crossfit_topology(tmp_path: Path) -> None:
    full_pool_root, cal_manifest, _, _ = _write_inputs(tmp_path)
    altered = tmp_path / "altered-runner.py"
    altered.write_text("print('not canonical')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical runner"):
        build_units(
            full_pool_root=full_pool_root,
            cal_manifest=cal_manifest,
            output_root=tmp_path / "outputs-a",
            runner=altered,
            stages=("delta",),
        )
    crossfit = full_pool_root / "matpes-e52-pool-100-crossfit.json"
    payload = json.loads(crossfit.read_text(encoding="utf-8"))
    payload["folds"][1]["query_systems"][0] = payload["folds"][0]["query_systems"][0]
    eligible = set(payload["eligible_systems"])
    payload["folds"][1]["fit_systems"] = sorted(
        eligible - set(payload["folds"][1]["query_systems"])
    )
    crossfit.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="pairwise disjoint"):
        build_units(
            full_pool_root=full_pool_root,
            cal_manifest=cal_manifest,
            output_root=tmp_path / "outputs-b",
            runner=Path(launcher.__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py"),
            stages=("delta",),
        )


def test_e55_cal_runner_manifest_reuses_exact_content_and_preserves_conflict(
    tmp_path: Path,
) -> None:
    full_pool_root, cal_manifest, task, _ = _write_inputs(tmp_path)
    cal = json.loads(cal_manifest.read_text(encoding="utf-8"))
    path = tmp_path / "outputs" / "cal-runner-manifests" / "fold1.json"
    barrier = Barrier(2)

    def write_exact() -> Path:
        barrier.wait()
        return _runner_manifest(
            path=path,
            task_sha256=_sha256(task),
            source_manifest_sha256=_sha256(cal_manifest),
            fold=cal["folds"][0],
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(lambda _: write_exact(), range(2))) == [path, path]
    original = path.read_bytes()
    with pytest.raises(ValueError, match="wrong identity"):
        _runner_manifest(
            path=path,
            task_sha256=_sha256(task),
            source_manifest_sha256="conflicting-manifest",
            fold=cal["folds"][0],
        )
    assert path.read_bytes() == original
    assert full_pool_root.is_dir()


def test_e55_competing_top_level_manifest_writes_create_once_and_exactly_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    units = _units(tmp_path, stages=("delta",))
    path = tmp_path / "outputs" / "e55-unit-manifest.json"
    original_write = launcher._write_json_exclusive
    created = []
    barrier = Barrier(2)

    def counted_write(target: Path, payload: dict[str, object]) -> None:
        try:
            original_write(target, payload)
        except FileExistsError:
            raise
        else:
            if target == path:
                created.append(target)

    monkeypatch.setattr(launcher, "_write_json_exclusive", counted_write)

    def write_manifest() -> None:
        barrier.wait()
        _write_top_level_manifest(path, units)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: write_manifest(), range(2)))

    assert created == [path]
    original = path.read_bytes()
    conflicting_units = list(units)
    conflicting_identity = dict(conflicting_units[0].identity)
    conflicting_identity["seed"] = 0
    conflicting_units[0] = launcher.Unit(
        command=conflicting_units[0].command,
        output=conflicting_units[0].output,
        log=conflicting_units[0].log,
        identity=conflicting_identity,
    )
    with pytest.raises(ValueError, match="wrong identity"):
        _write_top_level_manifest(path, conflicting_units)
    assert path.read_bytes() == original


def test_e55_competing_failure_marker_writes_preserve_the_first_payload(tmp_path: Path) -> None:
    unit = _units(tmp_path, stages=("delta",))[0]
    barrier = Barrier(2)

    def write_failure(reason: str) -> None:
        barrier.wait()
        _write_failure(unit, 1, reason)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(write_failure, ("first contender", "second contender")))

    path = unit.output.with_suffix(".failure.json")
    original = path.read_bytes()
    payload = json.loads(original)
    assert payload["reason"] in {"first contender", "second contender"}
    _write_failure(unit, 1, "later contender")
    assert path.read_bytes() == original


def test_e55_atomic_json_publication_hides_target_until_link_then_exposes_complete_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "published.json"
    payload = {"complete": [1, 2, 3]}
    before_link = Barrier(2)
    allow_link = Barrier(2)
    original_link = launcher.os.link

    def pause_before_link(source: str, target: str) -> None:
        before_link.wait()
        allow_link.wait()
        original_link(source, target)

    monkeypatch.setattr(launcher.os, "link", pause_before_link)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_write_json_exclusive, path, payload)
        before_link.wait()
        assert not path.exists()
        allow_link.wait()
        future.result()

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".published.json.*.tmp"))


def test_e55_atomic_json_publication_exposes_parseable_payload_immediately_after_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "published.json"
    payload = {"complete": [1, 2, 3]}
    after_link = Barrier(2)
    allow_cleanup = Barrier(2)
    original_link = launcher.os.link

    def pause_after_link(source: str, target: str) -> None:
        original_link(source, target)
        after_link.wait()
        allow_cleanup.wait()

    monkeypatch.setattr(launcher.os, "link", pause_after_link)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_write_json_exclusive, path, payload)
        after_link.wait()
        allow_cleanup.wait()
        future.result()

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert not list(tmp_path.glob(".published.json.*.tmp"))


def test_e55_atomic_json_publication_competition_preserves_one_complete_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "published.json"
    first = {"publisher": "first", "values": list(range(100))}
    second = {"publisher": "second", "values": list(range(100, 200))}
    barrier = Barrier(2)

    def publish(payload: dict[str, object]) -> str:
        barrier.wait()
        try:
            _write_json_exclusive(path, payload)
        except FileExistsError:
            return "exists"
        return "created"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, (first, second)))

    assert sorted(results) == ["created", "exists"]
    assert json.loads(path.read_text(encoding="utf-8")) in (first, second)
    assert not list(tmp_path.glob(".published.json.*.tmp"))
