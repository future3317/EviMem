from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from matmem import (
    CalibrationUtilityBuilder,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    PrequentialCausalEvaluator,
    ProtocolCompatibilityResolver,
    aggregate_prequential_prefix,
    frozen_grid_cells,
    paired_system_bootstrap,
)

from .test_matmem import _card, _query

MANIFEST_MODULE = runpy.run_path(
    str(Path(__file__).parents[1] / "tools" / "build_wbm_frozen_grid_manifest.py")
)


def test_frozen_grid_has_37_labels_and_15_physical_traces() -> None:
    cells = frozen_grid_cells()
    assert len(cells) == 37
    assert len({item.execution_key for item in cells}) == 15
    assert sum(item.physical_execution for item in cells) == 15
    assert all(item.capacity is None or item.capacity < item.budget for item in cells)
    joint = [item for item in cells if item.strategy == "joint_posterior_risk_one_swap"]
    assert [(item.budget, item.capacity) for item in joint] == [(8, 2), (12, 4)]
    assert not any(item.strategy == "survival_coreset" for item in cells)


def test_frozen_system_selection_keeps_all_candidates_and_never_mixes_systems() -> None:
    candidate_type = MANIFEST_MODULE["ObservableCandidate"]
    candidates = []
    systems = [
        *(tuple((f"B{index}", "X")) for index in range(10)),
        *(tuple((f"T{index}", "X", "Y")) for index in range(10)),
        *(tuple((f"Q{index}", "W", "X", "Y")) for index in range(3)),
    ]
    for system in systems:
        for index in range(16 + len(system)):
            candidates.append(
                candidate_type(
                    query_id=f"{'-'.join(system)}-{index}",
                    chemical_system=system,
                    composition=tuple((element, 1.0) for element in system),
                    exact_structure_sha256=f"sha256:{'-'.join(system)}:{index}",
                )
            )
    selection = MANIFEST_MODULE["select_frozen_grid_systems"](
        candidates, release_id="test-release"
    )
    assert selection["selected_system_count"] <= 24
    for pool in selection["pools"].values():
        system = tuple(pool["chemical_system"])
        expected = sum(item.chemical_system == system for item in candidates)
        assert pool["candidate_count"] == expected
        assert len(pool["candidates"]) == expected
        assert all(tuple(item["chemical_system"]) == system for item in pool["candidates"])
    assert all(
        item["selected_system_count"] <= 8
        for item in selection["strata"].values()
    )


def test_prequential_metrics_are_round_weighted_and_prefix_aggregated() -> None:
    builder = CalibrationUtilityBuilder(
        FixedKernelResidualGP(
            ProtocolCompatibilityResolver(),
            config=FixedKernelGPConfig(length_scale=0.35),
        )
    )
    queries = (
        _query("preq-a", embedding=(1.0, 0.0), base_energy=-1.02),
        _query("preq-b", embedding=(0.0, 1.0), base_energy=-0.96),
    )
    evaluator = PrequentialCausalEvaluator(
        builder,
        {"preq-a": -1.04, "preq-b": -0.92},
    )
    first = evaluator.evaluate(round_index=1, queries=queries, cards=())
    second = evaluator.evaluate(
        round_index=2,
        queries=queries,
        cards=(_card("preq-card", embedding=(1.0, 0.0), formation_energy=-1.04),),
        retention_seconds=0.25,
        parent_rss_bytes=100,
    )
    aggregate = aggregate_prequential_prefix((first, second), 2)
    assert aggregate["round_count"] == 2
    assert aggregate["boundary_weighted_causal_crps"] == pytest.approx(
        (first.boundary_weighted_causal_crps + second.boundary_weighted_causal_crps)
        / 2
    )
    assert aggregate["retention_seconds"] == pytest.approx(0.25)
    assert aggregate["peak_parent_rss_bytes"] == 100
    with pytest.raises(ValueError, match="exactly"):
        aggregate_prequential_prefix((first,), 2)


def test_system_clustered_bootstrap_is_paired_and_deterministic() -> None:
    dacc = {"A-B": 0.1, "C-D": 0.2, "E-F": 0.3}
    baseline = {"A-B": 0.2, "C-D": 0.1, "E-F": 0.4}
    left = paired_system_bootstrap(dacc, baseline, seed=17, iterations=1000)
    right = paired_system_bootstrap(dacc, baseline, seed=17, iterations=1000)
    assert left == right
    assert left["system_count"] == 3
    assert left["mean_paired_difference"] == pytest.approx(-1 / 30)
    with pytest.raises(ValueError, match="identical"):
        paired_system_bootstrap(dacc, {"A-B": 0.2}, seed=17, iterations=10)
