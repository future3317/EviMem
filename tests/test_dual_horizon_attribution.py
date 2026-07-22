from tools.run_dual_horizon_attribution import (
    _candidate_stratum,
    _chemistry_stratum,
    _group_summary,
)


def test_attribution_strata_are_deterministic() -> None:
    assert _chemistry_stratum("Li-O") == "binary"
    assert _chemistry_stratum("Li-Fe-O") == "ternary"
    assert _chemistry_stratum("Ba-Mg-Mn-O") == "quaternary_or_higher"
    assert _candidate_stratum(16) == "small_le_16"
    assert _candidate_stratum(17) == "medium_17_32"
    assert _candidate_stratum(32) == "medium_17_32"
    assert _candidate_stratum(33) == "large_gt_32"


def test_group_summary_reports_global_action_counts() -> None:
    records = [
        {
            "chemistry_stratum": "binary",
            "candidate_stratum": "small_le_16",
            "oracle_feasible_exists": True,
            "oracle_feasible": [True, False],
            "posterior_point_feasible": [True, True],
            "posterior_gate_feasible": [False, True],
        },
        {
            "chemistry_stratum": "binary",
            "candidate_stratum": "small_le_16",
            "oracle_feasible_exists": False,
            "oracle_feasible": [False],
            "posterior_point_feasible": [True],
            "posterior_gate_feasible": [False],
        },
    ]
    summary = _group_summary(records, "chemistry_stratum")["binary"]
    assert summary["state_count"] == 2
    assert summary["oracle_feasible_state_rate"] == 0.5
    assert summary["oracle_feasible_action_count"] == 1
    assert summary["posterior_feasible_action_recall"] == 1.0
    assert summary["point_feasible_action_count"] == 3
    assert summary["gate_feasible_action_count"] == 1
    assert summary["point_to_gate_rejection_rate"] == 2 / 3
