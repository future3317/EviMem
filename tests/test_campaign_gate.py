import numpy as np

from matmem.campaign_gate import campaign_gated_ic_sarr
from matmem.protocol_knowledge_gradient import FrozenProtocolRidgeTransport


def _model() -> FrozenProtocolRidgeTransport:
    return FrozenProtocolRidgeTransport(
        feature_mean=(0.0, 0.0, 0.0),
        feature_scale=(1.0, 1.0, 1.0),
        coefficients=(0.0, 0.0, 0.0, 0.0),
        precision=(
            (100.0, 0.0, 0.0, 0.0),
            (0.0, 100.0, 0.0, 0.0),
            (0.0, 0.0, 100.0, 0.0),
            (0.0, 0.0, 0.0, 100.0),
        ),
        within_system_variance=0.01,
        between_system_variance=0.01,
        ridge_penalty=1.0,
        fit_system_ids=("C-D", "E-F"),
        fit_element_ids=("A", "B", "C", "D", "E", "F"),
        fit_row_count=4,
    )


def _kwargs() -> dict[str, object]:
    return {
        "posterior_mean": np.asarray([-0.2, -0.1, -0.1]),
        "posterior_covariance": np.eye(3) * 0.01,
        "model": _model(),
        "query_compositions": (
            {"A": 0.5, "B": 0.5},
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        "query_source_energies": np.asarray([-0.2, -0.1, -0.1]),
        "query_ids": ("ab", "abb", "aab"),
        "query_features": np.asarray([[0.5, 0.5], [0.25, 0.75], [0.75, 0.25]]),
        "query_kernel_features": np.zeros((3, 1)),
        "reference_compositions": ({"A": 1.0}, {"B": 1.0}),
        "reference_energies": np.zeros(2),
        "budget": 2,
        "outer_sample_count": 8,
        "outer_seed": 9,
        "inner_stage_one_sample_count": 8,
        "inner_stage_two_sample_count": 16,
        "sobol_scramble_count": 4,
    }


def test_campaign_gate_is_deterministic_and_selects_a_complete_policy() -> None:
    first = campaign_gated_ic_sarr(**_kwargs())
    second = campaign_gated_ic_sarr(**_kwargs())
    assert first == second
    assert first.selected_policy in {"source_margin", "ic_sarr"}
    assert len(first.terminal_block_differences) == 4
    assert len(first.selected_history_block_differences) == 4
    assert np.isfinite(
        [
            first.terminal_advantage,
            first.selected_history_advantage,
            first.terminal_lower_bound,
            first.selected_history_lower_bound,
        ]
    ).all()
