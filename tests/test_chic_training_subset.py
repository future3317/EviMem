from __future__ import annotations

import numpy as np
import torch
from pymatgen.entries.computed_entries import ComputedEntry

from tools.run_chic_training_subset_exploratory import (
    ExperimentConfig,
    ResidualMLP,
    _selected_update,
)


def _rows(count: int) -> tuple[list[dict], dict[str, dict]]:
    rows = []
    outcomes = {}
    for index in range(count):
        pair_id = f"q{index}"
        rows.append(
            {
                "pair_id": pair_id,
                "chemical_system": "Fe-O",
                "composition": {"Fe": 1.0, "O": 1.0},
                "source_formation_energy_ev_per_atom": -0.5 + index / 100,
                "source_environment_embedding": [float(index), float(index % 2)],
            }
        )
        outcomes[pair_id] = {
            "target_formation_energy_ev_per_atom": -0.6 + index / 200,
        }
    return rows, outcomes


def test_proxy_gradient_match_only_backpropagates_selected_items() -> None:
    torch.manual_seed(1)
    rows, outcomes = _rows(5)
    model = ResidualMLP(input_dimension=3, hidden_dimension=4).double()
    mean = np.asarray([-0.48, 2.0, 0.4])
    scale = np.ones(3)
    config = ExperimentConfig(capacity=2, hidden_dimension=4)
    exact_entries = [
        ComputedEntry("Fe", 0.0),
        ComputedEntry("O", 0.0),
    ]
    _, matched = _selected_update(
        strategy="grad_match",
        model=model,
        archive_rows=rows,
        remaining_rows=[],
        exact_entries=exact_entries,
        outcomes=outcomes,
        feature_mean=mean,
        feature_scale=scale,
        config=config,
        system="Fe-O",
        round_index=5,
    )
    _, full = _selected_update(
        strategy="full_history",
        model=model,
        archive_rows=rows,
        remaining_rows=[],
        exact_entries=exact_entries,
        outcomes=outcomes,
        feature_mean=mean,
        feature_scale=scale,
        config=config,
        system="Fe-O",
        round_index=5,
    )
    assert matched["selected_size"] <= 2
    assert matched["sample_gradient_evaluations"] == matched["selected_size"]
    assert matched["proxy_gradient_evaluations"] == 5
    assert full["sample_gradient_evaluations"] == 5
    assert full["proxy_gradient_evaluations"] == 0
