from __future__ import annotations

import numpy as np
import pytest

from matmem.ceiling_diagnostics import (
    centered_kernel_target_alignment,
    gaussian_dispersion_diagnostics,
    kernel_moran_autocorrelation,
    nearest_neighbor_sign_agreement,
    regularized_effective_dimension,
    signal_kernel_matrix,
)
from matmem.residual_posterior import FixedKernelGPConfig
from tools.analyze_wbm_model_ceiling import _snapshot_arrays


def test_effective_dimension_respects_rank_and_regularization() -> None:
    config = FixedKernelGPConfig(
        kernel="rbf",
        length_scale=0.2,
        signal_std_ev_per_atom=1.0,
        noise_std_ev_per_atom=0.1,
        jitter=1e-12,
    )
    repeated = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    orthogonal = np.eye(3)
    repeated_dimension = regularized_effective_dimension(repeated, config)
    orthogonal_dimension = regularized_effective_dimension(orthogonal, config)
    assert 0.9 < repeated_dimension < 1.0
    assert orthogonal_dimension > 2.9


def test_alignment_and_neighbor_diagnostics_detect_local_signal() -> None:
    config = FixedKernelGPConfig(kernel="rbf", length_scale=0.3)
    embeddings = np.asarray(
        [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]], dtype=float
    )
    kernel = signal_kernel_matrix(embeddings, config)
    aligned = np.asarray([1.0, 0.8, -1.0, -0.8])
    scrambled = np.asarray([1.0, -1.0, 0.8, -0.8])
    assert centered_kernel_target_alignment(kernel, aligned) > centered_kernel_target_alignment(
        kernel, scrambled
    )
    assert kernel_moran_autocorrelation(kernel, aligned) > 0
    assert nearest_neighbor_sign_agreement(kernel, aligned) == pytest.approx(1.0)


def test_ceiling_snapshot_uses_explicit_threshold_and_rejects_legacy_inversion() -> None:
    record = {
        "query_id": "wbm-1-1",
        "true_residual_ev_per_atom": 0.02,
        "causal_stable_label": 1.0,
        "boundary_weight": 2.0,
        "residual_threshold_ev_per_atom": -0.03,
        # Deliberately saturated: the threshold must not be reconstructed from this.
        "stable_probability": 1.0,
    }
    snapshot = {"query_evaluations": [record]}
    embeddings = {"wbm-1-1": np.asarray([1.0, 0.0])}

    *_, thresholds = _snapshot_arrays(snapshot, embeddings)
    assert thresholds.tolist() == pytest.approx([-0.03])

    del record["residual_threshold_ev_per_atom"]
    with pytest.raises(KeyError, match="residual_threshold_ev_per_atom"):
        _snapshot_arrays(snapshot, embeddings)


def test_dispersion_diagnostic_detects_overconfidence_and_undercoverage() -> None:
    result = gaussian_dispersion_diagnostics(
        np.asarray([0.0, 1.0, -1.0]),
        np.zeros(3),
        np.asarray([0.1, 0.1, 0.1]),
    )
    assert result["mean_squared_standardized_residual"] > 10
    assert result["central_90_coverage"] < 0.9
    with pytest.raises(ValueError, match="positive"):
        gaussian_dispersion_diagnostics(
            np.asarray([0.0]),
            np.asarray([0.0]),
            np.asarray([0.0]),
        )
