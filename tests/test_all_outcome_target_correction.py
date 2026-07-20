from __future__ import annotations

import numpy as np
import pytest

from matmem import AllOutcomeTargetCorrectionState


def _state() -> AllOutcomeTargetCorrectionState:
    return AllOutcomeTargetCorrectionState(
        feature_mean=(0.0, 0.0),
        feature_scale=(1.0, 2.0),
        ridge_penalty=0.5,
        residual_variance_ev2_per_atom2=0.04,
    )


def test_all_target_outcomes_update_exact_natural_parameters() -> None:
    state = _state()
    observations = (((1.0, 2.0), 0.2), ((-1.0, 4.0), -0.1))
    for embedding, residual in observations:
        state.update(embedding, residual)
    features = np.asarray([[1.0, 1.0, 1.0], [1.0, -1.0, 2.0]])
    targets = np.asarray([0.2, -0.1])
    precision, eta = state.natural_parameters()
    assert np.allclose(precision, 0.5 * np.eye(3) + features.T @ features)
    assert np.allclose(eta, features.T @ targets)
    assert state.accepted_outcome_count == 2


def test_streaming_and_same_order_replay_are_bitwise_exact() -> None:
    observations = (((1.0, 2.0), 0.2), ((-1.0, 4.0), -0.1), ((0.5, 1.0), 0.05))
    streaming, replay = _state(), _state()
    for embedding, residual in observations:
        streaming.update(embedding, residual)
    for embedding, residual in observations:
        replay.update(embedding, residual)
    assert streaming.state_checksum() == replay.state_checksum()
    assert streaming.predict((0.2, 0.4)) == replay.predict((0.2, 0.4))


def test_state_size_is_archive_length_independent() -> None:
    state = _state()
    size = state.state_size_scalars
    for index in range(100):
        state.update((float(index), float(index + 1)), float(index) / 100)
    assert state.accepted_outcome_count == 100
    assert state.state_size_scalars == size


def test_feature_dimension_mismatch_fails_closed() -> None:
    with pytest.raises(ValueError, match="dimension"):
        _state().update((1.0,), 0.1)
