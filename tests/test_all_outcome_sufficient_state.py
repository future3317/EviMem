from __future__ import annotations

import numpy as np
import pytest

from matmem import (
    AllOutcomeLinearGaussianState,
    CompatibilityKind,
    ProtocolCompatibilityResolver,
    ProtocolTransportMap,
)

from .test_matmem import _card, _protocol, _query


def _state(
    resolver: ProtocolCompatibilityResolver | None = None,
    *,
    target=None,
) -> AllOutcomeLinearGaussianState:
    return AllOutcomeLinearGaussianState(
        resolver or ProtocolCompatibilityResolver(),
        target or _protocol("PBE"),
        feature_dimension=2,
        prior_std_ev_per_atom=0.2,
        observation_noise_std_ev_per_atom=0.03,
    )


def _transport() -> ProtocolTransportMap:
    return ProtocolTransportMap(
        source_protocol=_protocol("PBE"),
        target_protocol=_protocol("PBE+U"),
        slope=0.5,
        intercept_ev_per_atom=0.01,
        error_radius_ev_per_atom=0.02,
        matched_structure_count=12,
        calibration_group_checksum="sha256:" + "b" * 64,
        calibration_id="disjoint-transport-v1",
    )


def test_streaming_natural_parameters_equal_full_history_replay_exactly() -> None:
    cards = (
        _card("one", embedding=(1.0, 0.0), formation_energy=-0.98),
        _card("two", embedding=(0.0, 1.0), formation_energy=-0.95),
        _card("three", embedding=(1.0, 1.0), formation_energy=-1.01),
    )
    streaming = _state()
    for card in cards:
        streaming.update(card)
    replay = _state()
    replay.update_many(cards)
    left_precision, left_eta = streaming.natural_parameters()
    right_precision, right_eta = replay.natural_parameters()
    assert np.array_equal(left_precision, right_precision)
    assert np.array_equal(left_eta, right_eta)
    assert streaming.state_checksum() == replay.state_checksum()
    assert streaming.predict((_query(),)) == replay.predict((_query(),))


def test_state_size_does_not_grow_with_archive_length() -> None:
    state = _state()
    size = state.state_size_scalars
    state.update_many(
        _card(f"card-{index}", embedding=(1.0, float(index + 1)))
        for index in range(20)
    )
    assert state.accepted_outcome_count == 20
    assert state.state_size_scalars == size


def test_incompatible_outcome_is_archived_but_cannot_change_target_state() -> None:
    state = _state(target=_protocol("PBE+U"))
    before = state.state_checksum()
    update = state.update(_card("source", protocol=_protocol("PBE")))
    assert update.compatibility_kind is CompatibilityKind.REJECT
    assert state.accepted_outcome_count == 0
    assert state.rejected_outcome_count == 1
    assert state.state_checksum() != before  # audit counter changes, natural parameters do not
    precision, eta = state.natural_parameters()
    clean_precision, clean_eta = _state(target=_protocol("PBE+U")).natural_parameters()
    assert np.array_equal(precision, clean_precision)
    assert np.array_equal(eta, clean_eta)


def test_certified_transport_updates_all_outcome_state_with_extra_variance() -> None:
    target = _protocol("PBE+U")
    state = _state(
        ProtocolCompatibilityResolver([_transport()]),
        target=target,
    )
    update = state.update(
        _card(
            "source",
            protocol=_protocol("PBE"),
            formation_energy=-0.93,
            base_energy=-1.03,
        )
    )
    assert update.compatibility_kind is CompatibilityKind.TRANSPORTED
    assert update.transported_residual_ev_per_atom == pytest.approx(0.06)
    assert update.observation_variance == pytest.approx(0.03**2 + 0.02**2)


def test_natural_parameters_match_closed_form_batch_statistics() -> None:
    cards = (
        _card("one", embedding=(3.0, 4.0), formation_energy=-0.98),
        _card("two", embedding=(4.0, 3.0), formation_energy=-0.95),
    )
    state = _state()
    state.update_many(cards)
    features = np.asarray([[0.6, 0.8], [0.8, 0.6]])
    residuals = np.asarray([card.oracle_residual_ev_per_atom for card in cards])
    variance = 0.03**2
    expected_precision = np.eye(2) / 0.2**2 + features.T @ features / variance
    expected_eta = features.T @ residuals / variance
    precision, eta = state.natural_parameters()
    assert np.allclose(precision, expected_precision)
    assert np.allclose(eta, expected_eta)
