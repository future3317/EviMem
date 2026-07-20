from __future__ import annotations

import numpy as np
import pytest

from matmem.protocol_knowledge_gradient import (
    FrozenProtocolRidgeTransport,
    ProtocolTargetEnergyPosterior,
    delta_hull_active_search,
    fit_protocol_kernel_transport,
    fit_protocol_ridge_transport,
    protocol_hull_knowledge_gradient,
    protocol_hull_risk_reduction,
    protocol_target_energy_posterior,
)


def test_scrambled_sobol_gaussian_samples_are_deterministic_and_nested() -> None:
    from matmem.protocol_knowledge_gradient import _sample_gaussian

    mean = np.asarray([0.2, -0.1, 0.4])
    covariance = np.asarray(
        [
            [0.5, 0.1, 0.0],
            [0.1, 0.3, -0.05],
            [0.0, -0.05, 0.2],
        ]
    )
    small = _sample_gaussian(mean, covariance, sample_count=8, seed=17)
    repeated = _sample_gaussian(mean, covariance, sample_count=8, seed=17)
    large = _sample_gaussian(mean, covariance, sample_count=32, seed=17)
    other_seed = _sample_gaussian(mean, covariance, sample_count=8, seed=18)

    assert np.array_equal(small, repeated)
    assert np.array_equal(small, large[: len(small)])
    assert not np.array_equal(small, other_seed)
    assert np.isfinite(large).all()


def _transport_model():
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.2, 0.0],
            [0.8, 1.0],
            [1.0, 1.0],
        ]
    )
    source = np.asarray([-0.4, -0.2, -0.3, -0.1])
    target = source + np.asarray([0.04, 0.05, -0.03, -0.02])
    return fit_protocol_ridge_transport(
        features=features,
        source_energies=source,
        target_energies=target,
        system_ids=("A-B", "A-B", "C-D", "C-D"),
    )


def test_protocol_transport_is_frozen_and_system_disjoint() -> None:
    model = _transport_model()
    assert model.fit_system_ids == ("A-B", "C-D")
    assert model.fit_element_ids == ("A", "B", "C", "D")
    assert model.fit_row_count == 4
    assert model.identity_checksum.startswith("sha256:")
    assert model.within_system_variance > 0
    assert model.between_system_variance > 0


def test_hierarchical_kernel_transport_fits_only_registered_systems() -> None:
    features = np.asarray([[x, float(system)] for system in range(3) for x in (0.0, 0.3, 0.7, 1.0)])
    source = np.zeros(len(features))
    target = np.asarray(
        [
            0.15 * np.sin(4.0 * x) + 0.02 * system
            for system in range(3)
            for x in (0.0, 0.3, 0.7, 1.0)
        ]
    )
    systems = tuple(system for system in ("A-B", "C-D", "E-F") for _ in range(4))
    model = fit_protocol_kernel_transport(
        features=features,
        kernel_features=features[:, :1],
        source_energies=source,
        target_energies=target,
        system_ids=systems,
        kernel_feature_encoder="fixture-structure-encoder",
        kernel_feature_encoder_checksum="sha256:fixture",
    )
    assert model.local_kernel == "matern52"
    assert model.local_kernel_fit_system_count == 3
    assert model.local_kernel_signal_variance > 0
    assert model.local_kernel_noise_variance > 0
    assert model.local_kernel_nll_per_row is not None


def test_local_discrepancy_reveal_updates_nearby_candidate_more() -> None:
    model = FrozenProtocolRidgeTransport(
        feature_mean=(0.0, 0.0),
        feature_scale=(1.0, 1.0),
        coefficients=(0.0, 0.0, 0.0),
        precision=((1e8, 0.0, 0.0), (0.0, 1e8, 0.0), (0.0, 0.0, 1e8)),
        within_system_variance=0.0901,
        between_system_variance=1e-8,
        ridge_penalty=1.0,
        fit_system_ids=("A-B", "C-D"),
        fit_element_ids=("A", "B", "C", "D"),
        fit_row_count=8,
        local_kernel="matern52",
        local_kernel_signal_variance=0.09,
        local_kernel_noise_variance=0.0001,
        local_kernel_length_scale=0.3,
        local_kernel_fit_system_count=2,
        local_kernel_nll_per_row=0.0,
        kernel_feature_mean=(0.0,),
        kernel_feature_scale=(1.0,),
        kernel_feature_encoder="fixture-structure-encoder",
        kernel_feature_encoder_checksum="sha256:fixture",
    )
    posterior = protocol_target_energy_posterior(
        model,
        # The global-mean features are deliberately reversed relative to the
        # frozen structure embedding.  The local update must follow the latter.
        query_features=np.asarray([[2.0], [0.05]]),
        query_source_energies=np.zeros(2),
        history_features=np.asarray([[0.0]]),
        history_source_energies=np.zeros(1),
        history_target_energies=np.ones(1),
        query_kernel_features=np.asarray([[0.05], [2.0]]),
        history_kernel_features=np.asarray([[0.0]]),
    )
    assert posterior.mean[0] > posterior.mean[1] + 0.8
    assert posterior.covariance[0][0] < posterior.covariance[1][1]

    with pytest.raises(ValueError, match="local-kernel embeddings"):
        protocol_target_energy_posterior(
            model,
            query_features=np.asarray([[2.0], [0.05]]),
            query_source_energies=np.zeros(2),
            history_features=np.asarray([[0.0]]),
            history_source_energies=np.zeros(1),
            history_target_energies=np.ones(1),
        )


def test_every_revealed_outcome_updates_order_invariant_system_state() -> None:
    model = _transport_model()
    query_features = np.asarray([[0.4, 0.2], [0.6, 0.8]])
    query_source = np.asarray([-0.25, -0.15])
    history_features = np.asarray([[0.3, 0.1], [0.7, 0.9]])
    history_source = np.asarray([-0.30, -0.12])
    unconditioned = protocol_target_energy_posterior(
        model,
        query_features=query_features,
        query_source_energies=query_source,
        history_features=np.empty((0, 2)),
        history_source_energies=np.empty(0),
        history_target_energies=np.empty(0),
    )
    conditioned = protocol_target_energy_posterior(
        model,
        query_features=query_features,
        query_source_energies=query_source,
        history_features=history_features,
        history_source_energies=history_source,
        history_target_energies=history_source + 0.20,
    )
    reversed_conditioned = protocol_target_energy_posterior(
        model,
        query_features=query_features,
        query_source_energies=query_source,
        history_features=history_features[::-1],
        history_source_energies=history_source[::-1],
        history_target_energies=(history_source + 0.20)[::-1],
    )
    assert conditioned.history_count == 2
    assert conditioned.system_offset_mean > unconditioned.system_offset_mean
    assert np.asarray(conditioned.mean) == pytest.approx(reversed_conditioned.mean)
    assert not np.allclose(conditioned.mean, unconditioned.mean)
    np.testing.assert_allclose(
        np.asarray(conditioned.covariance),
        np.asarray(reversed_conditioned.covariance),
    )


def test_delta_hull_active_search_prefers_final_support_phase() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.5, 0.2),
        covariance=((1e-8, 0.0), (0.0, 1e-8)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    result = delta_hull_active_search(
        posterior,
        query_compositions=(
            {"A": 0.5, "B": 0.5},
            {"A": 0.5, "B": 0.5},
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        seed=7,
    )
    assert result.final_stability_probabilities == pytest.approx((1.0, 0.0))
    assert np.argmax(result.scores) == 0


def test_delta_hull_active_search_rejects_ratio_heuristic_costs() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.5, 0.2),
        covariance=((0.01, 0.0), (0.0, 0.01)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    with pytest.raises(ValueError, match="equal query costs"):
        delta_hull_active_search(
            posterior,
            query_compositions=(
                {"A": 0.5, "B": 0.5},
                {"A": 0.5, "B": 0.5},
            ),
            reference_compositions=({"A": 1.0}, {"B": 1.0}),
            reference_energies=np.zeros(2),
            costs=np.asarray([1.0, 2.0]),
        )


def test_delta_hull_scores_equal_manual_joint_final_membership_probability() -> None:
    from matmem.protocol_knowledge_gradient import (
        _final_hull_membership,
        _sample_gaussian,
    )

    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.25, -0.18, -0.08),
        covariance=(
            (0.02, 0.01, 0.0),
            (0.01, 0.03, 0.005),
            (0.0, 0.005, 0.01),
        ),
        system_offset_mean=0.0,
        system_offset_variance=0.01,
        history_count=2,
    )
    compositions = (
        {"A": 0.25, "B": 0.75},
        {"A": 0.5, "B": 0.5},
        {"A": 0.75, "B": 0.25},
    )
    result = delta_hull_active_search(
        posterior,
        query_compositions=compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(3),
        posterior_sample_count=32,
        seed=29,
    )
    manual = _final_hull_membership(
        query_compositions=compositions,
        sampled_query_energies=_sample_gaussian(
            np.asarray(posterior.mean),
            np.asarray(posterior.covariance),
            sample_count=32,
            seed=29,
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
    ).mean(axis=0)
    assert result.scores == pytest.approx(manual)
    assert result.final_stability_probabilities == pytest.approx(manual)


def test_two_step_protocol_hull_policy_rejects_nonuniform_costs() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-0.5, 0.2),
        covariance=((0.01, 0.0), (0.0, 0.01)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    with pytest.raises(ValueError, match="unit costs"):
        protocol_hull_knowledge_gradient(
            posterior,
            query_compositions=(
                {"A": 0.5, "B": 0.5},
                {"A": 0.5, "B": 0.5},
            ),
            reference_compositions=({"A": 1.0}, {"B": 1.0}),
            reference_energies=np.zeros(2),
            costs=np.asarray([1.0, 2.0]),
            remaining_budget=2.0,
        )


def test_hull_risk_reduction_remains_informative_when_membership_saturates() -> None:
    posterior = ProtocolTargetEnergyPosterior(
        mean=(-1.0, -1.0),
        covariance=((0.04, 0.0), (0.0, 0.0025)),
        system_offset_mean=0.0,
        system_offset_variance=0.0,
        history_count=0,
    )
    result = protocol_hull_risk_reduction(
        posterior,
        query_compositions=(
            {"A": 0.25, "B": 0.75},
            {"A": 0.75, "B": 0.25},
        ),
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=32,
        fantasy_count=3,
        seed=13,
    )
    assert result.current_hull_risk > 0
    assert result.risk_reductions[0] > result.risk_reductions[1]
    assert np.argmax(result.scores) == 0


def test_hull_risk_grid_is_invariant_when_query_becomes_reference() -> None:
    compositions = (
        {"A": 0.5, "B": 0.5},
        {"A": 0.25, "B": 0.75},
    )
    before = protocol_hull_risk_reduction(
        ProtocolTargetEnergyPosterior(
            mean=(-0.4, -0.3),
            covariance=((0.01, 0.0), (0.0, 0.01)),
            system_offset_mean=0.0,
            system_offset_variance=0.0,
            history_count=0,
        ),
        query_compositions=compositions,
        reference_compositions=({"A": 1.0}, {"B": 1.0}),
        reference_energies=np.zeros(2),
        costs=np.ones(2),
        posterior_sample_count=8,
        fantasy_count=1,
        seed=5,
    )
    after = protocol_hull_risk_reduction(
        ProtocolTargetEnergyPosterior(
            mean=(-0.3,),
            covariance=((0.01,),),
            system_offset_mean=0.0,
            system_offset_variance=0.0,
            history_count=1,
        ),
        query_compositions=(compositions[1],),
        reference_compositions=({"A": 1.0}, {"B": 1.0}, compositions[0]),
        reference_energies=np.asarray([0.0, 0.0, -0.4]),
        costs=np.ones(1),
        posterior_sample_count=8,
        fantasy_count=1,
        seed=5,
    )
    assert before.evaluation_composition_count == 4
    assert after.evaluation_composition_count == before.evaluation_composition_count
