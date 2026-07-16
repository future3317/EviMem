from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np

MODULE = runpy.run_path(
    str(
        Path(__file__).parents[1]
        / "tools"
        / "run_wbm_calibration_engineering_pilot.py"
    )
)


def test_exact_gram_embedding_preserves_all_pool_soap_inner_products() -> None:
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(8, 20))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    compact = MODULE["exact_gram_embedding"](vectors)
    assert compact.shape[0] == 8
    assert compact.shape[1] <= 8
    np.testing.assert_allclose(compact @ compact.T, vectors @ vectors.T, atol=1e-9)


def test_strategy_factorization_keeps_acquisition_matched() -> None:
    config = MODULE["FixedKernelGPConfig"](length_scale=0.35)
    fifo_policy, _ = MODULE["_strategy"]("fifo", 4, config)
    diversity_policy, _ = MODULE["_strategy"]("diversity", 4, config)
    variance_policy, _ = MODULE["_strategy"]("gp_variance_one_swap", 4, config)
    coreset_policy, _ = MODULE["_strategy"]("decision_coreset", 4, config)
    joint_policy, _ = MODULE["_strategy"](
        "joint_posterior_risk_one_swap", 4, config
    )
    assert (
        fifo_policy.policy
        == diversity_policy.policy
        == variance_policy.policy
        == coreset_policy.policy
        == joint_policy.policy
    )
    assert fifo_policy.identity_checksum == diversity_policy.identity_checksum
    frozen_fifo, _ = MODULE["_strategy"]("fifo", 4, config, acquisition="frozen")
    frozen_coreset, _ = MODULE["_strategy"](
        "decision_coreset", 4, config, acquisition="frozen"
    )
    frozen_joint, _ = MODULE["_strategy"](
        "joint_posterior_risk_one_swap", 4, config, acquisition="frozen"
    )
    assert (
        frozen_fifo.identity_checksum
        == frozen_coreset.identity_checksum
        == frozen_joint.identity_checksum
    )


def test_full_history_baseline_does_not_truncate_after_sixteen_witnesses() -> None:
    config = MODULE["FixedKernelGPConfig"](length_scale=0.35)
    _, evidence = MODULE["_strategy"]("full_history", 2, config)
    archive = tuple(range(32))
    assert evidence.active(archive) == archive
