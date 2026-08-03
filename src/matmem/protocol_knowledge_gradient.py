"""Compatibility shim for the original protocol_knowledge_gradient module.

The implementation has been split into four focused modules:

- :mod:`matmem.transport` — frozen ridge/kernel transport models
- :mod:`matmem.posterior` — Gaussian working posterior and sampling
- :mod:`matmem.hull_geometry` — shared hull-geometry primitives
- :mod:`matmem.protocol_acquisition` — acquisition and rollout policies

This module re-exports every public symbol that existed before the split so
existing imports continue to work.
"""

from __future__ import annotations

from .hull_geometry import (
    FixedCompositionHullTemplate,
    _CausalHullEnvelope,
    _final_hull_membership,
    fixed_composition_hull_membership,
)
from .posterior import (
    ProtocolTargetEnergyPosterior,
    _sample_gaussian,
    protocol_target_energy_posterior,
)
from .protocol_acquisition import (
    ConformalSourceRolloutCalibration,
    ConformalSourceRolloutResult,
    DeltaHullActiveSearchResult,
    DeltaHullAnchoredRolloutResult,
    DualHorizonSourceRolloutResult,
    IndependentConfirmationSourceRolloutResult,
    ProtocolHullKnowledgeGradientResult,
    ProtocolHullPosteriorSummary,
    ProtocolHullRiskReductionResult,
    SourceRolloutDeltaHullResult,
    _source_rollout_rewards,
    conformal_one_deviation_source_rollout,
    constrained_dual_horizon_source_rollout,
    delta_hull_active_search,
    delta_hull_anchored_rollout,
    fit_conformal_source_rollout_calibration,
    independent_confirmation_source_rollout,
    posterior_rank_diagnostics,
    protocol_hull_knowledge_gradient,
    protocol_hull_posterior_summary,
    protocol_hull_risk_reduction,
    source_margin_action_indices,
    source_rollout_delta_hull,
    source_rollout_system_score,
)
from .transport import (
    FrozenProtocolRidgeTransport,
    fit_protocol_kernel_transport,
    fit_protocol_ridge_transport,
)

__all__ = [
    "ConformalSourceRolloutCalibration",
    "ConformalSourceRolloutResult",
    "DeltaHullActiveSearchResult",
    "DualHorizonSourceRolloutResult",
    "FixedCompositionHullTemplate",
    "FrozenProtocolRidgeTransport",
    "IndependentConfirmationSourceRolloutResult",
    "ProtocolHullKnowledgeGradientResult",
    "ProtocolHullPosteriorSummary",
    "ProtocolHullRiskReductionResult",
    "ProtocolTargetEnergyPosterior",
    "SourceRolloutDeltaHullResult",
    "DeltaHullAnchoredRolloutResult",
    "conformal_one_deviation_source_rollout",
    "constrained_dual_horizon_source_rollout",
    "delta_hull_active_search",
    "fit_conformal_source_rollout_calibration",
    "fit_protocol_kernel_transport",
    "fit_protocol_ridge_transport",
    "fixed_composition_hull_membership",
    "independent_confirmation_source_rollout",
    "protocol_hull_knowledge_gradient",
    "protocol_hull_posterior_summary",
    "protocol_hull_risk_reduction",
    "protocol_target_energy_posterior",
    "source_margin_action_indices",
    "source_rollout_delta_hull",
    "delta_hull_anchored_rollout",
    "posterior_rank_diagnostics",
    "source_rollout_system_score",
]
