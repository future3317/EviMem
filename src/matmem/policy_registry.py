"""Canonical policy names for protocol and campaign execution.

Centralizing policy identifiers eliminates string-typo drift between the
closed-loop runner, the subprocess worker, and experiment scripts.
"""

from __future__ import annotations

from enum import StrEnum


class ProtocolPolicy(StrEnum):
    """Policies available to the protocol closed-loop runner."""

    RANDOM = "random"
    SOURCE_MARGIN = "source_margin"
    POSTERIOR_MEAN_TARGET_MARGIN = "posterior_mean_target_margin"
    DELTA_HULL_ACTIVE_SEARCH = "delta_hull_active_search"
    UNGATED_SOURCE_ROLLOUT = "ungated_source_rollout"
    SOURCE_ROLLOUT_DELTA_HULL = "source_rollout_delta_hull"
    DIAGONAL_IC_SARR = "diagonal_ic_sarr"
    INDEPENDENT_MC_IC_SARR = "independent_mc_ic_sarr"
    CONSTRAINED_DUAL_HORIZON_SOURCE_ROLLOUT = "constrained_dual_horizon_source_rollout"
    INDEPENDENT_CONFIRMATION_SOURCE_ROLLOUT = "independent_confirmation_source_rollout"
    CONFORMAL_SOURCE_ROLLOUT_DELTA_HULL = "conformal_source_rollout_delta_hull"
    PROTOCOL_HULL_KNOWLEDGE_GRADIENT = "protocol_hull_knowledge_gradient"
    PROTOCOL_HULL_RISK_REDUCTION = "protocol_hull_risk_reduction"
    RIDGE_MARGIN = "ridge_margin"
    RIDGE_UNCERTAINTY = "ridge_uncertainty"
    RIDGE_PREDICTED_FINAL_MARGIN = "ridge_predicted_final_margin"
    SOURCE_ONLINE_OFFSET = "source_online_offset"
    SOURCE_ONLINE_AFFINE = "source_online_affine"

    @classmethod
    def transport_required_policies(cls) -> set[ProtocolPolicy]:
        """Policies that need a frozen cross-protocol transport model."""

        return {
            cls.DELTA_HULL_ACTIVE_SEARCH,
            cls.POSTERIOR_MEAN_TARGET_MARGIN,
            cls.UNGATED_SOURCE_ROLLOUT,
            cls.SOURCE_ROLLOUT_DELTA_HULL,
            cls.DIAGONAL_IC_SARR,
            cls.INDEPENDENT_MC_IC_SARR,
            cls.CONSTRAINED_DUAL_HORIZON_SOURCE_ROLLOUT,
            cls.INDEPENDENT_CONFIRMATION_SOURCE_ROLLOUT,
            cls.CONFORMAL_SOURCE_ROLLOUT_DELTA_HULL,
        }

    @classmethod
    def all_hull_aware_policies(cls) -> set[ProtocolPolicy]:
        """All policies that operate on the target-protocol hull."""

        return cls.transport_required_policies() | {
            cls.PROTOCOL_HULL_KNOWLEDGE_GRADIENT,
            cls.PROTOCOL_HULL_RISK_REDUCTION,
        }

    @classmethod
    def ridge_policies(cls) -> set[ProtocolPolicy]:
        """Ridge-based acquisition policies that do not need a transport model."""

        return {
            cls.RIDGE_MARGIN,
            cls.RIDGE_UNCERTAINTY,
            cls.RIDGE_PREDICTED_FINAL_MARGIN,
        }


class CampaignPolicy(StrEnum):
    """Campaign-level strategy policies compared by the IC-SARR gate."""

    SOURCE_MARGIN = "source_margin"
    IC_SARR = "ic_sarr"


def requires_protocol_transport(policy: ProtocolPolicy | str) -> bool:
    """Return whether a policy requires a frozen cross-protocol posterior."""

    name = str(policy)
    if name.startswith("protocol_hull_"):
        return True
    try:
        return ProtocolPolicy(name) in ProtocolPolicy.transport_required_policies()
    except ValueError:
        return False


def is_hull_aware_protocol_policy(policy: ProtocolPolicy | str) -> bool:
    """Return whether a policy is one of the hull-aware protocol policies."""

    name = str(policy)
    if name.startswith("protocol_hull_"):
        return True
    try:
        return ProtocolPolicy(name) in ProtocolPolicy.all_hull_aware_policies()
    except ValueError:
        return False
