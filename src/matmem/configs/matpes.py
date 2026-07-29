"""Default experiment parameters for MatPES protocol-aware campaigns.

These defaults are frozen to match the original hard-coded values scattered
across ``tools/`` scripts.  Override them at call sites rather than editing
this module if you need a non-default experimental setting.
"""

from __future__ import annotations

from dataclasses import dataclass

from matmem.constants import (
    DEFAULT_BOUNDARY_TEMPERATURE,
    DEFAULT_CAMPAIGN_INNER_STAGE_ONE_SAMPLE_COUNT,
    DEFAULT_CAMPAIGN_INNER_STAGE_TWO_SAMPLE_COUNT,
    DEFAULT_CAMPAIGN_OUTER_SAMPLE_COUNT,
    DEFAULT_CAMPAIGN_OUTER_SEED,
    DEFAULT_CAMPAIGN_SOBOL_SCRAMBLE_COUNT,
    DEFAULT_EXPERIMENT_SEED,
    DEFAULT_FANTASY_COUNT,
    DEFAULT_INTEGRATION_CONFIDENCE,
    DEFAULT_MATPES_BUDGET,
    DEFAULT_POSTERIOR_SAMPLE_COUNT,
    DEFAULT_PRIOR_STANDARD_DEVIATION,
    DEFAULT_RIDGE_PENALTY,
    DEFAULT_SOBOL_SCRAMBLE_COUNT,
)
from matmem.policy_registry import ProtocolPolicy


@dataclass(frozen=True)
class MatPESDefaults:
    """Frozen defaults for MatPES exploratory and confirmatory experiments."""

    budget: int = DEFAULT_MATPES_BUDGET
    seed: int = DEFAULT_EXPERIMENT_SEED
    posterior_sample_count: int = DEFAULT_POSTERIOR_SAMPLE_COUNT
    sobol_scramble_count: int = DEFAULT_SOBOL_SCRAMBLE_COUNT
    fantasy_count: int = DEFAULT_FANTASY_COUNT
    integration_confidence: float = DEFAULT_INTEGRATION_CONFIDENCE
    ridge_penalty: float = DEFAULT_RIDGE_PENALTY
    prior_standard_deviation: float = DEFAULT_PRIOR_STANDARD_DEVIATION
    boundary_temperature: float = DEFAULT_BOUNDARY_TEMPERATURE
    campaign_outer_sample_count: int = DEFAULT_CAMPAIGN_OUTER_SAMPLE_COUNT
    campaign_outer_seed: int = DEFAULT_CAMPAIGN_OUTER_SEED
    campaign_inner_stage_one_sample_count: int = DEFAULT_CAMPAIGN_INNER_STAGE_ONE_SAMPLE_COUNT
    campaign_inner_stage_two_sample_count: int = DEFAULT_CAMPAIGN_INNER_STAGE_TWO_SAMPLE_COUNT
    campaign_sobol_scramble_count: int = DEFAULT_CAMPAIGN_SOBOL_SCRAMBLE_COUNT

    @property
    def policies(self) -> tuple[ProtocolPolicy, ...]:
        """Default exploratory policy roster."""

        return (
            ProtocolPolicy.SOURCE_MARGIN,
            ProtocolPolicy.DELTA_HULL_ACTIVE_SEARCH,
            ProtocolPolicy.SOURCE_ROLLOUT_DELTA_HULL,
            ProtocolPolicy.CONSTRAINED_DUAL_HORIZON_SOURCE_ROLLOUT,
            ProtocolPolicy.INDEPENDENT_CONFIRMATION_SOURCE_ROLLOUT,
            ProtocolPolicy.CONFORMAL_SOURCE_ROLLOUT_DELTA_HULL,
            ProtocolPolicy.PROTOCOL_HULL_KNOWLEDGE_GRADIENT,
            ProtocolPolicy.PROTOCOL_HULL_RISK_REDUCTION,
            ProtocolPolicy.RIDGE_MARGIN,
            ProtocolPolicy.RIDGE_UNCERTAINTY,
            ProtocolPolicy.RIDGE_PREDICTED_FINAL_MARGIN,
        )


MATPES_DEFAULTS = MatPESDefaults()
