"""Matched-access operation accounting for certified witness working sets.

The ledger intentionally does not score an acquisition policy. It evaluates the
pre-registered estimand in which persistent FIFO and on-demand reconstruction
have the same action and active-witness trajectory, so any difference is access
work rather than witness-selection quality.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .active import ActiveDiscoveryMetrics


class MatchedAccessOperationLedger(BaseModel):
    """Operation counts induced by a common, already-fixed action trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    oracle_admission_certifications: int = Field(ge=0)
    common_witness_scans: int = Field(ge=0)
    persistent_hull_recertifications: int = Field(ge=0)
    on_demand_archive_retrievals: int = Field(ge=0)
    on_demand_recertifications: int = Field(ge=0)

    @classmethod
    def from_metrics(cls, metrics: ActiveDiscoveryMetrics) -> MatchedAccessOperationLedger:
        """Count access work without assigning an economic price.

        At round ``t``, on-demand reconstruction reads and checks the active
        witness set carried by the preceding persistent round. Persistent state
        pays a re-certification only when a causal hull transition affects a
        later round. First certification of a newly revealed oracle card and
        query-time witness scans are common operations and are reported, but
        cancel from the matched access difference.
        """

        previous_active_size = 0
        common_scans = 0
        on_demand_retrievals = 0
        persistent_recertifications = 0
        for index, step in enumerate(metrics.steps):
            common_scans += previous_active_size
            on_demand_retrievals += previous_active_size
            if step.causal_hull_transition_after_observation and index + 1 < len(metrics.steps):
                persistent_recertifications += len(step.active_witness_ids_after_observation)
            previous_active_size = len(step.active_witness_ids_after_observation)
        return cls(
            oracle_admission_certifications=metrics.oracle_calls,
            common_witness_scans=common_scans,
            persistent_hull_recertifications=persistent_recertifications,
            on_demand_archive_retrievals=on_demand_retrievals,
            on_demand_recertifications=on_demand_retrievals,
        )


class MatchedAccessCostModel(BaseModel):
    """Prices only the operations that differ under matched access."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    archive_retrieval_cost: float = Field(ge=0)
    persistent_recertification_cost: float = Field(ge=0)
    on_demand_recertification_cost: float = Field(ge=0)

    def evaluate(self, ledger: MatchedAccessOperationLedger) -> MatchedAccessCost:
        persistent_cost = (
            ledger.persistent_hull_recertifications * self.persistent_recertification_cost
        )
        on_demand_cost = ledger.on_demand_archive_retrievals * self.archive_retrieval_cost + (
            ledger.on_demand_recertifications * self.on_demand_recertification_cost
        )
        return MatchedAccessCost(
            persistent_differential_cost=persistent_cost,
            on_demand_differential_cost=on_demand_cost,
            persistent_net_savings=on_demand_cost - persistent_cost,
        )


class MatchedAccessCost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    persistent_differential_cost: float = Field(ge=0)
    on_demand_differential_cost: float = Field(ge=0)
    persistent_net_savings: float
