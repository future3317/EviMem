"""Fixed decision-risk gains and facility-location calibration utilities."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from .cards import MaterialMemoryCard, MaterialQuery
from .residual_posterior import FixedKernelResidualGP


@dataclass(frozen=True)
class CalibrationUtilityMatrix:
    """One immutable ``G_t(u, m)`` matrix for a fixed round and query pool."""

    query_ids: tuple[str, ...]
    witness_ids: tuple[str, ...]
    gains: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.gains, dtype=float)
        if values.shape != (len(self.query_ids), len(self.witness_ids)):
            raise ValueError("calibration utility shape does not match its IDs")
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("calibration utility query IDs must be unique")
        if len(set(self.witness_ids)) != len(self.witness_ids):
            raise ValueError("calibration utility witness IDs must be unique")
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("calibration gains must be finite and non-negative")
        values = values.copy()
        values.setflags(write=False)
        object.__setattr__(self, "gains", values)

    def _indices(self, selected_ids: Collection[str]) -> tuple[int, ...]:
        unknown = set(selected_ids) - set(self.witness_ids)
        if unknown:
            raise KeyError(f"unknown calibration witnesses: {sorted(unknown)}")
        selected = set(selected_ids)
        return tuple(
            index for index, witness_id in enumerate(self.witness_ids) if witness_id in selected
        )

    def value(self, selected_ids: Collection[str]) -> float:
        indices = self._indices(selected_ids)
        if not indices or not self.query_ids:
            return 0.0
        return float(np.max(self.gains[:, indices], axis=1).sum())

    def marginal_gain(
        self,
        selected_ids: Collection[str],
        candidate_id: str,
    ) -> float:
        if candidate_id not in self.witness_ids:
            raise KeyError(f"unknown calibration witness: {candidate_id}")
        if candidate_id in selected_ids:
            return 0.0
        return self.value((*selected_ids, candidate_id)) - self.value(selected_ids)


class CalibrationUtilityBuilder:
    """Construct query-fixed gains from a frozen residual posterior.

    Hyperparameters come from the isolated calibration partition, while the
    online baseline contains no evaluation outcome.  Every candidate witness
    uses that same baseline and the same query-fixed boundary weights, so its
    value cannot change merely because another witness is in the candidate set.
    """

    def __init__(
        self,
        posterior_template: FixedKernelResidualGP,
        *,
        false_stable_cost: float = 5.0,
        false_unstable_cost: float = 1.0,
        boundary_scale_ev_per_atom: float = 0.05,
    ) -> None:
        if min(false_stable_cost, false_unstable_cost, boundary_scale_ev_per_atom) <= 0:
            raise ValueError("calibration risk costs and boundary scale must be positive")
        self.posterior_template = posterior_template
        self.false_stable_cost = false_stable_cost
        self.false_unstable_cost = false_unstable_cost
        self.boundary_scale_ev_per_atom = boundary_scale_ev_per_atom

    def _risks(
        self,
        queries: Sequence[MaterialQuery],
        cards: Sequence[MaterialMemoryCard],
    ) -> dict[str, float]:
        posterior = self.posterior_template.clone_unfit().fit(cards)
        return posterior.decision_risks(
            queries,
            false_stable_cost=self.false_stable_cost,
            false_unstable_cost=self.false_unstable_cost,
        )

    def _boundary_weight(self, query: MaterialQuery) -> float:
        margin = abs(
            query.base_hull_distance_ev_per_atom
            - query.stability_threshold_ev_per_atom
        )
        return float(np.exp(-margin / self.boundary_scale_ev_per_atom))

    def build(
        self,
        queries: Iterable[MaterialQuery],
        witnesses: Iterable[MaterialMemoryCard],
    ) -> tuple[CalibrationUtilityMatrix, float]:
        query_items = tuple(queries)
        witness_items = tuple(witnesses)
        if len({query.query_id for query in query_items}) != len(query_items):
            raise ValueError("calibration utility queries must have unique IDs")
        if len({card.card_id for card in witness_items}) != len(witness_items):
            raise ValueError("calibration utility witnesses must have unique IDs")
        baseline = self._risks(query_items, ())
        gains = np.zeros((len(query_items), len(witness_items)), dtype=float)
        for column, witness in enumerate(witness_items):
            conditioned = self.posterior_template.single_witness_decision_risks(
                query_items,
                witness,
                false_stable_cost=self.false_stable_cost,
                false_unstable_cost=self.false_unstable_cost,
            )
            for row, query in enumerate(query_items):
                gains[row, column] = self._boundary_weight(query) * max(
                    0.0,
                    baseline[query.query_id] - conditioned[query.query_id],
                )
        return (
            CalibrationUtilityMatrix(
                query_ids=tuple(query.query_id for query in query_items),
                witness_ids=tuple(card.card_id for card in witness_items),
                gains=gains,
            ),
            sum(baseline.values()),
        )
