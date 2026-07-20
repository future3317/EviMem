"""Fixed decision-risk gains and facility-location calibration utilities."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import ndtr

from .cards import MaterialMemoryCard, MaterialQuery
from .residual_posterior import FixedKernelResidualGP, ResidualPrediction


def _readonly_vector(values: Iterable[float], *, name: str) -> np.ndarray:
    vector = np.asarray(tuple(values), dtype=float)
    if vector.ndim != 1 or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    vector = vector.copy()
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True)
class ReferencePosteriorSnapshot:
    """A fixed full-evidence posterior shared by every candidate subset."""

    query_ids: tuple[str, ...]
    mean_ev_per_atom: np.ndarray
    std_ev_per_atom: np.ndarray
    stable_probability: np.ndarray
    residual_threshold_ev_per_atom: np.ndarray

    def __post_init__(self) -> None:
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("reference posterior query IDs must be unique")
        for field in (
            "mean_ev_per_atom",
            "std_ev_per_atom",
            "stable_probability",
            "residual_threshold_ev_per_atom",
        ):
            vector = _readonly_vector(getattr(self, field), name=field)
            if len(vector) != len(self.query_ids):
                raise ValueError("reference posterior vectors must match query IDs")
            object.__setattr__(self, field, vector)
        if np.any(self.std_ev_per_atom <= 0):
            raise ValueError("reference posterior standard deviations must be positive")
        if np.any((self.stable_probability < 0) | (self.stable_probability > 1)):
            raise ValueError("reference posterior probabilities must lie in [0, 1]")

    @classmethod
    def from_prediction(
        cls,
        queries: Sequence[MaterialQuery],
        prediction: ResidualPrediction,
    ) -> ReferencePosteriorSnapshot:
        items = tuple(queries)
        query_ids = tuple(item.query_id for item in items)
        if prediction.query_ids != query_ids:
            raise ValueError("reference prediction order does not match the query pool")
        return cls(
            query_ids=query_ids,
            mean_ev_per_atom=np.asarray(prediction.mean_ev_per_atom),
            std_ev_per_atom=np.asarray(prediction.std_ev_per_atom),
            stable_probability=np.asarray(prediction.stable_probability),
            residual_threshold_ev_per_atom=np.asarray(
                [
                    item.stability_threshold_ev_per_atom
                    - item.base_hull_distance_ev_per_atom
                    for item in items
                ]
            ),
        )


def bernoulli_brier_divergence(
    reference_probability: float,
    candidate_probability: float,
) -> float:
    """Strictly proper Brier excess risk for a Bernoulli forecast."""

    if not 0 <= reference_probability <= 1 or not 0 <= candidate_probability <= 1:
        raise ValueError("Bernoulli probabilities must lie in [0, 1]")
    return float((candidate_probability - reference_probability) ** 2)


def bernoulli_log_divergence(
    reference_probability: float,
    candidate_probability: float,
    *,
    clip: float = 1e-12,
) -> float:
    """Bernoulli KL induced by the proper logarithmic score."""

    if not 0 <= reference_probability <= 1 or not 0 <= candidate_probability <= 1:
        raise ValueError("Bernoulli probabilities must lie in [0, 1]")
    if not 0 < clip < 0.5:
        raise ValueError("probability clip must lie in (0, 0.5)")
    q = float(np.clip(reference_probability, clip, 1 - clip))
    p = float(np.clip(candidate_probability, clip, 1 - clip))
    value = q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
    return float(max(0.0, value))


def gaussian_kl_divergence(
    reference_mean: float,
    reference_std: float,
    candidate_mean: float,
    candidate_std: float,
) -> float:
    """KL(N_reference || N_candidate), including mean and scale fidelity."""

    values = (reference_mean, reference_std, candidate_mean, candidate_std)
    if not all(np.isfinite(values)) or min(reference_std, candidate_std) <= 0:
        raise ValueError("Gaussian moments must be finite with positive scales")
    value = (
        np.log(candidate_std / reference_std)
        + (reference_std**2 + (reference_mean - candidate_mean) ** 2)
        / (2 * candidate_std**2)
        - 0.5
    )
    return float(max(0.0, value))


def threshold_weighted_crps_divergence(
    reference_mean: float,
    reference_std: float,
    candidate_mean: float,
    candidate_std: float,
    *,
    threshold: float,
    bandwidth: float,
    weight_floor: float = 0.05,
    quadrature_points: int = 257,
) -> float:
    """Deterministic threshold-weighted CRPS divergence for two Gaussians."""

    values = (
        reference_mean,
        reference_std,
        candidate_mean,
        candidate_std,
        threshold,
        bandwidth,
        weight_floor,
    )
    if not all(np.isfinite(values)) or min(reference_std, candidate_std, bandwidth) <= 0:
        raise ValueError("weighted CRPS inputs must be finite with positive scales")
    if not 0 < weight_floor <= 1:
        raise ValueError("weighted CRPS floor must lie in (0, 1]")
    if quadrature_points < 33 or quadrature_points % 2 == 0:
        raise ValueError("weighted CRPS quadrature requires an odd count of at least 33")
    if reference_mean == candidate_mean and reference_std == candidate_std:
        return 0.0
    tail = 8.0 * max(reference_std, candidate_std, bandwidth)
    lower = min(reference_mean, candidate_mean, threshold) - tail
    upper = max(reference_mean, candidate_mean, threshold) + tail
    grid = np.linspace(lower, upper, quadrature_points)
    reference_cdf = ndtr((grid - reference_mean) / reference_std)
    candidate_cdf = ndtr((grid - candidate_mean) / candidate_std)
    weights = weight_floor + np.exp(-np.abs(grid - threshold) / bandwidth)
    value = np.trapezoid(weights * (reference_cdf - candidate_cdf) ** 2, grid)
    return float(max(0.0, value))


def reference_decision_regret(
    reference_probability: float,
    candidate_probability: float,
    *,
    false_stable_cost: float,
    false_unstable_cost: float,
) -> float:
    """Excess asymmetric decision cost under a fixed reference probability."""

    if min(false_stable_cost, false_unstable_cost) <= 0:
        raise ValueError("decision costs must be positive")
    if not 0 <= reference_probability <= 1 or not 0 <= candidate_probability <= 1:
        raise ValueError("Bernoulli probabilities must lie in [0, 1]")
    threshold = false_stable_cost / (false_stable_cost + false_unstable_cost)
    candidate_stable = candidate_probability >= threshold
    candidate_cost = (
        false_stable_cost * (1 - reference_probability)
        if candidate_stable
        else false_unstable_cost * reference_probability
    )
    reference_optimum = min(
        false_stable_cost * (1 - reference_probability),
        false_unstable_cost * reference_probability,
    )
    return float(max(0.0, candidate_cost - reference_optimum))


@dataclass(frozen=True)
class ProperPosteriorDivergence:
    """Frozen proper-score objective for posterior projection."""

    kind: Literal["brier", "log", "gaussian_kl", "threshold_weighted_crps"]
    threshold_bandwidth_ev_per_atom: float = 0.05
    weight_floor: float = 0.05
    reference_weight_floor: float = 0.05

    def __post_init__(self) -> None:
        if self.kind not in {
            "brier",
            "log",
            "gaussian_kl",
            "threshold_weighted_crps",
        }:
            raise ValueError("unsupported proper posterior divergence")
        if self.threshold_bandwidth_ev_per_atom <= 0:
            raise ValueError("threshold bandwidth must be positive")
        if not 0 < self.weight_floor <= 1 or not 0 < self.reference_weight_floor <= 1:
            raise ValueError("posterior-projection weight floors must lie in (0, 1]")

    def reference_weights(
        self,
        reference: ReferencePosteriorSnapshot,
        *,
        false_stable_cost: float,
        false_unstable_cost: float,
    ) -> np.ndarray:
        maximum = false_stable_cost * false_unstable_cost / (
            false_stable_cost + false_unstable_cost
        )
        bayes_risk = np.minimum(
            false_stable_cost * (1 - reference.stable_probability),
            false_unstable_cost * reference.stable_probability,
        )
        return np.asarray(
            self.reference_weight_floor
            + (1 - self.reference_weight_floor) * (bayes_risk / maximum),
            dtype=float,
        )

    def per_query(
        self,
        reference: ReferencePosteriorSnapshot,
        candidate: ResidualPrediction,
    ) -> np.ndarray:
        if candidate.query_ids != reference.query_ids:
            raise ValueError("candidate prediction order differs from the reference")
        candidate_mean = np.asarray(candidate.mean_ev_per_atom, dtype=float)
        candidate_std = np.asarray(candidate.std_ev_per_atom, dtype=float)
        candidate_probability = np.asarray(candidate.stable_probability, dtype=float)
        values = []
        for index in range(len(reference.query_ids)):
            if self.kind == "brier":
                value = bernoulli_brier_divergence(
                    float(reference.stable_probability[index]),
                    float(candidate_probability[index]),
                )
            elif self.kind == "log":
                value = bernoulli_log_divergence(
                    float(reference.stable_probability[index]),
                    float(candidate_probability[index]),
                )
            elif self.kind == "gaussian_kl":
                value = gaussian_kl_divergence(
                    float(reference.mean_ev_per_atom[index]),
                    float(reference.std_ev_per_atom[index]),
                    float(candidate_mean[index]),
                    float(candidate_std[index]),
                )
            else:
                value = threshold_weighted_crps_divergence(
                    float(reference.mean_ev_per_atom[index]),
                    float(reference.std_ev_per_atom[index]),
                    float(candidate_mean[index]),
                    float(candidate_std[index]),
                    threshold=float(reference.residual_threshold_ev_per_atom[index]),
                    bandwidth=self.threshold_bandwidth_ev_per_atom,
                    weight_floor=self.weight_floor,
                )
            values.append(value)
        return np.asarray(values, dtype=float)


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

    def boundary_weights(
        self, queries: Sequence[MaterialQuery]
    ) -> dict[str, float]:
        """Return the fixed query weights used by every objective in a round."""

        return {query.query_id: self._boundary_weight(query) for query in queries}

    def weighted_decision_risk(
        self,
        queries: Iterable[MaterialQuery],
        witnesses: Iterable[MaterialMemoryCard],
    ) -> float:
        """Evaluate the true joint-GP decision risk of one witness subset."""

        query_items = tuple(queries)
        weights = self.boundary_weights(query_items)
        risks = self._risks(query_items, tuple(witnesses))
        return float(
            sum(weights[query.query_id] * risks[query.query_id] for query in query_items)
        )

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
        weights = self.boundary_weights(query_items)
        gains = np.zeros((len(query_items), len(witness_items)), dtype=float)
        for column, witness in enumerate(witness_items):
            conditioned = self.posterior_template.single_witness_decision_risks(
                query_items,
                witness,
                false_stable_cost=self.false_stable_cost,
                false_unstable_cost=self.false_unstable_cost,
            )
            for row, query in enumerate(query_items):
                gains[row, column] = weights[query.query_id] * max(
                    0.0,
                    baseline[query.query_id] - conditioned[query.query_id],
                )
        return (
            CalibrationUtilityMatrix(
                query_ids=tuple(query.query_id for query in query_items),
                witness_ids=tuple(card.card_id for card in witness_items),
                gains=gains,
            ),
            float(
                sum(
                    weights[query.query_id] * baseline[query.query_id]
                    for query in query_items
                )
            ),
        )
