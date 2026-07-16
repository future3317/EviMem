"""Fixed-hyperparameter residual posteriors for bounded online calibration.

Hyperparameters are inputs to this module.  They must be frozen on disjoint
calibration chemical systems; the implementation never optimizes them on an
evaluation pool.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel

from .cards import MaterialMemoryCard, MaterialQuery
from .protocols import ProtocolCompatibilityResolver


class ResidualPrediction(BaseModel):
    """Posterior moments and induced stability probabilities in query order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_ids: tuple[str, ...]
    mean_ev_per_atom: tuple[float, ...]
    std_ev_per_atom: tuple[float, ...]
    stable_probability: tuple[float, ...]
    compatible_witness_count: tuple[int, ...]


class ResidualPosterior(Protocol):
    """Posterior interface used by calibration and acquisition."""

    def fit(self, cards: Sequence[MaterialMemoryCard]) -> Self: ...

    def predict(self, queries: Sequence[MaterialQuery]) -> ResidualPrediction: ...

    def sample_residuals(
        self,
        query: MaterialQuery,
        *,
        num_samples: int,
        seed: int,
    ) -> np.ndarray: ...


@dataclass(frozen=True)
class FixedKernelGPConfig:
    """Frozen GP settings; no optimizer or evaluation-pool fitting is allowed."""

    kernel: str = "matern52"
    length_scale: float = 1.0
    signal_std_ev_per_atom: float = 0.08
    noise_std_ev_per_atom: float = 0.01
    jitter: float = 1e-10

    def __post_init__(self) -> None:
        if self.kernel not in {"matern52", "rbf"}:
            raise ValueError("fixed residual GP kernel must be 'matern52' or 'rbf'")
        if min(
            self.length_scale,
            self.signal_std_ev_per_atom,
            self.noise_std_ev_per_atom,
            self.jitter,
        ) <= 0:
            raise ValueError("fixed residual GP hyperparameters must be positive")


class FixedKernelResidualGP:
    """Gaussian residual posterior over normalized SOAP-like embeddings.

    A query uses only certificate-compatible cards.  Models are cached by the
    exact compatible training view, so the common single-protocol WBM case pays
    one ``O(K^3)`` factorization per fit.  No unqueried outcome is accepted by
    this class.
    """

    def __init__(
        self,
        resolver: ProtocolCompatibilityResolver,
        *,
        config: FixedKernelGPConfig | None = None,
    ) -> None:
        self.resolver = resolver
        self.config = config or FixedKernelGPConfig()
        self._cards: tuple[MaterialMemoryCard, ...] = ()
        self._models: dict[
            tuple[tuple[str, float, float], ...], GaussianProcessRegressor
        ] = {}

    def clone_unfit(self) -> FixedKernelResidualGP:
        return type(self)(self.resolver, config=self.config)

    @staticmethod
    def _normalized_embedding(values: tuple[float, ...]) -> np.ndarray:
        vector = np.asarray(values, dtype=float)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("residual GP embeddings must have finite non-zero norm")
        return vector / norm

    def fit(self, cards: Sequence[MaterialMemoryCard]) -> Self:
        items = tuple(cards)
        if len({card.card_id for card in items}) != len(items):
            raise ValueError("residual GP training card IDs must be unique")
        dimensions = {len(card.embedding) for card in items}
        if len(dimensions) > 1:
            raise ValueError("residual GP training embeddings must share a dimension")
        self._cards = items
        self._models.clear()
        return self

    def _compatible_view(
        self, query: MaterialQuery
    ) -> tuple[tuple[MaterialMemoryCard, float, float], ...]:
        compatible: list[tuple[MaterialMemoryCard, float, float]] = []
        for card in self._cards:
            if len(card.embedding) != len(query.embedding):
                raise ValueError("residual GP query and witness dimensions differ")
            resolution = self.resolver.resolve(card.protocol, query.protocol)
            residual = resolution.transfer_residual(card.oracle_residual_ev_per_atom)
            if residual is None:
                continue
            compatible.append(
                (card, residual, resolution.uncertainty_radius_ev_per_atom)
            )
        return tuple(compatible)

    def _signal_kernel(self):
        base = (
            Matern(length_scale=self.config.length_scale, nu=2.5)
            if self.config.kernel == "matern52"
            else RBF(length_scale=self.config.length_scale)
        )
        return ConstantKernel(
            self.config.signal_std_ev_per_atom**2,
            constant_value_bounds="fixed",
        ) * base

    def _kernel(self):
        return self._signal_kernel() + WhiteKernel(
            noise_level=self.config.noise_std_ev_per_atom**2,
            noise_level_bounds="fixed",
        )

    def _model_for(
        self, view: tuple[tuple[MaterialMemoryCard, float, float], ...]
    ) -> GaussianProcessRegressor:
        key = tuple(
            (card.card_id, float(residual), float(radius))
            for card, residual, radius in view
        )
        cached = self._models.get(key)
        if cached is not None:
            return cached
        x_train = np.vstack(
            [self._normalized_embedding(card.embedding) for card, _, _ in view]
        )
        y_train = np.asarray([residual for _, residual, _ in view], dtype=float)
        alpha = np.asarray(
            [radius**2 + self.config.jitter for _, _, radius in view], dtype=float
        )
        model = GaussianProcessRegressor(
            kernel=self._kernel(),
            alpha=alpha,
            optimizer=None,
            normalize_y=False,
            random_state=0,
        )
        model.fit(x_train, y_train)
        self._models[key] = model
        return model

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def predict(self, queries: Sequence[MaterialQuery]) -> ResidualPrediction:
        items = tuple(queries)
        means: list[float] = []
        stds: list[float] = []
        probabilities: list[float] = []
        counts: list[int] = []
        for query in items:
            view = self._compatible_view(query)
            if not view:
                mean = 0.0
                std = math.sqrt(
                    self.config.signal_std_ev_per_atom**2
                    + self.config.noise_std_ev_per_atom**2
                )
            else:
                model = self._model_for(view)
                mean_array, std_array = model.predict(
                    self._normalized_embedding(query.embedding).reshape(1, -1),
                    return_std=True,
                )
                mean = float(mean_array[0])
                std = max(float(std_array[0]), math.sqrt(self.config.jitter))
            residual_threshold = (
                query.stability_threshold_ev_per_atom
                - query.base_hull_distance_ev_per_atom
            )
            probability = self._normal_cdf((residual_threshold - mean) / std)
            means.append(mean)
            stds.append(std)
            probabilities.append(min(1.0, max(0.0, probability)))
            counts.append(len(view))
        return ResidualPrediction(
            query_ids=tuple(query.query_id for query in items),
            mean_ev_per_atom=tuple(means),
            std_ev_per_atom=tuple(stds),
            stable_probability=tuple(probabilities),
            compatible_witness_count=tuple(counts),
        )

    def sample_residuals(
        self,
        query: MaterialQuery,
        *,
        num_samples: int,
        seed: int,
    ) -> np.ndarray:
        if num_samples < 1:
            raise ValueError("posterior sampling requires at least one fantasy")
        prediction = self.predict((query,))
        generator = np.random.default_rng(seed)
        return generator.normal(
            prediction.mean_ev_per_atom[0],
            prediction.std_ev_per_atom[0],
            size=num_samples,
        )

    def single_witness_decision_risks(
        self,
        queries: Sequence[MaterialQuery],
        witness: MaterialMemoryCard,
        *,
        false_stable_cost: float,
        false_unstable_cost: float,
    ) -> dict[str, float]:
        """Batch the exact one-observation GP risks used by ``G_t(u,m)``."""

        if min(false_stable_cost, false_unstable_cost) <= 0:
            raise ValueError("decision-risk costs must be positive")
        items = tuple(queries)
        if not items:
            return {}
        if any(len(query.embedding) != len(witness.embedding) for query in items):
            raise ValueError("single-witness GP dimensions differ")
        query_x = np.vstack(
            [self._normalized_embedding(query.embedding) for query in items]
        )
        witness_x = self._normalized_embedding(witness.embedding).reshape(1, -1)
        signal = self._signal_kernel()
        cross = np.asarray(signal(query_x, witness_x), dtype=float).reshape(-1)
        query_variance = np.asarray(signal.diag(query_x), dtype=float) + (
            self.config.noise_std_ev_per_atom**2
        )
        witness_variance = float(signal(witness_x, witness_x)[0, 0]) + (
            self.config.noise_std_ev_per_atom**2
        )
        risks: dict[str, float] = {}
        for index, query in enumerate(items):
            resolution = self.resolver.resolve(witness.protocol, query.protocol)
            residual = resolution.transfer_residual(
                witness.oracle_residual_ev_per_atom
            )
            if residual is None:
                prior_std = math.sqrt(query_variance[index])
                mean = 0.0
                std = prior_std
            else:
                denominator = (
                    witness_variance
                    + resolution.uncertainty_radius_ev_per_atom**2
                    + self.config.jitter
                )
                mean = cross[index] * residual / denominator
                variance = query_variance[index] - cross[index] ** 2 / denominator
                std = math.sqrt(max(variance, self.config.jitter))
            residual_threshold = (
                query.stability_threshold_ev_per_atom
                - query.base_hull_distance_ev_per_atom
            )
            probability = self._normal_cdf((residual_threshold - mean) / std)
            risks[query.query_id] = min(
                false_stable_cost * (1.0 - probability),
                false_unstable_cost * probability,
            )
        return risks

    def decision_risks(
        self,
        queries: Iterable[MaterialQuery],
        *,
        false_stable_cost: float,
        false_unstable_cost: float,
    ) -> dict[str, float]:
        if min(false_stable_cost, false_unstable_cost) <= 0:
            raise ValueError("decision-risk costs must be positive")
        items = tuple(queries)
        prediction = self.predict(items)
        return {
            query.query_id: min(
                false_stable_cost * (1.0 - probability),
                false_unstable_cost * probability,
            )
            for query, probability in zip(
                items, prediction.stable_probability, strict=True
            )
        }
