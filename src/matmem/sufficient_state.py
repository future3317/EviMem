"""All-outcome sufficient statistics for a fixed linear-Gaussian basis."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from .cards import MaterialMemoryCard, MaterialQuery
from .protocols import (
    CompatibilityKind,
    ProtocolCertificate,
    ProtocolCompatibilityResolver,
)
from .residual_posterior import ResidualPrediction


class SufficientStateUpdate(BaseModel):
    """Audit record for one archive outcome offered to a target state."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    card_id: str
    status: str
    compatibility_kind: CompatibilityKind
    transported_residual_ev_per_atom: float | None = None
    observation_variance: float | None = None


class AllOutcomeLinearGaussianState:
    """Fixed-rank state whose natural parameters include every legal outcome.

    For a frozen, outcome-independent feature map ``phi``, this class stores
    only ``Lambda`` and ``eta``. It never stores or selects a subset of outcome
    cards. Protocol-incompatible outcomes remain in the scientific archive but
    cannot influence this target-protocol state without a directed transport
    certificate.
    """

    def __init__(
        self,
        resolver: ProtocolCompatibilityResolver,
        target_protocol: ProtocolCertificate,
        *,
        feature_dimension: int,
        prior_std_ev_per_atom: float = 0.2,
        observation_noise_std_ev_per_atom: float = 0.03,
    ) -> None:
        if feature_dimension < 2:
            raise ValueError("sufficient-state feature dimension must be at least two")
        if min(prior_std_ev_per_atom, observation_noise_std_ev_per_atom) <= 0:
            raise ValueError("sufficient-state scales must be positive")
        if not all(
            math.isfinite(value)
            for value in (prior_std_ev_per_atom, observation_noise_std_ev_per_atom)
        ):
            raise ValueError("sufficient-state scales must be finite")
        self.resolver = resolver
        self.target_protocol = target_protocol
        self.feature_dimension = feature_dimension
        self.prior_std_ev_per_atom = prior_std_ev_per_atom
        self.observation_noise_std_ev_per_atom = observation_noise_std_ev_per_atom
        self._precision = np.eye(feature_dimension, dtype=np.float64) / (
            prior_std_ev_per_atom**2
        )
        self._eta = np.zeros(feature_dimension, dtype=np.float64)
        self._accepted_count = 0
        self._rejected_count = 0
        self._direct_count = 0
        self._transported_count = 0

    def _feature(self, embedding: tuple[float, ...]) -> np.ndarray:
        if len(embedding) != self.feature_dimension:
            raise ValueError("sufficient-state feature dimension differs from the frozen basis")
        vector = np.asarray(embedding, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm == 0 or not math.isfinite(norm):
            raise ValueError("sufficient-state feature must have finite non-zero norm")
        return vector / norm

    @property
    def accepted_outcome_count(self) -> int:
        return self._accepted_count

    @property
    def rejected_outcome_count(self) -> int:
        return self._rejected_count

    @property
    def state_size_scalars(self) -> int:
        """Fixed numerical representation size, independent of archive length."""

        return self.feature_dimension**2 + self.feature_dimension + 4

    def update(self, card: MaterialMemoryCard) -> SufficientStateUpdate:
        resolution = self.resolver.resolve(card.protocol, self.target_protocol)
        residual = resolution.transfer_residual(card.oracle_residual_ev_per_atom)
        if residual is None:
            self._rejected_count += 1
            return SufficientStateUpdate(
                card_id=card.card_id,
                status="rejected_no_transport_certificate",
                compatibility_kind=CompatibilityKind.REJECT,
            )
        feature = self._feature(card.embedding)
        variance = (
            self.observation_noise_std_ev_per_atom**2
            + resolution.uncertainty_radius_ev_per_atom**2
        )
        weight = 1.0 / variance
        self._precision += weight * np.outer(feature, feature)
        self._eta += weight * feature * residual
        self._accepted_count += 1
        if resolution.kind is CompatibilityKind.DIRECT:
            self._direct_count += 1
        else:
            self._transported_count += 1
        return SufficientStateUpdate(
            card_id=card.card_id,
            status="accepted_all_outcome_update",
            compatibility_kind=resolution.kind,
            transported_residual_ev_per_atom=residual,
            observation_variance=variance,
        )

    def update_many(
        self, cards: Iterable[MaterialMemoryCard]
    ) -> tuple[SufficientStateUpdate, ...]:
        return tuple(self.update(card) for card in cards)

    def natural_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return self._precision.copy(), self._eta.copy()

    def state_checksum(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.target_protocol.scientific_fingerprint.encode())
        digest.update(self._precision.tobytes(order="C"))
        digest.update(self._eta.tobytes(order="C"))
        digest.update(
            np.asarray(
                [
                    self._accepted_count,
                    self._rejected_count,
                    self._direct_count,
                    self._transported_count,
                ],
                dtype=np.int64,
            ).tobytes()
        )
        return "sha256:" + digest.hexdigest()

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))

    def predict(self, queries: Sequence[MaterialQuery]) -> ResidualPrediction:
        items = tuple(queries)
        if any(
            query.protocol.scientific_fingerprint
            != self.target_protocol.scientific_fingerprint
            for query in items
        ):
            raise ValueError("sufficient state can predict only its frozen target protocol")
        posterior_mean = np.linalg.solve(self._precision, self._eta)
        posterior_covariance = np.linalg.inv(self._precision)
        means: list[float] = []
        stds: list[float] = []
        probabilities: list[float] = []
        for query in items:
            feature = self._feature(query.embedding)
            mean = float(feature @ posterior_mean)
            variance = float(feature @ posterior_covariance @ feature)
            variance += self.observation_noise_std_ev_per_atom**2
            std = math.sqrt(max(variance, np.finfo(float).eps))
            threshold = (
                query.stability_threshold_ev_per_atom
                - query.base_hull_distance_ev_per_atom
            )
            probability = self._normal_cdf((threshold - mean) / std)
            means.append(mean)
            stds.append(std)
            probabilities.append(min(1.0, max(0.0, probability)))
        return ResidualPrediction(
            query_ids=tuple(query.query_id for query in items),
            mean_ev_per_atom=tuple(means),
            std_ev_per_atom=tuple(stds),
            stable_probability=tuple(probabilities),
            compatible_witness_count=(self._accepted_count,) * len(items),
        )


class AllOutcomeTargetCorrectionState:
    """All-target-outcome Bayesian ridge correction on a frozen representation.

    This state is the natural-parameter form of ridge regression.  It stores no
    cards and exposes no capacity or eviction operation.  Every target reveal
    contributes exactly one outer product and response vector update.
    """

    def __init__(
        self,
        *,
        feature_mean: tuple[float, ...],
        feature_scale: tuple[float, ...],
        ridge_penalty: float,
        residual_variance_ev2_per_atom2: float,
    ) -> None:
        if len(feature_mean) < 2 or len(feature_mean) != len(feature_scale):
            raise ValueError("target correction requires a fixed feature dimension")
        if any(
            not math.isfinite(value) for value in (*feature_mean, *feature_scale)
        ) or any(value <= 0 for value in feature_scale):
            raise ValueError("target correction feature standardization is invalid")
        if (
            not math.isfinite(ridge_penalty)
            or ridge_penalty <= 0
            or not math.isfinite(residual_variance_ev2_per_atom2)
            or residual_variance_ev2_per_atom2 <= 0
        ):
            raise ValueError("target correction scales must be finite and positive")
        self.feature_mean = np.asarray(feature_mean, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.ridge_penalty = ridge_penalty
        self.residual_variance_ev2_per_atom2 = residual_variance_ev2_per_atom2
        dimension = len(feature_mean) + 1
        self._precision = np.eye(dimension, dtype=np.float64) * ridge_penalty
        self._eta = np.zeros(dimension, dtype=np.float64)
        self._outcome_count = 0

    def _feature(self, embedding: tuple[float, ...]) -> np.ndarray:
        if len(embedding) != len(self.feature_mean):
            raise ValueError("target correction embedding dimension differs from freeze")
        vector = np.asarray(embedding, dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError("target correction embedding must be finite")
        return np.concatenate(([1.0], (vector - self.feature_mean) / self.feature_scale))

    @property
    def accepted_outcome_count(self) -> int:
        return self._outcome_count

    @property
    def state_size_scalars(self) -> int:
        dimension = len(self.feature_mean) + 1
        return dimension**2 + dimension + 1

    def update(
        self, embedding: tuple[float, ...], target_minus_base_ev_per_atom: float
    ) -> None:
        if not math.isfinite(target_minus_base_ev_per_atom):
            raise ValueError("target correction outcome must be finite")
        feature = self._feature(embedding)
        self._precision += np.outer(feature, feature)
        self._eta += feature * target_minus_base_ev_per_atom
        self._outcome_count += 1

    def natural_parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return self._precision.copy(), self._eta.copy()

    def state_checksum(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.feature_mean.tobytes(order="C"))
        digest.update(self.feature_scale.tobytes(order="C"))
        digest.update(self._precision.tobytes(order="C"))
        digest.update(self._eta.tobytes(order="C"))
        digest.update(np.asarray([self._outcome_count], dtype=np.int64).tobytes())
        return "sha256:" + digest.hexdigest()

    def predict(self, embedding: tuple[float, ...]) -> tuple[float, float]:
        feature = self._feature(embedding)
        mean = float(feature @ np.linalg.solve(self._precision, self._eta))
        leverage = 1.0 + float(feature @ np.linalg.solve(self._precision, feature))
        working_std = math.sqrt(
            max(
                self.residual_variance_ev2_per_atom2 * leverage,
                np.finfo(float).eps,
            )
        )
        return mean, working_std
