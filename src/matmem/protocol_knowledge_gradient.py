"""Myopic and source-rollout Delta-Hull active search.

The target is the final target-protocol hull over the visible fixed pool, not
the transient causal hull.  A transport model is fitted on disjoint exact
chemical systems.  Every target outcome revealed in the current system then
updates a hierarchical discrepancy posterior; no outcome is selected out of
the scientific state. Source-Rollout uses the strong source-margin policy as a
full-remaining-budget continuation rather than replacing it with a score mix.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from scipy.special import ndtri
from scipy.stats import qmc
from scipy.stats import t as student_t


class FrozenProtocolRidgeTransport(BaseModel):
    """System-balanced ridge transport fitted on disjoint exact systems."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    within_system_variance: float = Field(gt=0)
    between_system_variance: float = Field(gt=0)
    ridge_penalty: float = Field(gt=0)
    fit_system_ids: tuple[str, ...]
    fit_element_ids: tuple[str, ...]
    fit_row_count: int = Field(gt=0)
    local_kernel: Literal["independent", "matern52"] = "independent"
    local_kernel_signal_variance: float = Field(default=0.0, ge=0)
    local_kernel_noise_variance: float = Field(default=0.0, ge=0)
    local_kernel_length_scale: float = Field(default=1.0, gt=0)
    local_kernel_fit_system_count: int = Field(default=0, ge=0)
    local_kernel_nll_per_row: float | None = None
    local_kernel_optimizer_success: bool | None = None
    local_kernel_optimizer_status: int | None = None
    local_kernel_optimizer_message: str | None = None
    local_kernel_optimizer_gradient_norm: float | None = Field(default=None, ge=0)
    local_kernel_optimizer_bounds_active: tuple[str, ...] = ()
    kernel_feature_mean: tuple[float, ...] = ()
    kernel_feature_scale: tuple[float, ...] = ()
    kernel_feature_encoder: str | None = None
    kernel_feature_encoder_checksum: str | None = None

    @model_validator(mode="after")
    def _dimensions(self) -> FrozenProtocolRidgeTransport:
        feature_count = len(self.feature_mean)
        if feature_count == 0 or len(self.feature_scale) != feature_count:
            raise ValueError("transport feature normalization is inconsistent")
        if any(value <= 0 or not math.isfinite(value) for value in self.feature_scale):
            raise ValueError("transport feature scales must be finite and positive")
        if len(self.coefficients) != feature_count + 1:
            raise ValueError("transport coefficient dimension is inconsistent")
        if len(self.precision) != len(self.coefficients) or any(
            len(row) != len(self.coefficients) for row in self.precision
        ):
            raise ValueError("transport precision dimension is inconsistent")
        if not self.fit_system_ids or len(set(self.fit_system_ids)) != len(self.fit_system_ids):
            raise ValueError("transport fit systems must be unique and nonempty")
        if not self.fit_element_ids or len(set(self.fit_element_ids)) != len(self.fit_element_ids):
            raise ValueError("transport fit elements must be unique and nonempty")
        arrays = (
            np.asarray(self.feature_mean),
            np.asarray(self.feature_scale),
            np.asarray(self.coefficients),
            np.asarray(self.precision),
        )
        if any(not np.isfinite(values).all() for values in arrays):
            raise ValueError("transport parameters must be finite")
        if self.local_kernel == "independent":
            if (
                self.local_kernel_signal_variance != 0
                or self.kernel_feature_mean
                or self.kernel_feature_scale
                or self.kernel_feature_encoder is not None
                or self.kernel_feature_encoder_checksum is not None
            ):
                raise ValueError("independent transport cannot carry local-kernel state")
        elif (
            self.local_kernel_signal_variance <= 0
            or self.local_kernel_noise_variance <= 0
            or self.local_kernel_fit_system_count < 2
            or self.local_kernel_nll_per_row is None
            or not math.isfinite(self.local_kernel_nll_per_row)
            or not self.kernel_feature_mean
            or len(self.kernel_feature_scale) != len(self.kernel_feature_mean)
            or any(
                value <= 0 or not math.isfinite(value)
                for value in self.kernel_feature_scale
            )
            or not self.kernel_feature_encoder
            or not self.kernel_feature_encoder_checksum
        ):
            raise ValueError(
                "Matérn transport requires a frozen observable kernel representation"
            )
        return self

    @property
    def identity_checksum(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class ProtocolTargetEnergyPosterior(BaseModel):
    """Joint Gaussian working posterior over target-protocol pool energies."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]
    system_offset_mean: float
    system_offset_variance: float = Field(ge=0)
    history_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _dimensions(self) -> ProtocolTargetEnergyPosterior:
        size = len(self.mean)
        if (
            size == 0
            or len(self.covariance) != size
            or any(len(row) != size for row in self.covariance)
        ):
            raise ValueError("target posterior dimensions are inconsistent")
        mean = np.asarray(self.mean, dtype=float)
        covariance = np.asarray(self.covariance, dtype=float)
        if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
            raise ValueError("target posterior must be finite")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("target posterior covariance must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -1e-8:
            raise ValueError("target posterior covariance must be positive semidefinite")
        return self


class ProtocolHullKnowledgeGradientResult(BaseModel):
    """Two-step final-hull discovery values under a working posterior."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    final_stability_probabilities: tuple[float, ...]
    expected_second_step_values: tuple[float, ...]
    posterior_risk: float = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)
    fantasy_count: int = Field(gt=0)
    horizon: int = Field(ge=1, le=2)


class DeltaHullActiveSearchResult(BaseModel):
    """Myopic Bayes action values for target-protocol hull discovery."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    final_stability_probabilities: tuple[float, ...]
    posterior_sample_count: int = Field(gt=0)


class ProtocolHullRiskReductionResult(BaseModel):
    """Myopic Bayes-risk reduction for the target-protocol hull function."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    risk_reductions: tuple[float, ...]
    expected_posterior_risks: tuple[float, ...]
    current_hull_risk: float = Field(ge=0)
    evaluation_composition_count: int = Field(gt=0)
    posterior_sample_count: int = Field(gt=0)
    fantasy_count: int = Field(gt=0)


class ProtocolHullPosteriorSummary(BaseModel):
    """Posterior moments of the random target hull on the fixed pool grid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_compositions: tuple[dict[str, float], ...]
    mean_hull_energies: tuple[float, ...]
    hull_variances: tuple[float, ...]
    bayes_risk: float = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)


class SourceRolloutDeltaHullResult(BaseModel):
    """Full-budget rollout values using source margin as continuation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    block_scores: tuple[tuple[float, ...], ...]
    final_stability_probabilities: tuple[float, ...]
    paired_advantages_over_source: tuple[float, ...]
    paired_advantage_lower_bounds: tuple[float, ...]
    source_action_index: int = Field(ge=0)
    selected_action_index: int = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)
    sobol_scramble_count: int = Field(gt=1)
    simultaneous_comparison_count: int = Field(gt=0)
    horizon: int = Field(gt=0)
    fallback_reason: str | None = None


class ConformalSourceRolloutCalibration(BaseModel):
    """Exact-system calibration for a single source-relative deviation.

    ``radius`` is an upper quantile of the system-level maximum rollout
    over-estimation.  It is a deployment threshold, not a posterior standard
    deviation or a guarantee for arbitrary adaptive policies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    alpha: float = Field(gt=0, lt=1)
    system_ids: tuple[str, ...]
    system_scores: tuple[float, ...]
    order_statistic_one_based: int = Field(gt=0)
    radius: float = Field(ge=0)

    @model_validator(mode="after")
    def _calibration_dimensions(self) -> ConformalSourceRolloutCalibration:
        if not self.system_ids or len(set(self.system_ids)) != len(self.system_ids):
            raise ValueError("conformal rollout systems must be unique and nonempty")
        if len(self.system_scores) != len(self.system_ids):
            raise ValueError("conformal rollout scores and systems disagree")
        if any(not math.isfinite(value) or value < 0 for value in self.system_scores):
            raise ValueError("conformal rollout scores must be finite and non-negative")
        if not 1 <= self.order_statistic_one_based <= len(self.system_ids):
            raise ValueError("conformal rollout order statistic is out of range")
        if not math.isfinite(self.radius):
            raise ValueError("conformal rollout radius must be finite")
        return self

    @property
    def identity_checksum(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()


class ConformalSourceRolloutResult(BaseModel):
    """One-deviation source-rollout decision and its numerical diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scores: tuple[float, ...]
    paired_advantages_over_source: tuple[float, ...]
    rqmc_radii: tuple[float, ...]
    conformal_adjusted_advantages: tuple[float, ...]
    source_action_index: int = Field(ge=0)
    selected_action_index: int = Field(ge=0)
    deviation_used_before: bool
    deviation_selected: bool
    fallback_reason: str | None = None
    conformal_radius: float = Field(ge=0)
    posterior_sample_count: int = Field(gt=0)
    sobol_scramble_count: int = Field(gt=1)
    horizon: int = Field(gt=0)


@dataclass(frozen=True, slots=True)
class FixedCompositionHullTemplate:
    """Cached composition geometry for an action-equivalent lower-hull solver.

    The template contains no energies.  It can therefore be built before any
    oracle reveal and reused for every posterior sample and round whose union
    of candidate/reference compositions is unchanged.  Stability is computed
    from the same reduced-composition grouping, elemental-reference handling,
    formation-energy filter and Qhull convention as ``pymatgen.PhaseDiagram``.
    """

    elements: tuple[str, ...]
    normalized_composition_matrix: tuple[tuple[float, ...], ...]
    atom_counts: tuple[float, ...]
    candidate_indices: tuple[int, ...]
    reference_indices: tuple[int, ...]
    duplicate_composition_groups: tuple[tuple[int, ...], ...]
    element_reference_indices: tuple[tuple[str, int], ...]
    entry_names: tuple[str, ...]
    numerical_tolerance: float = 1e-11

    @classmethod
    def from_compositions(
        cls,
        *,
        query_compositions: Sequence[dict[str, float]],
        reference_compositions: Sequence[dict[str, float]],
        numerical_tolerance: float = 1e-11,
    ) -> FixedCompositionHullTemplate:
        from pymatgen.core import Composition

        if not query_compositions or not reference_compositions:
            raise ValueError("fixed-composition hull requires nonempty query and reference sets")
        parsed = [Composition(value) for value in reference_compositions] + [
            Composition(value) for value in query_compositions
        ]
        elements = tuple(sorted({str(element) for composition in parsed for element in composition.elements}))
        if len(elements) < 1:
            raise ValueError("fixed-composition hull requires at least one element")
        matrix = tuple(
            tuple(float(composition.get_atomic_fraction(element)) for element in elements)
            for composition in parsed
        )
        atom_counts = tuple(float(composition.num_atoms) for composition in parsed)
        entry_names = tuple(
            [f"reference:{index}" for index in range(len(reference_compositions))]
            + [f"query:{index}" for index in range(len(query_compositions))]
        )

        def composition_key(composition: Composition) -> tuple[tuple[str, float], ...]:
            reduced = composition.reduced_composition
            return tuple(
                (str(element), round(float(amount), 12))
                for element, amount in sorted(reduced.as_dict().items())
            )

        grouped: dict[tuple[tuple[str, float], ...], list[int]] = {}
        for index, composition in enumerate(parsed):
            grouped.setdefault(composition_key(composition), []).append(index)
        duplicate_groups = tuple(
            tuple(indices) for indices in sorted(grouped.values(), key=lambda group: group[0])
        )
        elemental: list[tuple[str, int]] = []
        for element in elements:
            candidates = [
                index
                for index, composition in enumerate(parsed)
                if composition.is_element and str(composition.elements[0]) == element
            ]
            if not candidates:
                raise ValueError(f"fixed-composition hull is missing elemental reference {element}")
            elemental.append((element, candidates[0]))
        return cls(
            elements=elements,
            normalized_composition_matrix=matrix,
            atom_counts=atom_counts,
            candidate_indices=tuple(range(len(reference_compositions), len(parsed))),
            reference_indices=tuple(range(len(reference_compositions))),
            duplicate_composition_groups=duplicate_groups,
            element_reference_indices=tuple(elemental),
            entry_names=entry_names,
            numerical_tolerance=float(numerical_tolerance),
        )

    @property
    def entry_count(self) -> int:
        return len(self.normalized_composition_matrix)

    def stable_candidate_mask(
        self,
        *,
        query_energies: np.ndarray,
        reference_energies: np.ndarray,
    ) -> np.ndarray:
        """Return the candidate stable mask for one complete energy vector."""

        values = np.concatenate(
            (
                np.asarray(reference_energies, dtype=np.float64).reshape(-1),
                np.asarray(query_energies, dtype=np.float64).reshape(-1),
            )
        )
        if len(values) != self.entry_count or not np.isfinite(values).all():
            raise ValueError("fixed-composition hull energy dimensions are inconsistent")
        selected: list[int] = []
        selected_by_group: dict[int, int] = {}
        for group in self.duplicate_composition_groups:
            chosen = min(group, key=lambda index: (values[index], self.entry_names[index]))
            selected.append(chosen)
            for index in group:
                selected_by_group[index] = chosen
        elemental_indices = tuple(
            selected_by_group[index] for _, index in self.element_reference_indices
        )
        element_energies = {
            element: float(values[selected_by_group[index]])
            for element, index in self.element_reference_indices
        }
        matrix = np.asarray(self.normalized_composition_matrix, dtype=np.float64)
        formation = np.asarray(
            [
                values[index]
                - sum(matrix[index, element_index] * element_energies[element] for element_index, element in enumerate(self.elements))
                for index in selected
            ],
            dtype=np.float64,
        )
        qhull_indices = [
            index for index, formation_energy in zip(selected, formation, strict=True)
            if formation_energy < -self.numerical_tolerance
        ]
        qhull_indices.extend(elemental_indices)
        # The PhaseDiagram implementation keeps this order and permits an
        # elemental entry to occur twice only if it was already negative, which
        # cannot happen for its own reference formation energy.
        qhull_indices = list(dict.fromkeys(qhull_indices))
        dimension = len(self.elements)
        qhull_data = np.column_stack((matrix[qhull_indices, 1:], values[qhull_indices]))
        extra_point = np.zeros(dimension, dtype=np.float64) + 1.0 / dimension
        extra_point[-1] = float(np.max(qhull_data[:, -1]) + 1.0)
        qhull_data = np.concatenate((qhull_data, extra_point[None, :]), axis=0)
        if dimension == 1:
            facets: list[np.ndarray] = [np.asarray([int(np.argmin(qhull_data[:, 0]))])]
        else:
            try:
                facets = list(ConvexHull(qhull_data, qhull_options="Qt i").simplices)
            except QhullError as exc:
                raise ValueError("fixed-composition hull Qhull failed") from exc
            final_facets: list[np.ndarray] = []
            for facet in facets:
                if int(np.max(facet)) == len(qhull_data) - 1:
                    continue
                facet_data = np.array(qhull_data[facet], copy=True)
                facet_data[:, -1] = 1.0
                if abs(float(np.linalg.det(facet_data))) > 1e-14:
                    final_facets.append(np.asarray(facet))
            facets = final_facets
        stable_qhull_indices = {
            int(index) for facet in facets for index in np.asarray(facet).reshape(-1)
        }
        stable_combined_indices = {
            qhull_indices[index] for index in stable_qhull_indices if index < len(qhull_indices)
        }
        return np.asarray(
            [index in stable_combined_indices for index in self.candidate_indices],
            dtype=bool,
        )


def fixed_composition_hull_membership(
    template: FixedCompositionHullTemplate,
    *,
    query_energies: np.ndarray,
    reference_energies: np.ndarray,
) -> np.ndarray:
    """Evaluate one or more sampled energy vectors with the cached backend."""

    samples = np.asarray(query_energies, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[1] != len(template.candidate_indices):
        raise ValueError("fixed-composition hull query samples have inconsistent dimensions")
    return np.asarray(
        [
            template.stable_candidate_mask(
                query_energies=sample,
                reference_energies=reference_energies,
            )
            for sample in samples
        ],
        dtype=bool,
    )


def _raw_features(features: np.ndarray, source_energies: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    source = np.asarray(source_energies, dtype=np.float64).reshape(-1)
    if values.ndim != 2 or not len(values) or len(source) != len(values):
        raise ValueError("protocol transport features and source energies disagree")
    if not np.isfinite(values).all() or not np.isfinite(source).all():
        raise ValueError("protocol transport inputs must be finite")
    return np.column_stack((values, source))


def fit_protocol_ridge_transport(
    *,
    features: np.ndarray,
    source_energies: np.ndarray,
    target_energies: np.ndarray,
    system_ids: Sequence[str],
    ridge_penalty: float = 1.0,
) -> FrozenProtocolRidgeTransport:
    """Fit a system-balanced source-to-target discrepancy model.

    Each exact system has equal total regression weight.  Variance components
    are then estimated from within-system residuals and held-out-system random
    intercepts instead of being tuned on a query trajectory.
    """

    raw = _raw_features(features, source_energies)
    target = np.asarray(target_energies, dtype=np.float64).reshape(-1)
    systems = tuple(str(value) for value in system_ids)
    if len(target) != len(raw) or len(systems) != len(raw):
        raise ValueError("protocol transport targets and systems disagree")
    if not np.isfinite(target).all():
        raise ValueError("protocol transport targets must be finite")
    if not math.isfinite(ridge_penalty) or ridge_penalty <= 0:
        raise ValueError("protocol transport ridge penalty must be positive")
    unique_systems = tuple(sorted(set(systems)))
    if len(unique_systems) < 2:
        raise ValueError("protocol transport requires at least two fit systems")

    feature_mean = raw.mean(axis=0)
    feature_scale = raw.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    standardized = (raw - feature_mean) / feature_scale
    design = np.column_stack((np.ones(len(raw)), standardized))
    counts = Counter(systems)
    weights = np.asarray(
        [len(raw) / (len(unique_systems) * counts[system]) for system in systems],
        dtype=np.float64,
    )
    penalty = ridge_penalty * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 1e-8
    precision = design.T @ (weights[:, None] * design) + penalty
    discrepancy = target - np.asarray(source_energies, dtype=np.float64)
    coefficients = np.linalg.solve(
        precision,
        design.T @ (weights * discrepancy),
    )
    residuals = discrepancy - design @ coefficients
    system_means = {
        system: float(np.mean(residuals[np.asarray(systems) == system]))
        for system in unique_systems
    }
    centered = np.asarray(
        [residual - system_means[system] for residual, system in zip(residuals, systems)],
        dtype=np.float64,
    )
    within_variance = max(float(np.mean(centered**2)), 1e-8)
    means = np.asarray([system_means[system] for system in unique_systems])
    sampling_variance = float(
        np.mean([within_variance / counts[system] for system in unique_systems])
    )
    between_variance = max(float(np.var(means, ddof=1)) - sampling_variance, 1e-8)
    return FrozenProtocolRidgeTransport(
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        coefficients=tuple(float(value) for value in coefficients),
        precision=tuple(tuple(float(value) for value in row) for row in precision),
        within_system_variance=within_variance,
        between_system_variance=between_variance,
        ridge_penalty=ridge_penalty,
        fit_system_ids=unique_systems,
        fit_element_ids=tuple(
            sorted({element for system in unique_systems for element in system.split("-")})
        ),
        fit_row_count=len(raw),
    )


def _matern52_covariance(
    left: np.ndarray,
    right: np.ndarray,
    *,
    length_scale: float,
) -> np.ndarray:
    """Unit-variance Matérn-5/2 covariance on standardized observables."""

    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if (
        left_values.ndim != 2
        or right_values.ndim != 2
        or left_values.shape[1] != right_values.shape[1]
        or not np.isfinite(left_values).all()
        or not np.isfinite(right_values).all()
        or not math.isfinite(length_scale)
        or length_scale <= 0
    ):
        raise ValueError("Matérn covariance inputs are inconsistent")
    distances = np.linalg.norm(
        left_values[:, None, :] - right_values[None, :, :],
        axis=2,
    )
    scaled = math.sqrt(5.0) * distances / length_scale
    return (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)


def fit_protocol_kernel_transport(
    *,
    features: np.ndarray,
    kernel_features: np.ndarray,
    source_energies: np.ndarray,
    target_energies: np.ndarray,
    system_ids: Sequence[str],
    kernel_feature_encoder: str,
    kernel_feature_encoder_checksum: str,
    ridge_penalty: float = 1.0,
) -> FrozenProtocolRidgeTransport:
    """Fit a hierarchical autoregressive discrepancy posterior.

    The cross-system mean remains the system-balanced ridge transport over
    composition and source-protocol observables.  A separate frozen structure
    representation defines a shared Matérn-5/2 covariance for the residual
    discrepancy inside each exact chemical system.  Kernel scales are
    empirical-Bayes estimates from the disjoint fit systems; no query-system
    outcome or acquisition trace participates in the fit.
    """

    model = fit_protocol_ridge_transport(
        features=features,
        source_energies=source_energies,
        target_energies=target_energies,
        system_ids=system_ids,
        ridge_penalty=ridge_penalty,
    )
    raw = _raw_features(features, source_energies)
    kernel_raw = np.asarray(kernel_features, dtype=np.float64)
    if (
        kernel_raw.ndim != 2
        or len(kernel_raw) != len(raw)
        or kernel_raw.shape[1] == 0
        or not np.isfinite(kernel_raw).all()
        or not kernel_feature_encoder.strip()
        or not kernel_feature_encoder_checksum.strip()
    ):
        raise ValueError("local protocol kernel features or provenance are inconsistent")
    target = np.asarray(target_energies, dtype=np.float64).reshape(-1)
    systems = np.asarray([str(value) for value in system_ids], dtype=object)
    standardized = (raw - np.asarray(model.feature_mean, dtype=np.float64)) / np.asarray(
        model.feature_scale, dtype=np.float64
    )
    kernel_feature_mean = kernel_raw.mean(axis=0)
    kernel_feature_scale = kernel_raw.std(axis=0)
    kernel_feature_scale[kernel_feature_scale < 1e-8] = 1.0
    kernel_standardized = (
        kernel_raw - kernel_feature_mean
    ) / kernel_feature_scale
    design = np.column_stack((np.ones(len(raw)), standardized))
    discrepancy = target - np.asarray(source_energies, dtype=np.float64)
    residuals = discrepancy - design @ np.asarray(model.coefficients)

    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    positive_distances: list[np.ndarray] = []
    for system in model.fit_system_ids:
        mask = systems == system
        system_x = kernel_standardized[mask]
        system_y = residuals[mask]
        if len(system_y) < 2:
            continue
        system_y = system_y - float(np.mean(system_y))
        blocks.append((system_x, system_y))
        distances = np.linalg.norm(system_x[:, None, :] - system_x[None, :, :], axis=2)
        values = distances[np.triu_indices(len(system_x), k=1)]
        positive = values[values > 1e-10]
        if len(positive):
            positive_distances.append(positive)
    if len(blocks) < 2 or not positive_distances:
        raise ValueError("local protocol kernel requires two non-degenerate fit systems")

    distances = np.concatenate(positive_distances)
    lower_length = max(float(np.quantile(distances, 0.05)), 1e-3)
    upper_length = max(float(np.quantile(distances, 0.95)), 2.0 * lower_length)
    initial_length = float(np.clip(np.median(distances), lower_length, upper_length))
    total_variance = max(model.within_system_variance, 1e-8)
    lower_variance = max(total_variance * 1e-4, 1e-10)
    upper_variance = max(total_variance * 10.0, lower_variance * 10.0)

    def objective(log_parameters: np.ndarray) -> float:
        length_scale, signal_variance, noise_variance = np.exp(log_parameters)
        values = []
        for system_x, system_y in blocks:
            covariance = signal_variance * _matern52_covariance(
                system_x,
                system_x,
                length_scale=length_scale,
            )
            covariance += noise_variance * np.eye(len(system_y))
            covariance += 1e-10 * np.eye(len(system_y))
            try:
                factor = np.linalg.cholesky(covariance)
                solved = np.linalg.solve(factor, system_y)
            except np.linalg.LinAlgError:
                return float("inf")
            nll = 0.5 * float(solved @ solved)
            nll += float(np.log(np.diag(factor)).sum())
            nll += 0.5 * len(system_y) * math.log(2.0 * math.pi)
            values.append(nll / len(system_y))
        return float(np.mean(values))

    initial = np.log(np.asarray([initial_length, total_variance / 2.0, total_variance / 2.0]))
    bounds = (
        (math.log(lower_length), math.log(upper_length)),
        (math.log(lower_variance), math.log(upper_variance)),
        (math.log(lower_variance), math.log(upper_variance)),
    )
    optimized = minimize(objective, initial, method="L-BFGS-B", bounds=bounds)
    if not np.isfinite(optimized.fun):
        raise RuntimeError("local protocol kernel marginal likelihood is non-finite")
    if not bool(optimized.success):
        raise RuntimeError(
            "local protocol kernel marginal likelihood optimizer failed: "
            f"status={optimized.status} message={optimized.message}"
        )
    length_scale, signal_variance, noise_variance = np.exp(optimized.x)
    gradient = np.asarray(getattr(optimized, "jac", ()), dtype=float)
    gradient_norm = (
        float(np.linalg.norm(gradient))
        if gradient.size and np.isfinite(gradient).all()
        else None
    )
    bound_names = ("length_scale", "signal_variance", "noise_variance")
    bounds_active = tuple(
        name
        for name, value, (lower, upper) in zip(
            bound_names, optimized.x, bounds, strict=True
        )
        if abs(float(value) - lower) <= 1e-8 or abs(float(value) - upper) <= 1e-8
    )
    payload = model.model_dump()
    payload.update(
        {
            "local_kernel": "matern52",
            "local_kernel_signal_variance": float(signal_variance),
            "local_kernel_noise_variance": float(noise_variance),
            "local_kernel_length_scale": float(length_scale),
            "local_kernel_fit_system_count": len(blocks),
            "local_kernel_nll_per_row": float(optimized.fun),
            "local_kernel_optimizer_success": bool(optimized.success),
            "local_kernel_optimizer_status": int(optimized.status),
            "local_kernel_optimizer_message": str(optimized.message),
            "local_kernel_optimizer_gradient_norm": gradient_norm,
            "local_kernel_optimizer_bounds_active": bounds_active,
            "kernel_feature_mean": tuple(float(value) for value in kernel_feature_mean),
            "kernel_feature_scale": tuple(float(value) for value in kernel_feature_scale),
            "kernel_feature_encoder": kernel_feature_encoder,
            "kernel_feature_encoder_checksum": kernel_feature_encoder_checksum,
        }
    )
    return FrozenProtocolRidgeTransport.model_validate(payload)


def protocol_target_energy_posterior(
    model: FrozenProtocolRidgeTransport,
    *,
    query_features: np.ndarray,
    query_source_energies: np.ndarray,
    history_features: np.ndarray,
    history_source_energies: np.ndarray,
    history_target_energies: np.ndarray,
    query_kernel_features: np.ndarray | None = None,
    history_kernel_features: np.ndarray | None = None,
) -> ProtocolTargetEnergyPosterior:
    """Condition the frozen transport on every revealed current-system outcome."""

    query_raw = _raw_features(query_features, query_source_energies)
    history_x = np.asarray(history_features, dtype=np.float64)
    history_source = np.asarray(history_source_energies, dtype=np.float64).reshape(-1)
    history_target = np.asarray(history_target_energies, dtype=np.float64).reshape(-1)
    if history_x.ndim != 2 or history_x.shape[1] != query_raw.shape[1] - 1:
        raise ValueError("protocol posterior history feature dimension disagrees")
    if len(history_x) != len(history_source) or len(history_x) != len(history_target):
        raise ValueError("protocol posterior history arrays disagree")
    if any(not np.isfinite(values).all() for values in (history_x, history_source, history_target)):
        raise ValueError("protocol posterior history must be finite")

    feature_mean = np.asarray(model.feature_mean)
    feature_scale = np.asarray(model.feature_scale)
    coefficients = np.asarray(model.coefficients)
    precision = np.asarray(model.precision)

    def design(features: np.ndarray, source: np.ndarray) -> np.ndarray:
        raw = np.column_stack((features, source))
        return np.column_stack((np.ones(len(raw)), (raw - feature_mean) / feature_scale))

    query_design = design(
        np.asarray(query_features, dtype=np.float64),
        np.asarray(query_source_energies, dtype=np.float64),
    )
    within = model.within_system_variance
    between = model.between_system_variance
    if model.local_kernel == "matern52":
        query_kernel = np.asarray(query_kernel_features, dtype=np.float64)
        history_kernel = np.asarray(history_kernel_features, dtype=np.float64)
        kernel_mean = np.asarray(model.kernel_feature_mean, dtype=np.float64)
        kernel_scale = np.asarray(model.kernel_feature_scale, dtype=np.float64)
        if (
            query_kernel.ndim != 2
            or query_kernel.shape != (len(query_raw), len(kernel_mean))
            or history_kernel.ndim != 2
            or history_kernel.shape != (len(history_x), len(kernel_mean))
            or not np.isfinite(query_kernel).all()
            or not np.isfinite(history_kernel).all()
        ):
            raise ValueError("frozen local-kernel embeddings are missing or inconsistent")
        coefficient_covariance = np.linalg.inv(precision) * (within + between)
        query_source = np.asarray(query_source_energies, dtype=np.float64)
        predicted_mean = query_source + query_design @ coefficients
        query_standardized = (query_kernel - kernel_mean) / kernel_scale
        covariance = query_design @ coefficient_covariance @ query_design.T
        covariance += between * np.ones((len(query_raw), len(query_raw)))
        covariance += model.local_kernel_signal_variance * _matern52_covariance(
            query_standardized,
            query_standardized,
            length_scale=model.local_kernel_length_scale,
        )
        covariance += model.local_kernel_noise_variance * np.eye(len(query_raw))
        system_offset_mean = 0.0
        system_offset_variance = between
        if len(history_x):
            history_design = design(history_x, history_source)
            history_standardized = (history_kernel - kernel_mean) / kernel_scale
            history_mean = history_source + history_design @ coefficients
            history_covariance = history_design @ coefficient_covariance @ history_design.T
            history_covariance += between * np.ones((len(history_x), len(history_x)))
            history_covariance += model.local_kernel_signal_variance * _matern52_covariance(
                history_standardized,
                history_standardized,
                length_scale=model.local_kernel_length_scale,
            )
            history_covariance += model.local_kernel_noise_variance * np.eye(len(history_x))
            history_covariance += 1e-10 * np.eye(len(history_x))
            cross_covariance = query_design @ coefficient_covariance @ history_design.T
            cross_covariance += between * np.ones((len(query_raw), len(history_x)))
            cross_covariance += model.local_kernel_signal_variance * _matern52_covariance(
                query_standardized,
                history_standardized,
                length_scale=model.local_kernel_length_scale,
            )
            factor = np.linalg.cholesky(history_covariance)
            innovation = history_target - history_mean
            solved_innovation = np.linalg.solve(factor.T, np.linalg.solve(factor, innovation))
            predicted_mean = predicted_mean + cross_covariance @ solved_innovation
            solved_cross = np.linalg.solve(factor, cross_covariance.T)
            covariance = covariance - solved_cross.T @ solved_cross
            ones = np.ones(len(history_x), dtype=np.float64)
            solved_ones = np.linalg.solve(factor.T, np.linalg.solve(factor, ones))
            system_offset_mean = float(between * ones @ solved_innovation)
            system_offset_variance = max(
                float(between - between**2 * ones @ solved_ones),
                0.0,
            )
        covariance = 0.5 * (covariance + covariance.T)
        covariance += 1e-12 * np.eye(len(query_raw))
        return ProtocolTargetEnergyPosterior(
            mean=tuple(float(value) for value in predicted_mean),
            covariance=tuple(tuple(float(value) for value in row) for row in covariance),
            system_offset_mean=system_offset_mean,
            system_offset_variance=system_offset_variance,
            history_count=len(history_x),
        )

    parameter_count = len(coefficients) + 1
    prior_precision = np.zeros((parameter_count, parameter_count), dtype=np.float64)
    prior_precision[:-1, :-1] = precision / (within + between)
    prior_precision[-1, -1] = 1.0 / between
    prior_mean = np.concatenate((coefficients, [0.0]))
    posterior_precision = prior_precision.copy()
    posterior_natural = prior_precision @ prior_mean
    if len(history_x):
        history_design = design(history_x, history_source)
        history_joint_design = np.column_stack((history_design, np.ones(len(history_design))))
        history_discrepancy = history_target - history_source
        posterior_precision += history_joint_design.T @ history_joint_design / within
        posterior_natural += history_joint_design.T @ history_discrepancy / within
    parameter_covariance = np.linalg.inv(posterior_precision)
    parameter_mean = parameter_covariance @ posterior_natural
    query_joint_design = np.column_stack((query_design, np.ones(len(query_design))))
    predicted_mean = np.asarray(query_source_energies, dtype=np.float64) + (
        query_joint_design @ parameter_mean
    )
    covariance = query_joint_design @ parameter_covariance @ query_joint_design.T
    covariance += within * np.eye(len(query_raw))
    covariance = 0.5 * (covariance + covariance.T)
    covariance += 1e-12 * np.eye(len(query_raw))
    return ProtocolTargetEnergyPosterior(
        mean=tuple(float(value) for value in predicted_mean),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        system_offset_mean=float(parameter_mean[-1]),
        system_offset_variance=float(parameter_covariance[-1, -1]),
        history_count=len(history_x),
    )


def _sample_gaussian(
    mean: np.ndarray,
    covariance: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> np.ndarray:
    """Draw a deterministic nested scrambled-Sobol Gaussian design.

    Hull membership is a discontinuous functional, so ordinary pseudo-random
    Monte Carlo can change an action merely because the requested sample count
    changed.  A scrambled Sobol design preserves randomized-QMC error control
    while making every power-of-two run a prefix of the next run for the same
    seed.  That gives the policy an observable numerical-convergence check
    without changing its posterior or acquisition objective.
    """

    if sample_count < 1:
        raise ValueError("Gaussian posterior sampling requires a positive count")
    dimension = len(mean)
    if dimension < 1:
        raise ValueError("Gaussian posterior sampling requires a nonempty mean")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    exponent = math.ceil(math.log2(sample_count))
    unit = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)
    unit = unit[:sample_count]
    epsilon = np.finfo(np.float64).eps
    normal = ndtri(np.clip(unit, epsilon, 1.0 - epsilon))
    return mean + normal @ factor.T


def _final_hull_membership(
    *,
    query_compositions: Sequence[dict[str, float]],
    sampled_query_energies: np.ndarray,
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    fixed_template: FixedCompositionHullTemplate | None = None,
) -> np.ndarray:
    if fixed_template is not None:
        expected = FixedCompositionHullTemplate.from_compositions(
            query_compositions=query_compositions,
            reference_compositions=reference_compositions,
            numerical_tolerance=fixed_template.numerical_tolerance,
        )
        if expected != fixed_template:
            raise ValueError("fixed-composition hull template does not match compositions")
        return fixed_composition_hull_membership(
            fixed_template,
            query_energies=sampled_query_energies,
            reference_energies=reference_energies,
        )
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    samples = np.asarray(sampled_query_energies, dtype=np.float64)
    reference_values = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or samples.shape[1] != len(query_compositions):
        raise ValueError("final-hull energy samples and candidates disagree")
    if len(reference_values) != len(reference_compositions):
        raise ValueError("final-hull reference arrays disagree")
    query_parsed = [Composition(value) for value in query_compositions]
    reference_parsed = [Composition(value) for value in reference_compositions]
    labels = np.zeros(samples.shape, dtype=bool)
    for sample_index, energies in enumerate(samples):
        entries = [
            ComputedEntry(
                composition,
                energy * composition.num_atoms,
                entry_id=f"reference:{index}",
            )
            for index, (composition, energy) in enumerate(
                zip(reference_parsed, reference_values, strict=True)
            )
        ]
        entries.extend(
            ComputedEntry(
                composition,
                energy * composition.num_atoms,
                entry_id=f"query:{index}",
            )
            for index, (composition, energy) in enumerate(zip(query_parsed, energies, strict=True))
        )
        stable_ids = {str(entry.entry_id) for entry in PhaseDiagram(entries).stable_entries}
        labels[sample_index] = [
            f"query:{index}" in stable_ids for index in range(len(query_parsed))
        ]
    return labels


def _normalized_composition_key(composition: dict[str, float]) -> tuple[tuple[str, float], ...]:
    total = float(sum(composition.values()))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("hull composition must have positive finite mass")
    return tuple(
        (element, round(float(amount) / total, 12))
        for element, amount in sorted(composition.items())
        if float(amount) > 0
    )


def source_margin_action_indices(
    *,
    source_energies: np.ndarray,
    competing_hull_energies: np.ndarray,
    query_ids: Sequence[str],
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Select source-margin actions with immutable-ID tie breaking.

    This vectorized primitive is shared by the deployed source policy and by
    posterior rollout continuations. Rows of ``competing_hull_energies`` are
    independent simulated causal-hull states; columns follow ``query_ids``.
    """

    source = np.asarray(source_energies, dtype=np.float64).reshape(-1)
    hull = np.asarray(competing_hull_energies, dtype=np.float64)
    if hull.ndim == 1:
        hull = hull[None, :]
    if hull.ndim != 2 or hull.shape[1] != len(source) or len(query_ids) != len(source):
        raise ValueError("source-margin arrays disagree")
    if not len(source) or not np.isfinite(source).all() or not np.isfinite(hull).all():
        raise ValueError("source-margin inputs must be nonempty and finite")
    if len(set(query_ids)) != len(query_ids) or any(not str(value) for value in query_ids):
        raise ValueError("source-margin query IDs must be unique and nonempty")
    mask = np.ones(hull.shape, dtype=bool)
    if eligible is not None:
        provided = np.asarray(eligible, dtype=bool)
        if provided.ndim == 1:
            provided = np.broadcast_to(provided[None, :], hull.shape)
        if provided.shape != hull.shape:
            raise ValueError("source-margin eligibility mask disagrees")
        mask = provided
    if np.any(~np.any(mask, axis=1)):
        raise ValueError("source-margin state has no eligible action")
    margins = source[None, :] - hull
    margins = np.where(mask, margins, np.inf)
    identifier_order = np.argsort(np.asarray(query_ids, dtype=str), kind="stable")
    ordered_actions = np.argmin(margins[:, identifier_order], axis=1)
    return identifier_order[ordered_actions]


@dataclass(frozen=True, slots=True)
class _CausalHullEnvelope:
    """Cached exact convex decompositions for one active composition set."""

    simplex_active_positions: np.ndarray
    simplex_weights: np.ndarray
    feasible: np.ndarray
    active_count: int
    query_count: int

    @classmethod
    def build(
        cls,
        *,
        query_compositions: Sequence[dict[str, float]],
        reference_compositions: Sequence[dict[str, float]],
        selected_query_indices: Sequence[int],
        tolerance: float = 1e-10,
    ) -> _CausalHullEnvelope:
        if not query_compositions or not reference_compositions:
            raise ValueError("causal-hull envelope requires queries and references")
        selected = tuple(sorted({int(index) for index in selected_query_indices}))
        if any(index < 0 or index >= len(query_compositions) for index in selected):
            raise ValueError("causal-hull selected index is out of range")
        elements = tuple(
            sorted(
                {
                    element
                    for composition in (*reference_compositions, *query_compositions)
                    for element, amount in composition.items()
                    if float(amount) > 0
                }
            )
        )
        dimension = len(elements)
        if dimension == 0:
            raise ValueError("causal-hull envelope has no elements")

        def fractions(composition: dict[str, float]) -> np.ndarray:
            total = float(sum(composition.values()))
            if not math.isfinite(total) or total <= 0:
                raise ValueError("causal-hull composition must have positive mass")
            return np.asarray(
                [float(composition.get(element, 0.0)) / total for element in elements],
                dtype=np.float64,
            )

        query_matrix = np.asarray([fractions(value) for value in query_compositions])
        active_compositions = [*reference_compositions]
        active_compositions.extend(query_compositions[index] for index in selected)
        active_matrix = np.asarray([fractions(value) for value in active_compositions])
        if len(active_matrix) < dimension:
            raise ValueError("causal hull lacks enough active phase compositions")
        simplex_positions: list[tuple[int, ...]] = []
        simplex_weights: list[np.ndarray] = []
        feasible_masks: list[np.ndarray] = []
        for positions in combinations(range(len(active_matrix)), dimension):
            matrix = active_matrix[np.asarray(positions)].T
            if abs(float(np.linalg.det(matrix))) <= 1e-12:
                continue
            weights = np.linalg.solve(matrix, query_matrix.T)
            weights[np.abs(weights) <= tolerance] = 0.0
            feasible = np.all(weights >= -tolerance, axis=0) & np.isclose(
                np.sum(weights, axis=0),
                1.0,
                atol=10.0 * tolerance,
            )
            if not np.any(feasible):
                continue
            simplex_positions.append(positions)
            simplex_weights.append(weights)
            feasible_masks.append(feasible)
        if not simplex_positions:
            raise ValueError("causal-hull envelope has no feasible decomposition")
        feasible = np.asarray(feasible_masks, dtype=bool)
        if np.any(~np.any(feasible, axis=0)):
            raise ValueError("causal-hull references do not span every query composition")
        return cls(
            simplex_active_positions=np.asarray(simplex_positions, dtype=np.int64),
            simplex_weights=np.asarray(simplex_weights, dtype=np.float64),
            feasible=feasible,
            active_count=len(active_matrix),
            query_count=len(query_matrix),
        )

    def competing_hull_energies(self, active_energies: np.ndarray) -> np.ndarray:
        values = np.asarray(active_energies, dtype=np.float64)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.active_count:
            raise ValueError("causal-hull active energies disagree with geometry")
        if not np.isfinite(values).all():
            raise ValueError("causal-hull active energies must be finite")
        hull = np.full((len(values), self.query_count), np.inf, dtype=np.float64)
        for positions, weights, feasible in zip(
            self.simplex_active_positions,
            self.simplex_weights,
            self.feasible,
            strict=True,
        ):
            candidate_values = values[:, positions] @ weights
            candidate_values[:, ~feasible] = np.inf
            np.minimum(hull, candidate_values, out=hull)
        if not np.isfinite(hull).all():
            raise ValueError("causal-hull energy is undefined for a query composition")
        return hull


def _source_rollout_rewards(
    *,
    sampled_query_energies: np.ndarray,
    final_hull_membership: np.ndarray,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    horizon: int,
) -> np.ndarray:
    """Evaluate every first action under a source-margin continuation."""

    samples = np.asarray(sampled_query_energies, dtype=np.float64)
    labels = np.asarray(final_hull_membership, dtype=bool)
    source = np.asarray(query_source_energies, dtype=np.float64).reshape(-1)
    references = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or labels.shape != samples.shape:
        raise ValueError("source rollout samples and final-hull labels disagree")
    if (
        samples.shape[1] != len(query_compositions)
        or len(source) != samples.shape[1]
        or len(query_ids) != samples.shape[1]
        or len(references) != len(reference_compositions)
    ):
        raise ValueError("source rollout arrays disagree")
    if horizon < 1 or horizon > samples.shape[1]:
        raise ValueError("source rollout horizon is invalid")
    if not np.isfinite(samples).all() or not np.isfinite(source).all() or not np.isfinite(references).all():
        raise ValueError("source rollout energies must be finite")

    sample_count, query_count = samples.shape
    rewards = np.empty((sample_count, query_count), dtype=np.float64)
    geometry_cache: dict[tuple[int, ...], _CausalHullEnvelope] = {}

    def geometry(selected: tuple[int, ...]) -> _CausalHullEnvelope:
        cached = geometry_cache.get(selected)
        if cached is None:
            cached = _CausalHullEnvelope.build(
                query_compositions=query_compositions,
                reference_compositions=reference_compositions,
                selected_query_indices=selected,
            )
            geometry_cache[selected] = cached
        return cached

    for first_action in range(query_count):
        selected = np.zeros((sample_count, query_count), dtype=bool)
        selected[:, first_action] = True
        for _ in range(1, horizon):
            groups: dict[tuple[int, ...], list[int]] = {}
            for sample_index in range(sample_count):
                key = tuple(int(index) for index in np.flatnonzero(selected[sample_index]))
                groups.setdefault(key, []).append(sample_index)
            for key, row_indices in groups.items():
                rows = np.asarray(row_indices, dtype=np.int64)
                envelope = geometry(key)
                active_energies = np.column_stack(
                    (
                        np.broadcast_to(references, (len(rows), len(references))),
                        samples[np.ix_(rows, np.asarray(key, dtype=np.int64))],
                    )
                )
                hull = envelope.competing_hull_energies(active_energies)
                eligible = np.ones(query_count, dtype=bool)
                eligible[np.asarray(key, dtype=np.int64)] = False
                next_actions = source_margin_action_indices(
                    source_energies=source,
                    competing_hull_energies=hull,
                    query_ids=query_ids,
                    eligible=eligible,
                )
                selected[rows, next_actions] = True
        rewards[:, first_action] = np.sum(selected & labels, axis=1)
    return rewards


def _simultaneous_paired_lower_bounds(
    block_differences: np.ndarray,
    *,
    confidence: float,
    comparison_count: int,
) -> np.ndarray:
    """Return one-sided Bonferroni-t lower bounds for all paired advantages.

    Rows are independent randomized-QMC blocks and columns are candidate
    advantages against the same source action.  The correction is applied to
    the non-source candidate family, so a positive returned bound supports a
    simultaneous source-relative statement rather than a collection of
    marginal tests.
    """

    values = np.asarray(block_differences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("simultaneous bounds require a block-by-candidate matrix")
    if not np.isfinite(values).all():
        raise ValueError("simultaneous bound inputs must be finite")
    if not 0.5 < confidence < 1.0:
        raise ValueError("simultaneous bound confidence must lie in (0.5, 1)")
    if comparison_count < 1:
        raise ValueError("simultaneous bound comparison count must be positive")
    alpha = (1.0 - confidence) / float(comparison_count)
    critical_value = float(student_t.ppf(1.0 - alpha, values.shape[0] - 1))
    if not math.isfinite(critical_value):
        raise ValueError("simultaneous bound critical value is not finite")
    means = values.mean(axis=0)
    standard_errors = values.std(axis=0, ddof=1) / math.sqrt(values.shape[0])
    return means - critical_value * standard_errors


def source_rollout_system_score(
    estimated_advantages: np.ndarray,
    counterfactual_advantages: np.ndarray,
) -> float:
    """Return one exact-system conformal score for rollout over-estimation.

    Both arrays contain source-relative advantages indexed by round and legal
    first action.  Calibration must use exact-system counterfactual oracle
    traces that are disjoint from deployment systems.  Clipping at zero makes
    this an upper-error nonconformity score rather than a signed effect.
    """

    estimated = np.asarray(estimated_advantages, dtype=np.float64)
    counterfactual = np.asarray(counterfactual_advantages, dtype=np.float64)
    if estimated.shape != counterfactual.shape or estimated.ndim < 1:
        raise ValueError("rollout advantage arrays must have the same nonempty shape")
    if not np.isfinite(estimated).all() or not np.isfinite(counterfactual).all():
        raise ValueError("rollout advantage arrays must be finite")
    return float(max(0.0, np.max(estimated - counterfactual)))


def fit_conformal_source_rollout_calibration(
    system_scores: Sequence[float],
    *,
    system_ids: Sequence[str],
    alpha: float = 0.1,
) -> ConformalSourceRolloutCalibration:
    """Fit a finite-sample exact-system split-conformal rollout threshold."""

    scores = tuple(float(value) for value in system_scores)
    identifiers = tuple(str(value) for value in system_ids)
    if len(scores) != len(identifiers) or not identifiers:
        raise ValueError("conformal rollout systems and scores disagree")
    if any(not value for value in identifiers):
        raise ValueError("conformal rollout system IDs must be nonempty")
    if not 0 < alpha < 1:
        raise ValueError("conformal rollout alpha must be in (0, 1)")
    if any(not math.isfinite(value) or value < 0 for value in scores):
        raise ValueError("conformal rollout scores must be finite and non-negative")
    order = math.ceil((len(scores) + 1) * (1.0 - alpha))
    if order > len(scores):
        raise ValueError("too few exact systems for a finite conformal rollout threshold")
    radius = sorted(scores)[order - 1]
    return ConformalSourceRolloutCalibration(
        alpha=alpha,
        system_ids=identifiers,
        system_scores=scores,
        order_statistic_one_based=order,
        radius=radius,
    )


def _fixed_evaluation_compositions(
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
) -> tuple[dict[str, float], ...]:
    """Return a composition grid invariant to query-to-reference transitions."""

    by_key: dict[tuple[tuple[str, float], ...], dict[str, float]] = {}
    for composition in (*reference_compositions, *query_compositions):
        key = _normalized_composition_key(composition)
        by_key.setdefault(key, {element: fraction for element, fraction in key})
    return tuple(by_key[key] for key in sorted(by_key))


def _final_hull_values(
    *,
    query_compositions: Sequence[dict[str, float]],
    sampled_query_energies: np.ndarray,
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    evaluation_compositions: Sequence[dict[str, float]],
) -> np.ndarray:
    """Evaluate sampled final hull functions on a fixed composition grid."""

    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    samples = np.asarray(sampled_query_energies, dtype=np.float64)
    reference_values = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or samples.shape[1] != len(query_compositions):
        raise ValueError("final-hull energy samples and candidates disagree")
    if len(reference_values) != len(reference_compositions):
        raise ValueError("final-hull reference arrays disagree")
    query_parsed = [Composition(value) for value in query_compositions]
    reference_parsed = [Composition(value) for value in reference_compositions]
    evaluation_parsed = [Composition(value) for value in evaluation_compositions]
    values = np.empty((len(samples), len(evaluation_parsed)), dtype=np.float64)
    for sample_index, energies in enumerate(samples):
        entries = [
            ComputedEntry(
                composition,
                energy * composition.num_atoms,
                entry_id=f"reference:{index}",
            )
            for index, (composition, energy) in enumerate(
                zip(reference_parsed, reference_values, strict=True)
            )
        ]
        entries.extend(
            ComputedEntry(
                composition,
                energy * composition.num_atoms,
                entry_id=f"query:{index}",
            )
            for index, (composition, energy) in enumerate(zip(query_parsed, energies, strict=True))
        )
        diagram = PhaseDiagram(entries)
        values[sample_index] = [
            float(diagram.get_hull_energy_per_atom(composition))
            for composition in evaluation_parsed
        ]
    return values


def _mean_hull_squared_error_risk(hull_values: np.ndarray) -> float:
    values = np.asarray(hull_values, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not values.shape[1]:
        raise ValueError("hull risk requires sampled hull functions")
    if not np.isfinite(values).all():
        raise ValueError("sampled hull functions must be finite")
    return float(np.var(values, axis=0, ddof=0).mean())


def protocol_hull_posterior_summary(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    posterior_sample_count: int = 16,
    seed: int = 0,
) -> ProtocolHullPosteriorSummary:
    """Summarize the random final hull without exposing target outcomes."""

    if posterior_sample_count < 4:
        raise ValueError("protocol hull posterior summary needs at least four samples")
    evaluation_compositions = _fixed_evaluation_compositions(
        query_compositions,
        reference_compositions,
    )
    hull_values = _final_hull_values(
        query_compositions=query_compositions,
        sampled_query_energies=_sample_gaussian(
            np.asarray(posterior.mean, dtype=np.float64),
            np.asarray(posterior.covariance, dtype=np.float64),
            sample_count=posterior_sample_count,
            seed=seed,
        ),
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        evaluation_compositions=evaluation_compositions,
    )
    means = np.mean(hull_values, axis=0)
    variances = np.var(hull_values, axis=0, ddof=0)
    return ProtocolHullPosteriorSummary(
        evaluation_compositions=evaluation_compositions,
        mean_hull_energies=tuple(float(value) for value in means),
        hull_variances=tuple(float(value) for value in variances),
        bayes_risk=float(np.mean(variances)),
        posterior_sample_count=posterior_sample_count,
    )


def delta_hull_active_search(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    costs: np.ndarray,
    posterior_sample_count: int = 16,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
) -> DeltaHullActiveSearchResult:
    """Return the exact one-step active-search objective under the posterior.

    The reward is one iff the queried configuration belongs to the final
    target-protocol hull over the complete visible fixed pool.  With one legal
    query remaining and equal query costs, maximizing its posterior membership
    probability is Bayes optimal.  Unequal costs are rejected rather than
    silently turning this finite-budget objective into a ratio heuristic.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    if len(query_compositions) != len(mean) or len(item_costs) != len(mean):
        raise ValueError("delta-hull active-search inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("delta-hull query costs must be finite and positive")
    if not np.allclose(item_costs, item_costs[0], atol=1e-12):
        raise ValueError("delta-hull active search requires equal query costs")
    if posterior_sample_count < 4:
        raise ValueError("delta-hull active search needs at least four posterior samples")

    labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=_sample_gaussian(
            mean,
            covariance,
            sample_count=posterior_sample_count,
            seed=seed,
        ),
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        fixed_template=fixed_template,
    )
    probabilities = labels.mean(axis=0)
    return DeltaHullActiveSearchResult(
        scores=tuple(float(value) for value in probabilities),
        final_stability_probabilities=tuple(float(value) for value in probabilities),
        posterior_sample_count=posterior_sample_count,
    )


def source_rollout_delta_hull(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    posterior_sample_count: int = 1024,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
    sobol_scramble_count: int = 16,
    integration_confidence: float = 0.95,
) -> SourceRolloutDeltaHullResult:
    """Improve source margin by a full-remaining-budget posterior rollout.

    Every candidate first action is followed by the deployed source-margin
    policy inside each complete target-energy sample. The sampled target
    energy of every simulated query is added to the composition-dependent
    causal hull before choosing the next action. A Bonferroni-simultaneous
    paired scrambled-Sobol lower bound prevents numerically unresolved gains
    from changing the strong source action; it is only an integration
    safeguard, not a calibration or real-world safety guarantee.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    source = np.asarray(query_source_energies, dtype=np.float64).reshape(-1)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    current_hull = np.asarray(current_competing_hull_energies, dtype=np.float64).reshape(-1)
    size = len(mean)
    if (
        len(query_compositions) != size
        or len(query_ids) != size
        or len(source) != size
        or len(item_costs) != size
        or len(current_hull) != size
    ):
        raise ValueError("source-rollout Delta-Hull inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("source-rollout query costs must be finite and positive")
    if not np.allclose(item_costs, item_costs[0], atol=1e-12):
        raise ValueError("source-rollout Delta-Hull requires equal query costs")
    if not math.isfinite(remaining_budget) or remaining_budget < item_costs[0]:
        raise ValueError("remaining protocol budget cannot pay for a rollout query")
    if sobol_scramble_count < 2 or posterior_sample_count % sobol_scramble_count:
        raise ValueError("posterior samples must divide into independent Sobol scrambles")
    block_size = posterior_sample_count // sobol_scramble_count
    if block_size < 2 or block_size & (block_size - 1):
        raise ValueError("each source-rollout Sobol block must have power-of-two size")
    if not 0.5 < integration_confidence < 1.0:
        raise ValueError("source-rollout integration confidence must lie in (0.5, 1)")
    horizon = min(size, int(math.floor((remaining_budget + 1e-12) / item_costs[0])))

    sample_blocks = tuple(
        _sample_gaussian(
            mean,
            covariance,
            sample_count=block_size,
            seed=seed + 104729 * block_index,
        )
        for block_index in range(sobol_scramble_count)
    )
    samples = np.concatenate(sample_blocks, axis=0)
    labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        fixed_template=fixed_template,
    )
    rewards = _source_rollout_rewards(
        sampled_query_energies=samples,
        final_hull_membership=labels,
        query_compositions=query_compositions,
        query_source_energies=source,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        horizon=horizon,
    )
    block_scores = rewards.reshape(sobol_scramble_count, block_size, size).mean(axis=1)
    scores = block_scores.mean(axis=0)
    source_action = int(
        source_margin_action_indices(
            source_energies=source,
            competing_hull_energies=current_hull,
            query_ids=query_ids,
        )[0]
    )
    differences = block_scores - block_scores[:, [source_action]]
    mean_advantages = differences.mean(axis=0)
    lower_bounds = _simultaneous_paired_lower_bounds(
        differences,
        confidence=integration_confidence,
        comparison_count=max(size - 1, 1),
    )
    lower_bounds[source_action] = 0.0
    improving = np.flatnonzero(lower_bounds > 0.0)
    if len(improving):
        selected_action = min(
            (int(index) for index in improving),
            key=lambda index: (-scores[index], str(query_ids[index])),
        )
        fallback_reason = None
    else:
        selected_action = source_action
        fallback_reason = (
            "source_is_only_legal_action"
            if size == 1
            else "no_positive_simultaneous_lower_bound"
        )
    probabilities = labels.mean(axis=0)
    return SourceRolloutDeltaHullResult(
        scores=tuple(float(value) for value in scores),
        block_scores=tuple(
            tuple(float(value) for value in row) for row in block_scores
        ),
        final_stability_probabilities=tuple(float(value) for value in probabilities),
        paired_advantages_over_source=tuple(float(value) for value in mean_advantages),
        paired_advantage_lower_bounds=tuple(float(value) for value in lower_bounds),
        source_action_index=source_action,
        selected_action_index=selected_action,
        posterior_sample_count=posterior_sample_count,
        sobol_scramble_count=sobol_scramble_count,
        simultaneous_comparison_count=max(size - 1, 1),
        horizon=horizon,
        fallback_reason=fallback_reason,
    )


def conformal_one_deviation_source_rollout(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    current_competing_hull_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    conformal_radius: float,
    deviation_used: bool = False,
    posterior_sample_count: int = 1024,
    seed: int = 0,
    fixed_template: FixedCompositionHullTemplate | None = None,
    sobol_scramble_count: int = 16,
    integration_confidence: float = 0.95,
) -> ConformalSourceRolloutResult:
    """Allow at most one calibrated deviation from source continuation.

    The rollout estimate is paired against the same source action and uses the
    existing simultaneous RQMC lower-bound radius as ``c_RQMC(x)``.  A
    non-source action is legal only when

    ``estimated_advantage - c_RQMC(x) > conformal_radius``.

    The conformal radius is calibrated on exact-system maxima of rollout
    over-estimation.  This function therefore supplies a safe, source-anchored
    policy rule; it does not turn posterior correctness into a distribution-free
    guarantee.  Once ``deviation_used`` is true, callers must execute the source
    policy directly for all remaining rounds.
    """

    radius = float(conformal_radius)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("conformal rollout radius must be finite and non-negative")
    rollout = source_rollout_delta_hull(
        posterior,
        query_compositions=query_compositions,
        query_source_energies=query_source_energies,
        query_ids=query_ids,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        current_competing_hull_energies=current_competing_hull_energies,
        costs=costs,
        remaining_budget=remaining_budget,
        posterior_sample_count=posterior_sample_count,
        seed=seed,
        fixed_template=fixed_template,
        sobol_scramble_count=sobol_scramble_count,
        integration_confidence=integration_confidence,
    )
    advantages = np.asarray(rollout.paired_advantages_over_source, dtype=np.float64)
    lower_bounds = np.asarray(rollout.paired_advantage_lower_bounds, dtype=np.float64)
    rqmc_radii = np.maximum(advantages - lower_bounds, 0.0)
    adjusted = advantages - rqmc_radii
    source_index = rollout.source_action_index
    adjusted[source_index] = 0.0
    eligible = np.flatnonzero(adjusted > radius)
    if deviation_used:
        selected = source_index
        selected_deviation = False
        reason = "deviation_already_used"
    elif len(eligible):
        selected = min(
            (int(index) for index in eligible),
            key=lambda index: (-adjusted[index], str(query_ids[index])),
        )
        selected_deviation = selected != source_index
        reason = None
    else:
        selected = source_index
        selected_deviation = False
        reason = "conformal_gate_not_positive"
    return ConformalSourceRolloutResult(
        scores=rollout.scores,
        paired_advantages_over_source=rollout.paired_advantages_over_source,
        rqmc_radii=tuple(float(value) for value in rqmc_radii),
        conformal_adjusted_advantages=tuple(float(value) for value in adjusted),
        source_action_index=source_index,
        selected_action_index=selected,
        deviation_used_before=deviation_used,
        deviation_selected=selected_deviation,
        fallback_reason=reason,
        conformal_radius=radius,
        posterior_sample_count=rollout.posterior_sample_count,
        sobol_scramble_count=rollout.sobol_scramble_count,
        horizon=rollout.horizon,
    )


def protocol_hull_risk_reduction(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    costs: np.ndarray,
    posterior_sample_count: int = 16,
    fantasy_count: int = 3,
    seed: int = 0,
) -> ProtocolHullRiskReductionResult:
    """Reduce Bayes risk of the complete target-protocol hull function.

    The terminal estimator is the posterior mean hull on the fixed union of
    initial, revealed and still-queryable compositions.  Under squared loss,
    its Bayes risk is the mean posterior variance of that random hull.  Each
    action is valued by the expected reduction in this risk per unit cost.
    This continuous objective remains informative when binary hull-membership
    probabilities saturate.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    size = len(mean)
    if len(query_compositions) != size or len(item_costs) != size:
        raise ValueError("protocol hull risk-reduction inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("protocol hull query costs must be finite and positive")
    if posterior_sample_count < 4 or fantasy_count < 1:
        raise ValueError("protocol hull Monte Carlo settings are too small")

    evaluation_compositions = _fixed_evaluation_compositions(
        query_compositions,
        reference_compositions,
    )
    current_samples = _sample_gaussian(
        mean,
        covariance,
        sample_count=posterior_sample_count,
        seed=seed,
    )
    current_hulls = _final_hull_values(
        query_compositions=query_compositions,
        sampled_query_energies=current_samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
        evaluation_compositions=evaluation_compositions,
    )
    current_risk = _mean_hull_squared_error_risk(current_hulls)
    nodes, weights = np.polynomial.hermite.hermgauss(fantasy_count)
    weights = weights / math.sqrt(math.pi)
    expected_risks = np.empty(size, dtype=np.float64)
    for query_index in range(size):
        variance = float(covariance[query_index, query_index])
        if variance <= 1e-15:
            expected_risks[query_index] = current_risk
            continue
        cross = covariance[:, query_index]
        conditional_covariance = covariance - np.outer(cross, cross) / variance
        conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
        conditional_covariance[query_index, :] = 0.0
        conditional_covariance[:, query_index] = 0.0
        expected_risk = 0.0
        for fantasy_index, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
            outcome = mean[query_index] + math.sqrt(2.0 * variance) * node
            conditional_mean = mean + cross * ((outcome - mean[query_index]) / variance)
            conditional_samples = _sample_gaussian(
                conditional_mean,
                conditional_covariance,
                sample_count=posterior_sample_count,
                seed=seed + 104729 * (fantasy_index + 1),
            )
            conditional_hulls = _final_hull_values(
                query_compositions=query_compositions,
                sampled_query_energies=conditional_samples,
                reference_compositions=reference_compositions,
                reference_energies=reference_energies,
                evaluation_compositions=evaluation_compositions,
            )
            expected_risk += float(weight) * _mean_hull_squared_error_risk(conditional_hulls)
        expected_risks[query_index] = expected_risk
    reductions = current_risk - expected_risks
    scores = reductions / item_costs
    return ProtocolHullRiskReductionResult(
        scores=tuple(float(value) for value in scores),
        risk_reductions=tuple(float(value) for value in reductions),
        expected_posterior_risks=tuple(float(value) for value in expected_risks),
        current_hull_risk=current_risk,
        evaluation_composition_count=len(evaluation_compositions),
        posterior_sample_count=posterior_sample_count,
        fantasy_count=fantasy_count,
    )


def protocol_hull_knowledge_gradient(
    posterior: ProtocolTargetEnergyPosterior,
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    costs: np.ndarray,
    remaining_budget: float,
    posterior_sample_count: int = 16,
    fantasy_count: int = 3,
    seed: int = 0,
) -> ProtocolHullKnowledgeGradientResult:
    """Evaluate the exact two-step Bayes objective under the working posterior.

    With unit query costs, the terminal utility is the number of queried phases
    that belong to the final target-protocol hull.  The first term is the
    current query's final-stability probability; the second is the expected
    optimal final-stability probability for one subsequent query.
    """

    mean = np.asarray(posterior.mean, dtype=np.float64)
    covariance = np.asarray(posterior.covariance, dtype=np.float64)
    item_costs = np.asarray(costs, dtype=np.float64).reshape(-1)
    size = len(mean)
    if len(query_compositions) != size or len(item_costs) != size:
        raise ValueError("protocol hull knowledge-gradient inputs disagree")
    if np.any(~np.isfinite(item_costs)) or np.any(item_costs <= 0):
        raise ValueError("protocol hull query costs must be finite and positive")
    if not np.allclose(item_costs, item_costs[0], atol=1e-12):
        raise ValueError("two-step protocol hull knowledge gradient requires unit costs")
    if posterior_sample_count < 4 or fantasy_count < 1:
        raise ValueError("protocol hull Monte Carlo settings are too small")
    if not math.isfinite(remaining_budget) or remaining_budget < item_costs[0]:
        raise ValueError("remaining protocol budget cannot pay for a query")

    current_samples = _sample_gaussian(
        mean,
        covariance,
        sample_count=posterior_sample_count,
        seed=seed,
    )
    current_labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=current_samples,
        reference_compositions=reference_compositions,
        reference_energies=reference_energies,
    )
    probabilities = current_labels.mean(axis=0)
    posterior_risk = float(np.minimum(probabilities, 1.0 - probabilities).sum())
    horizon = 2 if remaining_budget + 1e-12 >= 2.0 * item_costs[0] and size > 1 else 1
    expected_second = np.zeros(size, dtype=np.float64)
    if horizon == 2:
        nodes, weights = np.polynomial.hermite.hermgauss(fantasy_count)
        weights = weights / math.sqrt(math.pi)
        for query_index in range(size):
            variance = float(covariance[query_index, query_index])
            cross = covariance[:, query_index]
            conditional_covariance = covariance - np.outer(cross, cross) / variance
            conditional_covariance = 0.5 * (conditional_covariance + conditional_covariance.T)
            conditional_covariance[query_index, :] = 0.0
            conditional_covariance[:, query_index] = 0.0
            value = 0.0
            for fantasy_index, (node, weight) in enumerate(zip(nodes, weights, strict=True)):
                outcome = mean[query_index] + math.sqrt(2.0 * variance) * node
                conditional_mean = mean + cross * ((outcome - mean[query_index]) / variance)
                conditional_samples = _sample_gaussian(
                    conditional_mean,
                    conditional_covariance,
                    sample_count=posterior_sample_count,
                    seed=seed + 104729 * (fantasy_index + 1),
                )
                conditional_labels = _final_hull_membership(
                    query_compositions=query_compositions,
                    sampled_query_energies=conditional_samples,
                    reference_compositions=reference_compositions,
                    reference_energies=reference_energies,
                )
                next_probabilities = conditional_labels.mean(axis=0)
                next_probabilities[query_index] = -np.inf
                value += float(weight) * float(np.max(next_probabilities))
            expected_second[query_index] = value
    scores = (probabilities + expected_second) / item_costs
    return ProtocolHullKnowledgeGradientResult(
        scores=tuple(float(value) for value in scores),
        final_stability_probabilities=tuple(float(value) for value in probabilities),
        expected_second_step_values=tuple(float(value) for value in expected_second),
        posterior_risk=posterior_risk,
        posterior_sample_count=posterior_sample_count,
        fantasy_count=fantasy_count,
        horizon=horizon,
    )
