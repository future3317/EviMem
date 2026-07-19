"""Frozen WBM grid design, prequential evaluation, and system-level statistics."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .calibration_utility import CalibrationUtilityBuilder
from .cards import MaterialMemoryCard, MaterialQuery
from .residual_posterior import FixedKernelResidualGP

FROZEN_BUDGETS = (4, 8, 12)
FROZEN_CAPACITIES = (1, 2, 4)
PRIMARY_STRATEGIES = (
    "fifo",
    "diversity",
    "gp_variance_one_swap",
    "decision_coreset",
)
JOINT_RISK_SENTINELS = ((8, 2), (12, 4))


class FrozenGridCell(BaseModel):
    """One reported cell and the canonical physical trace that supplies it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str
    budget: int = Field(gt=0)
    capacity: int | None = Field(default=None, ge=0)
    physical_execution: bool
    execution_key: str
    canonical_budget: int = Field(gt=0)


def frozen_grid_cells() -> tuple[FrozenGridCell, ...]:
    """Return the preregistered 37 labels backed by 15 physical traces/system."""

    cells: list[FrozenGridCell] = []
    for strategy in PRIMARY_STRATEGIES:
        for capacity in FROZEN_CAPACITIES:
            eligible = tuple(budget for budget in FROZEN_BUDGETS if capacity < budget)
            canonical = max(eligible)
            key = f"{strategy}:k{capacity}:b{canonical}"
            cells.extend(
                FrozenGridCell(
                    strategy=strategy,
                    budget=budget,
                    capacity=capacity,
                    physical_execution=budget == canonical,
                    execution_key=key,
                    canonical_budget=canonical,
                )
                for budget in eligible
            )
    full_key = f"full_history:b{max(FROZEN_BUDGETS)}"
    cells.extend(
        FrozenGridCell(
            strategy="full_history",
            budget=budget,
            capacity=None,
            physical_execution=budget == max(FROZEN_BUDGETS),
            execution_key=full_key,
            canonical_budget=max(FROZEN_BUDGETS),
        )
        for budget in FROZEN_BUDGETS
    )
    cells.extend(
        FrozenGridCell(
            strategy="joint_posterior_risk_one_swap",
            budget=budget,
            capacity=capacity,
            physical_execution=True,
            execution_key=f"joint_posterior_risk_one_swap:k{capacity}:b{budget}",
            canonical_budget=budget,
        )
        for budget, capacity in JOINT_RISK_SENTINELS
    )
    return tuple(cells)


class PrequentialRoundMetrics(BaseModel):
    """Evaluator-only metrics after one reveal, hull transition, and retention."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(gt=0)
    remaining_candidate_count: int = Field(ge=0)
    active_witness_count: int = Field(ge=0)
    boundary_weight_sum: float = Field(ge=0)
    boundary_weighted_causal_crps: float | None = None
    boundary_weighted_causal_brier: float | None = None
    boundary_weighted_causal_log_loss: float | None = None
    residual_rmse_ev_per_atom: float | None = None
    residual_gaussian_nll: float | None = None
    boundary_weighted_false_stable_cost: float | None = None
    false_stable_count: int = Field(default=0, ge=0)
    posterior_fit_seconds: float = Field(ge=0)
    prediction_seconds: float = Field(ge=0)
    retention_seconds: float = Field(default=0, ge=0)
    round_pipeline_seconds: float = Field(default=0, ge=0)
    parent_rss_bytes: int | None = Field(default=None, ge=0)


class PrequentialCausalEvaluator:
    """Oracle-owned scorer; never pass this object or its labels to a policy."""

    def __init__(
        self,
        builder: CalibrationUtilityBuilder,
        oracle_formation_energy_by_id: Mapping[str, float],
    ) -> None:
        self.builder = builder
        self.oracle_formation_energy_by_id = dict(oracle_formation_energy_by_id)

    def evaluate(
        self,
        *,
        round_index: int,
        queries: Sequence[MaterialQuery],
        cards: Sequence[MaterialMemoryCard],
        retention_seconds: float = 0.0,
        parent_rss_bytes: int | None = None,
    ) -> PrequentialRoundMetrics:
        items = tuple(queries)
        witnesses = tuple(cards)
        if not items:
            return PrequentialRoundMetrics(
                round_index=round_index,
                remaining_candidate_count=0,
                active_witness_count=len(witnesses),
                boundary_weight_sum=0.0,
                posterior_fit_seconds=0.0,
                prediction_seconds=0.0,
                retention_seconds=retention_seconds,
                parent_rss_bytes=parent_rss_bytes,
            )
        missing = {item.query_id for item in items} - self.oracle_formation_energy_by_id.keys()
        if missing:
            raise ValueError(f"prequential oracle outcomes missing IDs: {sorted(missing)}")
        fit_started = time.perf_counter()
        posterior: FixedKernelResidualGP = self.builder.posterior_template.clone_unfit().fit(
            witnesses
        )
        fit_seconds = time.perf_counter() - fit_started
        prediction_started = time.perf_counter()
        prediction = posterior.predict(items)
        prediction_seconds = time.perf_counter() - prediction_started
        truth = np.asarray(
            [
                self.oracle_formation_energy_by_id[item.query_id]
                - item.base_predicted_formation_energy_ev_per_atom
                for item in items
            ],
            dtype=float,
        )
        labels = np.asarray(
            [
                item.hull_distance(self.oracle_formation_energy_by_id[item.query_id])
                <= item.stability_threshold_ev_per_atom
                for item in items
            ],
            dtype=float,
        )
        mean = np.asarray(prediction.mean_ev_per_atom, dtype=float)
        std = np.maximum(np.asarray(prediction.std_ev_per_atom, dtype=float), 1e-9)
        probabilities = np.asarray(prediction.stable_probability, dtype=float)
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        weights_by_id = self.builder.boundary_weights(items)
        weights = np.asarray([weights_by_id[item.query_id] for item in items], dtype=float)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0:
            raise ValueError("prequential boundary weights must have positive mass")
        z = (truth - mean) / std
        normal_pdf = np.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
        normal_cdf = 0.5 * (
            1 + np.asarray([math.erf(float(value) / math.sqrt(2)) for value in z])
        )
        crps = std * (
            z * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / math.sqrt(math.pi)
        )
        brier = (probabilities - labels) ** 2
        log_loss = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        stable_cutoff = self.builder.false_stable_cost / (
            self.builder.false_stable_cost + self.builder.false_unstable_cost
        )
        false_stable = (probabilities >= stable_cutoff) & (labels == 0)

        def weighted(values: np.ndarray) -> float:
            return float(np.sum(weights * values) / weight_sum)

        return PrequentialRoundMetrics(
            round_index=round_index,
            remaining_candidate_count=len(items),
            active_witness_count=len(witnesses),
            boundary_weight_sum=weight_sum,
            boundary_weighted_causal_crps=weighted(crps),
            boundary_weighted_causal_brier=weighted(brier),
            boundary_weighted_causal_log_loss=weighted(log_loss),
            residual_rmse_ev_per_atom=float(np.sqrt(np.mean((truth - mean) ** 2))),
            residual_gaussian_nll=float(
                np.mean(
                    0.5 * np.log(2 * math.pi * std**2)
                    + 0.5 * ((truth - mean) / std) ** 2
                )
            ),
            boundary_weighted_false_stable_cost=weighted(
                false_stable.astype(float) * self.builder.false_stable_cost
            ),
            false_stable_count=int(np.sum(false_stable)),
            posterior_fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
            retention_seconds=retention_seconds,
            parent_rss_bytes=parent_rss_bytes,
        )


def aggregate_prequential_prefix(
    rounds: Sequence[PrequentialRoundMetrics], budget: int
) -> dict[str, float | int | None]:
    """Average per-round losses so candidates within a system are not replicates."""

    selected = tuple(item for item in rounds if item.round_index <= budget)
    if len(selected) != budget:
        raise ValueError("prequential prefix does not contain exactly the requested rounds")
    metric_names = (
        "boundary_weighted_causal_crps",
        "boundary_weighted_causal_brier",
        "boundary_weighted_causal_log_loss",
        "residual_rmse_ev_per_atom",
        "residual_gaussian_nll",
        "boundary_weighted_false_stable_cost",
    )
    return {
        "round_count": len(selected),
        **{
            name: float(np.mean([getattr(item, name) for item in selected]))
            if all(getattr(item, name) is not None for item in selected)
            else None
            for name in metric_names
        },
        "posterior_fit_seconds": float(sum(item.posterior_fit_seconds for item in selected)),
        "prediction_seconds": float(sum(item.prediction_seconds for item in selected)),
        "retention_seconds": float(sum(item.retention_seconds for item in selected)),
        "round_pipeline_seconds": float(
            sum(item.round_pipeline_seconds for item in selected)
        ),
        "peak_parent_rss_bytes": max(
            (item.parent_rss_bytes for item in selected if item.parent_rss_bytes is not None),
            default=None,
        ),
    }


def paired_system_bootstrap(
    dacc_by_system: Mapping[str, float],
    baseline_by_system: Mapping[str, float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, float | int]:
    """Cluster bootstrap paired differences with exact systems as the units."""

    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if set(dacc_by_system) != set(baseline_by_system) or not dacc_by_system:
        raise ValueError("paired bootstrap requires identical nonempty system IDs")
    systems = tuple(sorted(dacc_by_system))
    differences = np.asarray(
        [dacc_by_system[item] - baseline_by_system[item] for item in systems], dtype=float
    )
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(systems), size=(iterations, len(systems)))
    means = np.mean(differences[draws], axis=1)
    return {
        "system_count": len(systems),
        "mean_paired_difference": float(np.mean(differences)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
    }


def paired_system_improvement_bootstrap(
    reference_by_system: Mapping[str, float],
    improved_by_system: Mapping[str, float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, float | int]:
    """Bootstrap a positive-is-better paired loss improvement by exact system."""

    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    if set(reference_by_system) != set(improved_by_system) or not reference_by_system:
        raise ValueError("paired bootstrap requires identical nonempty system IDs")
    systems = tuple(sorted(reference_by_system))
    improvements = np.asarray(
        [reference_by_system[item] - improved_by_system[item] for item in systems],
        dtype=float,
    )
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, len(systems), size=(iterations, len(systems)))
    means = np.mean(improvements[draws], axis=1)
    return {
        "system_count": len(systems),
        "mean_improvement": float(np.mean(improvements)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_seed": seed,
        "bootstrap_iterations": iterations,
    }
