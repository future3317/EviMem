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


class PosteriorQueryEvaluation(BaseModel):
    """Evaluator-owned per-query record for frozen P1 diagnostics only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    boundary_weight: float = Field(gt=0)
    true_residual_ev_per_atom: float
    causal_stable_label: float = Field(ge=0, le=1)
    posterior_mean_ev_per_atom: float
    posterior_std_ev_per_atom: float = Field(gt=0)
    residual_threshold_ev_per_atom: float = Field(allow_inf_nan=False)
    stable_probability: float = Field(ge=0, le=1)
    gaussian_crps: float
    causal_brier: float = Field(ge=0)
    causal_log_loss: float = Field(ge=0)
    gaussian_nll: float
    squared_error: float = Field(ge=0)
    posterior_variance: float = Field(gt=0)
    squared_standardized_error: float = Field(ge=0)


class PosteriorEvaluationSnapshot(BaseModel):
    """One named posterior evaluated against the same causal-time outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    witness_card_ids: tuple[str, ...]
    remaining_candidate_count: int = Field(ge=0)
    boundary_weight_sum: float = Field(ge=0)
    boundary_weighted_causal_crps: float | None = None
    boundary_weighted_causal_brier: float | None = None
    boundary_weighted_causal_log_loss: float | None = None
    residual_rmse_ev_per_atom: float | None = None
    residual_gaussian_nll: float | None = None
    query_evaluations: tuple[PosteriorQueryEvaluation, ...] = ()
    posterior_fit_seconds: float = Field(ge=0)
    prediction_seconds: float = Field(ge=0)


class GaussianNLLShapleyAttribution(BaseModel):
    """Symmetric mean/variance attribution for P3C-minus-GPV Gaussian NLL."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    p3c_minus_gpv_nll: float
    mean_attribution: float
    variance_attribution: float
    p3c_squared_error: float = Field(ge=0)
    gpv_squared_error: float = Field(ge=0)
    p3c_variance: float = Field(gt=0)
    gpv_variance: float = Field(gt=0)
    p3c_squared_standardized_error: float = Field(ge=0)
    gpv_squared_standardized_error: float = Field(ge=0)


class SelectionEffectRecord(BaseModel):
    """Observable card-level retention record for selective-inference diagnosis."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    card_id: str
    retained: bool
    signed_residual_ev_per_atom: float
    absolute_residual_ev_per_atom: float = Field(ge=0)
    residual_minus_union_mean_ev_per_atom: float
    residual_sign: int = Field(ge=-1, le=1)
    mean_kernel_similarity_to_queries: float = Field(ge=0, le=1)
    reference_mean_influence: float = Field(ge=0)
    reference_variance_influence: float = Field(ge=0)
    reference_stable_logit_influence: float = Field(ge=0)


def gaussian_nll_shapley_attribution(
    *,
    truth: float,
    p3c_mean: float,
    p3c_std: float,
    gpv_mean: float,
    gpv_std: float,
) -> GaussianNLLShapleyAttribution:
    """Attribute a Gaussian-NLL difference without ordering mean and variance."""

    if min(p3c_std, gpv_std) <= 0:
        raise ValueError("Gaussian-NLL attribution requires positive standard deviations")

    def nll(mean: float, std: float) -> float:
        return 0.5 * math.log(2 * math.pi * std**2) + 0.5 * ((truth - mean) / std) ** 2

    n_gg = nll(gpv_mean, gpv_std)
    n_pg = nll(p3c_mean, gpv_std)
    n_gp = nll(gpv_mean, p3c_std)
    n_pp = nll(p3c_mean, p3c_std)
    mean_attribution = 0.5 * ((n_pg - n_gg) + (n_pp - n_gp))
    variance_attribution = 0.5 * ((n_gp - n_gg) + (n_pp - n_pg))
    return GaussianNLLShapleyAttribution(
        p3c_minus_gpv_nll=n_pp - n_gg,
        mean_attribution=mean_attribution,
        variance_attribution=variance_attribution,
        p3c_squared_error=(truth - p3c_mean) ** 2,
        gpv_squared_error=(truth - gpv_mean) ** 2,
        p3c_variance=p3c_std**2,
        gpv_variance=gpv_std**2,
        p3c_squared_standardized_error=((truth - p3c_mean) / p3c_std) ** 2,
        gpv_squared_standardized_error=((truth - gpv_mean) / gpv_std) ** 2,
    )


def reference_headroom_recovery(
    *, reference_loss: float, projected_loss: float, comparator_loss: float
) -> dict[str, float | None]:
    """Return headroom, compression loss, and recovery for one loss metric."""

    values = (reference_loss, projected_loss, comparator_loss)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("headroom diagnostics require finite losses")
    headroom = comparator_loss - reference_loss
    compression_loss = projected_loss - reference_loss
    return {
        "reference_headroom": headroom,
        "compression_loss": compression_loss,
        "projected_minus_comparator": projected_loss - comparator_loss,
        "projection_recovery": (
            (comparator_loss - projected_loss) / headroom if headroom > 0 else None
        ),
    }


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
        snapshot = self.evaluate_snapshot(
            name="deployed",
            queries=queries,
            cards=cards,
        )
        return PrequentialRoundMetrics(
            round_index=round_index,
            remaining_candidate_count=snapshot.remaining_candidate_count,
            active_witness_count=len(tuple(cards)),
            boundary_weight_sum=snapshot.boundary_weight_sum,
            boundary_weighted_causal_crps=snapshot.boundary_weighted_causal_crps,
            boundary_weighted_causal_brier=snapshot.boundary_weighted_causal_brier,
            boundary_weighted_causal_log_loss=(snapshot.boundary_weighted_causal_log_loss),
            residual_rmse_ev_per_atom=snapshot.residual_rmse_ev_per_atom,
            residual_gaussian_nll=snapshot.residual_gaussian_nll,
            boundary_weighted_false_stable_cost=self._false_stable_cost(snapshot),
            false_stable_count=self._false_stable_count(snapshot),
            posterior_fit_seconds=snapshot.posterior_fit_seconds,
            prediction_seconds=snapshot.prediction_seconds,
            retention_seconds=retention_seconds,
            parent_rss_bytes=parent_rss_bytes,
        )

    def _false_stable_mask(self, snapshot: PosteriorEvaluationSnapshot) -> np.ndarray:
        stable_cutoff = self.builder.false_stable_cost / (
            self.builder.false_stable_cost + self.builder.false_unstable_cost
        )
        return np.asarray(
            [
                item.stable_probability >= stable_cutoff and item.causal_stable_label == 0
                for item in snapshot.query_evaluations
            ],
            dtype=bool,
        )

    def _false_stable_cost(self, snapshot: PosteriorEvaluationSnapshot) -> float | None:
        if not snapshot.query_evaluations:
            return None
        weights = np.asarray(
            [item.boundary_weight for item in snapshot.query_evaluations], dtype=float
        )
        return float(
            np.sum(
                weights
                * self._false_stable_mask(snapshot).astype(float)
                * self.builder.false_stable_cost
            )
            / np.sum(weights)
        )

    def _false_stable_count(self, snapshot: PosteriorEvaluationSnapshot) -> int:
        return int(np.sum(self._false_stable_mask(snapshot)))

    def evaluate_snapshot(
        self,
        *,
        name: str,
        queries: Sequence[MaterialQuery],
        cards: Sequence[MaterialMemoryCard],
    ) -> PosteriorEvaluationSnapshot:
        """Score a named posterior; oracle outcomes never leave this evaluator."""

        items = tuple(queries)
        witnesses = tuple(cards)
        if not items:
            return PosteriorEvaluationSnapshot(
                name=name,
                witness_card_ids=tuple(card.card_id for card in witnesses),
                remaining_candidate_count=0,
                boundary_weight_sum=0.0,
                posterior_fit_seconds=0.0,
                prediction_seconds=0.0,
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
        normal_cdf = 0.5 * (1 + np.asarray([math.erf(float(value) / math.sqrt(2)) for value in z]))
        crps = std * (z * (2 * normal_cdf - 1) + 2 * normal_pdf - 1 / math.sqrt(math.pi))
        brier = (probabilities - labels) ** 2
        log_loss = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))

        def weighted(values: np.ndarray) -> float:
            return float(np.sum(weights * values) / weight_sum)

        nll = 0.5 * np.log(2 * math.pi * std**2) + 0.5 * z**2
        return PosteriorEvaluationSnapshot(
            name=name,
            witness_card_ids=tuple(card.card_id for card in witnesses),
            remaining_candidate_count=len(items),
            boundary_weight_sum=weight_sum,
            boundary_weighted_causal_crps=weighted(crps),
            boundary_weighted_causal_brier=weighted(brier),
            boundary_weighted_causal_log_loss=weighted(log_loss),
            residual_rmse_ev_per_atom=float(np.sqrt(np.mean((truth - mean) ** 2))),
            residual_gaussian_nll=float(np.mean(nll)),
            query_evaluations=tuple(
                PosteriorQueryEvaluation(
                    query_id=item.query_id,
                    boundary_weight=float(weights[index]),
                    true_residual_ev_per_atom=float(truth[index]),
                    causal_stable_label=float(labels[index]),
                    posterior_mean_ev_per_atom=float(mean[index]),
                    posterior_std_ev_per_atom=float(std[index]),
                    residual_threshold_ev_per_atom=float(
                        item.stability_threshold_ev_per_atom
                        - item.base_hull_distance_ev_per_atom
                    ),
                    stable_probability=float(probabilities[index]),
                    gaussian_crps=float(crps[index]),
                    causal_brier=float(brier[index]),
                    causal_log_loss=float(log_loss[index]),
                    gaussian_nll=float(nll[index]),
                    squared_error=float((truth[index] - mean[index]) ** 2),
                    posterior_variance=float(std[index] ** 2),
                    squared_standardized_error=float(z[index] ** 2),
                )
                for index, item in enumerate(items)
            ),
            posterior_fit_seconds=fit_seconds,
            prediction_seconds=prediction_seconds,
        )

    def selection_effect_records(
        self,
        *,
        queries: Sequence[MaterialQuery],
        union_cards: Sequence[MaterialMemoryCard],
        retained_card_ids: Sequence[str],
    ) -> tuple[SelectionEffectRecord, ...]:
        """Measure whether a retention event depends on outcomes beyond geometry."""

        items = tuple(queries)
        cards = tuple(union_cards)
        if not cards or not items:
            return ()
        card_ids = tuple(card.card_id for card in cards)
        if len(set(card_ids)) != len(card_ids):
            raise ValueError("selection-effect cards must be unique")
        retained = set(retained_card_ids)
        if not retained.issubset(card_ids):
            raise ValueError("retained cards must belong to the audited union")
        weights_by_id = self.builder.boundary_weights(items)
        weights = np.asarray([weights_by_id[item.query_id] for item in items], dtype=float)
        weights /= np.sum(weights)
        reference = self.builder.posterior_template.clone_unfit().fit(cards).predict(items)
        reference_mean = np.asarray(reference.mean_ev_per_atom, dtype=float)
        reference_variance = np.asarray(reference.std_ev_per_atom, dtype=float) ** 2
        reference_probability = np.clip(
            np.asarray(reference.stable_probability, dtype=float), 1e-12, 1 - 1e-12
        )
        reference_logit = np.log(reference_probability / (1 - reference_probability))
        residuals = np.asarray([card.oracle_residual_ev_per_atom for card in cards], dtype=float)

        def normalized(values: tuple[float, ...]) -> np.ndarray:
            vector = np.asarray(values, dtype=float)
            return vector / np.linalg.norm(vector)

        query_vectors = np.vstack([normalized(item.embedding) for item in items])
        length_scale = self.builder.posterior_template.config.length_scale
        sqrt_five = math.sqrt(5.0)
        records: list[SelectionEffectRecord] = []
        for index, card in enumerate(cards):
            leave_one_out = tuple(item for item in cards if item.card_id != card.card_id)
            counterfactual = (
                self.builder.posterior_template.clone_unfit().fit(leave_one_out).predict(items)
            )
            counterfactual_mean = np.asarray(counterfactual.mean_ev_per_atom, dtype=float)
            counterfactual_variance = np.asarray(counterfactual.std_ev_per_atom, dtype=float) ** 2
            counterfactual_probability = np.clip(
                np.asarray(counterfactual.stable_probability, dtype=float),
                1e-12,
                1 - 1e-12,
            )
            counterfactual_logit = np.log(
                counterfactual_probability / (1 - counterfactual_probability)
            )
            distances = np.linalg.norm(query_vectors - normalized(card.embedding)[None, :], axis=1)
            scaled = sqrt_five * distances / length_scale
            similarities = (1 + scaled + scaled**2 / 3) * np.exp(-scaled)
            residual = float(residuals[index])
            records.append(
                SelectionEffectRecord(
                    card_id=card.card_id,
                    retained=card.card_id in retained,
                    signed_residual_ev_per_atom=residual,
                    absolute_residual_ev_per_atom=abs(residual),
                    residual_minus_union_mean_ev_per_atom=float(residual - np.mean(residuals)),
                    residual_sign=int(np.sign(residual)),
                    mean_kernel_similarity_to_queries=float(np.mean(similarities)),
                    reference_mean_influence=float(
                        np.sum(weights * np.abs(reference_mean - counterfactual_mean))
                    ),
                    reference_variance_influence=float(
                        np.sum(weights * np.abs(reference_variance - counterfactual_variance))
                    ),
                    reference_stable_logit_influence=float(
                        np.sum(weights * np.abs(reference_logit - counterfactual_logit))
                    ),
                )
            )
        return tuple(records)


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
        "round_pipeline_seconds": float(sum(item.round_pipeline_seconds for item in selected)),
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
