"""Campaign-level strategy gate for the frozen IC-SARR policy.

This module compares two complete adaptive campaigns, ``source_margin`` and
the frozen ``independent_confirmation_source_rollout`` (IC-SARR).  It does
not gate individual actions.  An outer posterior-energy draw is shared by the
two policies, while IC-SARR's own RQMC stream remains an independent inner
stream.  The result is posterior-relative and numerical; it is not an oracle
or deployment safety guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .protocol_knowledge_gradient import (
    FixedCompositionHullTemplate,
    FrozenProtocolRidgeTransport,
    _final_hull_membership,
    _sample_gaussian,
    _simultaneous_paired_lower_bounds,
    fixed_composition_hull_membership,
    independent_confirmation_source_rollout,
    protocol_target_energy_posterior,
    source_margin_action_indices,
)


@dataclass(frozen=True, slots=True)
class CampaignGatedICSARRResult:
    """Posterior campaign-level comparison and one-time strategy decision."""

    selected_policy: str
    terminal_advantage: float
    selected_history_advantage: float
    terminal_lower_bound: float
    selected_history_lower_bound: float
    terminal_block_differences: tuple[float, ...]
    selected_history_block_differences: tuple[float, ...]
    source_terminal_value: float
    ic_terminal_value: float
    source_selected_history_value: float
    ic_selected_history_value: float
    outer_sample_count: int
    outer_block_count: int
    inner_stage_one_sample_count: int
    inner_stage_two_sample_count: int


def _current_hull_energies(
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
) -> np.ndarray:
    """Compute competing-hull energies using the same phase-diagram backend."""

    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    entries = [
        ComputedEntry(
            Composition(composition),
            float(energy) * Composition(composition).num_atoms,
            entry_id=f"reference:{index}",
        )
        for index, (composition, energy) in enumerate(
            zip(reference_compositions, reference_energies, strict=True)
        )
    ]
    diagram = PhaseDiagram(entries)
    values: list[float] = []
    for composition in query_compositions:
        parsed = Composition(composition)
        hull = float(diagram.get_hull_energy_per_atom(parsed))
        fake = ComputedEntry(parsed, hull * parsed.num_atoms)
        values.append(float(diagram.get_form_energy_per_atom(fake)))
    return np.asarray(values, dtype=float)


def _simulate_campaign(
    *,
    policy: str,
    world: np.ndarray,
    model: FrozenProtocolRidgeTransport,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    query_features: np.ndarray,
    query_kernel_features: np.ndarray,
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    budget: int,
    policy_seed: int,
    inner_stage_one_sample_count: int,
    inner_stage_two_sample_count: int,
    sobol_scramble_count: int,
    integration_confidence: float,
) -> tuple[float, float, tuple[int, ...]]:
    """Simulate one complete policy under one shared posterior energy world."""

    if policy not in {"source_margin", "ic_sarr"}:
        raise ValueError("campaign policy must be source_margin or ic_sarr")
    source = np.asarray(query_source_energies, dtype=float).reshape(-1)
    world_values = np.asarray(world, dtype=float).reshape(-1)
    features = np.asarray(query_features, dtype=float)
    kernel = np.asarray(query_kernel_features, dtype=float)
    if len(world_values) != len(source) or features.shape[0] != len(source):
        raise ValueError("campaign world and query arrays disagree")
    if kernel.shape[0] != len(source):
        raise ValueError("campaign kernel and query arrays disagree")
    if budget < 1 or budget > len(source):
        raise ValueError("campaign budget must be within the query pool")

    remaining = list(range(len(source)))
    history: list[int] = []
    selected: list[int] = []
    for round_index in range(budget):
        query_comps = tuple(query_compositions[index] for index in remaining)
        query_ids_local = tuple(query_ids[index] for index in remaining)
        query_source_local = source[remaining]
        # The explicit lists below avoid mutating the caller's reference data.
        active_reference_compositions = list(reference_compositions)
        active_reference_energies = list(np.asarray(reference_energies, dtype=float))
        for index in history:
            active_reference_compositions.append(dict(query_compositions[index]))
            active_reference_energies.append(float(world_values[index]))
        active_reference_energies_array = np.asarray(active_reference_energies, dtype=float)
        current_hull = _current_hull_energies(
            query_compositions=query_comps,
            reference_compositions=active_reference_compositions,
            reference_energies=active_reference_energies_array,
        )
        if policy == "source_margin":
            local_index = int(
                source_margin_action_indices(
                    source_energies=query_source_local,
                    competing_hull_energies=current_hull,
                    query_ids=query_ids_local,
                )[0]
            )
        else:
            local_features = features[remaining]
            local_kernel = kernel[remaining]
            history_features = features[history]
            history_kernel = kernel[history]
            history_source = source[history]
            history_target = world_values[history]
            posterior = protocol_target_energy_posterior(
                model,
                query_features=local_features,
                query_source_energies=query_source_local,
                history_features=history_features.reshape(len(history), features.shape[1]),
                history_source_energies=history_source,
                history_target_energies=history_target,
                query_kernel_features=local_kernel,
                history_kernel_features=history_kernel.reshape(len(history), kernel.shape[1]),
            )
            local_template = FixedCompositionHullTemplate.from_compositions(
                query_compositions=query_comps,
                reference_compositions=active_reference_compositions,
            )
            result = independent_confirmation_source_rollout(
                posterior,
                query_compositions=query_comps,
                query_source_energies=query_source_local,
                query_ids=query_ids_local,
                reference_compositions=active_reference_compositions,
                reference_energies=active_reference_energies_array,
                current_competing_hull_energies=current_hull,
                costs=np.ones(len(remaining), dtype=float),
                remaining_budget=float(budget - round_index),
                stage_one_posterior_sample_count=inner_stage_one_sample_count,
                stage_two_posterior_sample_count=inner_stage_two_sample_count,
                seed=policy_seed + 1009 * round_index,
                fixed_template=local_template,
                sobol_scramble_count=sobol_scramble_count,
                integration_confidence=integration_confidence,
            )
            local_index = int(result.selected_action_index)
        global_index = remaining.pop(local_index)
        selected.append(global_index)
        history.append(global_index)

    all_template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=query_compositions,
        reference_compositions=reference_compositions,
    )
    labels = _final_hull_membership(
        query_compositions=query_compositions,
        sampled_query_energies=world_values[None, :],
        reference_compositions=reference_compositions,
        reference_energies=np.asarray(reference_energies, dtype=float),
        fixed_template=all_template,
    )[0]
    terminal_value = float(np.sum(labels[np.asarray(selected, dtype=int)]))
    selected_compositions = tuple(query_compositions[index] for index in selected)
    selected_template = FixedCompositionHullTemplate.from_compositions(
        query_compositions=selected_compositions,
        reference_compositions=reference_compositions,
    )
    selected_labels = fixed_composition_hull_membership(
        selected_template,
        query_energies=world_values[np.asarray(selected, dtype=int)][None, :],
        reference_energies=np.asarray(reference_energies, dtype=float),
    )[0]
    selected_history_value = float(np.sum(selected_labels))
    return terminal_value, selected_history_value, tuple(selected)


def campaign_gated_ic_sarr(
    *,
    posterior_mean: np.ndarray,
    posterior_covariance: np.ndarray,
    model: FrozenProtocolRidgeTransport,
    query_compositions: Sequence[dict[str, float]],
    query_source_energies: np.ndarray,
    query_ids: Sequence[str],
    query_features: np.ndarray,
    query_kernel_features: np.ndarray,
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    budget: int,
    outer_sample_count: int = 128,
    outer_seed: int = 0,
    inner_stage_one_sample_count: int = 64,
    inner_stage_two_sample_count: int = 128,
    sobol_scramble_count: int = 8,
    integration_confidence: float = 0.95,
) -> CampaignGatedICSARRResult:
    """Select IC-SARR or source once using a campaign-level paired gate."""

    mean = np.asarray(posterior_mean, dtype=float).reshape(-1)
    covariance = np.asarray(posterior_covariance, dtype=float)
    size = len(mean)
    if covariance.shape != (size, size) or not np.isfinite(covariance).all():
        raise ValueError("campaign posterior covariance is inconsistent")
    if outer_sample_count < 4 or outer_sample_count % sobol_scramble_count:
        raise ValueError("outer samples must divide into Sobol blocks")
    if outer_sample_count // sobol_scramble_count < 2:
        raise ValueError("outer Sobol blocks must contain at least two samples")
    if len(query_compositions) != size or len(query_ids) != size:
        raise ValueError("campaign posterior and query arrays disagree")
    blocks = outer_sample_count // sobol_scramble_count
    world_blocks = tuple(
        _sample_gaussian(
            mean,
            covariance,
            sample_count=blocks,
            seed=outer_seed + 104729 * block_index,
        )
        for block_index in range(sobol_scramble_count)
    )
    paired_t: list[float] = []
    paired_f: list[float] = []
    source_t: list[float] = []
    ic_t: list[float] = []
    source_f: list[float] = []
    ic_f: list[float] = []
    for worlds in world_blocks:
        block_t: list[float] = []
        block_f: list[float] = []
        for world in worlds:
            source_values = _simulate_campaign(
                policy="source_margin",
                world=world,
                model=model,
                query_compositions=query_compositions,
                query_source_energies=query_source_energies,
                query_ids=query_ids,
                query_features=query_features,
                query_kernel_features=query_kernel_features,
                reference_compositions=reference_compositions,
                reference_energies=reference_energies,
                budget=budget,
                policy_seed=outer_seed + 700001,
                inner_stage_one_sample_count=inner_stage_one_sample_count,
                inner_stage_two_sample_count=inner_stage_two_sample_count,
                sobol_scramble_count=sobol_scramble_count,
                integration_confidence=integration_confidence,
            )
            ic_values = _simulate_campaign(
                policy="ic_sarr",
                world=world,
                model=model,
                query_compositions=query_compositions,
                query_source_energies=query_source_energies,
                query_ids=query_ids,
                query_features=query_features,
                query_kernel_features=query_kernel_features,
                reference_compositions=reference_compositions,
                reference_energies=reference_energies,
                budget=budget,
                policy_seed=outer_seed + 700001,
                inner_stage_one_sample_count=inner_stage_one_sample_count,
                inner_stage_two_sample_count=inner_stage_two_sample_count,
                sobol_scramble_count=sobol_scramble_count,
                integration_confidence=integration_confidence,
            )
            source_t.append(source_values[0])
            source_f.append(source_values[1])
            ic_t.append(ic_values[0])
            ic_f.append(ic_values[1])
            block_t.append(ic_values[0] - source_values[0])
            block_f.append(ic_values[1] - source_values[1])
        paired_t.append(float(np.mean(block_t)))
        paired_f.append(float(np.mean(block_f)))
    terminal_bounds = _simultaneous_paired_lower_bounds(
        np.asarray(paired_t, dtype=float).reshape(-1, 1),
        confidence=integration_confidence,
        comparison_count=2,
    )
    history_bounds = _simultaneous_paired_lower_bounds(
        np.asarray(paired_f, dtype=float).reshape(-1, 1),
        confidence=integration_confidence,
        comparison_count=2,
    )
    terminal_advantage = float(np.mean(paired_t))
    history_advantage = float(np.mean(paired_f))
    selected_policy = (
        "ic_sarr" if float(terminal_bounds[0]) > 0 and float(history_bounds[0]) >= 0 else "source_margin"
    )
    return CampaignGatedICSARRResult(
        selected_policy=selected_policy,
        terminal_advantage=terminal_advantage,
        selected_history_advantage=history_advantage,
        terminal_lower_bound=float(terminal_bounds[0]),
        selected_history_lower_bound=float(history_bounds[0]),
        terminal_block_differences=tuple(paired_t),
        selected_history_block_differences=tuple(paired_f),
        source_terminal_value=float(np.mean(source_t)),
        ic_terminal_value=float(np.mean(ic_t)),
        source_selected_history_value=float(np.mean(source_f)),
        ic_selected_history_value=float(np.mean(ic_f)),
        outer_sample_count=outer_sample_count,
        outer_block_count=sobol_scramble_count,
        inner_stage_one_sample_count=inner_stage_one_sample_count,
        inner_stage_two_sample_count=inner_stage_two_sample_count,
    )
