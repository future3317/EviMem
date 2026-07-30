"""Shared hull-geometry primitives for protocol and WBM decisions.

This module centralizes composition handling, PhaseDiagram construction,
stable-membership checks, and cached competing-hull computations.  It is
intentionally dependency-light so that both protocol and WBM subsystems can
share it without pulling in acquisition logic.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from .constants import HULL_NUMERICAL_TOLERANCE
from .utils import _normalized_composition_key

if TYPE_CHECKING:
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry


class FixedCompositionHullTemplate(BaseModel):
    """Cached composition geometry for an action-equivalent lower-hull solver.

    The template contains no energies.  It can therefore be built before any
    oracle reveal and reused for every posterior sample and round whose union
    of candidate/reference compositions is unchanged.  Stability is computed
    from the same reduced-composition grouping, elemental-reference handling,
    formation-energy filter and Qhull convention as ``pymatgen.PhaseDiagram``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    elements: tuple[str, ...]
    normalized_composition_matrix: tuple[tuple[float, ...], ...]
    atom_counts: tuple[float, ...]
    candidate_indices: tuple[int, ...]
    reference_indices: tuple[int, ...]
    duplicate_composition_groups: tuple[tuple[int, ...], ...]
    element_reference_indices: tuple[tuple[str, int], ...]
    entry_names: tuple[str, ...]
    numerical_tolerance: float = HULL_NUMERICAL_TOLERANCE

    @model_validator(mode="after")
    def _dimensions(self) -> FixedCompositionHullTemplate:
        if not self.elements:
            raise ValueError("fixed-composition hull requires at least one element")
        if len(self.normalized_composition_matrix) != len(self.atom_counts):
            raise ValueError("fixed-composition matrix and atom counts disagree")
        if len(self.entry_names) != len(self.normalized_composition_matrix):
            raise ValueError("fixed-composition entry names disagree")
        if len(self.candidate_indices) + len(self.reference_indices) != len(
            self.normalized_composition_matrix
        ):
            raise ValueError("fixed-composition candidate/reference indices disagree")
        return self

    @classmethod
    def from_compositions(
        cls,
        *,
        query_compositions: Sequence[dict[str, float]],
        reference_compositions: Sequence[dict[str, float]],
        numerical_tolerance: float = HULL_NUMERICAL_TOLERANCE,
    ) -> FixedCompositionHullTemplate:
        from pymatgen.core import Composition

        if not query_compositions or not reference_compositions:
            raise ValueError("fixed-composition hull requires nonempty query and reference sets")
        parsed = [Composition(value) for value in reference_compositions] + [
            Composition(value) for value in query_compositions
        ]
        elements = tuple(
            sorted({str(element) for composition in parsed for element in composition.elements})
        )
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

        return fixed_composition_hull_membership(
            self,
            query_energies=np.asarray(query_energies, dtype=np.float64),
            reference_energies=np.asarray(reference_energies, dtype=np.float64),
            runtime_plan=FixedHullRuntimePlan.from_template(self),
        )[0]


@dataclass(frozen=True, slots=True)
class FixedHullRuntimePlan:
    """Process-local NumPy companion for one immutable hull template.

    The Pydantic template is the audited, serializable geometry contract.  This
    plan contains only prevalidated arrays and index maps derived from it, so a
    posterior world never needs to rebuild composition matrices, duplicate
    dictionaries, or elemental-reference lookups.  It intentionally has no
    energy-dependent hull state.
    """

    template: FixedCompositionHullTemplate
    composition_matrix: np.ndarray
    candidate_indices: np.ndarray
    reference_indices: np.ndarray
    duplicate_groups_by_entry_name: tuple[np.ndarray, ...]
    elemental_group_indices: np.ndarray
    candidate_group_indices: np.ndarray
    binary_group_order: np.ndarray

    @classmethod
    def from_template(cls, template: FixedCompositionHullTemplate) -> FixedHullRuntimePlan:
        entry_to_group: dict[int, int] = {}
        groups: list[np.ndarray] = []
        for group_index, group in enumerate(template.duplicate_composition_groups):
            ordered = np.asarray(
                sorted(group, key=lambda index: template.entry_names[index]), dtype=np.int64
            )
            groups.append(ordered)
            entry_to_group.update({int(index): group_index for index in ordered})
        return cls(
            template=template,
            composition_matrix=np.asarray(template.normalized_composition_matrix, dtype=np.float64),
            candidate_indices=np.asarray(template.candidate_indices, dtype=np.int64),
            reference_indices=np.asarray(template.reference_indices, dtype=np.int64),
            duplicate_groups_by_entry_name=tuple(groups),
            elemental_group_indices=np.asarray(
                [entry_to_group[index] for _, index in template.element_reference_indices],
                dtype=np.int64,
            ),
            candidate_group_indices=np.asarray(
                [entry_to_group[index] for index in template.candidate_indices], dtype=np.int64
            ),
            binary_group_order=np.asarray(
                sorted(
                    range(len(groups)),
                    key=lambda group_index: (
                        float(template.normalized_composition_matrix[groups[group_index][0]][1]),
                        template.entry_names[int(groups[group_index][0])],
                    ),
                )
                if len(template.elements) == 2
                else (),
                dtype=np.int64,
            ),
        )


def fixed_composition_hull_membership(
    template: FixedCompositionHullTemplate,
    *,
    query_energies: np.ndarray,
    reference_energies: np.ndarray,
    runtime_plan: FixedHullRuntimePlan | None = None,
) -> np.ndarray:
    """Evaluate one or more sampled energy vectors with the cached backend."""

    samples = np.asarray(query_energies, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[None, :]
    if samples.ndim != 2 or samples.shape[1] != len(template.candidate_indices):
        raise ValueError("fixed-composition hull query samples have inconsistent dimensions")
    reference_values = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if (
        len(reference_values) != len(template.reference_indices)
        or not np.isfinite(reference_values).all()
    ):
        raise ValueError("fixed-composition hull reference energies are inconsistent")
    if not np.isfinite(samples).all():
        raise ValueError("fixed-composition hull query energies must be finite")
    plan = FixedHullRuntimePlan.from_template(template) if runtime_plan is None else runtime_plan
    if plan.template != template:
        raise ValueError("fixed-composition runtime plan does not match template")
    combined = np.empty((len(samples), template.entry_count), dtype=np.float64)
    combined[:, plan.reference_indices] = reference_values
    combined[:, plan.candidate_indices] = samples
    return np.asarray(
        [
            _fixed_stable_candidate_mask(
                combined_energies=sample,
                runtime_plan=plan,
            )
            for sample in combined
        ],
        dtype=bool,
    )


def _fixed_stable_candidate_mask(
    *,
    combined_energies: np.ndarray,
    runtime_plan: FixedHullRuntimePlan,
) -> np.ndarray:
    """Evaluate one cached-geometry lower hull from its complete energy vector.

    This is deliberately the same Qhull path used by
    :meth:`FixedCompositionHullTemplate.stable_candidate_mask`.  The separate
    helper lets batched posterior evaluation share the immutable composition
    matrix rather than rebuilding it once per RQMC draw.
    """

    from scipy.spatial import ConvexHull, QhullError

    template = runtime_plan.template
    values = combined_energies
    matrix = runtime_plan.composition_matrix
    selected_indices = np.empty(len(runtime_plan.duplicate_groups_by_entry_name), dtype=np.int64)
    for group_index, group in enumerate(runtime_plan.duplicate_groups_by_entry_name):
        # Groups are ordered by entry name, therefore argmin exactly reproduces
        # the historical (energy, entry_name) duplicate tie break.
        selected_indices[group_index] = group[int(np.argmin(values[group]))]
    elemental_indices = selected_indices[runtime_plan.elemental_group_indices]
    element_energies = values[elemental_indices]
    formation = values[selected_indices] - matrix[selected_indices] @ element_energies
    dimension = len(template.elements)
    if dimension == 2:
        active_groups = formation < -template.numerical_tolerance
        active_groups[runtime_plan.elemental_group_indices] = True
        ordered_groups = runtime_plan.binary_group_order[
            active_groups[runtime_plan.binary_group_order]
        ]
        lower_chain: list[int] = []
        for group_index in ordered_groups:
            index = int(selected_indices[group_index])
            while len(lower_chain) >= 2:
                left = int(selected_indices[lower_chain[-2]])
                middle = int(selected_indices[lower_chain[-1]])
                cross = (matrix[middle, 1] - matrix[left, 1]) * (values[index] - values[left]) - (
                    values[middle] - values[left]
                ) * (matrix[index, 1] - matrix[left, 1])
                if cross <= 1e-14:
                    lower_chain.pop()
                else:
                    break
            lower_chain.append(int(group_index))
        stable_groups = np.zeros(len(selected_indices), dtype=bool)
        stable_groups[np.asarray(lower_chain, dtype=np.int64)] = True
        candidate_winners = selected_indices[runtime_plan.candidate_group_indices]
        return np.asarray(
            stable_groups[runtime_plan.candidate_group_indices]
            & (candidate_winners == runtime_plan.candidate_indices),
            dtype=bool,
        )
    qhull_indices = [
        int(index)
        for index, formation_energy in zip(selected_indices, formation, strict=True)
        if formation_energy < -template.numerical_tolerance
    ]
    qhull_indices.extend(int(index) for index in elemental_indices)
    qhull_indices = list(dict.fromkeys(qhull_indices))
    qhull_data = np.empty((len(qhull_indices), dimension), dtype=np.float64)
    qhull_data[:, :-1] = matrix[qhull_indices, 1:]
    qhull_data[:, -1] = values[qhull_indices]
    if dimension == 3 and len(qhull_data) >= 4:
        # For ternaries, Qhull already exposes the oriented facet equations.
        # A negative energy-axis normal identifies the lower facets directly,
        # avoiding the synthetic high-energy point and the Python determinant
        # loop used by pymatgen's dimension-generic construction.  Degenerate
        # inputs retain that reference path below.
        try:
            hull = ConvexHull(qhull_data, qhull_options="Qt i")
        except QhullError:
            pass
        else:
            lower = hull.simplices[hull.equations[:, -2] < -1e-14]
            if len(lower):
                stable_qhull_indices = {
                    int(index) for index in np.asarray(lower, dtype=np.int64).reshape(-1)
                }
                stable_combined_indices = {qhull_indices[index] for index in stable_qhull_indices}
                return np.asarray(
                    [int(index) in stable_combined_indices for index in runtime_plan.candidate_indices],
                    dtype=bool,
                )
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
        [int(index) in stable_combined_indices for index in runtime_plan.candidate_indices],
        dtype=bool,
    )


def _build_computed_entries(
    *,
    compositions: Sequence[dict[str, float]],
    energies: np.ndarray,
    entry_id_prefix: str,
) -> list[ComputedEntry]:
    """Build a list of ComputedEntry objects with total energies in eV.

    ``energies`` are assumed to be per-atom; each entry stores
    ``energy * composition.num_atoms``.
    """

    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    if len(energies) != len(compositions):
        raise ValueError("computed-entry energies and compositions disagree")
    parsed = [Composition(value) for value in compositions]
    return [
        ComputedEntry(
            composition,
            float(energy) * composition.num_atoms,
            entry_id=f"{entry_id_prefix}:{index}",
        )
        for index, (composition, energy) in enumerate(zip(parsed, energies, strict=True))
    ]


def _current_hull_energies(
    *,
    query_compositions: Sequence[dict[str, float]],
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
) -> np.ndarray:
    """Compute competing-hull energies using the pymatgen phase-diagram backend."""

    from pymatgen.analysis.phase_diagram import PhaseDiagram
    from pymatgen.core import Composition
    from pymatgen.entries.computed_entries import ComputedEntry

    reference_values = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if len(reference_values) != len(reference_compositions):
        raise ValueError("current-hull reference arrays disagree")
    entries = _build_computed_entries(
        compositions=reference_compositions,
        energies=reference_values,
        entry_id_prefix="reference",
    )
    diagram = PhaseDiagram(entries)
    values: list[float] = []
    for composition in query_compositions:
        parsed = Composition(composition)
        hull = float(diagram.get_hull_energy_per_atom(parsed))
        fake = ComputedEntry(parsed, hull * parsed.num_atoms)
        values.append(float(diagram.get_form_energy_per_atom(fake)))
    return np.asarray(values, dtype=float)


def _final_hull_membership(
    *,
    query_compositions: Sequence[dict[str, float]],
    sampled_query_energies: np.ndarray,
    reference_compositions: Sequence[dict[str, float]],
    reference_energies: np.ndarray,
    fixed_template: FixedCompositionHullTemplate | None = None,
    fixed_runtime_plan: FixedHullRuntimePlan | None = None,
) -> np.ndarray:
    if fixed_template is not None:
        if fixed_runtime_plan is None:
            expected = FixedCompositionHullTemplate.from_compositions(
                query_compositions=query_compositions,
                reference_compositions=reference_compositions,
                numerical_tolerance=fixed_template.numerical_tolerance,
            )
            if expected != fixed_template:
                raise ValueError("fixed-composition hull template does not match compositions")
        elif fixed_runtime_plan.template != fixed_template:
            raise ValueError("fixed-composition runtime plan does not match template")
        return fixed_composition_hull_membership(
            fixed_template,
            query_energies=sampled_query_energies,
            reference_energies=reference_energies,
            runtime_plan=fixed_runtime_plan,
        )

    from pymatgen.analysis.phase_diagram import PhaseDiagram

    samples = np.asarray(sampled_query_energies, dtype=np.float64)
    reference_values = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or samples.shape[1] != len(query_compositions):
        raise ValueError("final-hull energy samples and candidates disagree")
    if len(reference_values) != len(reference_compositions):
        raise ValueError("final-hull reference arrays disagree")
    labels = np.zeros(samples.shape, dtype=bool)
    for sample_index, energies in enumerate(samples):
        entries = _build_computed_entries(
            compositions=reference_compositions,
            energies=reference_values,
            entry_id_prefix="reference",
        )
        entries.extend(
            _build_computed_entries(
                compositions=query_compositions,
                energies=energies,
                entry_id_prefix="query",
            )
        )
        stable_ids = {str(entry.entry_id) for entry in PhaseDiagram(entries).stable_entries}
        labels[sample_index] = [
            f"query:{index}" in stable_ids for index in range(len(query_compositions))
        ]
    return labels


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

    samples = np.asarray(sampled_query_energies, dtype=np.float64)
    reference_values = np.asarray(reference_energies, dtype=np.float64).reshape(-1)
    if samples.ndim != 2 or samples.shape[1] != len(query_compositions):
        raise ValueError("final-hull energy samples and candidates disagree")
    if len(reference_values) != len(reference_compositions):
        raise ValueError("final-hull reference arrays disagree")
    evaluation_parsed = [Composition(value) for value in evaluation_compositions]
    values = np.empty((len(samples), len(evaluation_parsed)), dtype=np.float64)
    for sample_index, energies in enumerate(samples):
        entries = _build_computed_entries(
            compositions=reference_compositions,
            energies=reference_values,
            entry_id_prefix="reference",
        )
        entries.extend(
            _build_computed_entries(
                compositions=query_compositions,
                energies=energies,
                entry_id_prefix="query",
            )
        )
        diagram = PhaseDiagram(entries)
        values[sample_index] = [
            float(diagram.get_hull_energy_per_atom(composition))
            for composition in evaluation_parsed
        ]
    return values


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
