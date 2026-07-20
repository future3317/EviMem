from __future__ import annotations

import pytest

from matmem import CompositionAwareProtocolTransportMap, MatchedEnergyPair

from .test_matmem import _protocol

OFFSETS = {"Li": -0.2, "Na": 0.1, "O": -0.3, "F": 0.2}


def _pair(system: str, index: int, source_energy: float) -> MatchedEnergyPair:
    elements = system.split("-")
    fractions = {element: 1 / len(elements) for element in elements}
    target = 0.9 * source_energy + sum(
        fractions[element] * OFFSETS[element] for element in elements
    )
    return MatchedEnergyPair(
        exact_calculation_id=f"{system}-calculation-{index}",
        canonical_structure_id=f"{system}-canonical-{index}",
        chemical_system=system,
        element_fractions=fractions,
        source_energy_ev_per_atom=source_energy,
        target_energy_ev_per_atom=target,
    )


def test_composition_aware_transport_fits_reference_offsets_and_cluster_radius() -> None:
    fit = [
        _pair(system, index, -0.2 - 0.1 * index)
        for system in ("Li-O", "Li-F", "Na-O")
        for index in range(3)
    ]
    radius = [
        _pair(system, index, -0.25 - 0.1 * index)
        for system in ("Na-F", "Li-Na", "F-O")
        for index in range(3)
    ]
    transport = CompositionAwareProtocolTransportMap.fit_same_structure_system_split(
        _protocol("PBE"),
        _protocol("PBE+U"),
        fit,
        radius,
        calibration_id="composition-aware-disjoint-v1",
        alpha=0.3,
        held_out_canonical_structure_ids=("held-out-evaluation",),
    )
    prediction = transport.transport(-0.4, {"Li": 0.5, "O": 0.5})
    assert prediction == pytest.approx(0.9 * -0.4 - 0.25, abs=2e-3)
    assert transport.error_radius_ev_per_atom < 2e-3
    assert transport.transport(-0.4, {"Xe": 1.0}) is None


def test_composition_aware_transport_rejects_system_overlap() -> None:
    fit = [_pair("Li-O", index, -0.1 * (index + 1)) for index in range(3)] + [
        _pair("Li-F", index, -0.2 * (index + 1)) for index in range(3)
    ]
    radius = [_pair("Li-O", index + 10, -0.3 * (index + 1)) for index in range(3)] + [
        _pair("Na-O", index, -0.4 * (index + 1)) for index in range(3)
    ]
    with pytest.raises(ValueError, match="overlap"):
        CompositionAwareProtocolTransportMap.fit_same_structure_system_split(
            _protocol("PBE"),
            _protocol("PBE+U"),
            fit,
            radius,
            calibration_id="invalid-overlap",
            alpha=0.4,
            held_out_canonical_structure_ids=("held-out",),
        )
