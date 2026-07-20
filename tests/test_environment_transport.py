from __future__ import annotations

import pytest

from matmem import (
    EnvironmentConditionalProtocolTransportMap,
    EnvironmentTransportStatus,
    MatchedEnvironmentEnergyPair,
)

from .test_matmem import _protocol


def _pair(
    system: str,
    index: int,
    *,
    source: float,
    environment: float,
) -> MatchedEnvironmentEnergyPair:
    elements = system.split("-")
    fractions = {element: 1 / len(elements) for element in elements}
    target = source + 0.25 * environment - 0.05 * source
    return MatchedEnvironmentEnergyPair(
        exact_calculation_id=f"{system}-calculation-{index}",
        canonical_structure_id=f"{system}-structure-{index}",
        chemical_system=system,
        element_fractions=fractions,
        source_descriptor=(environment, environment**2),
        source_energy_ev_per_atom=source,
        target_energy_ev_per_atom=target,
    )


def _transport() -> EnvironmentConditionalProtocolTransportMap:
    fit = [
        _pair(
            system,
            index,
            source=-0.2 - 0.04 * index + 0.03 * system_index,
            environment=0.1 * index,
        )
        for system_index, system in enumerate(("Li-O", "Li-F", "Na-O"))
        for index in range(6)
    ]
    radius = [
        _pair(
            system,
            index + 20,
            source=-0.22 - 0.03 * index + 0.02 * system_index,
            environment=0.08 * index,
        )
        for system_index, system in enumerate(("Na-F", "Li-Na", "F-O", "Li-O-F"))
        for index in range(5)
    ]
    return EnvironmentConditionalProtocolTransportMap.fit_same_candidate_system_split(
        _protocol("PBE"),
        _protocol("PBE+U"),
        fit,
        radius,
        calibration_id="environment-conditional-disjoint-v1",
        alpha=0.3,
        ridge_penalty=1e-6,
        held_out_canonical_structure_ids=("held-out-evaluation",),
    )


def test_environment_transport_recovers_source_conditioned_delta() -> None:
    transport = _transport()
    prediction = transport.predict(-0.3, (0.2, 0.04), {"Li": 0.5, "O": 0.5})
    assert prediction.status is EnvironmentTransportStatus.CERTIFIED
    assert prediction.target_energy_ev_per_atom == pytest.approx(-0.235, abs=2e-3)
    assert prediction.lower_energy_ev_per_atom <= -0.235
    assert prediction.upper_energy_ev_per_atom >= -0.235


def test_environment_transport_fails_closed_for_unseen_elements_and_widens_ood() -> None:
    transport = _transport()
    unseen = transport.predict(-0.3, (0.2, 0.04), {"Xe": 1.0})
    ood = transport.predict(-20.0, (100.0, 10000.0), {"Li": 0.5, "O": 0.5})
    assert unseen.status is EnvironmentTransportStatus.REJECT_UNSEEN_ELEMENT
    assert ood.status is EnvironmentTransportStatus.CERTIFIED
    assert unseen.target_energy_ev_per_atom is None
    assert ood.interval_half_width_ev_per_atom is not None
    supported = transport.predict(-0.3, (0.2, 0.04), {"Li": 0.5, "O": 0.5})
    assert supported.interval_half_width_ev_per_atom is not None
    assert ood.interval_half_width_ev_per_atom > supported.interval_half_width_ev_per_atom


def test_environment_transport_rejects_exact_system_overlap() -> None:
    fit = [
        _pair(system, index, source=-0.1 * (index + 1), environment=0.1 * index)
        for system in ("Li-O", "Li-F")
        for index in range(3)
    ]
    radius = [
        _pair(system, index + 10, source=-0.2 * (index + 1), environment=0.1 * index)
        for system in ("Li-O", "Na-O")
        for index in range(3)
    ]
    with pytest.raises(ValueError, match="overlap"):
        EnvironmentConditionalProtocolTransportMap.fit_same_candidate_system_split(
            _protocol("PBE"),
            _protocol("PBE+U"),
            fit,
            radius,
            calibration_id="invalid-overlap",
            alpha=0.4,
            held_out_canonical_structure_ids=("held-out",),
        )
