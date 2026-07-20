from __future__ import annotations

import pytest

from matmem import (
    CompatibilityKind,
    ProtocolAwareActivator,
    ProtocolCompatibilityResolver,
    ProtocolTransportMap,
    StructureArtifactIdentity,
    StructureStage,
)

from .test_matmem import _card, _protocol, _query


def _transport(radius: float = 0.02) -> ProtocolTransportMap:
    return ProtocolTransportMap(
        source_protocol=_protocol("PBE"),
        target_protocol=_protocol("PBE+U"),
        slope=1.0,
        intercept_ev_per_atom=0.0,
        error_radius_ev_per_atom=radius,
        matched_structure_count=12,
        calibration_group_checksum="sha256:" + "a" * 64,
        calibration_id="disjoint-same-structure-calibration-v1",
    )


def test_low_fidelity_relaxed_structure_is_causal_for_target_query() -> None:
    identity = StructureArtifactIdentity.low_fidelity_relaxed("q", "hash")
    assert identity.stage is StructureStage.LOW_FIDELITY_RELAXED
    assert identity.causal_available_before_query is True


def test_homogeneous_zero_cost_null_is_exact_full_history() -> None:
    protocol = _protocol("PBE")
    query = _query(protocol=protocol)
    archive = (
        _card("first", protocol=protocol),
        _card("second", protocol=protocol),
    )
    activation = ProtocolAwareActivator(
        ProtocolCompatibilityResolver(),
        max_transport_uncertainty_ev_per_atom=None,
    ).activate(query, archive)
    assert activation.cards == archive
    assert activation.audit.active_card_ids == activation.audit.archive_card_ids
    assert activation.audit.full_history_equivalent is True


def test_activation_never_drops_direct_or_certified_history() -> None:
    source, target = _protocol("PBE"), _protocol("PBE+U")
    resolver = ProtocolCompatibilityResolver([_transport()])
    query = _query(protocol=target)
    direct = _card("direct", protocol=target)
    source_card = _card("source", protocol=source)
    activation = ProtocolAwareActivator(resolver).activate(query, (source_card, direct))
    assert activation.audit.active_card_ids == ("source", "direct")
    assert resolver.resolve(source, target).kind is CompatibilityKind.TRANSPORTED


def test_all_certified_transports_remain_active() -> None:
    source, target = _protocol("PBE"), _protocol("PBE+U")
    query = _query(protocol=target)
    canonical = query.identity.canonical_structure_id
    unrelated = _card("a-unrelated", embedding=query.embedding, protocol=source)
    same = _card("z-same", embedding=(0.0, 1.0), protocol=source).model_copy(
        update={
            "identity": unrelated.identity.model_copy(
                update={"canonical_structure_id": canonical}
            )
        }
    )
    activation = ProtocolAwareActivator(ProtocolCompatibilityResolver([_transport()])).activate(
        query, (unrelated, same)
    )
    assert activation.audit.transported_card_ids == ("a-unrelated", "z-same")


def test_activation_has_no_similarity_or_capacity_selection_path() -> None:
    source, target = _protocol("PBE"), _protocol("PBE+U")
    with pytest.raises(TypeError):
        ProtocolAwareActivator(  # type: ignore[call-arg]
            ProtocolCompatibilityResolver([_transport()]),
            minimum_similarity=0.9,
        )
    with pytest.raises(TypeError):
        ProtocolAwareActivator(  # type: ignore[call-arg]
            ProtocolCompatibilityResolver([_transport()]),
            capacity=1,
        )

    query = _query(protocol=target)
    anti_aligned = _card("anti", embedding=(-1.0, 0.0), protocol=source)
    activation = ProtocolAwareActivator(
        ProtocolCompatibilityResolver([_transport()])
    ).activate(query, (anti_aligned,))
    assert activation.audit.transported_card_ids == ("anti",)


def test_transport_uncertainty_gate_fails_closed() -> None:
    source, target = _protocol("PBE"), _protocol("PBE+U")
    source_card = _card("source", protocol=source)
    activation = ProtocolAwareActivator(
        ProtocolCompatibilityResolver([_transport(radius=0.2)]),
        max_transport_uncertainty_ev_per_atom=0.05,
    ).activate(_query(protocol=target), (source_card,))
    assert activation.cards == ()
    assert activation.audit.rejected_uncertainty_card_ids == ("source",)
