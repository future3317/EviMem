from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from matmem import (
    DataLicenseDecision,
    ExternalDataArtifact,
    FrozenPredictionSOAPCache,
    FrozenPredictionSOAPRecord,
    MaterialIdentity,
    MPCausalHullBuilder,
    MPPhaseRecord,
    ProtocolCertificate,
    SOAPCacheConfig,
    WBMObservableRecord,
    audit_external_data_artifacts,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _license(name: str) -> DataLicenseDecision:
    return DataLicenseDecision(
        dataset_name=name,
        release_id="frozen-v1",
        source_url=f"https://example.test/{name}",
        license_spdx="CC-BY-4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
        research_use_permitted=True,
        redistribution_permitted=False,
        attribution_required=True,
        reviewed_by="test-reviewer",
        reviewed_at=START,
    )


def _protocol() -> ProtocolCertificate:
    return ProtocolCertificate(
        functional="PBE",
        pseudopotential_set="MP-PBE-54",
        correction_scheme="MP2020",
        relaxation_protocol="MPRelaxSet",
        calculation_code="VASP",
    )


def _observable(query_id: str) -> WBMObservableRecord:
    return WBMObservableRecord(
        query_id=query_id,
        structure_hash=f"structure-{query_id}",
        identity=MaterialIdentity(
            exact_calculation_id=f"calc-{query_id}",
            canonical_structure_id=f"canonical-{query_id}",
            composition_family="A-B",
            prototype_family=f"prototype-{query_id}",
        ),
        composition="AB",
        chemical_system=("A", "B"),
        protocol=_protocol(),
        as_of=START + timedelta(days=2),
    )


def _mp_phases() -> tuple[MPPhaseRecord, ...]:
    return (
        MPPhaseRecord(
            phase_id="mp-A",
            composition_fractions={"A": 1.0},
            formation_energy_ev_per_atom=0.0,
            source_release="MP-2022.10.28",
            protocol=_protocol(),
            known_at=START,
        ),
        MPPhaseRecord(
            phase_id="mp-B",
            composition_fractions={"B": 1.0},
            formation_energy_ev_per_atom=0.0,
            source_release="MP-2022.10.28",
            protocol=_protocol(),
            known_at=START,
        ),
        MPPhaseRecord(
            phase_id="mp-AB",
            composition_fractions={"A": 0.5, "B": 0.5},
            formation_energy_ev_per_atom=-1.0,
            source_release="MP-2022.10.28",
            protocol=_protocol(),
            known_at=START,
        ),
    )


def _hull():
    return MPCausalHullBuilder().build(
        {"A": 0.5, "B": 0.5},
        _mp_phases(),
        built_at=START + timedelta(days=1),
        source_release="MP-2022.10.28",
        protocol=_protocol(),
    )


def _cache(query_ids: tuple[str, ...]) -> FrozenPredictionSOAPCache:
    vectors = {
        "a": (1.0, 0.0),
        "b": (0.0, 1.0),
        "c": (math.sqrt(0.5), math.sqrt(0.5)),
    }
    predictions = {"a": -1.02, "b": -1.01, "c": -1.00}
    return FrozenPredictionSOAPCache.create(
        predictor_id="frozen-chgnet-test",
        predictor_artifact_sha256="sha256:" + "a" * 64,
        config=SOAPCacheConfig(species=("B", "A")),
        records=(
            FrozenPredictionSOAPRecord(
                query_id=query_id,
                structure_hash=f"structure-{query_id}",
                predicted_formation_energy_ev_per_atom=predictions[query_id],
                soap_vector=vectors[query_id],
            )
            for query_id in query_ids
        ),
    )


def test_external_data_audit_requires_licensed_checksummed_files_outside_repo(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    external = tmp_path / "external"
    repository.mkdir()
    external.mkdir()
    artifacts = []
    for role in ("wbm", "materials_project", "prediction", "structure"):
        payload = f"{role}-artifact".encode()
        path = external / f"{role}.bin"
        path.write_bytes(payload)
        artifacts.append(
            ExternalDataArtifact(
                role=role,
                path=path,
                expected_sha256=_sha256(payload),
                license=_license(role),
            )
        )
    report = audit_external_data_artifacts(artifacts, repository_root=repository)
    assert report.passed
    assert report.artifact_count == 4
    assert report.findings == ()

    contaminated_path = repository / "wbm.bin"
    contaminated_path.write_bytes(b"do-not-commit")
    contaminated = artifacts[0].model_copy(
        update={"path": contaminated_path, "expected_sha256": _sha256(b"do-not-commit")}
    )
    failed = audit_external_data_artifacts(
        (contaminated, *artifacts[1:]),
        repository_root=repository,
    )
    assert not failed.passed
    assert {finding.code for finding in failed.findings} == {"artifact-inside-repository"}


def test_mp_initial_hull_is_composition_specific_and_mp_only() -> None:
    builder = MPCausalHullBuilder()
    hull = builder.build(
        {"A": 0.25, "B": 0.75},
        _mp_phases(),
        built_at=START + timedelta(days=1),
        source_release="MP-2022.10.28",
        protocol=_protocol(),
    )
    assert hull.reference_hull_energy_ev_per_atom == pytest.approx(-0.5)
    assert hull.chemical_system == ("A", "B")
    assert hull.source_version == "MaterialsProject:MP-2022.10.28"
    assert hull.phase_set_checksum.startswith("sha256:")

    with pytest.raises(ValueError, match="cannot span"):
        builder.build(
            {"A": 0.5, "B": 0.5},
            _mp_phases()[:1],
            built_at=START + timedelta(days=1),
            source_release="MP-2022.10.28",
            protocol=_protocol(),
        )

    incompatible = _protocol().model_copy(update={"correction_scheme": "none"})
    with pytest.raises(ValueError, match="system and protocol"):
        builder.build(
            {"A": 0.5, "B": 0.5},
            _mp_phases(),
            built_at=START + timedelta(days=1),
            source_release="MP-2022.10.28",
            protocol=incompatible,
        )


def test_frozen_prediction_soap_cache_is_order_invariant_and_identity_checked() -> None:
    first = _cache(("a", "b", "c"))
    second = _cache(("c", "b", "a"))
    assert first.cache_checksum == second.cache_checksum
    assert first.config.species == ("A", "B")
    assert first.query(_observable("a"), _hull()).embedding == (1.0, 0.0)

    bad_observable = _observable("a").model_copy(update={"structure_hash": "different"})
    with pytest.raises(KeyError, match="identity mismatch"):
        first.query(bad_observable, _hull())
    with pytest.raises(ValidationError, match="must be normalized"):
        FrozenPredictionSOAPRecord(
            query_id="bad",
            structure_hash="bad",
            predicted_formation_energy_ev_per_atom=0.0,
            soap_vector=(1.0, 1.0),
        )
