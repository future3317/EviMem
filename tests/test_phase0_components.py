from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evimem.contracts import CandidateObservation, ProposerProvenance, ScientificClaim
from evimem.domains import DomainValidator, list_domain_packs, load_domain_pack
from evimem.evidence import EvidenceBinder, EvidenceBlockStore, EvidenceReleaseManager

from .evimem_helpers import claim


def test_all_packaged_domain_packs_load_with_stable_identity() -> None:
    assert list_domain_packs() == ("piezoelectric", "superconductors", "thermoelectric")
    packs = [load_domain_pack(domain_id) for domain_id in list_domain_packs()]
    assert [len(pack.properties) for pack in packs] == [24, 11, 12]
    assert all(pack.pack_hash.startswith("sha256:") for pack in packs)
    assert load_domain_pack("piezoelectric").pack_hash == packs[0].pack_hash


def test_domain_validator_checks_property_unit_range_and_context() -> None:
    validator = DomainValidator(load_domain_pack("piezoelectric"))
    result = validator.validate(claim())
    assert result.passed
    assert result.canonical_property == "d33"
    invalid = validator.validate(claim().model_copy(update={"value_num": 9000.0}))
    assert invalid.reason_codes == ("value_outside_expected_range",)


def test_controller_package_has_no_publication_write_dependency() -> None:
    controller_dir = Path(__file__).parents[1] / "src" / "evimem" / "controller"
    source = "\n".join(path.read_text(encoding="utf-8") for path in controller_dir.glob("*.py"))
    assert "evimem.publication" not in source


def test_binding_can_verify_one_tuple_from_multiple_immutable_blocks(tmp_path) -> None:
    manager = EvidenceReleaseManager(tmp_path / "evidence")
    manager.create_release(
        [
            {
                "doi": "10.1000/multiblock",
                "source": "fixture",
                "block_id": "material-property",
                "text": "BaTiO3 was characterized for its d33 coefficient.",
            },
            {
                "doi": "10.1000/multiblock",
                "source": "fixture",
                "block_id": "value-condition",
                "text": "The measured value was 190 pC/N at room temperature.",
            },
        ],
        release_id="multi-block-release",
    )
    store = EvidenceBlockStore(manager)
    claim_value = ScientificClaim(
        property_key="d33",
        value_raw="190",
        value_num=190.0,
        unit_raw="pC/N",
        material_raw="BaTiO3",
        conditions_raw="room temperature",
    )
    candidate = CandidateObservation(
        candidate_id="multi-block-candidate",
        run_id="multi-block-run",
        doi="10.1000/multiblock",
        claim=claim_value,
        proposed_evidence=list(store.refs_for_doi("multi-block-release", "10.1000/multiblock")),
        proposer_provenance=ProposerProvenance(
            provider="test",
            model="test",
            extraction_schema_version="evimem.v1",
            prompt_hash="test",
            extraction_timestamp=datetime(2026, 7, 13, tzinfo=UTC),
        ),
    )
    binding = EvidenceBinder(store, load_domain_pack("piezoelectric")).bind(candidate)
    assert binding.binding_method == "multi_block_slot_match"
    assert binding.support_tier == "verified_strong"
    assert {ref.block_id for ref in binding.resolved_evidence} == {
        "material-property",
        "value-condition",
    }
