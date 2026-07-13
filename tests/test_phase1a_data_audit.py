from __future__ import annotations

import json
from pathlib import Path

import pytest

from evimem.benchmark import (
    DatasetRegistry,
    DataView,
    EvidenceInferenceAdapter,
    MeasEvalAdapter,
    QasperAdapter,
    SciFactAdapter,
    SciRexAdapter,
    ViewSample,
    leakage_report,
)
from evimem.benchmark.views import ConversionOrigin, exact_evidence
from evimem.contracts import ScientificClaimRecord, UpdateOperation

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "public_data"


def _fixture(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("fixture", "adapter", "expected_task"),
    [
        ("scirex.json", SciRexAdapter(), "n_ary_relation_retrieval"),
        ("qasper.json", QasperAdapter(), "question_evidence_retrieval"),
        ("scifact.json", SciFactAdapter(), "claim_rationale_retrieval"),
        (
            "evidence_inference.json",
            EvidenceInferenceAdapter(),
            "ico_evidence_retrieval",
        ),
        ("measeval.json", MeasEvalAdapter(), "slot_extraction_only"),
    ],
)
def test_real_public_schema_fixture(fixture, adapter, expected_task) -> None:
    row = _fixture(fixture)
    if isinstance(row, list):
        row = row[0]
    split = row.pop("split", "train")
    samples = adapter.convert_views(row, split=split)
    assert samples
    assert samples[0].native_task == expected_task


def test_evidence_round_trip_and_physical_oracle_separation() -> None:
    sample = QasperAdapter().convert_views(_fixture("qasper.json"), split="train")[0]
    for evidence in sample.evidence:
        assert sample.source_text[evidence.start : evidence.end] == evidence.text
    model_input = sample.model_input().model_dump()
    assert "target" not in model_input
    assert "evidence" not in model_input
    assert "admission" not in model_input
    assert sample.oracle_payload().evidence
    with pytest.raises(ValueError, match="not an exact substring"):
        exact_evidence(
            sample.source_text,
            "fabricated evidence",
            source_document_id=sample.source_document_id,
            source_field="test",
        )


def test_component_license_gate_and_scifact_multi_license_propagation() -> None:
    registry = DatasetRegistry.load(ROOT / "configs" / "datasets.json")
    registry.assert_training_allowed("SciREX", DataView.RETRIEVAL)
    registry.assert_training_allowed("SciFact", DataView.RETRIEVAL)
    with pytest.raises(ValueError, match="annotations:license_checksum_missing"):
        registry.assert_training_allowed("QASPER", DataView.RETRIEVAL)
    scifact = registry.get("SciFact")
    assert scifact.licenses.annotations.spdx_identifier == "CC-BY-4.0"
    assert scifact.licenses.source_text.spdx_identifier == "ODC-By-1.0"
    assert scifact.licenses.code.spdx_identifier == "Apache-2.0"
    assert scifact.licenses.derived_artifacts.spdx_identifier == "LicenseRef-SciFact-Composite"


def test_split_leakage_detects_document_and_claim_family_crossing() -> None:
    base = QasperAdapter().convert_views(_fixture("qasper.json"), split="train")[0]
    leaked = base.model_copy(update={"sample_id": "leaked", "split": "test"})
    report = leakage_report([base, leaked])
    assert report["passed"] is False
    assert base.source_document_id in report["source_document_cross_split"]
    assert report["exact_normalized_claim_or_question_family_cross_split"]


def test_no_field_fabrication_and_scifact_has_no_update_gold() -> None:
    meas = MeasEvalAdapter().convert_views(_fixture("measeval.json"), split="train")[0]
    assert meas.view is None
    assert meas.target["measured_property"] is None
    assert meas.target["unit"] is None
    assert meas.admission is None
    assert meas.memory_operation is None
    scifact = SciFactAdapter().convert_views(_fixture("scifact.json"), split="train")[0]
    assert scifact.admission is None
    assert scifact.memory_operation is None
    assert scifact.claim is not None
    assert scifact.claim.object is None
    assert scifact.claim.value is None
    assert scifact.claim.unit is None
    assert scifact.claim.condition is None
    assert scifact.claim.qualifiers is None


def test_scirex_official_filtered_relation_is_separate() -> None:
    samples = SciRexAdapter().convert_views(_fixture("scirex.json"), split="train")
    assert len(samples) == 2
    retrievable, table_missing = samples
    assert retrievable.view == DataView.RETRIEVAL
    assert retrievable.evidence
    assert table_missing.view is None
    assert table_missing.native_task == "filtered_relation_table_missing"
    assert table_missing.metadata["table_missing_relation"] is True
    assert table_missing.metadata["missing_mention_types"] == ["Metric"]


def test_illegal_supersede_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="human-reviewed"):
        ViewSample(
            sample_id="illegal",
            dataset_name="SciFact",
            source_document_id="doc",
            split="train",
            view=DataView.UPDATE,
            native_task="claim_veracity",
            query_text="claim",
            source_text="source",
            claim=ScientificClaimRecord(subject="claim", relation="veracity"),
            memory_operation=UpdateOperation.SUPERSEDE,
            origin=ConversionOrigin.DETERMINISTIC_DERIVED,
            label_source="native_scifact_contradiction",
        )


def test_retrieval_view_cannot_smuggle_admission_or_update_labels() -> None:
    sample = SciFactAdapter().convert_views(_fixture("scifact.json"), split="train")[0]
    with pytest.raises(ValueError, match="retrieval_view"):
        ViewSample.model_validate(
            {
                **sample.model_dump(),
                "memory_operation": "IGNORE",
            }
        )
