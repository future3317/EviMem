"""Tests for the AI-adjudicated silver annotation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evimem.phase1b.ai_adjudication import (
    AdjudicationPacket,
    CriticReview,
    validate_ai_adjudication_label,
)
from evimem.phase1b.ai_adjudication.validate import (
    assert_no_sampling_or_gold_fields,
    load_packets,
    read_jsonl_records,
    validate_canonical_jsonl,
    write_jsonl_records,
)


@pytest.fixture
def crossref_external_safe() -> dict[str, object]:
    return {
        "id": "crossref:abc123",
        "data": {
            "pair_id": "crossref:abc123",
            "left_claim": "Work metadata title: Correction notice",
            "right_claim": "Source-level correction status recorded for DOI 10.1000/original; claim-level effect is not yet annotated.",
            "left_source": "doi:10.1000/original",
            "right_source": "doi:10.1000/update",
            "left_evidence_locator": "Crossref REST item.update-to",
            "right_evidence_locator": "Crossref REST item.update-to",
            "source_level_update_type": "correction",
            "source_level_update_notice": "Document-level metadata only; never claim-level supersession authority.",
        },
        "meta": {
            "left_evidence_checksum": "sha256:" + "a" * 64,
            "right_evidence_checksum": "sha256:" + "b" * 64,
            "license_components": ["Crossref metadata", "Retraction Watch metadata"],
            "source_dataset": "Crossref/Retraction Watch",
            "source_level_update": {
                "source": "publisher",
                "update_type": "correction",
                "timestamp": "2026-01-02T00:00:00Z",
                "original_doi": "10.1000/original",
                "update_doi": "10.1000/update",
                "api_response_checksum": "sha256:" + "c" * 64,
                "status_scope": "source_level_only",
                "claim_level_status": "awaiting_human_evidence_annotation",
            },
        },
    }


@pytest.fixture
def scirex_external_safe() -> dict[str, object]:
    return {
        "id": "scirex:0001",
        "data": {
            "pair_id": "scirex:0001",
            "left_claim": "FRRN Mean_IoU for Semantic_Segmentation = 71.8% under {\"dataset\": \"Cityscapes\"}",
            "right_claim": "U-Net Mean_IoU for Cell_Segmentation = 0.9203 under {\"dataset\": \"PhC-U373\"}",
            "left_source": "000f90380d768a85e2316225854fc377c079b5c4",
            "right_source": "07045f87709d0b7b998794e9fa912c0aba912281",
            "left_evidence_locator": "coref.Material:1466-1476;coref.Method:0-35;coref.Metric:1415-1448;coref.Task:40-61",
            "right_evidence_locator": "coref.Material:826-840;coref.Method:11-18;coref.Metric:13110-13113;coref.Task:65-77",
            "source_level_update_type": "none",
            "source_level_update_notice": "none",
        },
        "meta": {
            "left_evidence_checksum": "sha256:" + "d" * 64,
            "right_evidence_checksum": "sha256:" + "e" * 64,
            "license_components": ["annotations:Apache-2.0", "source_text:Apache-2.0"],
            "source_dataset": "SciREX",
            "source_level_update": None,
        },
    }


@pytest.fixture
def scirex_packet(scirex_external_safe: dict[str, object]) -> AdjudicationPacket:
    return AdjudicationPacket.from_external_safe_record(
        scirex_external_safe, provenance="packet:external_safe"
    )


@pytest.fixture
def crossref_packet(crossref_external_safe: dict[str, object]) -> AdjudicationPacket:
    return AdjudicationPacket.from_external_safe_record(
        crossref_external_safe, provenance="packet:external_safe"
    )


@pytest.fixture
def scirex_same_method_packet(scirex_external_safe: dict[str, object]) -> AdjudicationPacket:
    modified = json.loads(json.dumps(scirex_external_safe))
    # Replace right claim method with the same method as left for SAME_SCOPE tests.
    modified["data"]["right_claim"] = (
        "FRRN Mean_IoU for Semantic_Segmentation = 0.9203 under {\"dataset\": \"PhC-U373\"}"
    )
    return AdjudicationPacket.from_external_safe_record(
        modified, provenance="packet:external_safe"
    )


PROMPT_CHECKSUM = "sha256:" + "a" * 64


def base_juror_dict(task_id: str, packet_checksum: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "semantic_relation": "COMPATIBLE_DISTINCT",
        "scope_relation": "DIFFERENT_SCOPE",
        "authority_relation": "NOT_APPLICABLE",
        "evidence_sufficiency": "SUFFICIENT",
        "evidence_note": "Left and right differ in dataset and task.",
        "uncertainty_note": "None.",
        "annotation_provenance": "ai_juror",
        "annotator_id": "claude-opus-4-7-juror-a",
        "model_id": "claude-opus-4-7",
        "prompt_checksum": PROMPT_CHECKSUM,
        "packet_checksum": packet_checksum,
        "schema_version": "phase1b-v3",
        "gold_status": "not_gold",
        "juror_run_id": "juror-a-001",
    }


def base_judge_dict(task_id: str, packet_checksum: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "semantic_relation": "COMPATIBLE_DISTINCT",
        "scope_relation": "DIFFERENT_SCOPE",
        "authority_relation": "NOT_APPLICABLE",
        "evidence_sufficiency": "SUFFICIENT",
        "evidence_note": "Left and right differ in dataset and task.",
        "uncertainty_note": "None.",
        "annotation_provenance": "ai_adjudicated_silver",
        "annotator_id": "claude-opus-4-7-judge",
        "model_id": "claude-opus-4-7",
        "prompt_checksum": PROMPT_CHECKSUM,
        "packet_checksum": packet_checksum,
        "schema_version": "phase1b-v3",
        "gold_status": "not_gold",
        "juror_run_ids": ["juror-a-001", "juror-b-001"],
        "critic_run_id": "critic-001",
        "adjudication_path": "juror_a+juror_b+critic->judge",
        "evidence_locator_refs": ["left_evidence_locator", "right_evidence_locator"],
        "requires_higher_tier_ai_review": False,
    }


class TestPacketGeneration:
    def test_external_safe_packet_excludes_sampling_and_gold_fields(
        self, scirex_external_safe: dict[str, object]
    ) -> None:
        assert_no_sampling_or_gold_fields(scirex_external_safe)

    def test_packet_checksum_matches_content(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        assert scirex_packet.packet_checksum.startswith("sha256:")
        # Mutating the checksum should fail validation.
        bad = scirex_packet.model_dump(mode="json")
        bad["packet_checksum"] = "sha256:" + "0" * 64
        with pytest.raises(ValueError, match="packet_checksum mismatch"):
            AdjudicationPacket.model_validate(bad)

    def test_unlabeled_record_is_rejected_for_packet_input(self) -> None:
        unlabeled = {
            "id": "scirex:0001",
            "data": {"pair_id": "scirex:0001"},
            "meta": {
                "sampling_stratum_not_gold": "possible_same_scope_conflict_not_gold",
                "candidate_is_gold": False,
            },
        }
        with pytest.raises(ValueError, match="forbidden"):
            assert_no_sampling_or_gold_fields(unlabeled)


class TestHardRules:
    def test_crossref_non_conservative_semantic_rejected(
        self, crossref_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(crossref_packet.task_id, crossref_packet.packet_checksum)
        record["semantic_relation"] = "CONTRADICTORY"
        record["scope_relation"] = "SAME_SCOPE"
        record["authority_relation"] = "NEWER_MORE_AUTHORITATIVE"
        record["evidence_sufficiency"] = "SUFFICIENT"
        with pytest.raises(ValueError, match="Crossref"):
            validate_ai_adjudication_label(record, packet=crossref_packet)

    def test_crossref_conservative_accepted(
        self, crossref_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(crossref_packet.task_id, crossref_packet.packet_checksum)
        record["semantic_relation"] = "INSUFFICIENT_CONTEXT"
        record["scope_relation"] = "UNKNOWN_SCOPE"
        record["authority_relation"] = "UNRESOLVED"
        record["evidence_sufficiency"] = "INSUFFICIENT"
        validated = validate_ai_adjudication_label(record, packet=crossref_packet)
        assert validated["semantic_relation"] == "INSUFFICIENT_CONTEXT"

    def test_contradictory_same_scope_not_applicable_rejected(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        record["semantic_relation"] = "CONTRADICTORY"
        record["scope_relation"] = "SAME_SCOPE"
        record["authority_relation"] = "NOT_APPLICABLE"
        with pytest.raises(ValueError, match=r"CONTRADICTORY \+ SAME_SCOPE"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    @pytest.mark.parametrize(
        "authority",
        ["EQUAL_AUTHORITY", "NEWER_MORE_AUTHORITATIVE", "OLDER_MORE_AUTHORITATIVE"],
    )
    def test_contradictory_same_scope_non_unresolved_authority_rejected(
        self, scirex_packet: AdjudicationPacket, authority: str
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        record["semantic_relation"] = "CONTRADICTORY"
        record["scope_relation"] = "SAME_SCOPE"
        record["authority_relation"] = authority
        with pytest.raises(ValueError, match=r"CONTRADICTORY \+ SAME_SCOPE"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    def test_contradictory_same_scope_unresolved_accepted(
        self, scirex_same_method_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(
            scirex_same_method_packet.task_id, scirex_same_method_packet.packet_checksum
        )
        record["semantic_relation"] = "CONTRADICTORY"
        record["scope_relation"] = "SAME_SCOPE"
        record["authority_relation"] = "UNRESOLVED"
        record["evidence_sufficiency"] = "SUFFICIENT"
        validated = validate_ai_adjudication_label(record, packet=scirex_same_method_packet)
        assert validated["authority_relation"] == "UNRESOLVED"

    @pytest.mark.parametrize(
        "authority",
        ["UNRESOLVED", "EQUAL_AUTHORITY", "NEWER_MORE_AUTHORITATIVE"],
    )
    def test_non_same_scope_relation_requires_not_applicable_authority(
        self, scirex_packet: AdjudicationPacket, authority: str
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        record["authority_relation"] = authority
        with pytest.raises(ValueError, match="authority_relation must be NOT_APPLICABLE"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    def test_scirex_different_methods_same_scope_rejected(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        record["scope_relation"] = "SAME_SCOPE"
        with pytest.raises(ValueError, match="different methods"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    def test_scirex_same_task_unrelated_rejected(
        self, scirex_external_safe: dict[str, object]
    ) -> None:
        external_safe = json.loads(json.dumps(scirex_external_safe))
        external_safe["data"]["right_claim"] = (
            "U-Net Mean_IoU for Semantic_Segmentation = 0.9203 under "
            '{"dataset": "PhC-U373"}'
        )
        packet = AdjudicationPacket.from_external_safe_record(
            external_safe, provenance="packet:external_safe"
        )
        record = base_juror_dict(packet.task_id, packet.packet_checksum)
        record["semantic_relation"] = "UNRELATED"
        record["scope_relation"] = "DIFFERENT_SCOPE"
        with pytest.raises(ValueError, match="same visible task cannot be UNRELATED"):
            validate_ai_adjudication_label(record, packet=packet)

    def test_missing_canonical_field_rejected(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        del record["evidence_note"]
        with pytest.raises(ValueError, match="missing canonical fields"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    def test_operation_field_rejected(self, scirex_packet: AdjudicationPacket) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        record["update_operation"] = "MERGE"
        with pytest.raises(ValueError, match="forbidden operation key"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    def test_operation_label_in_note_rejected(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        record["evidence_note"] = "This should MERGE with existing memory."
        with pytest.raises(ValueError, match="forbidden operation label"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    def test_human_reviewed_provenance_rejected(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        record["annotation_provenance"] = "human-reviewed"
        with pytest.raises(ValueError, match="forbidden provenance term"):
            validate_ai_adjudication_label(record, packet=scirex_packet)

    def test_gold_provenance_term_in_note_rejected(self) -> None:
        record = {
            "task_id": "x",
            "semantic_relation": "EQUIVALENT",
            "scope_relation": "SAME_SCOPE",
            "authority_relation": "EQUAL_AUTHORITY",
            "evidence_sufficiency": "SUFFICIENT",
            "evidence_note": "The visible evidence does not resolve an alias.",
            "uncertainty_note": "gold standard match",
            "annotation_provenance": "ai_juror",
            "annotator_id": "x",
            "model_id": "x",
            "prompt_checksum": PROMPT_CHECKSUM,
            "packet_checksum": "sha256:" + "b" * 64,
            "schema_version": "phase1b-v3",
            "gold_status": "not_gold",
            "juror_run_id": "juror-a-001",
        }
        with pytest.raises(ValueError, match="forbidden provenance term"):
            validate_ai_adjudication_label(record)

    def test_packet_checksum_mismatch_rejected(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        record = base_juror_dict(scirex_packet.task_id, "sha256:" + "f" * 64)
        with pytest.raises(ValueError, match="packet_checksum does not match"):
            validate_ai_adjudication_label(record, packet=scirex_packet)


class TestNormalization:
    def test_normalize_label_studio_nested(self, tmp_path: Path) -> None:
        nested = {
            "id": "scirex:0001",
            "data": {"pair_id": "scirex:0001"},
            "annotations": [
                {
                    "result": [
                        {
                            "from_name": "semantic_relation",
                            "value": {"choices": ["COMPATIBLE_DISTINCT"]},
                        },
                        {
                            "from_name": "scope_relation",
                            "value": {"choices": ["DIFFERENT_SCOPE"]},
                        },
                        {
                            "from_name": "authority_relation",
                            "value": {"choices": ["NOT_APPLICABLE"]},
                        },
                        {
                            "from_name": "evidence_sufficiency",
                            "value": {"choices": ["SUFFICIENT"]},
                        },
                        {
                            "from_name": "evidence_note",
                            "value": {"text": ["Datasets differ."]},
                        },
                        {
                            "from_name": "uncertainty_note",
                            "value": {"text": ["None."]},
                        },
                    ]
                }
            ],
            "meta": {
                "annotation_provenance": "ai_juror",
                "annotator_id": "claude-juror",
                "model_id": "claude-opus-4-7",
                "prompt_checksum": PROMPT_CHECKSUM,
                "packet_checksum": "sha256:" + "b" * 64,
                "schema_version": "phase1b-v3",
                "gold_status": "not_gold",
            },
        }
        input_path = tmp_path / "nested.jsonl"
        output_path = tmp_path / "canonical.jsonl"
        write_jsonl_records(input_path, [nested])

        # Without --label-studio, validation should fail because of missing fields.
        with pytest.raises(ValueError, match="missing task_id"):
            validate_canonical_jsonl(input_path)

        # With normalization enabled, it should succeed.
        records = read_jsonl_records(input_path)
        normalized = [
            validate_ai_adjudication_label(records[0], allow_label_studio_nested=True)
        ]
        write_jsonl_records(output_path, normalized)

        result = read_jsonl_records(output_path)[0]
        assert result["semantic_relation"] == "COMPATIBLE_DISTINCT"
        assert result["scope_relation"] == "DIFFERENT_SCOPE"
        assert result["evidence_note"] == "Datasets differ."

        summary = validate_canonical_jsonl(input_path, allow_label_studio_nested=True)
        assert summary["record_count"] == 1

    def test_normalize_is_lossless(self, tmp_path: Path) -> None:
        canonical = {
            "task_id": "scirex:0001",
            "semantic_relation": "COMPATIBLE_DISTINCT",
            "scope_relation": "DIFFERENT_SCOPE",
            "authority_relation": "NOT_APPLICABLE",
            "evidence_sufficiency": "SUFFICIENT",
            "evidence_note": "Datasets differ.",
            "uncertainty_note": "None.",
            "annotation_provenance": "ai_juror",
            "annotator_id": "claude-juror",
            "model_id": "claude-opus-4-7",
            "prompt_checksum": PROMPT_CHECKSUM,
            "packet_checksum": "sha256:" + "b" * 64,
            "schema_version": "phase1b-v3",
            "gold_status": "not_gold",
            "juror_run_id": "juror-a-001",
        }
        input_path = tmp_path / "in.jsonl"
        output_path = tmp_path / "out.jsonl"
        write_jsonl_records(input_path, [canonical])

        records = read_jsonl_records(input_path)
        normalized = [validate_ai_adjudication_label(records[0])]
        write_jsonl_records(output_path, normalized)

        result = read_jsonl_records(output_path)[0]
        assert result["semantic_relation"] == canonical["semantic_relation"]
        assert result["scope_relation"] == canonical["scope_relation"]
        assert result["evidence_note"] == canonical["evidence_note"]


class TestJudgeMajorityVote:
    def test_judge_rejects_unsupported_majority_agreement(
        self, scirex_same_method_packet: AdjudicationPacket
    ) -> None:
        # Two jurors agree on EQUIVALENT+SAME_SCOPE but evidence is partial.
        draft = base_judge_dict(
            scirex_same_method_packet.task_id, scirex_same_method_packet.packet_checksum
        )
        draft["semantic_relation"] = "EQUIVALENT"
        draft["scope_relation"] = "SAME_SCOPE"
        draft["authority_relation"] = "NOT_APPLICABLE"
        draft["evidence_sufficiency"] = "PARTIAL"
        draft["requires_higher_tier_ai_review"] = False
        with pytest.raises(ValueError, match=r"EQUIVALENT \+ SAME_SCOPE"):
            validate_ai_adjudication_label(draft, packet=scirex_same_method_packet)

    def test_judge_downgrades_to_requires_higher_tier_review(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        draft = base_judge_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        draft["evidence_sufficiency"] = "INSUFFICIENT"
        draft["requires_higher_tier_ai_review"] = True
        validated = validate_ai_adjudication_label(draft, packet=scirex_packet)
        assert validated["requires_higher_tier_ai_review"] is True

    def test_judge_cannot_be_sufficient_and_require_review(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        draft = base_judge_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        draft["evidence_sufficiency"] = "SUFFICIENT"
        draft["requires_higher_tier_ai_review"] = True
        with pytest.raises(ValueError, match="requires_higher_tier_ai_review"):
            validate_ai_adjudication_label(draft, packet=scirex_packet)


class TestCriticReview:
    def test_critic_review_schema(self, scirex_packet: AdjudicationPacket) -> None:
        review = {
            "task_id": scirex_packet.task_id,
            "critic_run_id": "critic-001",
            "juror_run_ids": ("juror-a-001", "juror-b-001"),
            "packet_checksum": scirex_packet.packet_checksum,
            "issues": [
                {
                    "axis": "scope",
                    "issue_type": "scope_omission",
                    "evidence_locator_ref": "right_evidence_locator",
                    "explanation": "Right side lacks measurement temperature.",
                }
            ],
            "model_id": "test-critic-model",
            "prompt_checksum": PROMPT_CHECKSUM,
            "annotation_provenance": "ai_critic",
            "schema_version": "phase1b-v3",
        }
        validated = CriticReview.model_validate(review)
        assert validated.task_id == scirex_packet.task_id

    def test_critic_review_rejects_operation_language(
        self, scirex_packet: AdjudicationPacket
    ) -> None:
        review = {
            "task_id": scirex_packet.task_id,
            "critic_run_id": "critic-001",
            "juror_run_ids": ["juror-a-001", "juror-b-001"],
            "packet_checksum": scirex_packet.packet_checksum,
            "issues": [
                {
                    "axis": "note",
                    "issue_type": "operation leakage",
                    "evidence_locator_ref": scirex_packet.left.evidence_locator,
                    "explanation": "The note suggests SUPERSEDE, which is forbidden.",
                }
            ],
            "model_id": "test-critic-model",
            "prompt_checksum": PROMPT_CHECKSUM,
            "annotation_provenance": "ai_critic",
            "schema_version": "phase1b-v3",
        }
        with pytest.raises(ValueError, match="forbidden operation label"):
            CriticReview.model_validate(review)


class TestCliPacket:
    def test_cli_packet_generates_one_file_per_task(
        self, tmp_path: Path, scirex_external_safe: dict[str, object]
    ) -> None:
        input_path = tmp_path / "external_safe.jsonl"
        output_dir = tmp_path / "packets"
        write_jsonl_records(input_path, [scirex_external_safe])

        from tools.run_ai_adjudication import main

        argv = [
            "packet",
            "--input",
            str(input_path),
            "--output",
            str(output_dir),
        ]
        assert main(argv) == 0

        packets = load_packets(output_dir)
        assert len(packets) == 1
        packet = packets["scirex:0001"]
        assert packet.source_dataset == "SciREX"
        assert "sampling_stratum" not in packet.model_dump_json()

    def test_cli_packet_jsonl_mode(
        self, tmp_path: Path, scirex_external_safe: dict[str, object]
    ) -> None:
        input_path = tmp_path / "external_safe.jsonl"
        output_path = tmp_path / "packets.jsonl"
        write_jsonl_records(input_path, [scirex_external_safe])

        from tools.run_ai_adjudication import main

        argv = [
            "packet",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--jsonl",
        ]
        assert main(argv) == 0

        records = read_jsonl_records(output_path)
        assert len(records) == 1
        assert records[0]["task_id"] == "scirex:0001"

    def test_cli_packet_explicit_task_subset_is_label_free(
        self, tmp_path: Path, scirex_external_safe: dict[str, object]
    ) -> None:
        input_path = tmp_path / "external_safe.jsonl"
        output_dir = tmp_path / "packets"
        other = json.loads(json.dumps(scirex_external_safe))
        other["id"] = "scirex:0002"
        other["data"]["pair_id"] = "scirex:0002"
        write_jsonl_records(input_path, [scirex_external_safe, other])

        from tools.run_ai_adjudication import main

        assert main(
            [
                "packet",
                "--input",
                str(input_path),
                "--output",
                str(output_dir),
                "--task-id",
                "scirex:0002",
            ]
        ) == 0

        packets = load_packets(output_dir)
        assert set(packets) == {"scirex:0002"}


class TestCliValidate:
    def test_cli_validate_canonical_jsonl(
        self, tmp_path: Path, scirex_packet: AdjudicationPacket
    ) -> None:
        input_path = tmp_path / "canonical.jsonl"
        packet_dir = tmp_path / "packets"
        record = base_juror_dict(scirex_packet.task_id, scirex_packet.packet_checksum)
        write_jsonl_records(input_path, [record])
        packet_dir.mkdir(parents=True, exist_ok=True)
        (packet_dir / "scirex_0001.json").write_text(
            json.dumps(scirex_packet.model_dump(mode="json"), ensure_ascii=False),
            encoding="utf-8",
        )

        from tools.run_ai_adjudication import main

        argv = [
            "validate",
            "--input",
            str(input_path),
            "--packets",
            str(packet_dir),
        ]
        assert main(argv) == 0

    def test_cli_validate_rejects_duplicate_task_ids(self, tmp_path: Path) -> None:
        input_path = tmp_path / "canonical.jsonl"
        record = base_juror_dict("task-1", "sha256:" + "b" * 64)
        write_jsonl_records(input_path, [record, record])

        from tools.run_ai_adjudication import main

        argv = ["validate", "--input", str(input_path)]
        assert main(argv) == 1
