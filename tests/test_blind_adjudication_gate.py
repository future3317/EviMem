"""Tests for fail-closed local gating of blind model candidates."""

from __future__ import annotations

import json
from pathlib import Path

from evimem.phase1b.ai_adjudication.blind_gate import (
    CandidateExport,
    build_blind_gate_records,
    load_blind_candidate_export,
)
from evimem.phase1b.ai_adjudication.schema import AdjudicationPacket
from evimem.phase1b.ai_adjudication.validate import write_jsonl_records
from tools.run_blind_adjudication_gate import main


def make_packet() -> AdjudicationPacket:
    return AdjudicationPacket.from_external_safe_record(
        {
            "id": "scirex:0001",
            "data": {
                "pair_id": "scirex:0001",
                "left_claim": "MethodA Metric for Task = 0.1 under {\"dataset\": \"D1\"}",
                "right_claim": "MethodB Metric for Task = 0.2 under {\"dataset\": \"D1\"}",
                "left_source": "source-left",
                "right_source": "source-right",
                "left_evidence_locator": "left:0-10",
                "right_evidence_locator": "right:0-10",
                "source_level_update_type": "none",
                "source_level_update_notice": "none",
            },
            "meta": {
                "left_evidence_checksum": "sha256:" + "a" * 64,
                "right_evidence_checksum": "sha256:" + "b" * 64,
                "source_dataset": "SciREX",
            },
        },
        provenance="packet:external_safe",
    )


def candidate(packet: AdjudicationPacket, *, model_id: str, evidence: str = "SUFFICIENT") -> dict[str, object]:
    return {
        "task_id": packet.task_id,
        "semantic_relation": "COMPATIBLE_DISTINCT",
        "scope_relation": "DIFFERENT_SCOPE",
        "authority_relation": "NOT_APPLICABLE",
        "evidence_sufficiency": evidence,
        "evidence_note": "The methods differ.",
        "uncertainty_note": "",
        "annotation_provenance": "ai_juror",
        "annotator_id": f"{model_id}:blind",
        "model_id": model_id,
        "prompt_checksum": "sha256:" + "c" * 64,
        "packet_checksum": packet.packet_checksum,
        "schema_version": "phase1b-v3",
        "gold_status": "not_gold",
        "juror_run_id": f"{model_id}:blind",
    }


def export(
    packet: AdjudicationPacket,
    *,
    model_id: str,
    run_id: str | None = None,
    **changes: object,
) -> CandidateExport:
    record = candidate(packet, model_id=model_id)
    if run_id is not None:
        record["juror_run_id"] = run_id
    record.update(changes)
    return CandidateExport(
        source_name=f"{model_id}.jsonl",
        source_checksum="sha256:" + model_id.encode().hex().ljust(64, "0")[:64],
        records={packet.task_id: record},
    )


def test_two_models_are_held_for_insufficient_model_diversity() -> None:
    packet = make_packet()
    records = build_blind_gate_records(
        {packet.task_id: packet},
        [export(packet, model_id="model-a"), export(packet, model_id="model-b")],
    )

    assert records[0]["gate_status"] == "requires_human_review"
    assert "insufficient_distinct_model_diversity" in records[0]["risk_reasons"]
    assert records[0]["axis_agreement"] == {
        "semantic_relation": True,
        "scope_relation": True,
        "authority_relation": True,
        "evidence_sufficiency": True,
    }


def test_three_model_sufficient_consensus_is_only_provisional() -> None:
    packet = make_packet()
    records = build_blind_gate_records(
        {packet.task_id: packet},
        [
            export(packet, model_id="model-a"),
            export(packet, model_id="model-b"),
            export(packet, model_id="third-family"),
        ],
    )

    assert records[0]["gate_status"] == "provisional_multi_model_consensus_candidate"
    assert records[0]["gold_status"] == "not_gold"
    assert "semantic_relation" not in records[0]


def test_disagreement_or_insufficient_evidence_forces_review() -> None:
    packet = make_packet()
    disagreement = export(packet, model_id="pro", semantic_relation="UNRELATED")
    records = build_blind_gate_records(
        {packet.task_id: packet},
        [export(packet, model_id="model-a"), disagreement],
        min_distinct_models=2,
    )
    assert "model_disagreement:semantic_relation" in records[0]["risk_reasons"]
    assert records[0]["gate_status"] == "requires_human_review"

    insufficient = export(packet, model_id="pro", evidence_sufficiency="INSUFFICIENT")
    records = build_blind_gate_records(
        {packet.task_id: packet},
        [export(packet, model_id="model-a"), insufficient],
        min_distinct_models=2,
    )
    assert "evidence_not_unanimously_sufficient" in records[0]["risk_reasons"]


def test_cli_writes_no_label_winner(tmp_path: Path) -> None:
    packet = make_packet()
    packets_dir = tmp_path / "packets"
    packets_dir.mkdir()
    (packets_dir / "scirex_0001.json").write_text(
        packet.model_dump_json(), encoding="utf-8"
    )
    candidate_paths: list[Path] = []
    for model_id in ("model-a", "model-b"):
        path = tmp_path / f"{model_id}.jsonl"
        write_jsonl_records(path, [candidate(packet, model_id=model_id)])
        candidate_paths.append(path)

    assert main(
        [
            "--packets",
            str(packets_dir),
            "--candidate",
            str(candidate_paths[0]),
            "--candidate",
            str(candidate_paths[1]),
            "--output",
            str(tmp_path / "gate"),
            "--min-distinct-models",
            "2",
        ]
    ) == 0

    record = json.loads(
        (tmp_path / "gate" / "gate_records.jsonl").read_text(encoding="utf-8").strip()
    )
    assert record["gate_status"] == "provisional_multi_model_consensus_candidate"
    assert "semantic_relation" not in record


def test_superset_candidate_is_locally_restricted_to_safe_packet_subset(
    tmp_path: Path,
) -> None:
    packet = make_packet()
    path = tmp_path / "superset.jsonl"
    unrelated = candidate(packet, model_id="model-a")
    unrelated["task_id"] = "scirex:outside-selected-subset"
    write_jsonl_records(path, [candidate(packet, model_id="model-a"), unrelated])

    export_record = load_blind_candidate_export(
        path,
        {packet.task_id: packet},
        allow_superset=True,
    )

    assert set(export_record.records) == {packet.task_id}


def test_same_model_repeat_is_only_a_stability_candidate() -> None:
    packet = make_packet()
    records = build_blind_gate_records(
        {packet.task_id: packet},
        [
            export(
                packet,
                model_id="deepseek-v4-pro",
                run_id="pro-pass-a:juror-a",
            ),
            export(
                packet,
                model_id="deepseek-v4-pro",
                run_id="pro-pass-b:juror-a",
            ),
        ],
        same_model_repeat_mode=True,
    )

    assert records[0]["gate_status"] == "same_model_repeat_consistent_candidate"
    assert records[0]["distinct_model_count"] == 1
    assert records[0]["distinct_blind_run_count"] == 2
    assert records[0]["gold_status"] == "not_gold"
    assert "semantic_relation" not in records[0]


def test_same_model_repeat_keeps_any_risk_in_review_queue() -> None:
    packet = make_packet()
    records = build_blind_gate_records(
        {packet.task_id: packet},
        [
            export(
                packet,
                model_id="deepseek-v4-pro",
                run_id="pro-pass-a:juror-a",
            ),
            export(
                packet,
                model_id="deepseek-v4-pro",
                run_id="pro-pass-b:juror-a",
                evidence_sufficiency="PARTIAL",
            ),
        ],
        same_model_repeat_mode=True,
    )

    assert records[0]["gate_status"] == "requires_human_review"
    assert "evidence_not_unanimously_sufficient" in records[0]["risk_reasons"]


def test_same_model_repeat_rejects_reused_blind_run() -> None:
    packet = make_packet()
    try:
        build_blind_gate_records(
            {packet.task_id: packet},
            [
                export(packet, model_id="deepseek-v4-pro"),
                export(packet, model_id="deepseek-v4-pro"),
            ],
            same_model_repeat_mode=True,
        )
    except ValueError as exc:
        assert "distinct blind juror_run_id" in str(exc)
    else:
        raise AssertionError("same-model duplicate run must fail closed")


def test_same_model_cli_writes_stability_routing_only(tmp_path: Path) -> None:
    packet = make_packet()
    packets_dir = tmp_path / "packets"
    packets_dir.mkdir()
    (packets_dir / "scirex_0001.json").write_text(
        packet.model_dump_json(), encoding="utf-8"
    )
    candidate_paths: list[Path] = []
    for suffix in ("a", "b"):
        record = candidate(packet, model_id="deepseek-v4-pro")
        record["juror_run_id"] = f"pro-pass-{suffix}:juror-a"
        path = tmp_path / f"pro-{suffix}.jsonl"
        write_jsonl_records(path, [record])
        candidate_paths.append(path)

    output = tmp_path / "gate"
    assert main(
        [
            "--packets",
            str(packets_dir),
            "--candidate",
            str(candidate_paths[0]),
            "--candidate",
            str(candidate_paths[1]),
            "--same-model-repeat",
            "--output",
            str(output),
        ]
    ) == 0

    record = json.loads(
        (output / "same_model_repeat_consistent_candidates.jsonl")
        .read_text(encoding="utf-8")
        .strip()
    )
    assert record["gate_status"] == "same_model_repeat_consistent_candidate"
    assert "semantic_relation" not in record
    assert not (output / "provisional_multi_model_consensus.jsonl").read_text(
        encoding="utf-8"
    ).strip()
