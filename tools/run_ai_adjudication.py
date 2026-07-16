"""CLI for the evimem-ai-adjudication Claude Code skill.

This script provides deterministic scaffolding for packet generation,
normalization, validation, and canonical formatting of juror/critic/judge
outputs. Reasoning for juror/critic/judge is performed by Claude following
the skill instructions; this tool validates and serializes the results.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evimem.phase1b.ai_adjudication.schema import (
    AdjudicationPacket,
    AiAdjudicatedSilverLabel,
    CriticReview,
    JurorAnnotation,
    canonical_json_checksum,
    validate_ai_adjudication_label,
)
from evimem.phase1b.ai_adjudication.validate import (
    assert_no_sampling_or_gold_fields,
    load_packets,
    read_jsonl_records,
    validate_canonical_jsonl,
    write_jsonl_records,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evimem-ai-adjudication")
    sub = parser.add_subparsers(dest="command", required=True)

    packet = sub.add_parser("packet", help="Generate minimal packets from external-safe input")
    packet.add_argument("--input", type=Path, required=True)
    packet.add_argument("--output", type=Path, required=True)
    packet.add_argument("--jsonl", action="store_true", help="Write one JSONL file instead of one file per packet")
    packet.add_argument("--provenance", type=str, default="packet:external_safe")
    packet.add_argument(
        "--task-id",
        action="append",
        dest="task_ids",
        help="Include one explicit external-safe task ID; repeat to build a label-free subset.",
    )

    juror = sub.add_parser("juror", help="Validate and compile juror annotations")
    juror.add_argument("--run-id", type=str, required=True, dest="run_id")
    juror.add_argument("--input", type=Path, required=True, help="Directory containing packet files")
    juror.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    juror.add_argument("--draft", type=Path, required=True, help="Draft JSONL to validate and compile")

    critic = sub.add_parser("critic", help="Validate and compile critic reviews")
    critic.add_argument("--input", type=Path, required=True, help="Directory containing packet files")
    critic.add_argument("--juror-a", type=Path, required=True, dest="juror_a")
    critic.add_argument("--juror-b", type=Path, required=True, dest="juror_b")
    critic.add_argument("--output", type=Path, required=True)
    critic.add_argument("--draft", type=Path, required=True, help="Draft JSONL to validate and compile")

    judge = sub.add_parser("judge", help="Validate and compile judge labels")
    judge.add_argument("--input", type=Path, required=True, help="Directory containing packet files")
    judge.add_argument("--juror-a", type=Path, required=True, dest="juror_a")
    judge.add_argument("--juror-b", type=Path, required=True, dest="juror_b")
    judge.add_argument("--critic", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--draft", type=Path, required=True, help="Draft JSONL to validate and compile")

    normalize = sub.add_parser("normalize", help="Losslessly normalize Label Studio or canonical JSONL")
    normalize.add_argument("--input", type=Path, required=True)
    normalize.add_argument("--output", type=Path, required=True)
    normalize.add_argument("--label-studio", action="store_true", dest="label_studio")

    validate = sub.add_parser("validate", help="Fail-closed validation of canonical JSONL")
    validate.add_argument("--input", type=Path, required=True)
    validate.add_argument("--packets", type=Path, default=None)
    validate.add_argument("--label-studio", action="store_true", dest="label_studio")

    return parser


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _load_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    records = read_jsonl_records(path)
    index: dict[str, dict[str, Any]] = {}
    for record in records:
        task_id = record.get("task_id")
        if not task_id:
            raise ValueError("record missing task_id")
        if task_id in index:
            raise ValueError(f"duplicate task_id: {task_id}")
        index[task_id] = record
    return index


def _safe_filename(task_id: str) -> str:
    return task_id.replace(":", "_").replace("/", "_")


def cmd_packet(args: argparse.Namespace) -> int:
    records = read_jsonl_records(args.input)
    requested_task_ids = args.task_ids or []
    requested = set(requested_task_ids)
    if len(requested) != len(requested_task_ids):
        raise ValueError("duplicate --task-id")
    packets: list[dict[str, Any]] = []
    observed: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        task_id = record.get("id")
        if requested and task_id not in requested:
            continue
        if not isinstance(task_id, str):
            raise ValueError(f"line {line_number}: external-safe record missing string id")
        if task_id in observed:
            raise ValueError(f"line {line_number}: duplicate selected task_id {task_id}")
        observed.add(task_id)
        assert_no_sampling_or_gold_fields(record)
        try:
            packet = AdjudicationPacket.from_external_safe_record(
                record, provenance=args.provenance
            )
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        packets.append(packet.model_dump(mode="json"))

    missing = requested - observed
    if missing:
        raise ValueError(f"requested external-safe task_id not found: {sorted(missing)[0]}")

    if args.jsonl:
        write_jsonl_records(args.output, packets)
        print(f"Wrote {len(packets)} packets to {args.output}")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
        for packet in packets:
            safe_name = _safe_filename(packet["task_id"])
            path = args.output / f"{safe_name}.json"
            _write_json(path, packet)
        print(f"Wrote {len(packets)} packets to {args.output}/")
    return 0


def cmd_juror(args: argparse.Namespace) -> int:
    packets = load_packets(args.input)
    output_records: list[dict[str, Any]] = []

    drafts = _load_jsonl_index(args.draft)
    for task_id in sorted(packets):
        packet = packets[task_id]
        draft = drafts.get(task_id)
        if draft is None:
            raise ValueError(f"missing draft for task {task_id}")
        if draft.get("juror_run_id") != args.run_id:
            draft["juror_run_id"] = args.run_id
        if draft.get("annotator_id") != args.run_id:
            draft["annotator_id"] = args.run_id
        validated = validate_ai_adjudication_label(draft, packet=packet)
        annotation = JurorAnnotation.model_validate(validated)
        output_records.append(annotation.model_dump(mode="json"))

    write_jsonl_records(args.output, output_records)
    print(f"Wrote {len(output_records)} juror annotations to {args.output}")
    return 0


def cmd_critic(args: argparse.Namespace) -> int:
    packets = load_packets(args.input)
    juror_a_records = _load_jsonl_index(args.juror_a)
    juror_b_records = _load_jsonl_index(args.juror_b)
    output_records: list[dict[str, Any]] = []

    drafts = _load_jsonl_index(args.draft)
    for task_id in sorted(packets):
        packet = packets[task_id]
        juror_a = JurorAnnotation.model_validate(
            validate_ai_adjudication_label(juror_a_records[task_id], packet=packet)
        )
        juror_b = JurorAnnotation.model_validate(
            validate_ai_adjudication_label(juror_b_records[task_id], packet=packet)
        )
        draft = drafts.get(task_id)
        if draft is None:
            raise ValueError(f"missing draft for task {task_id}")
        if draft.get("task_id") != task_id:
            raise ValueError("draft task_id does not match packet")
        expected_run_ids = {
            juror_a.juror_run_id or juror_a.annotator_id,
            juror_b.juror_run_id or juror_b.annotator_id,
        }
        if set(draft.get("juror_run_ids", [])) != expected_run_ids:
            raise ValueError("draft juror_run_ids do not match the supplied juror outputs")
        if draft.get("packet_checksum") != packet.packet_checksum:
            raise ValueError("draft packet_checksum does not match packet")
        review = CriticReview.model_validate(draft)
        output_records.append(review.model_dump(mode="json"))

    write_jsonl_records(args.output, output_records)
    print(f"Wrote {len(output_records)} critic reviews to {args.output}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    packets = load_packets(args.input)
    juror_a_records = _load_jsonl_index(args.juror_a)
    juror_b_records = _load_jsonl_index(args.juror_b)
    critic_records = _load_jsonl_index(args.critic)
    output_records: list[dict[str, Any]] = []

    drafts = _load_jsonl_index(args.draft)
    for task_id in sorted(packets):
        packet = packets[task_id]
        juror_a = JurorAnnotation.model_validate(
            validate_ai_adjudication_label(juror_a_records[task_id], packet=packet)
        )
        juror_b = JurorAnnotation.model_validate(
            validate_ai_adjudication_label(juror_b_records[task_id], packet=packet)
        )
        critic = CriticReview.model_validate(critic_records[task_id])
        if critic.task_id != task_id or critic.packet_checksum != packet.packet_checksum:
            raise ValueError("critic review does not match packet")
        draft = drafts.get(task_id)
        if draft is None:
            raise ValueError(f"missing draft for task {task_id}")
        expected_run_ids = {
            juror_a.juror_run_id or juror_a.annotator_id,
            juror_b.juror_run_id or juror_b.annotator_id,
        }
        if set(draft.get("juror_run_ids", [])) != expected_run_ids:
            raise ValueError("draft juror_run_ids do not match the supplied juror outputs")
        if draft.get("critic_run_id") != critic.critic_run_id:
            raise ValueError("draft critic_run_id does not match critic")
        validated = validate_ai_adjudication_label(draft, packet=packet)
        label = AiAdjudicatedSilverLabel.model_validate(validated)
        output_records.append(label.model_dump(mode="json"))

    write_jsonl_records(args.output, output_records)
    print(f"Wrote {len(output_records)} judge labels to {args.output}")
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    records = read_jsonl_records(args.input)
    normalized: list[dict[str, Any]] = []
    for line_number, record in enumerate(records, start=1):
        original_checksum = canonical_json_checksum(record)
        try:
            validated = validate_ai_adjudication_label(
                record, allow_label_studio_nested=args.label_studio
            )
        except ValueError as exc:
            raise ValueError(f"line {line_number}: {exc}") from exc
        if "original_checksum" not in validated:
            validated["original_checksum"] = original_checksum
        validated["normalization_provenance"] = (
            "normalize:label_studio" if args.label_studio else "normalize:canonical"
        )
        normalized.append(validated)

    write_jsonl_records(args.output, normalized)
    print(f"Wrote {len(normalized)} normalized records to {args.output}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    packets = None
    if args.packets:
        packets = load_packets(args.packets)
    summary = validate_canonical_jsonl(
        args.input,
        packets=packets,
        allow_label_studio_nested=args.label_studio,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "packet":
            return cmd_packet(args)
        if args.command == "juror":
            return cmd_juror(args)
        if args.command == "critic":
            return cmd_critic(args)
        if args.command == "judge":
            return cmd_judge(args)
        if args.command == "normalize":
            return cmd_normalize(args)
        if args.command == "validate":
            return cmd_validate(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
