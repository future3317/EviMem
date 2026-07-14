"""File-level validation helpers for AI-adjudicated annotation artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evimem.phase1b.ai_adjudication.schema import (
    AdjudicationPacket,
    CriticReview,
    canonical_json_checksum,
    validate_ai_adjudication_label,
)


def read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"line {line_number} is not a JSON object")
            records.append(record)
    return records


def write_jsonl_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_packets(packet_dir: Path) -> dict[str, AdjudicationPacket]:
    """Load all packet files from a directory, keyed by task_id."""
    packets: dict[str, AdjudicationPacket] = {}
    if packet_dir.is_file():
        if packet_dir.suffix == ".jsonl":
            records = read_jsonl_records(packet_dir)
        else:
            records = [json.loads(packet_dir.read_text(encoding="utf-8"))]
    else:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(packet_dir.glob("*.json"))
        ]
    for record in records:
        packet = AdjudicationPacket.model_validate(record)
        if packet.task_id in packets:
            raise ValueError(f"duplicate task_id in packets: {packet.task_id}")
        packets[packet.task_id] = packet
    return packets


def validate_canonical_jsonl(
    path: Path,
    *,
    packets: dict[str, AdjudicationPacket] | None = None,
    allow_label_studio_nested: bool = False,
) -> dict[str, Any]:
    """Validate a canonical JSONL file fail-closed.

    Returns a summary dict with counts and task_ids. Raises ValueError on the
    first violation.
    """
    records = read_jsonl_records(path)
    if not records:
        raise ValueError("no records found")

    task_ids: list[str] = []
    seen: set[str] = set()
    for line_number, record in enumerate(records, start=1):
        if allow_label_studio_nested:
            data = record.get("data", {})
            task_id = record.get("id") or (
                data.get("pair_id") if isinstance(data, dict) else None
            )
        else:
            task_id = record.get("task_id")
        if not task_id:
            raise ValueError(f"line {line_number}: missing task_id")
        if task_id in seen:
            raise ValueError(f"line {line_number}: duplicate task_id {task_id}")
        seen.add(task_id)
        task_ids.append(task_id)

        packet = packets.get(task_id) if packets else None
        if packets is not None and packet is None:
            raise ValueError(f"line {line_number}: task {task_id} is not in the supplied packets")
        try:
            if "issues" in record:
                review = CriticReview.model_validate(record)
                if packet is not None:
                    if review.task_id != packet.task_id:
                        raise ValueError("critic task_id does not match supplied packet")
                    if review.packet_checksum != packet.packet_checksum:
                        raise ValueError("critic packet_checksum does not match supplied packet")
            else:
                validate_ai_adjudication_label(
                    record,
                    packet=packet,
                    allow_label_studio_nested=allow_label_studio_nested,
                )
        except ValueError as exc:
            raise ValueError(f"line {line_number} (task {task_id}): {exc}") from exc

    return {
        "record_count": len(records),
        "unique_task_ids": len(seen),
        "task_ids": task_ids,
        "source_path": str(path),
    }


def validate_packet_directory(packet_dir: Path) -> dict[str, Any]:
    """Validate all packet files in a directory."""
    packets = load_packets(packet_dir)
    if not packets:
        raise ValueError("no packets found")
    return {
        "packet_count": len(packets),
        "task_ids": sorted(packets.keys()),
        "source_path": str(packet_dir),
    }


def assert_no_sampling_or_gold_fields(record: dict[str, Any]) -> None:
    """Assert that an external-safe record contains no sampling/gold/native metadata."""
    forbidden = {
        "sampling_stratum_not_gold",
        "candidate_is_gold",
        "native_support",
        "native_contradict",
        "compiled_operation",
        "update_operation",
        "operation",
    }
    keys = set(record.keys())
    if "meta" in record and isinstance(record["meta"], dict):
        keys |= set(record["meta"].keys())
    if "data" in record and isinstance(record["data"], dict):
        keys |= set(record["data"].keys())
    violations = forbidden & keys
    if violations:
        raise ValueError(f"forbidden sampling/gold/operation fields present: {sorted(violations)}")


def build_record_checksum(record: dict[str, Any]) -> str:
    """Compute checksum for a canonical record, excluding volatile provenance fields."""
    canonical = {k: v for k, v in record.items() if k not in {"annotation_provenance", "annotator_id"}}
    return canonical_json_checksum(canonical)
