"""Build a fail-closed local gate from independently produced blind candidates.

This tool performs no model calls.  It deliberately does not select a winning
four-axis label.  Its role is to compare already validated blind candidates
locally and route disagreements, insufficient evidence, same-scope
contradictions, and insufficient model diversity to a human/high-risk queue.

Example:

    conda run --no-capture-output -n llm python tools/run_blind_adjudication_gate.py `
      --packets runs/review/packets `
      --candidate runs/deepseek-pro-pass-a/votes/juror-a.jsonl `
      --candidate runs/deepseek-pro-pass-b/votes/juror-a.jsonl `
      --same-model-repeat --output runs/blind-gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from evimem.phase1b.ai_adjudication.blind_gate import (
    BlindGateError,
    build_blind_gate_records,
    load_blind_candidate_export,
    summarize_gate_records,
)
from evimem.phase1b.ai_adjudication.validate import load_packets, write_jsonl_records


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl_records(temporary, records)
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        required=True,
        help="Validated ai_juror JSONL export; specify at least twice.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-distinct-models", type=int, default=3)
    parser.add_argument(
        "--same-model-repeat",
        action="store_true",
        help=(
            "Treat two or more blind calls to one model as a stability check, "
            "never as multi-model consensus."
        ),
    )
    parser.add_argument(
        "--allow-candidate-superset",
        action="store_true",
        help="Locally restrict a larger blind candidate export to the packet task set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(args.candidate) < 2:
        print("ERROR: at least two --candidate paths are required", file=sys.stderr)
        return 1
    if args.min_distinct_models < 2:
        print("ERROR: --min-distinct-models must be at least two", file=sys.stderr)
        return 1
    if args.output.exists() and any(args.output.iterdir()):
        print("ERROR: --output must be new or empty", file=sys.stderr)
        return 1

    try:
        packets = load_packets(args.packets)
        if not packets:
            raise BlindGateError("no packets found")
        candidates = [
            load_blind_candidate_export(
                path, packets, allow_superset=args.allow_candidate_superset
            )
            for path in args.candidate
        ]
        records = build_blind_gate_records(
            packets,
            candidates,
            min_distinct_models=args.min_distinct_models,
            same_model_repeat_mode=args.same_model_repeat,
        )
    except (BlindGateError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    review_queue = [
        record for record in records if record["gate_status"] == "requires_human_review"
    ]
    provisional = [
        record
        for record in records
        if record["gate_status"] == "provisional_multi_model_consensus_candidate"
    ]
    same_model_repeat_consistent = [
        record
        for record in records
        if record["gate_status"] == "same_model_repeat_consistent_candidate"
    ]
    _atomic_write_jsonl(args.output / "gate_records.jsonl", records)
    _atomic_write_jsonl(args.output / "requires_human_review.jsonl", review_queue)
    _atomic_write_jsonl(
        args.output / "provisional_multi_model_consensus.jsonl", provisional
    )
    _atomic_write_jsonl(
        args.output / "same_model_repeat_consistent_candidates.jsonl",
        same_model_repeat_consistent,
    )
    summary = summarize_gate_records(records)
    summary["candidate_sources"] = [
        {
            "name": candidate.source_name,
            "checksum": candidate.source_checksum,
            "model_ids": sorted(
                {record["model_id"] for record in candidate.records.values()}
            ),
        }
        for candidate in candidates
    ]
    summary["min_distinct_models"] = args.min_distinct_models
    summary["same_model_repeat_mode"] = args.same_model_repeat
    _atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
