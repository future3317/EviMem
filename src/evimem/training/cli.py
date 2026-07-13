"""Validation utilities for EviMem data preparation; no dataset downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evimem.benchmark import DatasetRegistry

from .dataset import load_manager_examples_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evimem-data")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-licenses")
    audit.add_argument("--manifest", type=Path, default=Path("configs/datasets.json"))
    validate = subparsers.add_parser("validate-manager-data")
    validate.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-licenses":
        result = DatasetRegistry.load(args.manifest).audit()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["ok"] else 1
    examples = load_manager_examples_jsonl(args.path)
    print(json.dumps({"ok": True, "example_count": len(examples)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
