"""Freeze an outcome-independent MAD-1.5 acquisition-curve system manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SELECTION_NAMESPACE = "MAD-1.5-curve-protocol-v1"
STRATA = ("binary", "ternary", "quaternary_or_higher")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


def _stratum(system: str) -> str:
    count = len(system.split("-"))
    return "binary" if count == 2 else "ternary" if count == 3 else "quaternary_or_higher"


def _opened_systems(payload: dict[str, Any]) -> set[str]:
    systems = payload.get("query_systems", payload.get("development_systems"))
    if systems is None:
        systems = list(payload.get("systems", {}))
    return {str(system) for system in systems}


def build(
    *,
    task_path: Path,
    opened_result_paths: tuple[Path, ...],
    output_path: Path,
    system_count: int = 96,
    fold_count: int = 6,
    minimum_candidates: int = 8,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    if system_count < fold_count or system_count % fold_count:
        raise ValueError("system_count must be divisible by fold_count")
    if fold_count < 2 or minimum_candidates < 6:
        raise ValueError("invalid curve manifest dimensions")
    task = json.loads(task_path.read_text(encoding="utf-8"))
    opened_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in opened_result_paths]
    opened_by_source = {
        str(path): sorted(_opened_systems(payload))
        for path, payload in zip(opened_result_paths, opened_payloads, strict=True)
    }
    opened = set().union(*(set(values) for values in opened_by_source.values()))
    rows = list(task["development_pairs"])
    by_system: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_system.setdefault(str(row["chemical_system"]), []).append(row)
    eligible = {
        system
        for system, system_rows in by_system.items()
        if len(system_rows) >= minimum_candidates
        and len({tuple(sorted(row["composition"].items())) for row in system_rows}) >= 3
    }
    available = sorted(
        eligible - opened,
        key=lambda system: _stable_hash(str(task["release_id"]), SELECTION_NAMESPACE, system),
    )
    if len(available) < system_count:
        raise ValueError("not enough unopened eligible systems for the curve manifest")
    selected = available[:system_count]
    folds: list[list[str]] = [[] for _ in range(fold_count)]
    for stratum in STRATA:
        systems = sorted(
            (system for system in selected if _stratum(system) == stratum),
            key=lambda system: _stable_hash(SELECTION_NAMESPACE, "fold", stratum, system),
        )
        for index, system in enumerate(systems):
            folds[index % fold_count].append(system)
    fold_payload = [
        {
            "fold_index": index,
            "query_systems": sorted(systems),
            "system_count": len(systems),
            "stratum_counts": {stratum: sum(_stratum(system) == stratum for system in systems) for stratum in STRATA},
        }
        for index, systems in enumerate(folds)
    ]
    if set().union(*(set(fold["query_systems"]) for fold in fold_payload)) != set(selected):
        raise AssertionError("curve folds do not cover the selected systems exactly")
    payload = {
        "schema_version": 1,
        "status": "mad15_curve_system_manifest_frozen",
        "task_sha256": _sha256(task_path),
        "release_id": task["release_id"],
        "selection_namespace": SELECTION_NAMESPACE,
        "opened_result_sha256": {str(path): _sha256(path) for path in opened_result_paths},
        "opened_systems_by_source": opened_by_source,
        "opened_system_count_union": len(opened),
        "assignment_uses_target_outcomes": False,
        "selection_rule": (
            "eligible exact chemical systems with >= minimum_candidates and >= 3 compositions; "
            "exclude the union of supplied opened manifests; sort by SHA256(release_id || "
            "selection_namespace || exact_system) and take the first system_count"
        ),
        "minimum_candidates": minimum_candidates,
        "system_count": system_count,
        "selected_systems": selected,
        "fold_count": fold_count,
        "fold_rule": "within each element-count stratum, sort SHA256(selection_namespace || fold || stratum || exact_system), then round-robin",
        "folds": fold_payload,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--opened-result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system-count", type=int, default=96)
    parser.add_argument("--fold-count", type=int, default=6)
    parser.add_argument("--minimum-candidates", type=int, default=8)
    args = parser.parse_args()
    payload = build(
        task_path=args.task,
        opened_result_paths=tuple(args.opened_result),
        output_path=args.output,
        system_count=args.system_count,
        fold_count=args.fold_count,
        minimum_candidates=args.minimum_candidates,
    )
    print(json.dumps({key: payload[key] for key in ("system_count", "fold_count", "opened_system_count_union")}, indent=2))


if __name__ == "__main__":
    main()
