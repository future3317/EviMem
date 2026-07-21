"""Audit fixed-composition hull parity before enabling the cached backend.

The audit compares complete Delta-Hull action traces under both backends and
also compares 1,024 deterministic posterior-like energy samples per selected
system.  A single mismatch fails closed; the fixed backend remains opt-in.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from matmem.protocol_knowledge_gradient import (
    FixedCompositionHullTemplate,
    _final_hull_membership,
    _sample_gaussian,
    fixed_composition_hull_membership,
)


def _load_runner():
    path = Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py")
    spec = importlib.util.spec_from_file_location("matpes_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load MatPES runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(
    *,
    task_path: Path,
    vault_path: Path,
    output_dir: Path,
    max_systems: int = 24,
    minimum_candidates: int = 16,
    budget: int = 6,
    sample_count: int = 1024,
    seed: int = 20270720,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir.resolve().is_relative_to(repo_root):
        raise ValueError("parity outputs must remain outside Git")
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = _load_runner()
    common = dict(
        max_systems=max_systems,
        minimum_candidates=minimum_candidates,
        maximum_budget=budget,
        seed=seed,
        posterior_sample_count=sample_count,
        policies=("delta_hull_active_search",),
        split="development",
    )
    results: dict[str, dict[str, Any]] = {}
    for backend in ("pymatgen", "fixed_composition"):
        output = output_dir / f"parity-{backend}.json"
        config = runner.ExperimentConfig(hull_backend=backend, **common)
        runner.run(
            task_path=task_path,
            development_vault_path=vault_path,
            output_path=output,
            config=config,
        )
        results[backend] = json.loads(output.read_text(encoding="utf-8"))
    systems = sorted(set(results["pymatgen"]["systems"]) | set(results["fixed_composition"]["systems"]))
    mismatches: list[dict[str, Any]] = []
    for system in systems:
        left = results["pymatgen"]["systems"][system]["strategies"]["delta_hull_active_search"]
        right = results["fixed_composition"]["systems"][system]["strategies"]["delta_hull_active_search"]
        if left["selected_pair_ids"] != right["selected_pair_ids"]:
            mismatches.append({"system": system, "kind": "action_trace"})

    # Independent state-level parity uses the same fixed composition geometry,
    # but a fresh deterministic sample stream for each system.
    task = json.loads(task_path.read_text(encoding="utf-8"))
    pairs = {system: rows for system, rows in _group_rows(task["development_pairs"]).items()}
    for system in systems:
        rows = pairs[system]
        references = task["development_initial_phase_entries"][system]
        query_compositions = [row["composition"] for row in rows]
        reference_compositions = [row["composition"] for row in references]
        template = FixedCompositionHullTemplate.from_compositions(
            query_compositions=query_compositions,
            reference_compositions=reference_compositions,
        )
        samples = _sample_gaussian(
            np.zeros(len(rows)),
            np.eye(len(rows)) * 0.01,
            sample_count=sample_count,
            seed=seed + 7919 * (systems.index(system) + 1),
        )
        reference_energies = np.zeros(len(references))
        expected = _final_hull_membership(
            query_compositions=query_compositions,
            sampled_query_energies=samples,
            reference_compositions=reference_compositions,
            reference_energies=reference_energies,
        )
        actual = fixed_composition_hull_membership(
            template,
            query_energies=samples,
            reference_energies=reference_energies,
        )
        if not np.array_equal(expected, actual):
            mismatches.append({"system": system, "kind": "sample_membership"})
    manifest = {
        "schema_version": 1,
        "status": "passed" if not mismatches else "failed_closed",
        "task_sha256": _sha256(task_path),
        "vault_sha256": _sha256(vault_path),
        "sample_count": sample_count,
        "seed": seed,
        "systems": systems,
        "mismatches": mismatches,
        "fixed_backend_authorized": not mismatches,
    }
    (output_dir / "fixed-hull-parity-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mismatches:
        raise RuntimeError(f"fixed-composition parity failed: {mismatches[0]}")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def _group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["chemical_system"], []).append(row)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-systems", type=int, default=24)
    parser.add_argument("--minimum-candidates", type=int, default=16)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20270720)
    args = parser.parse_args()
    audit(
        task_path=args.task,
        vault_path=args.vault,
        output_dir=args.output_dir,
        max_systems=args.max_systems,
        minimum_candidates=args.minimum_candidates,
        budget=args.budget,
        sample_count=args.sample_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
