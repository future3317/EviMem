"""Measure independent Sobol-scramble stability on development systems."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_runner():
    path = Path(__file__).with_name("run_matpes_protocol_closed_loop_exploratory.py")
    spec = importlib.util.spec_from_file_location("matpes_runner", path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load MatPES runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def audit(
    *,
    task_path: Path,
    vault_path: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = (11, 29, 47, 71),
    max_systems: int = 24,
    minimum_candidates: int = 16,
    budget: int = 6,
    sample_count: int = 1024,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[1]
    if output_dir.resolve().is_relative_to(repo_root):
        raise ValueError("seed audit outputs must remain outside Git")
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("seed list must be nonempty and unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = _load_runner()
    traces: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        output = output_dir / f"seed-{seed}.json"
        runner.run(
            task_path=task_path,
            development_vault_path=vault_path,
            output_path=output,
            config=runner.ExperimentConfig(
                max_systems=max_systems,
                minimum_candidates=minimum_candidates,
                maximum_budget=budget,
                seed=seed,
                posterior_sample_count=sample_count,
                policies=("delta_hull_active_search",),
            ),
        )
        traces[seed] = json.loads(output.read_text(encoding="utf-8"))
    systems = sorted(traces[seeds[0]]["systems"])
    per_system: dict[str, Any] = {}
    for system in systems:
        selected = {
            str(seed): traces[seed]["systems"][system]["strategies"]["delta_hull_active_search"][
                "selected_pair_ids"
            ]
            for seed in seeds
        }
        per_system[system] = {
            "first_action_agreement_fraction": sum(
                selected[str(seeds[0])][0] == selected[str(seed)][0] for seed in seeds[1:]
            )
            / max(len(seeds) - 1, 1),
            "full_trace_agreement_fraction": sum(
                selected[str(seeds[0])] == selected[str(seed)] for seed in seeds[1:]
            )
            / max(len(seeds) - 1, 1),
            "selected_pair_ids_by_seed": selected,
            "oracle_discoveries_by_seed": {
                str(seed): traces[seed]["systems"][system]["strategies"][
                    "delta_hull_active_search"
                ]["oracle_pool_confirmed_discoveries"]
                for seed in seeds
            },
        }
    report = {
        "schema_version": 1,
        "status": "development_seed_diagnostic",
        "seeds": list(seeds),
        "sample_count": sample_count,
        "systems": per_system,
        "evaluation_systems_accessed": False,
    }
    (output_dir / "seed-stability-manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--vault", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 29, 47, 71))
    parser.add_argument("--max-systems", type=int, default=24)
    parser.add_argument("--minimum-candidates", type=int, default=16)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--sample-count", type=int, default=1024)
    args = parser.parse_args()
    audit(
        task_path=args.task,
        vault_path=args.vault,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        max_systems=args.max_systems,
        minimum_candidates=args.minimum_candidates,
        budget=args.budget,
        sample_count=args.sample_count,
    )


if __name__ == "__main__":
    main()
