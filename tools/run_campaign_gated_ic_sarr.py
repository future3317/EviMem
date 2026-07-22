"""Run a small data-backed Campaign-Gated IC-SARR development pilot.

This runner consumes observable task/model fields only. It samples the frozen
posterior and never reads a target oracle vault. Outputs are development smoke
artifacts and must remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from matmem.campaign_gate import campaign_gated_ic_sarr
from matmem.protocol_knowledge_gradient import (
    FrozenProtocolRidgeTransport,
    protocol_target_energy_posterior,
)

OPENED_ATTRIBUTION_SYSTEMS = {
    "Ag-S",
    "Al-O-P",
    "B-Fe-Li-Mn-O",
    "B-Li",
    "Ba-Mg-Mn-O",
    "Ba-Mn-O",
    "Bi-Li-O-P",
    "C-Ca",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(
    *,
    task_path: Path,
    transport_path: Path,
    output_path: Path,
    systems: tuple[str, ...] = (),
    max_systems: int = 2,
    budget: int = 6,
    outer_sample_count: int = 16,
    inner_stage_one_sample_count: int = 32,
    inner_stage_two_sample_count: int = 64,
    sobol_scramble_count: int = 4,
    seed: int = 20270722,
) -> dict[str, object]:
    if output_path.exists():
        raise FileExistsError(output_path)
    task = json.loads(task_path.read_text(encoding="utf-8"))
    transport_payload = json.loads(transport_path.read_text(encoding="utf-8"))
    transport = FrozenProtocolRidgeTransport.model_validate(
        transport_payload.get("model", transport_payload)
    )
    rows = list(task["development_pairs"])
    by_system: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_system.setdefault(str(row["chemical_system"]), []).append(row)
    selected_systems = tuple(systems) if systems else tuple(
        system
        for system in sorted(by_system)
        if system not in OPENED_ATTRIBUTION_SYSTEMS
    )[:max_systems]
    if not selected_systems:
        raise ValueError("no development systems selected")
    records: list[dict[str, object]] = []
    for system_index, system in enumerate(selected_systems):
        system_rows = sorted(by_system[system], key=lambda row: str(row["pair_id"]))
        if len(system_rows) < budget:
            raise ValueError(f"system {system} has fewer candidates than budget")
        query_compositions = tuple(dict(row["composition"]) for row in system_rows)
        query_source = np.asarray(
            [float(row["source_formation_energy_ev_per_atom"]) for row in system_rows],
            dtype=float,
        )
        query_ids = tuple(str(row["pair_id"]) for row in system_rows)
        query_features = np.asarray(
            [row["source_environment_embedding"] for row in system_rows], dtype=float
        )
        query_kernel = np.asarray(
            [row["source_local_environment_embedding"] for row in system_rows], dtype=float
        )
        reference_compositions = tuple(
            dict(entry["composition"])
            for entry in task["development_initial_phase_entries"][system]
        )
        reference_energies = np.asarray(
            [
                float(entry["corrected_total_energy_ev"])
                / float(sum(entry["composition"].values()))
                for entry in task["development_initial_phase_entries"][system]
            ],
            dtype=float,
        )
        empty_features = np.empty((0, query_features.shape[1]), dtype=float)
        empty_kernel = np.empty((0, query_kernel.shape[1]), dtype=float)
        posterior = protocol_target_energy_posterior(
            transport,
            query_features=query_features,
            query_source_energies=query_source,
            history_features=empty_features,
            history_source_energies=np.empty(0, dtype=float),
            history_target_energies=np.empty(0, dtype=float),
            query_kernel_features=query_kernel,
            history_kernel_features=empty_kernel,
        )
        result = campaign_gated_ic_sarr(
            posterior_mean=np.asarray(posterior.mean),
            posterior_covariance=np.asarray(posterior.covariance),
            model=transport,
            query_compositions=query_compositions,
            query_source_energies=query_source,
            query_ids=query_ids,
            query_features=query_features,
            query_kernel_features=query_kernel,
            reference_compositions=reference_compositions,
            reference_energies=reference_energies,
            budget=budget,
            outer_sample_count=outer_sample_count,
            outer_seed=seed + 1009 * system_index,
            inner_stage_one_sample_count=inner_stage_one_sample_count,
            inner_stage_two_sample_count=inner_stage_two_sample_count,
            sobol_scramble_count=sobol_scramble_count,
        )
        records.append(
            {
                "system": system,
                "candidate_count": len(system_rows),
                "selected_policy": result.selected_policy,
                "terminal_advantage": result.terminal_advantage,
                "selected_history_advantage": result.selected_history_advantage,
                "terminal_lower_bound": result.terminal_lower_bound,
                "selected_history_lower_bound": result.selected_history_lower_bound,
                "source_terminal_value": result.source_terminal_value,
                "ic_terminal_value": result.ic_terminal_value,
                "source_selected_history_value": result.source_selected_history_value,
                "ic_selected_history_value": result.ic_selected_history_value,
                "outer_sample_count": outer_sample_count,
                "inner_stage_one_sample_count": inner_stage_one_sample_count,
                "inner_stage_two_sample_count": inner_stage_two_sample_count,
            }
        )
    payload = {
        "schema_version": 1,
        "status": "development_smoke_only",
        "method": "campaign_gated_ic_sarr",
        "task_sha256": _sha256(task_path),
        "transport_sha256": _sha256(transport_path),
        "systems": selected_systems,
        "budget": budget,
        "seed": seed,
        "records": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--transport-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--system", action="append", dest="systems", default=[])
    parser.add_argument("--max-systems", type=int, default=2)
    parser.add_argument("--budget", type=int, default=6)
    parser.add_argument("--outer-sample-count", type=int, default=16)
    parser.add_argument("--inner-stage-one-sample-count", type=int, default=32)
    parser.add_argument("--inner-stage-two-sample-count", type=int, default=64)
    parser.add_argument("--sobol-scramble-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20270722)
    args = parser.parse_args()
    payload = run(
        task_path=args.task,
        transport_path=args.transport_model,
        output_path=args.output,
        systems=tuple(args.systems),
        max_systems=args.max_systems,
        budget=args.budget,
        outer_sample_count=args.outer_sample_count,
        inner_stage_one_sample_count=args.inner_stage_one_sample_count,
        inner_stage_two_sample_count=args.inner_stage_two_sample_count,
        sobol_scramble_count=args.sobol_scramble_count,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
