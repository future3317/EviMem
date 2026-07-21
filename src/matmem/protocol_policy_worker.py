"""Subprocess-only acquisition policies for the secure protocol runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

import numpy as np
from chic import (
    linear_ridge_hull_influence_acquisition,
    linear_ridge_predicted_final_hull_acquisition,
)
from protocol_knowledge_gradient import (
    FixedCompositionHullTemplate,
    FrozenProtocolRidgeTransport,
    delta_hull_active_search,
    protocol_hull_knowledge_gradient,
    protocol_hull_risk_reduction,
    protocol_target_energy_posterior,
)


def _source_offset(history: list[dict[str, object]]) -> float:
    if not history:
        return 0.0
    deltas = [
        float(row["revealed_target_formation_energy_ev_per_atom"])
        - float(row["source_formation_energy_ev_per_atom"])
        for row in history
    ]
    return float(np.median(np.asarray(deltas, dtype=float)))


def _source_affine(history: list[dict[str, object]]) -> tuple[float, float]:
    if len(history) < 2:
        return 1.0, _source_offset(history)
    source = np.asarray(
        [float(row["source_formation_energy_ev_per_atom"]) for row in history],
        dtype=float,
    )
    target = np.asarray(
        [float(row["revealed_target_formation_energy_ev_per_atom"]) for row in history],
        dtype=float,
    )
    if float(np.ptp(source)) <= 1e-12:
        return 1.0, _source_offset(history)
    design = np.column_stack((source, np.ones(len(source), dtype=float)))
    slope, intercept = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(slope), float(intercept)


def _composition_matrix(
    rows: list[dict[str, object]], elements: tuple[str, ...]
) -> np.ndarray:
    return np.asarray(
        [
            [float(dict(row["composition"]).get(element, 0.0)) for element in elements]
            for row in rows
        ],
        dtype=float,
    )


def select(
    payload: dict[str, object],
    *,
    policy: str,
    seed: int,
    ridge_penalty: float = 1.0,
    prior_standard_deviation: float = 0.1,
    boundary_temperature: float = 0.05,
    transport_model: FrozenProtocolRidgeTransport | None = None,
    posterior_sample_count: int = 16,
    fantasy_count: int = 3,
    hull_backend: str = "pymatgen",
) -> str:
    queries = list(payload["queries"])
    history = list(payload["revealed_history"])
    if not queries:
        raise ValueError("protocol policy received no legal queries")
    if policy == "random":
        return min(
            queries,
            key=lambda row: hashlib.sha256(
                f"{seed}|{payload['round_index']}|{row['pair_id']}".encode()
            ).hexdigest(),
        )["pair_id"]
    if policy in {
        "ridge_margin",
        "ridge_uncertainty",
        "chic_hull_influence",
        "ridge_predicted_final_margin",
        "delta_hull_active_search",
        "protocol_hull_knowledge_gradient",
        "protocol_hull_risk_reduction",
    }:
        query_features = np.asarray(
            [row["source_environment_embedding"] for row in queries], dtype=float
        )
        history_features = np.asarray(
            [row["source_environment_embedding"] for row in history], dtype=float
        ).reshape(len(history), query_features.shape[1])
        arguments = dict(
            query_features=query_features,
            query_source_energies=np.asarray(
                [row["source_formation_energy_ev_per_atom"] for row in queries]
            ),
            current_competing_hull_energies=np.asarray(
                [row["current_competing_hull_ev_per_atom"] for row in queries]
            ),
            history_features=history_features,
            history_source_energies=np.asarray(
                [row["source_formation_energy_ev_per_atom"] for row in history]
            ),
            history_target_energies=np.asarray(
                [row["revealed_target_formation_energy_ev_per_atom"] for row in history]
            ),
            costs=np.asarray([row["oracle_cost"] for row in queries]),
            ridge_penalty=ridge_penalty,
            prior_standard_deviation=prior_standard_deviation,
            boundary_temperature=boundary_temperature,
        )
        if policy == "delta_hull_active_search" or policy.startswith("protocol_hull_"):
            if transport_model is None:
                raise ValueError("protocol hull policy has no frozen transport model")
            query_elements = set(queries[0]["chemical_system"])
            if not query_elements <= set(transport_model.fit_element_ids):
                values = tuple(
                    float(row["current_competing_hull_ev_per_atom"])
                    - float(row["source_formation_energy_ev_per_atom"])
                    for row in queries
                )
            else:
                phases = list(payload["causal_hull_phases"])
                if hull_backend not in {"pymatgen", "fixed_composition"}:
                    raise ValueError("unknown protocol hull backend")
                fixed_template = (
                    FixedCompositionHullTemplate.from_compositions(
                        query_compositions=tuple(dict(row["composition"]) for row in queries),
                        reference_compositions=tuple(dict(row["composition"]) for row in phases),
                    )
                    if hull_backend == "fixed_composition"
                    else None
                )
                kernel_dimension = len(transport_model.kernel_feature_mean)
                query_kernel_rows = [
                    row.get("source_local_environment_embedding") for row in queries
                ]
                history_kernel_rows = [
                    row.get("source_local_environment_embedding") for row in history
                ]
                if (
                    transport_model.local_kernel == "matern52"
                    and (
                        kernel_dimension == 0
                        or any(row is None for row in query_kernel_rows)
                        or any(row is None for row in history_kernel_rows)
                    )
                ):
                    raise ValueError(
                        "protocol hull policy requires frozen local-environment embeddings"
                    )
                posterior = protocol_target_energy_posterior(
                    transport_model,
                    query_features=query_features,
                    query_source_energies=arguments["query_source_energies"],
                    history_features=history_features,
                    history_source_energies=arguments["history_source_energies"],
                    history_target_energies=arguments["history_target_energies"],
                    query_kernel_features=(
                        None
                        if transport_model.local_kernel == "independent"
                        else np.asarray(query_kernel_rows, dtype=float)
                    ),
                    history_kernel_features=(
                        None
                        if transport_model.local_kernel == "independent"
                        else np.asarray(history_kernel_rows, dtype=float).reshape(
                            len(history_kernel_rows), kernel_dimension
                        )
                    ),
                )
                hull_arguments = dict(
                    query_compositions=tuple(
                        dict(row["composition"]) for row in queries
                    ),
                    reference_compositions=tuple(
                        dict(row["composition"]) for row in phases
                    ),
                    reference_energies=np.asarray(
                        [row["formation_energy_ev_per_atom"] for row in phases]
                    ),
                    costs=arguments["costs"],
                    posterior_sample_count=posterior_sample_count,
                    fantasy_count=fantasy_count,
                    seed=seed + 1009 * int(payload["round_index"]),
                )
                if policy == "delta_hull_active_search":
                    result = delta_hull_active_search(
                        posterior,
                        query_compositions=hull_arguments["query_compositions"],
                        reference_compositions=hull_arguments["reference_compositions"],
                        reference_energies=hull_arguments["reference_energies"],
                        costs=hull_arguments["costs"],
                        posterior_sample_count=hull_arguments["posterior_sample_count"],
                        seed=hull_arguments["seed"],
                        fixed_template=fixed_template,
                    )
                    values = result.scores
                elif policy == "protocol_hull_risk_reduction":
                    result = protocol_hull_risk_reduction(
                        posterior,
                        **hull_arguments,
                    )
                    values = result.scores
                else:
                    result = protocol_hull_knowledge_gradient(
                        posterior,
                        remaining_budget=float(payload["remaining_budget"]),
                        **hull_arguments,
                    )
                    values = result.scores
        elif policy == "ridge_predicted_final_margin":
            phases = list(payload["causal_hull_phases"])
            elements = tuple(queries[0]["chemical_system"])
            result = linear_ridge_predicted_final_hull_acquisition(
                query_features=query_features,
                query_source_energies=arguments["query_source_energies"],
                query_compositions=_composition_matrix(queries, elements),
                reference_compositions=_composition_matrix(phases, elements),
                reference_energies=np.asarray(
                    [row["formation_energy_ev_per_atom"] for row in phases]
                ),
                history_features=history_features,
                history_source_energies=arguments["history_source_energies"],
                history_target_energies=arguments["history_target_energies"],
                costs=arguments["costs"],
                ridge_penalty=ridge_penalty,
                prior_standard_deviation=prior_standard_deviation,
            )
            values = result.scores
        else:
            result = linear_ridge_hull_influence_acquisition(**arguments)
            if policy == "ridge_uncertainty":
                values = result.predictive_standard_deviations
            elif policy == "ridge_margin":
                values = tuple(
                    float(row["current_competing_hull_ev_per_atom"]) - prediction
                    for row, prediction in zip(
                        queries, result.predicted_target_energies, strict=True
                    )
                )
            else:
                values = result.scores
        return min(
            zip(queries, values, strict=True),
            key=lambda item: (-item[1], item[0]["pair_id"]),
        )[0]["pair_id"]
    if policy == "source_online_affine":
        slope, intercept = _source_affine(history)
    else:
        slope = 1.0
        intercept = _source_offset(history) if policy == "source_online_offset" else 0.0
    return min(
        queries,
        key=lambda row: (
            slope * float(row["source_formation_energy_ev_per_atom"])
            + intercept
            - float(row["current_competing_hull_ev_per_atom"]),
            row["pair_id"],
        ),
    )["pair_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=(
            "source_margin",
            "random",
            "source_online_offset",
            "source_online_affine",
            "ridge_margin",
            "ridge_uncertainty",
            "chic_hull_influence",
            "ridge_predicted_final_margin",
            "delta_hull_active_search",
            "protocol_hull_knowledge_gradient",
            "protocol_hull_risk_reduction",
        ),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-penalty", type=float, default=1.0)
    parser.add_argument("--prior-standard-deviation", type=float, default=0.1)
    parser.add_argument("--boundary-temperature", type=float, default=0.05)
    parser.add_argument("--posterior-sample-count", type=int, default=16)
    parser.add_argument("--fantasy-count", type=int, default=3)
    parser.add_argument("--hull-backend", choices=("pymatgen", "fixed_composition"), default="pymatgen")
    parser.add_argument("--serve-jsonl", action="store_true")
    args = parser.parse_args()
    def respond(payload: dict[str, object]) -> None:
        model_payload = payload.pop("transport_model", None)
        transport_model = (
            None
            if model_payload is None
            else FrozenProtocolRidgeTransport.model_validate(model_payload)
        )
        print(
            select(
                payload,
                policy=args.policy,
                seed=args.seed,
                ridge_penalty=args.ridge_penalty,
                prior_standard_deviation=args.prior_standard_deviation,
                boundary_temperature=args.boundary_temperature,
                transport_model=transport_model,
                posterior_sample_count=args.posterior_sample_count,
                fantasy_count=args.fantasy_count,
                hull_backend=args.hull_backend,
            ),
            flush=True,
        )

    if args.serve_jsonl:
        for line in sys.stdin:
            if line.strip():
                respond(json.loads(line))
    else:
        respond(json.load(sys.stdin))


if __name__ == "__main__":
    main()
