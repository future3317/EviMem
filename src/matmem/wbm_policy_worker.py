"""Policy-only subprocess worker for secure WBM execution."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from matmem import (  # noqa: E402
    CalibrationUtilityBuilder,
    FacilityLocationCoresetPlanner,
    FixedKernelGPConfig,
    FixedKernelResidualGP,
    FrozenHullDistanceAcquisition,
    HullSnapshot,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    PosteriorUncertaintyAcquisition,
    ProtocolCompatibilityResolver,
    SeededRandomAcquisition,
    SourceProvenance,
    SurvivalConditionedAcquisition,
)
from matmem.wbm_secure import PolicyState  # noqa: E402

POLICY_TIME = datetime(2000, 1, 1, tzinfo=UTC)


def _material_views(
    state: PolicyState,
) -> tuple[tuple[MaterialQuery, ...], tuple[MaterialMemoryCard, ...]]:
    queries = tuple(
        MaterialQuery(
            query_id=item.query_id,
            structure_hash=item.structure_hash,
            identity=MaterialIdentity(
                exact_calculation_id=f"policy:{item.query_id}",
                canonical_structure_id=f"policy:{item.structure_hash}",
                composition_family="-".join(item.chemical_system),
            ),
            composition=item.composition,
            embedding=item.embedding,
            protocol=item.protocol,
            hull_snapshot=HullSnapshot(
                snapshot_id=item.hull_snapshot_id,
                chemical_system=item.chemical_system,
                reference_hull_energy_ev_per_atom=(
                    item.hull_reference_energy_ev_per_atom
                ),
                phase_set_checksum=item.hull_phase_checksum,
                known_through=POLICY_TIME,
                built_at=POLICY_TIME,
                source_version="serialized-policy-view",
            ),
            base_predicted_formation_energy_ev_per_atom=(
                item.frozen_prediction_ev_per_atom
            ),
            stability_threshold_ev_per_atom=(
                item.stability_threshold_ev_per_atom
            ),
            oracle_cost=item.oracle_cost,
            as_of=POLICY_TIME,
        )
        for item in state.queries
    )
    if not queries:
        raise ValueError("policy state contains no queries")
    representative = queries[0]
    witnesses = tuple(
        MaterialMemoryCard(
            card_id=item.witness_id,
            material_id=item.witness_id,
            structure_hash=item.structure_hash,
            identity=MaterialIdentity(
                exact_calculation_id=f"policy:{item.witness_id}",
                canonical_structure_id=f"policy:{item.structure_hash}",
                composition_family="-".join(
                    representative.hull_snapshot.chemical_system
                ),
            ),
            composition=item.composition,
            embedding=item.embedding,
            protocol=item.protocol,
            provenance=SourceProvenance(
                source_name="serialized-policy-state",
                source_version="v1",
                record_locator=item.witness_id,
                retrieved_at=POLICY_TIME,
            ),
            formation_energy_ev_per_atom=item.residual_ev_per_atom,
            base_predicted_formation_energy_ev_per_atom=0.0,
            oracle_residual_ev_per_atom=item.residual_ev_per_atom,
            hull_snapshot=representative.hull_snapshot,
            quality_weight=item.quality_weight,
            observed_at=POLICY_TIME,
        )
        for item in state.witnesses
    )
    return queries, witnesses


def _calibration_policy(state: PolicyState, args: argparse.Namespace) -> str:
    queries, witnesses = _material_views(state)
    resolver = ProtocolCompatibilityResolver()
    config = FixedKernelGPConfig(
        kernel=args.kernel,
        length_scale=args.length_scale,
        signal_std_ev_per_atom=args.signal_std,
        noise_std_ev_per_atom=args.noise_std,
        jitter=args.jitter,
    )
    posterior = FixedKernelResidualGP(resolver, config=config)
    proposal = PosteriorUncertaintyAcquisition(posterior)
    if args.policy == "gp_uncertainty":
        return proposal.rank(queries, witnesses)[0].query_id
    planner = FacilityLocationCoresetPlanner(
        state.active_witness_capacity,
        CalibrationUtilityBuilder(
            FixedKernelResidualGP(resolver, config=config)
        ),
    )
    acquisition = SurvivalConditionedAcquisition(
        proposal,
        posterior,
        planner,
        proposal_size=args.proposal_size,
        num_fantasies=args.num_fantasies,
        survival_weight=args.survival_weight,
        seed=args.seed,
    )
    return acquisition.rank(queries, witnesses)[0].query_id


def _baseline_policy(state: PolicyState, args: argparse.Namespace) -> str:
    queries, witnesses = _material_views(state)
    acquisition = (
        FrozenHullDistanceAcquisition()
        if args.policy == "frozen"
        else SeededRandomAcquisition(args.seed)
    )
    return acquisition.rank(queries, witnesses)[0].query_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        choices=("frozen", "random", "gp_uncertainty", "survival_coreset"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--kernel", choices=("matern52", "rbf"), default="matern52")
    parser.add_argument("--length-scale", type=float, default=1.0)
    parser.add_argument("--signal-std", type=float, default=0.08)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--jitter", type=float, default=1e-10)
    parser.add_argument("--proposal-size", type=int, default=32)
    parser.add_argument("--num-fantasies", type=int, default=8)
    parser.add_argument("--survival-weight", type=float, default=1.0)
    args = parser.parse_args()
    state = PolicyState.model_validate_json(sys.stdin.read())
    if args.policy in {"frozen", "random"}:
        selected = _baseline_policy(state, args)
    else:
        selected = _calibration_policy(state, args)
    sys.stdout.write(selected)


if __name__ == "__main__":
    main()
