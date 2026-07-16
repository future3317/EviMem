"""Run a deterministic synthetic falsification pilot for dual-budget MatMem.

This script downloads no data and writes no experiment artifact. It prints
aggregate mechanism results for recurring residual structure and two negative
controls. The synthetic pilot is not a materials benchmark result.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from evimem.matmem import (
    ActiveDiscoveryEvaluator,
    BaseBoundaryAcquisition,
    BoundaryRiskConfig,
    BoundaryRiskPotential,
    BoundaryRiskRetention,
    BoundaryUncertaintyAcquisition,
    CardOracleVault,
    CompatibilityKind,
    DeterministicReservoirMemory,
    DiversityBoundedMemory,
    FIFOBoundedMemory,
    FullHistoryMemory,
    HullSnapshot,
    LegacyTwoScenarioAcquisition,
    MaterialIdentity,
    MaterialMemoryCard,
    MaterialQuery,
    OnDemandKNNArchiveAcquisition,
    ProtocolAwareBoundaryAcquisition,
    ProtocolCertificate,
    ProtocolCompatibility,
    ProtocolCompatibilityResolver,
    ResidualPriorityMemory,
    SeededRandomAcquisition,
    SourceProvenance,
    SyntheticMinHullEngine,
)

START = datetime(2026, 1, 1, tzinfo=UTC)


@dataclass(frozen=True)
class SyntheticCase:
    query: MaterialQuery
    oracle_card: MaterialMemoryCard


def oracle_vault(candidates: list[SyntheticCase]) -> CardOracleVault:
    return CardOracleVault(
        {candidate.query.query_id: candidate.oracle_card for candidate in candidates}
    )


def evaluate_candidates(
    evaluator: ActiveDiscoveryEvaluator,
    candidates: list[SyntheticCase],
):
    return evaluator.evaluate(
        (candidate.query for candidate in candidates),
        oracle_vault(candidates),
    )


def protocol(functional: str = "PBE") -> ProtocolCertificate:
    return ProtocolCertificate(
        functional=functional,
        pseudopotential_set="SYNTHETIC-PAW",
        correction_scheme="none",
        relaxation_protocol="synthetic-fixed",
        calculation_code="synthetic-oracle",
    )


def snapshot() -> HullSnapshot:
    return HullSnapshot(
        snapshot_id="synthetic-hull-v1",
        chemical_system=("A", "B"),
        reference_hull_energy_ev_per_atom=-1.0,
        phase_set_checksum="sha256:" + "d" * 64,
        known_through=START,
        built_at=START,
        source_version="synthetic-v1",
    )


def item(
    query_id: str,
    *,
    embedding: tuple[float, ...],
    base_energy: float,
    oracle_energy: float,
    selected_protocol: ProtocolCertificate | None = None,
) -> SyntheticCase:
    current_protocol = selected_protocol or protocol()
    identity = MaterialIdentity(
        exact_calculation_id=f"calc-{query_id}",
        canonical_structure_id=f"canonical-{query_id}",
        composition_family="A-B",
        prototype_family=query_id.split("-")[0],
    )
    current_snapshot = snapshot()
    query = MaterialQuery(
        query_id=query_id,
        structure_hash=f"structure-{query_id}",
        identity=identity,
        composition="AB",
        embedding=embedding,
        protocol=current_protocol,
        hull_snapshot=current_snapshot,
        base_predicted_formation_energy_ev_per_atom=base_energy,
        as_of=START + timedelta(days=1),
    )
    card = MaterialMemoryCard(
        card_id=f"card-{query_id}",
        material_id=f"material-{query_id}",
        structure_hash=query.structure_hash,
        identity=identity,
        composition="AB",
        embedding=embedding,
        protocol=current_protocol,
        provenance=SourceProvenance(
            source_name="dual-budget-synthetic-pilot",
            source_version="v1",
            record_locator=query_id,
        ),
        formation_energy_ev_per_atom=oracle_energy,
        base_predicted_formation_energy_ev_per_atom=base_energy,
        oracle_residual_ev_per_atom=oracle_energy - base_energy,
        hull_snapshot=current_snapshot,
        recorded_hull_distance_ev_per_atom=oracle_energy + 1.0,
        observed_at=START + timedelta(days=2),
    )
    return SyntheticCase(query=query, oracle_card=card)


def recurring_pool(seed: int, size: int) -> list[SyntheticCase]:
    """Residuals recur by structure cluster; memory should be useful."""

    rng = random.Random(seed)
    false_count = size // 2
    stable_count = size - false_count
    candidates = [
        item(
            f"false-{seed}-{index:03d}",
            embedding=(1.0, 0.0, 0.0),
            base_energy=-1.055 + rng.uniform(-0.003, 0.003),
            oracle_energy=-0.94 + rng.uniform(-0.004, 0.004),
        )
        for index in range(false_count)
    ]
    candidates.extend(
        item(
            f"stable-{seed}-{index:03d}",
            embedding=(0.0, 1.0, 0.0),
            base_energy=-1.035 + rng.uniform(-0.003, 0.003),
            oracle_energy=-1.20 + rng.uniform(-0.004, 0.004),
        )
        for index in range(stable_count)
    )
    rng.shuffle(candidates)
    return candidates


def local_boundary_pool(seed: int, size: int) -> list[SyntheticCase]:
    """A smooth local residual field around the hull boundary."""

    rng = random.Random(seed)
    candidates = []
    for index in range(size):
        angle = 2 * math.pi * index / size
        embedding = (math.cos(angle), math.sin(angle), 0.2)
        base_energy = -1.015 + 0.01 * math.cos(angle) + rng.uniform(-0.002, 0.002)
        residual = 0.055 * math.sin(angle) + rng.uniform(-0.004, 0.004)
        candidates.append(
            item(
                f"local-{seed}-{index:03d}",
                embedding=embedding,
                base_energy=base_energy,
                oracle_energy=base_energy + residual,
            )
        )
    rng.shuffle(candidates)
    return candidates


def iid_pool(seed: int, size: int) -> list[SyntheticCase]:
    """Residual sign is independent of recurring embeddings."""

    rng = random.Random(seed)
    candidates = []
    for index in range(size):
        embedding = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))[index % 2]
        base_energy = -1.045 + rng.uniform(-0.01, 0.01)
        stable = rng.random() < 0.5
        oracle_energy = (-1.04 if stable else -0.96) + rng.uniform(-0.005, 0.005)
        candidates.append(
            item(
                f"iid-{seed}-{index:03d}",
                embedding=embedding,
                base_energy=base_energy,
                oracle_energy=oracle_energy,
            )
        )
    rng.shuffle(candidates)
    return candidates


def nonrecurring_pool(seed: int, size: int) -> list[SyntheticCase]:
    """Every candidate is orthogonal, so past residuals cannot transfer."""

    rng = random.Random(seed)
    candidates = []
    for index in range(size):
        embedding = tuple(float(position == index) for position in range(size))
        base_energy = -1.05 + rng.uniform(-0.005, 0.005)
        oracle_energy = (-1.04 if index % 2 else -0.96) + rng.uniform(-0.003, 0.003)
        candidates.append(
            item(
                f"unique-{seed}-{index:03d}",
                embedding=embedding,
                base_energy=base_energy,
                oracle_energy=oracle_energy,
            )
        )
    rng.shuffle(candidates)
    return candidates


def hull_revision_pool(seed: int, size: int) -> list[SyntheticCase]:
    """One newly observed deep phase revises the causal hull for later calls."""

    rng = random.Random(seed)
    candidates = [
        item(
            f"deep-{seed}",
            embedding=(1.0, 0.0, 0.0),
            base_energy=-1.18,
            oracle_energy=-1.20,
        )
    ]
    for index in range(size - 1):
        candidates.append(
            item(
                f"revision-{seed}-{index:03d}",
                embedding=(0.0, 1.0, 0.0),
                base_energy=-1.03 + rng.uniform(-0.004, 0.004),
                oracle_energy=-1.02 + rng.uniform(-0.004, 0.004),
            )
        )
    rng.shuffle(candidates)
    return candidates


def protocol_shift_pool(seed: int, size: int) -> list[SyntheticCase]:
    """The same structural neighborhood has opposite residuals by protocol."""

    rng = random.Random(seed)
    candidates = []
    for index in range(size):
        scan = index % 2 == 1
        selected_protocol = protocol("SCAN" if scan else "PBE")
        base_energy = -1.04 + rng.uniform(-0.003, 0.003)
        oracle_energy = (-1.12 if scan else -0.94) + rng.uniform(-0.003, 0.003)
        candidates.append(
            item(
                f"protocol-{seed}-{index:03d}",
                embedding=(1.0, 0.0, 0.0),
                base_energy=base_energy,
                oracle_energy=oracle_energy,
                selected_protocol=selected_protocol,
            )
        )
    rng.shuffle(candidates)
    return candidates


def retention_competition_pool(seed: int, size: int) -> list[SyntheticCase]:
    """Singleton bait competes with observations valuable to a recurring pool."""

    rng = random.Random(seed)
    candidates = [
        item(
            f"bait-{seed}",
            embedding=(1.0, 0.0, 0.0, 0.0),
            base_energy=-1.10,
            oracle_energy=-0.90,
        )
    ]
    false_count = max(2, (size - 1) // 2)
    for index in range(false_count):
        candidates.append(
            item(
                f"informative-false-{seed}-{index:03d}",
                embedding=(0.0, 1.0, 0.0, 0.0),
                base_energy=-1.04 + rng.uniform(-0.002, 0.002),
                oracle_energy=-0.94 + rng.uniform(-0.003, 0.003),
            )
        )
    for index in range(size - len(candidates)):
        candidates.append(
            item(
                f"stable-family-{seed}-{index:03d}",
                embedding=(0.0, 0.0, 1.0, 0.0),
                base_energy=-1.03 + rng.uniform(-0.002, 0.002),
                oracle_energy=-1.20 + rng.uniform(-0.003, 0.003),
            )
        )
    rng.shuffle(candidates)
    return candidates


class OfflineOracleRetention:
    """Evaluation-only future-label upper bound; forbidden in online policies."""

    def __init__(
        self,
        capacity: int,
        potential: BoundaryRiskPotential,
        candidates: list[SyntheticCase],
    ) -> None:
        self.capacity = capacity
        self.potential = potential
        self._cards: dict[str, MaterialMemoryCard] = {}
        self._oracle = {item.query.query_id: item.oracle_card for item in candidates}

    def cards(self) -> tuple[MaterialMemoryCard, ...]:
        return tuple(self._cards[key] for key in sorted(self._cards))

    def _future_loss(
        self,
        cards: list[MaterialMemoryCard],
        queries: tuple[MaterialQuery, ...],
    ) -> float:
        loss = 0.0
        config = self.potential.config
        for query in queries:
            estimate = self.potential.estimate(query, cards)
            oracle = self._oracle[query.query_id]
            actual_stable = (
                oracle.hull_distance(query.hull_snapshot)
                <= query.stability_threshold_ev_per_atom
            )
            if estimate.predicted_stable and not actual_stable:
                loss += config.false_stable_cost
            elif not estimate.predicted_stable and actual_stable:
                loss += config.false_unstable_cost
        return loss

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: tuple[MaterialQuery, ...] | object,
    ) -> None:
        queries = tuple(query_pool)
        self._cards[card.card_id] = card
        while len(self._cards) > self.capacity:
            choices = []
            for card_id in self._cards:
                retained = [item for key, item in self._cards.items() if key != card_id]
                choices.append((self._future_loss(retained, queries), card_id))
            _, evicted = min(choices, key=lambda item: (item[0], item[1]))
            del self._cards[evicted]


class UnsafeProtocolAgnosticResolver:
    """Isolated negative-transfer diagnostic; never a deployment policy."""

    def resolve(
        self,
        source: ProtocolCertificate,
        target: ProtocolCertificate,
    ) -> ProtocolCompatibility:
        del source, target
        return ProtocolCompatibility(
            kind=CompatibilityKind.DIRECT,
            uncertainty_radius_ev_per_atom=0.0,
            reason="unsafe_protocol_agnostic_diagnostic",
        )


def policy_factories(
    seed: int,
    capacity: int,
    budget: int,
    candidates: list[SyntheticCase],
) -> dict[str, Callable[[], tuple[object, object]]]:
    def components() -> tuple[ProtocolCompatibilityResolver, BoundaryRiskPotential]:
        resolver = ProtocolCompatibilityResolver()
        potential = BoundaryRiskPotential(
            resolver,
            BoundaryRiskConfig(
                residual_lipschitz_ev_per_atom=0.08,
                prior_radius_ev_per_atom=0.15,
                calibration_radius_ev_per_atom=0.01,
            ),
        )
        return resolver, potential

    def compatible_fifo() -> tuple[object, object]:
        resolver, _ = components()
        return ProtocolAwareBoundaryAcquisition(
            resolver,
            prior_strength=0.1,
            discovery_reward=5.0,
            false_stable_cost=1.0,
            exploration_weight=0.25,
        ), FIFOBoundedMemory(capacity)

    def compatible_residual() -> tuple[object, object]:
        resolver, _ = components()
        return ProtocolAwareBoundaryAcquisition(
            resolver,
            prior_strength=0.1,
        ), ResidualPriorityMemory(capacity)

    def uncertainty_fifo() -> tuple[object, object]:
        _, potential = components()
        return BoundaryUncertaintyAcquisition(potential), FIFOBoundedMemory(capacity)

    def fixed_boundary_retention() -> tuple[object, object]:
        _, potential = components()
        return BaseBoundaryAcquisition(), BoundaryRiskRetention(capacity, potential)

    def retention_aware_fifo() -> tuple[object, object]:
        _, potential = components()
        return LegacyTwoScenarioAcquisition(
            potential,
            active_witness_budget=capacity,
            discovery_weight=5.0,
        ), FIFOBoundedMemory(capacity)

    def decoupled_boundary() -> tuple[object, object]:
        resolver, potential = components()
        return ProtocolAwareBoundaryAcquisition(
            resolver,
            prior_strength=0.1,
        ), BoundaryRiskRetention(capacity, potential)

    def joint() -> tuple[object, object]:
        _, potential = components()
        return LegacyTwoScenarioAcquisition(
            potential,
            active_witness_budget=capacity,
            discovery_weight=5.0,
            information_weight=1.0,
        ), BoundaryRiskRetention(capacity, potential)

    def full_history() -> tuple[object, object]:
        resolver, _ = components()
        return ProtocolAwareBoundaryAcquisition(
            resolver,
            prior_strength=0.1,
        ), FullHistoryMemory(budget)

    def archive_topk() -> tuple[object, object]:
        resolver, _ = components()
        return OnDemandKNNArchiveAcquisition(
            ProtocolAwareBoundaryAcquisition(resolver, prior_strength=0.1),
            active_witness_budget=capacity,
        ), FullHistoryMemory(budget)

    def oracle_retention() -> tuple[object, object]:
        resolver, potential = components()
        return ProtocolAwareBoundaryAcquisition(
            resolver,
            prior_strength=0.1,
        ), OfflineOracleRetention(capacity, potential, candidates)

    return {
        "random": lambda: (SeededRandomAcquisition(seed), FIFOBoundedMemory(capacity)),
        "base_only": lambda: (BaseBoundaryAcquisition(), FIFOBoundedMemory(capacity)),
        "unsafe_protocol_agnostic": lambda: (
            ProtocolAwareBoundaryAcquisition(UnsafeProtocolAgnosticResolver(), prior_strength=0.1),
            FIFOBoundedMemory(capacity),
        ),
        "reservoir": lambda: (
            ProtocolAwareBoundaryAcquisition(ProtocolCompatibilityResolver(), prior_strength=0.1),
            DeterministicReservoirMemory(capacity, seed),
        ),
        "diversity": lambda: (
            ProtocolAwareBoundaryAcquisition(ProtocolCompatibilityResolver(), prior_strength=0.1),
            DiversityBoundedMemory(capacity),
        ),
        "uncertainty_fifo": uncertainty_fifo,
        "compatible_knn_fifo": compatible_fifo,
        "compatible_knn_residual": compatible_residual,
        "fixed_acq_boundary_retention": fixed_boundary_retention,
        "retention_aware_fifo": retention_aware_fifo,
        "decoupled_boundary": decoupled_boundary,
        "caw_joint": joint,
        "compatible_knn_full_history": full_history,
        "compatible_knn_archive_topk": archive_topk,
        "offline_oracle_retention": oracle_retention,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    scenarios = {
        "local_boundary_correlation": local_boundary_pool,
        "recurring_residual": recurring_pool,
        "iid_residual": iid_pool,
        "nonrecurring_chemistry": nonrecurring_pool,
        "causal_hull_revision": hull_revision_pool,
        "unsupported_protocol_shift": protocol_shift_pool,
        "retention_competition": retention_competition_pool,
    }
    if args.scenario:
        scenarios = {name: scenarios[name] for name in args.scenario}
    metrics = (
        "cumulative_true_discoveries",
        "unstable_oracle_calls",
        "false_stable_oracle_calls",
        "cumulative_discovery_regret",
        "cost_per_true_discovery",
    )
    output: dict[str, object] = {
        "scope": "synthetic_mechanism_pilot_not_materials_benchmark",
        "seeds": args.seeds,
        "candidate_count": args.candidates,
        "oracle_budget": args.budget,
        "active_witness_budget": args.capacity,
        "scenarios": {},
    }
    for scenario_name, builder in scenarios.items():
        rows: dict[str, dict[str, list[float]]] = {}
        for seed in range(args.seeds):
            candidates = builder(seed, args.candidates)
            factories = policy_factories(
                seed,
                args.capacity,
                args.budget,
                candidates,
            )
            if args.policy:
                factories = {name: factories[name] for name in args.policy}
            for policy_name, factory in factories.items():
                acquisition_policy, retention_policy = factory()
                started = time.perf_counter()
                result = evaluate_candidates(ActiveDiscoveryEvaluator(
                    acquisition_policy,
                    retention_policy,
                    oracle_budget=args.budget,
                    hull_engine=(
                        SyntheticMinHullEngine()
                        if scenario_name == "causal_hull_revision"
                        else None
                    ),
                ), candidates)
                runtime = time.perf_counter() - started
                policy_rows = rows.setdefault(policy_name, {metric: [] for metric in metrics})
                for metric in metrics:
                    value = getattr(result, metric)
                    policy_rows[metric].append(float(value) if value is not None else float(args.budget))
                policy_rows.setdefault("runtime_seconds", []).append(runtime)
                score_mass = sum(
                    abs(step.acquisition_score) * step.oracle_cost for step in result.steps
                )
                lookahead_mass = sum(
                    step.downstream_risk_reduction for step in result.steps
                )
                policy_rows.setdefault("lookahead_fraction_of_score", []).append(
                    lookahead_mass / score_mass if score_mass else 0.0
                )
        output["scenarios"][scenario_name] = {
            policy: {metric: summarize(values) for metric, values in policy_metrics.items()}
            for policy, policy_metrics in rows.items()
        }
    return output


def print_markdown(result: dict[str, object]) -> None:
    print("Synthetic mechanism pilot (mean over seeds; not a materials benchmark result)")
    print()
    scenarios = result["scenarios"]
    assert isinstance(scenarios, dict)
    for scenario, rows in scenarios.items():
        print(f"## {scenario}")
        print("| Policy | Discoveries | Unstable calls | Discovery regret | Cost/discovery |")
        print("|---|---:|---:|---:|---:|")
        assert isinstance(rows, dict)
        for policy, metrics in rows.items():
            assert isinstance(metrics, dict)
            values = [
                metrics[name]["mean"]
                for name in (
                    "cumulative_true_discoveries",
                    "unstable_oracle_calls",
                    "cumulative_discovery_regret",
                    "cost_per_true_discovery",
                )
            ]
            print(f"| {policy} | " + " | ".join(f"{value:.3f}" for value in values) + " |")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--candidates", type=int, default=40)
    parser.add_argument("--budget", type=int, default=12)
    parser.add_argument("--capacity", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--policy",
        action="append",
        choices=tuple(policy_factories(0, 1, 1, [recurring_pool(0, 1)[0]])),
        help="run only the selected policy; repeat to select multiple",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=(
            "local_boundary_correlation",
            "recurring_residual",
            "iid_residual",
            "nonrecurring_chemistry",
            "causal_hull_revision",
            "unsupported_protocol_shift",
            "retention_competition",
        ),
        help="run only the selected scenario; repeat to select multiple",
    )
    args = parser.parse_args()
    if min(args.seeds, args.candidates, args.budget) < 1 or args.capacity < 0:
        parser.error("seeds, candidates, and budget must be positive; capacity may be zero")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_markdown(result)


if __name__ == "__main__":
    main()
