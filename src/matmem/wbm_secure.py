"""Capability-isolated, causal WBM execution.

This is the sole closed-loop execution path for real WBM records. Policies run
in a subprocess and receive a serialized, allow-listed state. The evaluator
alone owns the oracle vault, corrected total-energy entries, phase diagrams,
and the append-only action ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .cards import HullSnapshot, MaterialMemoryCard, MaterialQuery, SourceProvenance
from .coreset import (
    StreamingCalibrationCoreset,
    StreamingJointPosteriorRiskCoreset,
    StreamingPosteriorProjectionCoreset,
)
from .identity import StructureArtifactIdentity, StructureStage
from .protocols import ProtocolCertificate
from .residual_posterior import FixedKernelGPConfig
from .wbm import OracleEnergySource, WBMOracleRecord


def _checksum(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _entry_total_energy(entry: object) -> float:
    try:
        value = float(entry.energy)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError("corrected phase entry must expose finite total energy") from exc
    if not math.isfinite(value):
        raise ValueError("corrected phase total energy must be finite")
    return value


def _entry_system(entry: object) -> tuple[str, ...]:
    try:
        return tuple(sorted(str(element) for element in entry.composition.elements))
    except AttributeError as exc:
        raise TypeError("corrected phase entry must expose a composition") from exc


class PolicyQuery(BaseModel):
    """Allow-listed observable candidate state sent across the process boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    structure_hash: str
    structure_identity: StructureArtifactIdentity
    composition: str
    chemical_system: tuple[str, ...]
    embedding: tuple[float, ...]
    protocol: ProtocolCertificate
    frozen_prediction_ev_per_atom: float
    base_hull_distance_ev_per_atom: float
    hull_reference_energy_ev_per_atom: float
    stability_threshold_ev_per_atom: float
    hull_snapshot_id: str
    hull_phase_checksum: str
    oracle_cost: float = Field(gt=0)

    @model_validator(mode="after")
    def _initial_structure_only(self) -> PolicyQuery:
        if (
            self.structure_identity.query_id != self.query_id
            or self.structure_identity.structure_hash != self.structure_hash
            or self.structure_identity.stage is not StructureStage.INITIAL
            or not self.structure_identity.causal_available_before_query
        ):
            raise ValueError("policy query requires its pre-query initial structure")
        return self


class PolicyWitness(BaseModel):
    """Previously revealed residual evidence; never a corrected phase entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    witness_id: str
    material_id: str
    structure_hash: str
    structure_identity: StructureArtifactIdentity
    composition: str
    residual_ev_per_atom: float
    embedding: tuple[float, ...]
    protocol: ProtocolCertificate
    quality_weight: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _initial_structure_only(self) -> PolicyWitness:
        if (
            self.structure_identity.query_id != self.material_id
            or self.structure_identity.structure_hash != self.structure_hash
            or self.structure_identity.stage is not StructureStage.INITIAL
            or not self.structure_identity.causal_available_before_query
        ):
            raise ValueError("policy witness requires its pre-query initial structure")
        return self


class PolicyState(BaseModel):
    """The complete and exclusive serialized capability given to a policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(ge=1)
    remaining_budget: float = Field(ge=0)
    queries: tuple[PolicyQuery, ...]
    witnesses: tuple[PolicyWitness, ...]
    history_query_ids: tuple[str, ...]
    active_witness_capacity: int = Field(ge=0)
    policy_identity_checksum: str
    state_checksum: str

    @classmethod
    def create(
        cls,
        *,
        round_index: int,
        remaining_budget: float,
        queries: Iterable[MaterialQuery],
        witnesses: Iterable[MaterialMemoryCard],
        history_query_ids: Iterable[str],
        active_witness_capacity: int,
        policy_identity_checksum: str,
    ) -> PolicyState:
        query_items = tuple(
            sorted(
                (
                    PolicyQuery(
                        query_id=query.query_id,
                        structure_hash=query.structure_hash,
                        structure_identity=query.structure_identity,
                        composition=query.composition,
                        chemical_system=query.hull_snapshot.chemical_system,
                        embedding=query.embedding,
                        protocol=query.protocol,
                        frozen_prediction_ev_per_atom=(
                            query.base_predicted_formation_energy_ev_per_atom
                        ),
                        base_hull_distance_ev_per_atom=(query.base_hull_distance_ev_per_atom),
                        hull_reference_energy_ev_per_atom=(
                            query.hull_snapshot.reference_hull_energy_ev_per_atom
                        ),
                        stability_threshold_ev_per_atom=(query.stability_threshold_ev_per_atom),
                        hull_snapshot_id=query.hull_snapshot.snapshot_id,
                        hull_phase_checksum=query.hull_snapshot.phase_set_checksum,
                        oracle_cost=query.oracle_cost,
                    )
                    for query in queries
                ),
                key=lambda item: item.query_id,
            )
        )
        witness_items = tuple(
            sorted(
                (
                    PolicyWitness(
                        witness_id=card.card_id,
                        material_id=card.material_id,
                        structure_hash=card.structure_hash,
                        structure_identity=card.structure_identity,
                        composition=card.composition,
                        residual_ev_per_atom=card.oracle_residual_ev_per_atom,
                        embedding=card.embedding,
                        protocol=card.protocol,
                        quality_weight=card.quality_weight,
                    )
                    for card in witnesses
                ),
                key=lambda item: item.witness_id,
            )
        )
        history = tuple(history_query_ids)
        content = {
            "round_index": round_index,
            "remaining_budget": remaining_budget,
            "queries": [item.model_dump(mode="json") for item in query_items],
            "witnesses": [item.model_dump(mode="json") for item in witness_items],
            "history_query_ids": history,
            "active_witness_capacity": active_witness_capacity,
            "policy_identity_checksum": policy_identity_checksum,
        }
        return cls(
            round_index=round_index,
            remaining_budget=remaining_budget,
            queries=query_items,
            witnesses=witness_items,
            history_query_ids=history,
            active_witness_capacity=active_witness_capacity,
            policy_identity_checksum=policy_identity_checksum,
            state_checksum=_checksum(content),
        )

    def serialized_for_policy(self) -> str:
        payload = self.model_dump_json()
        forbidden = (
            "oracle_card",
            "oracle_record",
            "corrected_phase",
            "corrected_total_energy",
            "corrected_formation_energy",
            "stable_label",
            "phase_diagram",
            "e_above_hull",
        )
        lowered = payload.lower()
        found = [field for field in forbidden if field in lowered]
        if found:
            raise ValueError(f"policy payload contains forbidden fields: {found}")
        return payload


class RevealedObservation(BaseModel):
    """Archive-safe per-atom observation produced only after selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    structure_hash: str
    corrected_formation_energy_ev_per_atom: float
    residual_ev_per_atom: float
    source_record_locator: str


@dataclass(frozen=True)
class CorrectedPhaseEntry:
    """Hull-only total-energy entry, never serialized into policy state."""

    query_id: str
    corrected_total_energy_ev: float
    energy_source: OracleEnergySource
    entry: Any

    def __post_init__(self) -> None:
        if not self.query_id.strip():
            raise ValueError("corrected phase query ID must be non-empty")
        if not math.isfinite(self.corrected_total_energy_ev):
            raise ValueError("corrected phase total energy must be finite")
        if self.energy_source is not OracleEnergySource.FROZEN_PARITY_CORRECTED:
            raise ValueError("corrected phase entry requires frozen parity energy")
        if not math.isclose(
            _entry_total_energy(self.entry),
            self.corrected_total_energy_ev,
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError("corrected phase entry energy must be total energy in eV, not eV/atom")


@dataclass(frozen=True)
class WBMReveal:
    observation: RevealedObservation
    card: MaterialMemoryCard
    corrected_phase: CorrectedPhaseEntry


class WBMActionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["action"] = "action"
    round_index: int = Field(ge=1)
    selected_query_id: str
    pre_reveal_state_checksum: str
    previous_record_checksum: str | None
    action_checksum: str


class WBMRevealRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reveal"] = "reveal"
    round_index: int = Field(ge=1)
    selected_query_id: str
    action_checksum: str
    causal_discovery: bool
    active_witness_ids: tuple[str, ...]
    post_reveal_hull_checksum: str
    archive_checksum: str
    reveal_checksum: str


@dataclass(frozen=True)
class _ActionAuthorization:
    record_index: int
    selected_query_id: str
    action_checksum: str


class AppendOnlyWBMEventLog:
    """Durable JSONL ledger that persists an action before oracle access."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise FileExistsError(
                "WBM event log is append-only and cannot overwrite a run"
            ) from exc
        self._records: list[WBMActionRecord | WBMRevealRecord] = []
        self._pending_action: WBMActionRecord | None = None

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> AppendOnlyWBMEventLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def records(self) -> tuple[WBMActionRecord | WBMRevealRecord, ...]:
        return tuple(self._records)

    def _persist(self, record: WBMActionRecord | WBMRevealRecord) -> None:
        self._handle.write(record.model_dump_json() + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._records.append(record)

    def append_action(
        self,
        *,
        round_index: int,
        selected_query_id: str,
        pre_reveal_state_checksum: str,
    ) -> _ActionAuthorization:
        if self._pending_action is not None:
            raise RuntimeError("previous WBM action has no reveal record")
        previous = None
        if self._records:
            previous = (
                self._records[-1]
                .model_dump()
                .get(
                    "reveal_checksum",
                    self._records[-1].model_dump().get("action_checksum"),
                )
            )
        payload = {
            "round_index": round_index,
            "selected_query_id": selected_query_id,
            "pre_reveal_state_checksum": pre_reveal_state_checksum,
            "previous_record_checksum": previous,
        }
        record = WBMActionRecord(
            **payload,
            action_checksum=_checksum(payload),
        )
        self._persist(record)
        self._pending_action = record
        return _ActionAuthorization(
            record_index=len(self._records) - 1,
            selected_query_id=selected_query_id,
            action_checksum=record.action_checksum,
        )

    def authorize_reveal(
        self,
        authorization: _ActionAuthorization,
        selected_query_id: str,
    ) -> WBMActionRecord:
        if not isinstance(authorization, _ActionAuthorization):
            raise RuntimeError("vault reveal requires a persisted action authorization")
        if self._pending_action is None:
            raise RuntimeError("vault reveal is forbidden before a persisted action")
        record = self._pending_action
        if (
            authorization.record_index != len(self._records) - 1
            or authorization.selected_query_id != selected_query_id
            or authorization.action_checksum != record.action_checksum
            or record.selected_query_id != selected_query_id
        ):
            raise RuntimeError("vault reveal authorization does not match persisted action")
        if not self.path.is_file() or self.path.stat().st_size <= 0:
            raise RuntimeError("persisted action record is unavailable")
        return record

    def append_reveal(
        self,
        *,
        round_index: int,
        selected_query_id: str,
        action_checksum: str,
        causal_discovery: bool,
        active_witness_ids: tuple[str, ...],
        post_reveal_hull_checksum: str,
        archive_checksum: str,
    ) -> WBMRevealRecord:
        if self._pending_action is None:
            raise RuntimeError("cannot persist reveal without a pending action")
        if (
            self._pending_action.round_index != round_index
            or self._pending_action.selected_query_id != selected_query_id
            or self._pending_action.action_checksum != action_checksum
        ):
            raise RuntimeError("reveal record does not match its action")
        payload = {
            "round_index": round_index,
            "selected_query_id": selected_query_id,
            "action_checksum": action_checksum,
            "causal_discovery": causal_discovery,
            "active_witness_ids": active_witness_ids,
            "post_reveal_hull_checksum": post_reveal_hull_checksum,
            "archive_checksum": archive_checksum,
        }
        record = WBMRevealRecord(**payload, reveal_checksum=_checksum(payload))
        self._persist(record)
        self._pending_action = None
        return record


class WBMOracleVault:
    """No-enumeration, single-use vault with ledger-gated reveals."""

    def __init__(
        self,
        records: Iterable[WBMOracleRecord],
        phase_entries: Mapping[str, object],
        *,
        source_version: str,
    ) -> None:
        items = tuple(records)
        if not source_version.strip():
            raise ValueError("WBM oracle vault requires a source version")
        if len({item.query_id for item in items}) != len(items):
            raise ValueError("WBM oracle query IDs must be unique")
        if set(phase_entries) != {item.query_id for item in items}:
            raise ValueError("oracle records and corrected phase entries must align")
        for item in items:
            if item.energy_source is not OracleEnergySource.FROZEN_PARITY_CORRECTED:
                raise ValueError("WBM oracle vault requires frozen parity energy")
            entry = phase_entries[item.query_id]
            if not math.isclose(
                _entry_total_energy(entry),
                item.corrected_total_energy_ev,
                rel_tol=0.0,
                abs_tol=1e-8,
            ):
                raise ValueError("oracle total energy and corrected phase entry energy disagree")
        self._records = {item.query_id: item for item in items}
        self._phase_entries = dict(phase_entries)
        self._revealed: list[str] = []
        self.source_version = source_version.strip()

    @property
    def reveal_count(self) -> int:
        return len(self._revealed)

    @property
    def revealed_query_ids(self) -> tuple[str, ...]:
        return tuple(self._revealed)

    def reveal(
        self,
        query: MaterialQuery,
        *,
        authorization: _ActionAuthorization,
        event_log: AppendOnlyWBMEventLog,
    ) -> WBMReveal:
        event_log.authorize_reveal(authorization, query.query_id)
        if query.query_id in self._revealed:
            raise ValueError("WBM oracle result has already been revealed")
        record = self._records.get(query.query_id)
        if record is None:
            raise KeyError("selected query is absent from WBM oracle vault")
        if record.structure_hash != query.structure_hash:
            raise ValueError("selected query and WBM oracle structure hashes differ")
        if record.observed_at < query.as_of:
            raise ValueError("WBM oracle observation predates the query state")
        entry = self._phase_entries[query.query_id]
        if _entry_system(entry) != query.hull_snapshot.chemical_system:
            raise ValueError("selected query and corrected phase entry systems differ")
        self._revealed.append(query.query_id)
        energy = record.corrected_formation_energy_ev_per_atom
        residual = energy - query.base_predicted_formation_energy_ev_per_atom
        observation = RevealedObservation(
            query_id=query.query_id,
            structure_hash=query.structure_hash,
            corrected_formation_energy_ev_per_atom=energy,
            residual_ev_per_atom=residual,
            source_record_locator=record.source_record_locator,
        )
        card = MaterialMemoryCard(
            card_id=f"wbm-card:{query.query_id}",
            material_id=query.query_id,
            structure_hash=query.structure_hash,
            structure_identity=query.structure_identity,
            identity=query.identity,
            composition=query.composition,
            embedding=query.embedding,
            protocol=query.protocol,
            provenance=SourceProvenance(
                source_name="WBM",
                source_version=self.source_version,
                record_locator=record.source_record_locator,
                retrieved_at=record.observed_at,
            ),
            formation_energy_ev_per_atom=energy,
            base_predicted_formation_energy_ev_per_atom=(
                query.base_predicted_formation_energy_ev_per_atom
            ),
            oracle_residual_ev_per_atom=residual,
            hull_snapshot=query.hull_snapshot,
            recorded_hull_distance_ev_per_atom=query.hull_distance(energy),
            observed_at=record.observed_at,
        )
        return WBMReveal(
            observation=observation,
            card=card,
            corrected_phase=CorrectedPhaseEntry(
                query_id=query.query_id,
                corrected_total_energy_ev=record.corrected_total_energy_ev,
                energy_source=record.energy_source,
                entry=entry,
            ),
        )


class PolicySubprocess:
    """One-shot subprocess receiving only JSON and returning one opaque ID."""

    def __init__(
        self,
        policy: Literal[
            "frozen",
            "random",
            "gp_uncertainty",
            "survival_coreset",
        ],
        *,
        seed: int = 0,
        gp_config: FixedKernelGPConfig | None = None,
        proposal_size: int = 32,
        num_fantasies: int = 8,
        survival_weight: float = 1.0,
    ) -> None:
        if proposal_size < 1 or num_fantasies < 1 or survival_weight < 0:
            raise ValueError("WBM policy reranking parameters are invalid")
        self.policy = policy
        self.seed = seed
        self.gp_config = gp_config or FixedKernelGPConfig()
        self.proposal_size = proposal_size
        self.num_fantasies = num_fantasies
        self.survival_weight = survival_weight

    @property
    def identity_checksum(self) -> str:
        return _checksum(
            {
                "policy": self.policy,
                "seed": self.seed,
                "gp_config": self.gp_config.__dict__,
                "proposal_size": self.proposal_size,
                "num_fantasies": self.num_fantasies,
                "survival_weight": self.survival_weight,
            }
        )

    def select(self, state: PolicyState) -> str:
        worker = Path(__file__).with_name("wbm_policy_worker.py")
        command = [
            sys.executable,
            str(worker),
            "--policy",
            self.policy,
            "--seed",
            str(self.seed),
            "--kernel",
            self.gp_config.kernel,
            "--length-scale",
            str(self.gp_config.length_scale),
            "--signal-std",
            str(self.gp_config.signal_std_ev_per_atom),
            "--noise-std",
            str(self.gp_config.noise_std_ev_per_atom),
            "--jitter",
            str(self.gp_config.jitter),
            "--proposal-size",
            str(self.proposal_size),
            "--num-fantasies",
            str(self.num_fantasies),
            "--survival-weight",
            str(self.survival_weight),
        ]
        result = subprocess.run(
            command,
            input=state.serialized_for_policy(),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"policy subprocess failed: {result.stderr.strip()}")
        selected = result.stdout.strip()
        if selected not in {item.query_id for item in state.queries}:
            raise RuntimeError("policy subprocess returned an unknown query ID")
        return selected


class EvidenceAccessStrategy(Protocol):
    capacity: int

    def active(self, archive: tuple[MaterialMemoryCard, ...]) -> tuple[MaterialMemoryCard, ...]: ...

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> object: ...


class PersistentFIFOEvidence:
    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("FIFO evidence capacity cannot be negative")
        self.capacity = capacity
        self._active: list[MaterialMemoryCard] = []

    def active(self, archive: tuple[MaterialMemoryCard, ...]) -> tuple[MaterialMemoryCard, ...]:
        del archive
        return tuple(self._active)

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery] = (),
    ) -> None:
        del query_pool
        if any(item.card_id == card.card_id for item in self._active):
            raise ValueError("persistent FIFO cannot admit a duplicate witness")
        self._active.append(card)
        if len(self._active) > self.capacity:
            del self._active[0]


class ReconstructedFIFOEvidence:
    """Free on-demand reconstruction of exactly the persistent FIFO selector."""

    def __init__(self, capacity: int) -> None:
        if capacity < 0:
            raise ValueError("FIFO evidence capacity cannot be negative")
        self.capacity = capacity

    def active(self, archive: tuple[MaterialMemoryCard, ...]) -> tuple[MaterialMemoryCard, ...]:
        if self.capacity == 0:
            return ()
        return archive[-self.capacity :]

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery] = (),
    ) -> None:
        del card, query_pool


class StreamingCoresetEvidence:
    """Expose the new calibration coreset through the sole WBM access API."""

    def __init__(
        self,
        coreset: (
            StreamingCalibrationCoreset
            | StreamingJointPosteriorRiskCoreset
            | StreamingPosteriorProjectionCoreset
        ),
    ) -> None:
        self.coreset = coreset
        self.capacity = coreset.capacity

    def active(self, archive: tuple[MaterialMemoryCard, ...]) -> tuple[MaterialMemoryCard, ...]:
        del archive
        return self.coreset.cards()

    def admit(
        self,
        card: MaterialMemoryCard,
        query_pool: Iterable[MaterialQuery],
    ) -> object:
        return self.coreset.admit(card, query_pool)


class CompositionHullState:
    """Evaluator-owned causal hull for exactly one chemical system."""

    def __init__(
        self,
        initial_phase_diagram: object,
        *,
        chemical_system: tuple[str, ...],
        source_version: str,
    ) -> None:
        self.chemical_system = tuple(sorted(chemical_system))
        if not self.chemical_system:
            raise ValueError("composition hull requires a chemical system")
        self._initial_entries = tuple(initial_phase_diagram.all_entries)
        if any(
            not set(_entry_system(entry)).issubset(self.chemical_system)
            for entry in self._initial_entries
        ):
            raise ValueError("initial phase diagram contains another chemical system")
        self._selected: list[CorrectedPhaseEntry] = []
        self._phase_diagram = initial_phase_diagram
        self.source_version = source_version

    def _diagram(self, selected: Iterable[CorrectedPhaseEntry]) -> object:
        from pymatgen.analysis.phase_diagram import PhaseDiagram

        return PhaseDiagram([*self._initial_entries, *(item.entry for item in selected)])

    @property
    def selected_query_ids(self) -> tuple[str, ...]:
        return tuple(item.query_id for item in self._selected)

    @property
    def phase_set_checksum(self) -> str:
        initial_ids = [
            str(entry.entry_id)
            if getattr(entry, "entry_id", None) is not None
            else _checksum(
                {
                    "composition": str(entry.composition),
                    "total_energy_ev": _entry_total_energy(entry),
                }
            )
            for entry in self._initial_entries
        ]
        return _checksum(
            {
                "initial_phase_ids": sorted(initial_ids),
                "revealed_query_ids": self.selected_query_ids,
            }
        )

    def copy(self) -> CompositionHullState:
        copied = CompositionHullState(
            self._diagram(()),
            chemical_system=self.chemical_system,
            source_version=self.source_version,
        )
        copied._selected = list(self._selected)
        copied._phase_diagram = copied._diagram(copied._selected)
        return copied

    def hypothetical(self, phase: CorrectedPhaseEntry) -> CompositionHullState:
        copied = self.copy()
        copied.add_revealed(phase)
        return copied

    def add_revealed(self, phase: CorrectedPhaseEntry) -> None:
        if phase.query_id in self.selected_query_ids:
            raise ValueError("corrected phase was already added to causal hull")
        if _entry_system(phase.entry) != self.chemical_system:
            raise ValueError("corrected phase belongs to another chemical system")
        self._selected.append(phase)
        self._phase_diagram = self._diagram(self._selected)

    def _formation_hull_energy_per_atom(self, composition: object) -> float:
        from pymatgen.core import Composition, Element

        target = Composition(composition)
        raw_hull = float(self._phase_diagram.get_hull_energy_per_atom(target))
        fractions = target.fractional_composition.as_dict()
        elemental_reference = sum(
            float(fraction) * float(self._phase_diagram.el_refs[Element(element)].energy_per_atom)
            for element, fraction in fractions.items()
        )
        return raw_hull - elemental_reference

    def snapshot(
        self,
        query: MaterialQuery,
        *,
        round_index: int,
        built_at: datetime,
    ) -> HullSnapshot:
        if query.hull_snapshot.chemical_system != self.chemical_system:
            raise ValueError("query belongs to another chemical system")
        reference = self._formation_hull_energy_per_atom(query.composition)
        return query.hull_snapshot.model_copy(
            update={
                "snapshot_id": (f"causal:{self.source_version}:{round_index}:{query.query_id}"),
                "reference_hull_energy_ev_per_atom": reference,
                "phase_set_checksum": self.phase_set_checksum,
                "known_through": built_at,
                "built_at": built_at,
            }
        )

    def selected_final_stability(self) -> dict[str, bool]:
        diagram = self._diagram(self._selected)
        return {
            item.query_id: bool(diagram.get_e_above_hull(item.entry, allow_negative=True) <= 1e-8)
            for item in self._selected
        }

    def oracle_final_stability(
        self,
        oracle_universe: Iterable[CorrectedPhaseEntry],
    ) -> dict[str, bool]:
        universe = tuple(oracle_universe)
        if not universe:
            raise ValueError("oracle-final hull requires the frozen exact-system universe")
        if any(_entry_system(item.entry) != self.chemical_system for item in universe):
            raise ValueError("oracle-final universe crosses chemical systems")
        by_id = {item.query_id: item for item in universe}
        if len(by_id) != len(universe):
            raise ValueError("oracle-final universe query IDs must be unique")
        missing = set(self.selected_query_ids) - set(by_id)
        if missing:
            raise ValueError(f"oracle-final universe omits selected queries: {sorted(missing)}")
        diagram = self._diagram(universe)
        return {
            query_id: bool(
                diagram.get_e_above_hull(by_id[query_id].entry, allow_negative=True) <= 1e-8
            )
            for query_id in self.selected_query_ids
        }


class WBMEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int
    selected_query_id: str
    pre_reveal_state_checksum: str
    action_checksum: str
    causal_discovery: bool
    active_witness_ids: tuple[str, ...]
    post_reveal_hull_checksum: str
    archive_checksum: str
    reveal_checksum: str


class BudgetPrefixParityRecord(BaseModel):
    """Behavioral parity of one state under an alternative budget label."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int
    canonical_budget: float
    compared_budget: float
    spent_before_round: float
    canonical_selected_query_id: str
    compared_selected_query_id: str
    actions_match: bool
    active_witness_ids: tuple[str, ...]
    post_reveal_hull_checksum: str
    prequential_metric_input_checksum: str


class WBMPhaseTiming(BaseModel):
    """Non-semantic timing record excluded from deterministic trace checksums."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(gt=0)
    hull_update_seconds: float = Field(ge=0)


class WBMRunResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_query_ids: tuple[str, ...]
    causal_discoveries: int
    selected_final_confirmations: int
    selected_history_invalidations: int
    oracle_final_true_discoveries: int
    benchmark_false_confirmations: int
    events: tuple[WBMEvent, ...]
    budget_prefix_parity: tuple[BudgetPrefixParityRecord, ...] = ()
    phase_timings: tuple[WBMPhaseTiming, ...] = ()
    trace_checksum: str


class SecureWBMRunner:
    """Execute one exact-system trajectory behind a subprocess boundary."""

    def __init__(
        self,
        *,
        queries: Iterable[MaterialQuery],
        vault: WBMOracleVault,
        hull_state: CompositionHullState,
        policy: PolicySubprocess,
        evidence_access: EvidenceAccessStrategy,
        oracle_universe: Iterable[CorrectedPhaseEntry],
        event_log: AppendOnlyWBMEventLog,
    ) -> None:
        items = tuple(queries)
        if not items or len({item.query_id for item in items}) != len(items):
            raise ValueError("secure WBM runner requires unique nonempty queries")
        systems = {item.hull_snapshot.chemical_system for item in items}
        if systems != {hull_state.chemical_system}:
            raise ValueError("one secure WBM runner handles exactly one chemical system")
        self.queries = {item.query_id: item for item in items}
        self.vault = vault
        self.hull_state = hull_state
        self.policy = policy
        self.evidence_access = evidence_access
        self.oracle_universe = tuple(oracle_universe)
        self.event_log = event_log

    def run(
        self,
        *,
        oracle_budget: float,
        budget_prefix_checks: tuple[float, ...] = (),
    ) -> WBMRunResult:
        if oracle_budget <= 0:
            raise ValueError("oracle budget must be positive")
        if any(value <= 0 or value > oracle_budget for value in budget_prefix_checks):
            raise ValueError("prefix parity budgets must be in (0, oracle_budget]")
        remaining = dict(self.queries)
        archive: list[MaterialMemoryCard] = []
        causal_by_query: dict[str, bool] = {}
        events: list[WBMEvent] = []
        budget_prefix_parity: list[BudgetPrefixParityRecord] = []
        phase_timings: list[WBMPhaseTiming] = []
        spent = 0.0
        round_index = 1
        while remaining:
            affordable = tuple(
                query
                for query in remaining.values()
                if query.oracle_cost <= oracle_budget - spent + 1e-12
            )
            if not affordable:
                break
            pre_reveal_active = self.evidence_access.active(tuple(archive))
            state = PolicyState.create(
                round_index=round_index,
                remaining_budget=oracle_budget - spent,
                queries=affordable,
                witnesses=pre_reveal_active,
                history_query_ids=(item.material_id for item in archive),
                active_witness_capacity=self.evidence_access.capacity,
                policy_identity_checksum=self.policy.identity_checksum,
            )
            selected_id = self.policy.select(state)
            alternate_actions = {
                compared_budget: self.policy.select(
                    PolicyState.create(
                        round_index=round_index,
                        remaining_budget=compared_budget - spent,
                        queries=affordable,
                        witnesses=pre_reveal_active,
                        history_query_ids=(item.material_id for item in archive),
                        active_witness_capacity=self.evidence_access.capacity,
                        policy_identity_checksum=self.policy.identity_checksum,
                    )
                )
                for compared_budget in sorted(set(budget_prefix_checks))
                if compared_budget != oracle_budget and spent < compared_budget
            }
            authorization = self.event_log.append_action(
                round_index=round_index,
                selected_query_id=selected_id,
                pre_reveal_state_checksum=state.state_checksum,
            )
            query = remaining.pop(selected_id)
            revealed = self.vault.reveal(
                query,
                authorization=authorization,
                event_log=self.event_log,
            )
            spent += query.oracle_cost
            causal = (
                query.hull_distance(revealed.observation.corrected_formation_energy_ev_per_atom)
                <= query.stability_threshold_ev_per_atom
            )
            causal_by_query[selected_id] = causal
            archive.append(revealed.card)
            hull_update_started = time.perf_counter()
            self.hull_state.add_revealed(revealed.corrected_phase)
            for query_id, future in tuple(remaining.items()):
                snapshot = self.hull_state.snapshot(
                    future,
                    round_index=round_index + 1,
                    built_at=revealed.card.observed_at,
                )
                remaining[query_id] = future.model_copy(
                    update={"hull_snapshot": snapshot, "as_of": snapshot.built_at}
                )
            phase_timings.append(
                WBMPhaseTiming(
                    round_index=round_index,
                    hull_update_seconds=time.perf_counter() - hull_update_started,
                )
            )
            self.evidence_access.admit(revealed.card, tuple(remaining.values()))
            active = self.evidence_access.active(tuple(archive))
            prequential_metric_input_checksum = _checksum(
                {
                    "remaining_queries": [
                        item.model_dump(mode="json")
                        for item in PolicyState.create(
                            round_index=round_index + 1,
                            remaining_budget=0.0,
                            queries=remaining.values(),
                            witnesses=active,
                            history_query_ids=(item.material_id for item in archive),
                            active_witness_capacity=self.evidence_access.capacity,
                            policy_identity_checksum=self.policy.identity_checksum,
                        ).queries
                    ],
                    "active_witness_ids": [item.card_id for item in active],
                    "hull_phase_checksum": self.hull_state.phase_set_checksum,
                }
            )
            archive_checksum = _checksum([item.card_id for item in archive])
            reveal_record = self.event_log.append_reveal(
                round_index=round_index,
                selected_query_id=selected_id,
                action_checksum=authorization.action_checksum,
                causal_discovery=causal,
                active_witness_ids=tuple(item.card_id for item in active),
                post_reveal_hull_checksum=self.hull_state.phase_set_checksum,
                archive_checksum=archive_checksum,
            )
            action_record = self.event_log.records[-2]
            assert isinstance(action_record, WBMActionRecord)
            events.append(
                WBMEvent(
                    round_index=round_index,
                    selected_query_id=selected_id,
                    pre_reveal_state_checksum=state.state_checksum,
                    action_checksum=action_record.action_checksum,
                    causal_discovery=causal,
                    active_witness_ids=reveal_record.active_witness_ids,
                    post_reveal_hull_checksum=reveal_record.post_reveal_hull_checksum,
                    archive_checksum=archive_checksum,
                    reveal_checksum=reveal_record.reveal_checksum,
                )
            )
            budget_prefix_parity.extend(
                BudgetPrefixParityRecord(
                    round_index=round_index,
                    canonical_budget=oracle_budget,
                    compared_budget=compared_budget,
                    spent_before_round=spent - query.oracle_cost,
                    canonical_selected_query_id=selected_id,
                    compared_selected_query_id=alternative_id,
                    actions_match=alternative_id == selected_id,
                    active_witness_ids=tuple(item.card_id for item in active),
                    post_reveal_hull_checksum=self.hull_state.phase_set_checksum,
                    prequential_metric_input_checksum=prequential_metric_input_checksum,
                )
                for compared_budget, alternative_id in alternate_actions.items()
            )
            round_index += 1
        selected_final = self.hull_state.selected_final_stability()
        oracle_final = self.hull_state.oracle_final_stability(self.oracle_universe)
        selected_ids = tuple(event.selected_query_id for event in events)
        invalidations = sum(
            causal_by_query[item] and not selected_final[item] for item in selected_ids
        )
        false_confirmations = sum(
            selected_final[item] and not oracle_final[item] for item in selected_ids
        )
        payload = [event.model_dump(mode="json") for event in events]
        return WBMRunResult(
            selected_query_ids=selected_ids,
            causal_discoveries=sum(causal_by_query.values()),
            selected_final_confirmations=sum(selected_final.values()),
            selected_history_invalidations=invalidations,
            oracle_final_true_discoveries=sum(oracle_final.values()),
            benchmark_false_confirmations=false_confirmations,
            events=tuple(events),
            budget_prefix_parity=tuple(budget_prefix_parity),
            phase_timings=tuple(phase_timings),
            trace_checksum=_checksum(payload),
        )


class WBMReplayAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    action_count: int
    trace_checksum: str


def replay_wbm_event_log(
    path: Path,
    *,
    initial_phase_diagram: object,
    chemical_system: tuple[str, ...],
    source_version: str,
    phase_entries: Mapping[str, CorrectedPhaseEntry],
) -> WBMReplayAudit:
    """Replay ledger ordering and causal-hull checksums without running policy."""

    raw = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(raw) % 2:
        raise ValueError("WBM event log has an unmatched action")
    hull = CompositionHullState(
        initial_phase_diagram,
        chemical_system=chemical_system,
        source_version=source_version,
    )
    events: list[WBMEvent] = []
    previous_checksum: str | None = None
    for offset in range(0, len(raw), 2):
        action = WBMActionRecord.model_validate(raw[offset])
        reveal = WBMRevealRecord.model_validate(raw[offset + 1])
        if action.round_index != offset // 2 + 1:
            raise ValueError("WBM action rounds are not contiguous")
        if action.previous_record_checksum != previous_checksum:
            raise ValueError("WBM event checksum chain is broken")
        expected_action = _checksum(
            {
                "round_index": action.round_index,
                "selected_query_id": action.selected_query_id,
                "pre_reveal_state_checksum": action.pre_reveal_state_checksum,
                "previous_record_checksum": action.previous_record_checksum,
            }
        )
        if action.action_checksum != expected_action:
            raise ValueError("WBM action checksum is invalid")
        if (
            reveal.round_index != action.round_index
            or reveal.selected_query_id != action.selected_query_id
            or reveal.action_checksum != action.action_checksum
        ):
            raise ValueError("WBM reveal is not paired with its action")
        phase = phase_entries.get(action.selected_query_id)
        if phase is None:
            raise KeyError("replay phase registry omits a selected query")
        hull.add_revealed(phase)
        if reveal.post_reveal_hull_checksum != hull.phase_set_checksum:
            raise ValueError("replayed causal hull checksum differs from recorded hull")
        expected_reveal = _checksum(
            {
                "round_index": reveal.round_index,
                "selected_query_id": reveal.selected_query_id,
                "action_checksum": reveal.action_checksum,
                "causal_discovery": reveal.causal_discovery,
                "active_witness_ids": reveal.active_witness_ids,
                "post_reveal_hull_checksum": reveal.post_reveal_hull_checksum,
                "archive_checksum": reveal.archive_checksum,
            }
        )
        if reveal.reveal_checksum != expected_reveal:
            raise ValueError("WBM reveal checksum is invalid")
        previous_checksum = reveal.reveal_checksum
        events.append(
            WBMEvent(
                round_index=action.round_index,
                selected_query_id=action.selected_query_id,
                pre_reveal_state_checksum=action.pre_reveal_state_checksum,
                action_checksum=action.action_checksum,
                causal_discovery=reveal.causal_discovery,
                active_witness_ids=reveal.active_witness_ids,
                post_reveal_hull_checksum=reveal.post_reveal_hull_checksum,
                archive_checksum=reveal.archive_checksum,
                reveal_checksum=reveal.reveal_checksum,
            )
        )
    return WBMReplayAudit(
        passed=True,
        action_count=len(events),
        trace_checksum=_checksum([event.model_dump(mode="json") for event in events]),
    )


class ExactEmulationAudit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool
    persistent_checksum: str
    reconstructed_checksum: str
    round_count: int


def assert_exact_emulation(
    persistent: WBMRunResult,
    reconstructed: WBMRunResult,
) -> ExactEmulationAudit:
    persistent_semantics = persistent.model_dump(exclude={"phase_timings"})
    reconstructed_semantics = reconstructed.model_dump(exclude={"phase_timings"})
    if persistent_semantics != reconstructed_semantics:
        for index, (left, right) in enumerate(
            zip(persistent.events, reconstructed.events, strict=False),
            start=1,
        ):
            if left != right:
                differing = sorted(
                    key
                    for key, value in left.model_dump().items()
                    if value != right.model_dump()[key]
                )
                raise AssertionError(
                    f"exact emulation mismatch at round {index}: {', '.join(differing)}"
                )
        raise AssertionError("exact emulation mismatch outside paired rounds")
    return ExactEmulationAudit(
        passed=True,
        persistent_checksum=persistent.trace_checksum,
        reconstructed_checksum=reconstructed.trace_checksum,
        round_count=len(persistent.events),
    )
