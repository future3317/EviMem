"""Oracle-isolated, action-driven execution for multi-protocol discovery.

This module intentionally separates *query selection* from any future training
subset selector.  Every paid target outcome is appended to the revealed
history and causal hull.  A policy runs in a subprocess, receives only an
allow-listed observable state, and its persisted action is the sole capability
that can reveal an oracle outcome.

The frozen JARVIS--MP v4 runner remains a historical fixed-trace evaluator.
This is the native boundary for fresh action-driven tasks; it is not a
compatibility wrapper around that evaluator.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .identity import StructureArtifactIdentity, StructureStage
from .protocol_knowledge_gradient import (
    FrozenProtocolRidgeTransport,
    source_margin_action_indices,
)
from .protocols import ProtocolCertificate


def requires_protocol_transport(policy: str) -> bool:
    """Return whether a policy requires a frozen cross-protocol posterior."""

    return policy in {
        "delta_hull_active_search",
        "source_rollout_delta_hull",
        "constrained_dual_horizon_source_rollout",
        "independent_confirmation_source_rollout",
        "conformal_source_rollout_delta_hull",
    } or policy.startswith("protocol_hull_")


def _checksum(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validated_composition(values: dict[str, float]) -> dict[str, float]:
    cleaned = {str(key).strip(): float(value) for key, value in values.items()}
    if (
        not cleaned
        or any(not key or not math.isfinite(value) or value < 0 for key, value in cleaned.items())
        or sum(cleaned.values()) <= 0
    ):
        raise ValueError("composition fractions must be finite and non-negative")
    return dict(sorted(cleaned.items()))


def _normalized_composition(values: dict[str, float]) -> dict[str, float]:
    cleaned = _validated_composition(values)
    total = sum(cleaned.values())
    return dict(sorted((key, value / total) for key, value in cleaned.items()))


class ProtocolCandidate(BaseModel):
    """Observable candidate available before an expensive target query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    source_structure_hash: str
    source_structure_identity: StructureArtifactIdentity
    chemical_system: tuple[str, ...]
    composition: dict[str, float]
    source_formation_energy_ev_per_atom: float
    source_environment_embedding: tuple[float, ...]
    source_local_environment_embedding: tuple[float, ...] | None = None
    source_protocol: ProtocolCertificate
    target_protocol: ProtocolCertificate
    oracle_cost: float = Field(default=1.0, gt=0)

    @field_validator("pair_id", "source_structure_hash")
    @classmethod
    def _identity(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("protocol candidate identity must be non-empty")
        return value

    @field_validator("chemical_system")
    @classmethod
    def _system(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(sorted({value.strip() for value in values if value.strip()}))
        if not normalized:
            raise ValueError("protocol candidate requires a chemical system")
        return normalized

    @field_validator("composition")
    @classmethod
    def _composition(cls, values: dict[str, float]) -> dict[str, float]:
        # Preserve stoichiometric atom count: corrected_total_energy_ev is for
        # this exact composition, not for a normalized one-atom formula.
        return _validated_composition(values)

    @field_validator(
        "source_formation_energy_ev_per_atom",
        "source_environment_embedding",
        "source_local_environment_embedding",
    )
    @classmethod
    def _finite(cls, value: float | tuple[float, ...] | None) -> float | tuple[float, ...] | None:
        if value is None:
            return value
        values = (value,) if isinstance(value, (float, int)) else value
        if not values or any(not math.isfinite(float(item)) for item in values):
            raise ValueError("protocol candidate numeric fields must be finite")
        return value

    @model_validator(mode="after")
    def _causal_source_contract(self) -> ProtocolCandidate:
        identity = self.source_structure_identity
        if (
            identity.query_id != self.pair_id
            or identity.structure_hash != self.source_structure_hash
            or not identity.causal_available_before_query
            or identity.stage is StructureStage.RELAXED
        ):
            raise ValueError("protocol candidate requires a causal pre-query structure")
        if set(self.composition) != set(self.chemical_system):
            raise ValueError("candidate composition and exact chemical system disagree")
        return self


class ProtocolOracleOutcome(BaseModel):
    """Target-protocol outcome owned only by the reveal boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    source_structure_hash: str
    chemical_system: tuple[str, ...]
    composition: dict[str, float]
    target_corrected_total_energy_ev: float
    target_formation_energy_ev_per_atom: float
    split: Literal["calibration", "evaluation", "development", "confirmatory", "fixture"]

    @field_validator("composition")
    @classmethod
    def _composition(cls, values: dict[str, float]) -> dict[str, float]:
        # The reveal boundary must preserve the atom count associated with the
        # corrected total energy used to update the phase diagram.
        return _validated_composition(values)

    @field_validator(
        "target_corrected_total_energy_ev",
        "target_formation_energy_ev_per_atom",
    )
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("oracle energies must be finite")
        return value

    @model_validator(mode="after")
    def _system_matches_composition(self) -> ProtocolOracleOutcome:
        if set(self.composition) != set(self.chemical_system):
            raise ValueError("oracle composition and exact chemical system disagree")
        return self


class ObservableProtocolQuery(BaseModel):
    """Allow-listed dynamic candidate state sent to a policy subprocess."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    source_structure_hash: str
    chemical_system: tuple[str, ...]
    composition: dict[str, float]
    source_formation_energy_ev_per_atom: float
    source_environment_embedding: tuple[float, ...]
    source_local_environment_embedding: tuple[float, ...] | None = None
    current_competing_hull_ev_per_atom: float
    source_protocol_fingerprint: str
    target_protocol_fingerprint: str
    oracle_cost: float = Field(gt=0)

    @field_validator("composition")
    @classmethod
    def _composition(cls, values: dict[str, float]) -> dict[str, float]:
        # Policy-facing energies are per atom, so fractional composition is the
        # sufficient, representation-invariant quantity here.
        return _normalized_composition(values)


class ObservableProtocolPhase(BaseModel):
    """A target-protocol phase already legal to use in the causal hull."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    entry_id: str
    composition: dict[str, float]
    formation_energy_ev_per_atom: float

    @field_validator("composition")
    @classmethod
    def _composition(cls, values: dict[str, float]) -> dict[str, float]:
        return _normalized_composition(values)

    @field_validator("formation_energy_ev_per_atom")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("observable phase energy must be finite")
        return value


class RevealedProtocolObservation(BaseModel):
    """One legally revealed target outcome visible to later policy rounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    pair_id: str
    source_formation_energy_ev_per_atom: float
    revealed_target_formation_energy_ev_per_atom: float
    source_environment_embedding: tuple[float, ...]
    source_local_environment_embedding: tuple[float, ...] | None = None


class ProtocolPolicyState(BaseModel):
    """Exclusive JSON capability passed to an acquisition policy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int = Field(ge=1)
    remaining_budget: float = Field(ge=0)
    queries: tuple[ObservableProtocolQuery, ...]
    causal_hull_phases: tuple[ObservableProtocolPhase, ...]
    revealed_history: tuple[RevealedProtocolObservation, ...]
    conformal_deviation_used: bool = False
    policy_identity_checksum: str
    state_checksum: str

    @classmethod
    def create(
        cls,
        *,
        round_index: int,
        remaining_budget: float,
        queries: Iterable[ObservableProtocolQuery],
        causal_hull_phases: Iterable[ObservableProtocolPhase],
        revealed_history: Iterable[RevealedProtocolObservation],
        conformal_deviation_used: bool = False,
        policy_identity_checksum: str,
    ) -> ProtocolPolicyState:
        query_items = tuple(sorted(queries, key=lambda item: item.pair_id))
        history = tuple(revealed_history)
        phases = tuple(sorted(causal_hull_phases, key=lambda item: item.entry_id))
        content = {
            "round_index": round_index,
            "remaining_budget": remaining_budget,
            "queries": [item.model_dump(mode="json") for item in query_items],
            "causal_hull_phases": [item.model_dump(mode="json") for item in phases],
            "revealed_history": [item.model_dump(mode="json") for item in history],
            "conformal_deviation_used": conformal_deviation_used,
            "policy_identity_checksum": policy_identity_checksum,
        }
        return cls(**content, state_checksum=_checksum(content))

    def serialized_for_policy(self) -> str:
        payload = self.model_dump_json()
        candidate_payload = json.loads(payload)["queries"]
        forbidden = (
            "target_corrected_total_energy",
            "target_formation_energy",
            "oracle_outcome",
            "oracle_vault",
            "stable_label",
            "target_structure",
        )
        encoded_candidates = json.dumps(candidate_payload, sort_keys=True).lower()
        found = [field for field in forbidden if field in encoded_candidates]
        if found:
            raise ValueError(f"unrevealed policy candidates contain forbidden fields: {found}")
        return payload


class ProtocolActionRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["action"] = "action"
    round_index: int = Field(ge=1)
    selected_pair_id: str
    pre_reveal_state_checksum: str
    previous_record_checksum: str | None
    action_checksum: str


class ProtocolRevealRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reveal"] = "reveal"
    round_index: int = Field(ge=1)
    selected_pair_id: str
    action_checksum: str
    outcome_checksum: str
    post_reveal_hull_checksum: str
    archive_checksum: str
    reveal_checksum: str


@dataclass(frozen=True)
class _RevealAuthorization:
    record_index: int
    selected_pair_id: str
    action_checksum: str


class AppendOnlyProtocolEventLog:
    """Persist every selected action before granting a reveal capability."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        except FileExistsError as exc:
            raise FileExistsError("protocol event log cannot overwrite a run") from exc
        self._records: list[ProtocolActionRecord | ProtocolRevealRecord] = []
        self._pending: ProtocolActionRecord | None = None

    @property
    def records(self) -> tuple[ProtocolActionRecord | ProtocolRevealRecord, ...]:
        return tuple(self._records)

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def __enter__(self) -> AppendOnlyProtocolEventLog:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _persist(self, record: ProtocolActionRecord | ProtocolRevealRecord) -> None:
        self._handle.write(record.model_dump_json() + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._records.append(record)

    def append_action(
        self,
        *,
        round_index: int,
        selected_pair_id: str,
        pre_reveal_state_checksum: str,
    ) -> _RevealAuthorization:
        if self._pending is not None:
            raise RuntimeError("previous protocol action has no reveal record")
        previous = None
        if self._records:
            last = self._records[-1].model_dump()
            previous = last.get("reveal_checksum", last.get("action_checksum"))
        content = {
            "round_index": round_index,
            "selected_pair_id": selected_pair_id,
            "pre_reveal_state_checksum": pre_reveal_state_checksum,
            "previous_record_checksum": previous,
        }
        record = ProtocolActionRecord(**content, action_checksum=_checksum(content))
        self._persist(record)
        self._pending = record
        return _RevealAuthorization(
            record_index=len(self._records) - 1,
            selected_pair_id=selected_pair_id,
            action_checksum=record.action_checksum,
        )

    def authorize(self, authorization: _RevealAuthorization, pair_id: str) -> None:
        if not isinstance(authorization, _RevealAuthorization):
            raise RuntimeError("protocol reveal requires a persisted action authorization")
        record = self._pending
        if (
            record is None
            or authorization.record_index != len(self._records) - 1
            or authorization.selected_pair_id != pair_id
            or authorization.action_checksum != record.action_checksum
            or record.selected_pair_id != pair_id
        ):
            raise RuntimeError("protocol reveal does not match the persisted action")
        if not self.path.is_file() or self.path.stat().st_size <= 0:
            raise RuntimeError("persisted protocol action is unavailable")

    def append_reveal(
        self,
        *,
        round_index: int,
        selected_pair_id: str,
        action_checksum: str,
        outcome_checksum: str,
        post_reveal_hull_checksum: str,
        archive_checksum: str,
    ) -> ProtocolRevealRecord:
        record = self._pending
        if (
            record is None
            or record.round_index != round_index
            or record.selected_pair_id != selected_pair_id
            or record.action_checksum != action_checksum
        ):
            raise RuntimeError("protocol reveal record does not match its action")
        content = {
            "round_index": round_index,
            "selected_pair_id": selected_pair_id,
            "action_checksum": action_checksum,
            "outcome_checksum": outcome_checksum,
            "post_reveal_hull_checksum": post_reveal_hull_checksum,
            "archive_checksum": archive_checksum,
        }
        reveal = ProtocolRevealRecord(**content, reveal_checksum=_checksum(content))
        self._persist(reveal)
        self._pending = None
        return reveal


class ProtocolOracleVault:
    """Non-enumerable target-outcome store with single-use authorized reveals."""

    def __init__(
        self,
        outcomes: Iterable[ProtocolOracleOutcome],
        *,
        expected_split: Literal[
            "calibration", "evaluation", "development", "confirmatory", "fixture"
        ],
    ) -> None:
        items = tuple(outcomes)
        if not items or any(item.split != expected_split for item in items):
            raise ValueError("protocol vault rows do not match the authorized split")
        if len({item.pair_id for item in items}) != len(items):
            raise ValueError("protocol oracle pair IDs must be unique")
        self._outcomes = {item.pair_id: item for item in items}
        self._revealed: list[str] = []
        self.expected_split = expected_split

    @property
    def revealed_pair_ids(self) -> tuple[str, ...]:
        return tuple(self._revealed)

    @property
    def reveal_count(self) -> int:
        return len(self._revealed)

    def reveal(
        self,
        candidate: ProtocolCandidate,
        *,
        authorization: _RevealAuthorization,
        event_log: AppendOnlyProtocolEventLog,
    ) -> ProtocolOracleOutcome:
        event_log.authorize(authorization, candidate.pair_id)
        if candidate.pair_id in self._revealed:
            raise ValueError("protocol oracle outcome has already been revealed")
        outcome = self._outcomes.get(candidate.pair_id)
        if outcome is None:
            raise KeyError("selected pair is absent from the protocol oracle vault")
        if (
            outcome.source_structure_hash != candidate.source_structure_hash
            or outcome.chemical_system != candidate.chemical_system
            or outcome.composition != candidate.composition
        ):
            raise ValueError("selected candidate and protocol oracle identity disagree")
        self._revealed.append(candidate.pair_id)
        return outcome


class ProtocolPolicySubprocess:
    """One-shot acquisition subprocess with no oracle-vault capability."""

    def __init__(
        self,
        policy: Literal[
            "source_margin",
            "random",
            "source_online_offset",
            "source_online_affine",
            "ridge_margin",
            "ridge_uncertainty",
            "chic_hull_influence",
            "ridge_predicted_final_margin",
            "delta_hull_active_search",
            "source_rollout_delta_hull",
            "constrained_dual_horizon_source_rollout",
            "independent_confirmation_source_rollout",
            "conformal_source_rollout_delta_hull",
            "protocol_hull_knowledge_gradient",
            "protocol_hull_risk_reduction",
        ],
        *,
        seed: int = 0,
        ridge_penalty: float = 1.0,
        prior_standard_deviation: float = 0.1,
        boundary_temperature: float = 0.05,
        transport_model: FrozenProtocolRidgeTransport | None = None,
        posterior_sample_count: int = 16,
        fantasy_count: int = 3,
        conformal_threshold: float | None = None,
        hull_backend: Literal["pymatgen", "fixed_composition"] = "pymatgen",
        selection_timeout_seconds: float = 30.0,
        worker_path: Path | None = None,
    ) -> None:
        self.policy = policy
        self.seed = seed
        self.ridge_penalty = ridge_penalty
        self.prior_standard_deviation = prior_standard_deviation
        self.boundary_temperature = boundary_temperature
        self.transport_model = transport_model
        self.posterior_sample_count = posterior_sample_count
        self.fantasy_count = fantasy_count
        self.conformal_threshold = conformal_threshold
        self.hull_backend = hull_backend
        self.selection_timeout_seconds = selection_timeout_seconds
        if (
            not math.isfinite(ridge_penalty)
            or ridge_penalty <= 0
            or not math.isfinite(prior_standard_deviation)
            or prior_standard_deviation <= 0
            or not math.isfinite(boundary_temperature)
            or boundary_temperature <= 0
        ):
            raise ValueError("protocol policy scales must be finite and positive")
        if posterior_sample_count < 4 or fantasy_count < 1:
            raise ValueError("protocol hull Monte Carlo settings are too small")
        if requires_protocol_transport(policy) and transport_model is None:
            raise ValueError("protocol hull policies require a frozen transport model")
        if policy in {
            "source_rollout_delta_hull",
            "constrained_dual_horizon_source_rollout",
            "conformal_source_rollout_delta_hull",
        }:
            rollout_block_size = posterior_sample_count // 16
            if (
                posterior_sample_count % 16
                or rollout_block_size < 2
                or rollout_block_size & (rollout_block_size - 1)
            ):
                raise ValueError("source rollout requires sixteen power-of-two Sobol blocks")
        if policy == "conformal_source_rollout_delta_hull" and (
            conformal_threshold is None
            or not math.isfinite(conformal_threshold)
            or conformal_threshold < 0
        ):
            raise ValueError("conformal source rollout requires a finite non-negative threshold")
        if hull_backend not in {"pymatgen", "fixed_composition"}:
            raise ValueError("unknown protocol hull backend")
        if not math.isfinite(selection_timeout_seconds) or selection_timeout_seconds <= 0:
            raise ValueError("protocol policy timeout must be finite and positive")
        self._persistent = worker_path is None
        self.worker_path = (
            worker_path or Path(__file__).with_name("protocol_policy_worker.py")
        ).resolve()
        if not self.worker_path.is_file():
            raise FileNotFoundError("protocol policy worker is unavailable")
        self._process: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[str | None] = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=50)
        self._last_selection_diagnostics: dict[str, Any] | None = None

    @property
    def last_selection_diagnostics(self) -> dict[str, Any] | None:
        """Observable policy-side diagnostics for the most recent selection.

        These values are computed before the reveal boundary. They are kept
        separately from the append-only action/reveal log so an audit can
        inspect numerical decision evidence without granting oracle access to
        the policy subprocess.
        """

        return self._last_selection_diagnostics

    @property
    def identity_checksum(self) -> str:
        return _checksum(
            {
                "policy": self.policy,
                "seed": self.seed,
                "ridge_penalty": self.ridge_penalty,
                "prior_standard_deviation": self.prior_standard_deviation,
                "boundary_temperature": self.boundary_temperature,
                "transport_model_checksum": (
                    None if self.transport_model is None else self.transport_model.identity_checksum
                ),
                "posterior_sample_count": self.posterior_sample_count,
                "fantasy_count": self.fantasy_count,
                "conformal_threshold": self.conformal_threshold,
                "hull_backend": self.hull_backend,
                "execution_mode": ("persistent_jsonl" if self._persistent else "one_shot_custom"),
                "worker_sha256": hashlib.sha256(self.worker_path.read_bytes()).hexdigest(),
            }
        )

    def _command(self) -> list[str]:
        command = [
            sys.executable,
            str(self.worker_path),
            "--policy",
            self.policy,
            "--seed",
            str(self.seed),
            "--ridge-penalty",
            str(self.ridge_penalty),
            "--prior-standard-deviation",
            str(self.prior_standard_deviation),
            "--boundary-temperature",
            str(self.boundary_temperature),
            "--posterior-sample-count",
            str(self.posterior_sample_count),
            "--fantasy-count",
            str(self.fantasy_count),
            "--hull-backend",
            self.hull_backend,
        ]
        if self.conformal_threshold is not None:
            command.extend(["--conformal-threshold", str(self.conformal_threshold)])
        return command

    def _serialized_request(self, state: ProtocolPolicyState) -> str:
        payload = json.loads(state.serialized_for_policy())
        if self.transport_model is not None:
            payload["transport_model"] = self.transport_model.model_dump(mode="json")
        payload["hull_backend"] = self.hull_backend
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _start_persistent(self) -> None:
        if self._process is not None:
            return
        process = subprocess.Popen(
            [*self._command(), "--serve-jsonl"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        if process.stdin is None or process.stdout is None or process.stderr is None:
            process.kill()
            raise RuntimeError("persistent protocol policy pipes are unavailable")
        self._process = process

        def read_stdout() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self._responses.put(line.rstrip("\r\n"))
            self._responses.put(None)

        def read_stderr() -> None:
            assert process.stderr is not None
            for line in process.stderr:
                self._stderr.append(line.rstrip("\r\n"))

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()

    def _select_one_shot(self, state: ProtocolPolicyState) -> str:
        result = subprocess.run(
            self._command(),
            input=self._serialized_request(state),
            text=True,
            capture_output=True,
            check=False,
            timeout=self.selection_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(f"protocol policy subprocess failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def _select_persistent(self, state: ProtocolPolicyState) -> str:
        self._start_persistent()
        process = self._process
        assert process is not None and process.stdin is not None
        if process.poll() is not None:
            raise RuntimeError("persistent protocol policy exited: " + "\n".join(self._stderr))
        try:
            process.stdin.write(self._serialized_request(state) + "\n")
            process.stdin.flush()
            selected = self._responses.get(timeout=self.selection_timeout_seconds)
        except (BrokenPipeError, queue.Empty) as exc:
            self.close()
            raise RuntimeError(
                "persistent protocol policy timed out or closed: " + "\n".join(self._stderr)
            ) from exc
        if selected is None:
            self.close()
            raise RuntimeError(
                "persistent protocol policy returned EOF: " + "\n".join(self._stderr)
            )
        return selected.strip()

    def select(self, state: ProtocolPolicyState) -> str:
        response = (
            self._select_persistent(state) if self._persistent else self._select_one_shot(state)
        )
        self._last_selection_diagnostics = None
        try:
            payload = json.loads(response)
        except json.JSONDecodeError:
            selected = response.strip()
        else:
            if not isinstance(payload, dict) or not isinstance(
                payload.get("selected_pair_id"), str
            ):
                raise RuntimeError("protocol policy subprocess returned an invalid response")
            selected = payload["selected_pair_id"]
            diagnostics = payload.get("diagnostics")
            if diagnostics is not None:
                if not isinstance(diagnostics, dict):
                    raise RuntimeError("protocol policy diagnostics must be an object")
                self._last_selection_diagnostics = diagnostics
        if selected not in {item.pair_id for item in state.queries}:
            raise RuntimeError("protocol policy subprocess returned an unknown pair ID")
        return selected

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


class ProtocolCausalHull:
    """Target-protocol causal hull updated only by authorized reveals."""

    def __init__(self, initial_entries: Iterable[Any], *, chemical_system: tuple[str, ...]) -> None:
        from pymatgen.analysis.phase_diagram import PhaseDiagram

        entries = list(initial_entries)
        if not entries:
            raise ValueError("protocol causal hull requires initial target entries")
        self.chemical_system = tuple(sorted(chemical_system))
        self._entries = entries
        self._diagram = PhaseDiagram(entries)

    @property
    def phase_set_checksum(self) -> str:
        payload = [
            {
                "entry_id": str(getattr(entry, "entry_id", "")),
                "composition": entry.composition.as_dict(),
                "total_energy_ev": float(entry.energy),
            }
            for entry in self._entries
        ]
        return _checksum(payload)

    @property
    def observable_phases(self) -> tuple[ObservableProtocolPhase, ...]:
        return tuple(
            ObservableProtocolPhase(
                entry_id=str(getattr(entry, "entry_id", "")),
                composition=entry.composition.as_dict(),
                formation_energy_ev_per_atom=float(self._diagram.get_form_energy_per_atom(entry)),
            )
            for entry in self._entries
        )

    def competing_hull_formation_energy(self, composition: dict[str, float]) -> float:
        from pymatgen.core import Composition
        from pymatgen.entries.computed_entries import ComputedEntry

        parsed = Composition(composition)
        if tuple(sorted(str(element) for element in parsed.elements)) != self.chemical_system:
            raise ValueError("candidate belongs to another exact chemical system")
        total_per_atom = float(self._diagram.get_hull_energy_per_atom(parsed))
        fake = ComputedEntry(parsed, total_per_atom * parsed.num_atoms)
        return float(self._diagram.get_form_energy_per_atom(fake))

    def add_revealed(self, outcome: ProtocolOracleOutcome) -> None:
        from pymatgen.analysis.phase_diagram import PhaseDiagram
        from pymatgen.entries.computed_entries import ComputedEntry

        if outcome.chemical_system != self.chemical_system:
            raise ValueError("revealed outcome belongs to another exact chemical system")
        self._entries.append(
            ComputedEntry(
                outcome.composition,
                outcome.target_corrected_total_energy_ev,
                entry_id=outcome.pair_id,
            )
        )
        self._diagram = PhaseDiagram(self._entries)


class ProtocolClosedLoopEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    round_index: int
    selected_pair_id: str
    pre_reveal_state_checksum: str
    action_checksum: str
    reveal_checksum: str
    post_reveal_hull_checksum: str
    archive_checksum: str
    selection_diagnostics: dict[str, Any] | None = None


class ProtocolClosedLoopResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selected_pair_ids: tuple[str, ...]
    revealed_pair_ids: tuple[str, ...]
    training_archive_policy: Literal["full_history"] = "full_history"
    events: tuple[ProtocolClosedLoopEvent, ...]
    trace_checksum: str


class SecureProtocolQueryRunner:
    """Run a real action-driven query trajectory for one exact system.

    The policy's selected ID is persisted before access and is then the only ID
    the vault can reveal.  All revealed outcomes remain in the full history;
    this runner does not claim or implement an expensive-model training
    coreset.
    """

    def __init__(
        self,
        *,
        candidates: Iterable[ProtocolCandidate],
        vault: ProtocolOracleVault,
        causal_hull: ProtocolCausalHull,
        policy: ProtocolPolicySubprocess,
        event_log: AppendOnlyProtocolEventLog,
    ) -> None:
        items = tuple(candidates)
        if not items or len({item.pair_id for item in items}) != len(items):
            raise ValueError("secure protocol runner requires unique nonempty candidates")
        systems = {item.chemical_system for item in items}
        if systems != {causal_hull.chemical_system}:
            raise ValueError("one protocol runner handles exactly one chemical system")
        self.candidates = {item.pair_id: item for item in items}
        self.vault = vault
        self.causal_hull = causal_hull
        self.policy = policy
        self.event_log = event_log

    def _observable_query(self, candidate: ProtocolCandidate) -> ObservableProtocolQuery:
        return ObservableProtocolQuery(
            pair_id=candidate.pair_id,
            source_structure_hash=candidate.source_structure_hash,
            chemical_system=candidate.chemical_system,
            composition=candidate.composition,
            source_formation_energy_ev_per_atom=(candidate.source_formation_energy_ev_per_atom),
            source_environment_embedding=candidate.source_environment_embedding,
            source_local_environment_embedding=(candidate.source_local_environment_embedding),
            current_competing_hull_ev_per_atom=(
                self.causal_hull.competing_hull_formation_energy(candidate.composition)
            ),
            source_protocol_fingerprint=candidate.source_protocol.scientific_fingerprint,
            target_protocol_fingerprint=candidate.target_protocol.scientific_fingerprint,
            oracle_cost=candidate.oracle_cost,
        )

    def run(self, *, oracle_budget: float) -> ProtocolClosedLoopResult:
        try:
            return self._run_with_open_policy(oracle_budget=oracle_budget)
        finally:
            self.policy.close()

    def _run_with_open_policy(self, *, oracle_budget: float) -> ProtocolClosedLoopResult:
        if oracle_budget <= 0:
            raise ValueError("protocol oracle budget must be positive")
        remaining = dict(self.candidates)
        history: list[RevealedProtocolObservation] = []
        events: list[ProtocolClosedLoopEvent] = []
        spent = 0.0
        round_index = 1
        conformal_deviation_used = False
        while remaining:
            affordable = tuple(
                candidate
                for candidate in remaining.values()
                if candidate.oracle_cost <= oracle_budget - spent + 1e-12
            )
            if not affordable:
                break
            state = ProtocolPolicyState.create(
                round_index=round_index,
                remaining_budget=oracle_budget - spent,
                queries=(self._observable_query(item) for item in affordable),
                causal_hull_phases=self.causal_hull.observable_phases,
                revealed_history=history,
                conformal_deviation_used=conformal_deviation_used,
                policy_identity_checksum=self.policy.identity_checksum,
            )
            selected_id = self.policy.select(state)
            selection_diagnostics = self.policy.last_selection_diagnostics
            if self.policy.policy == "conformal_source_rollout_delta_hull":
                source_index = int(
                    source_margin_action_indices(
                        source_energies=[
                            item.source_formation_energy_ev_per_atom for item in state.queries
                        ],
                        competing_hull_energies=[
                            item.current_competing_hull_ev_per_atom for item in state.queries
                        ],
                        query_ids=[item.pair_id for item in state.queries],
                    )[0]
                )
                source_id = state.queries[source_index].pair_id
                if selected_id != source_id:
                    conformal_deviation_used = True
            authorization = self.event_log.append_action(
                round_index=round_index,
                selected_pair_id=selected_id,
                pre_reveal_state_checksum=state.state_checksum,
            )
            candidate = remaining.pop(selected_id)
            outcome = self.vault.reveal(
                candidate,
                authorization=authorization,
                event_log=self.event_log,
            )
            spent += candidate.oracle_cost
            history.append(
                RevealedProtocolObservation(
                    pair_id=selected_id,
                    source_formation_energy_ev_per_atom=(
                        candidate.source_formation_energy_ev_per_atom
                    ),
                    revealed_target_formation_energy_ev_per_atom=(
                        outcome.target_formation_energy_ev_per_atom
                    ),
                    source_environment_embedding=candidate.source_environment_embedding,
                    source_local_environment_embedding=(
                        candidate.source_local_environment_embedding
                    ),
                )
            )
            self.causal_hull.add_revealed(outcome)
            archive_checksum = _checksum([item.model_dump(mode="json") for item in history])
            reveal = self.event_log.append_reveal(
                round_index=round_index,
                selected_pair_id=selected_id,
                action_checksum=authorization.action_checksum,
                outcome_checksum=_checksum(outcome.model_dump(mode="json")),
                post_reveal_hull_checksum=self.causal_hull.phase_set_checksum,
                archive_checksum=archive_checksum,
            )
            events.append(
                ProtocolClosedLoopEvent(
                    round_index=round_index,
                    selected_pair_id=selected_id,
                    pre_reveal_state_checksum=state.state_checksum,
                    action_checksum=authorization.action_checksum,
                    reveal_checksum=reveal.reveal_checksum,
                    post_reveal_hull_checksum=self.causal_hull.phase_set_checksum,
                    archive_checksum=archive_checksum,
                    selection_diagnostics=selection_diagnostics,
                )
            )
            round_index += 1
        selected = tuple(event.selected_pair_id for event in events)
        if selected != self.vault.revealed_pair_ids:
            raise AssertionError("selected actions and oracle reveals diverged")
        content = {
            "selected_pair_ids": selected,
            "revealed_pair_ids": self.vault.revealed_pair_ids,
            "training_archive_policy": "full_history",
            "events": [item.model_dump(mode="json") for item in events],
        }
        return ProtocolClosedLoopResult(**content, trace_checksum=_checksum(content))
