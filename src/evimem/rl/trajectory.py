"""Candidate trajectory recording and run-level audit projection."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from evimem.core.contracts import CurationStep, CurationTrajectory
from evimem.core.ids import deterministic_id

if TYPE_CHECKING:
    from evimem.controller.state import ActionRecord, ControllerState


class TrajectoryRecorder:
    """Builds a canonical trajectory from executor-owned action records."""

    def __init__(self, *, run_id: str, initial_state: ControllerState):
        self.run_id = run_id
        self.initial_state = initial_state
        self._steps: list[CurationStep] = []

    def append(self, *, prior_state: ControllerState, record: ActionRecord) -> None:
        self._steps.append(
            CurationStep(
                step=len(self._steps),
                state_hash=prior_state.state_hash(),
                action=record.action.type.value,
                action_args=record.action.arguments,
                rationale_code=record.action.rationale_code,
                result_digest=record.result_digest,
                cost=record.cost,
                verifier_delta=record.verifier_delta,
            )
        )

    def build(
        self,
        *,
        terminal_action: str,
        final_certificate_id: str | None = None,
        total_reward: float | None = None,
    ) -> CurationTrajectory:
        state = self.initial_state
        trajectory_id = deterministic_id(
            self.run_id,
            state.candidate.candidate_id,
            state.state_hash(),
            length=32,
            namespace="traj",
        )
        return CurationTrajectory(
            trajectory_id=trajectory_id,
            run_id=self.run_id,
            candidate_id=state.candidate.candidate_id,
            evidence_release_id=state.claim_state.evidence_release_id,
            domain_pack_id=state.claim_state.domain_pack_id,
            domain_pack_version=state.claim_state.domain_pack_version,
            domain_pack_hash=state.claim_state.domain_pack_hash,
            initial_state_hash=state.state_hash(),
            steps=tuple(self._steps),
            terminal_action=terminal_action,
            final_certificate_id=final_certificate_id,
            total_reward=total_reward,
        )


@dataclass(frozen=True)
class RunAuditEvent:
    event_id: str
    sequence: int
    event_type: str
    run_id: str
    doi: str
    stage: str
    actor: str
    action: str
    status: str
    success: bool
    decision_id: str = ""
    correlation_id: str = ""
    attempt: int | None = None
    elapsed_seconds: float | None = None
    reasoning: str = ""
    errors: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "sequence": self.sequence,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "doi": self.doi,
            "stage": self.stage,
            "actor": self.actor,
            "action": self.action,
            "status": self.status,
            "success": self.success,
            "decision_id": self.decision_id,
            "correlation_id": self.correlation_id,
            "attempt": self.attempt,
            "elapsed_seconds": self.elapsed_seconds,
            "reasoning": self.reasoning,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class RunAuditTrajectory:
    run_id: str
    doi: str
    final_state: str
    success: bool
    events: list[RunAuditEvent]
    schema_version: str = "evimem.run_audit.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "doi": self.doi,
            "final_state": self.final_state,
            "success": self.success,
            "event_count": len(self.events),
            "events": [event.to_dict() for event in self.events],
        }


def _iter_decisions(value: Any) -> Iterable[tuple[str, dict[str, Any]]]:
    if isinstance(value, dict):
        if value.get("decision_id") and value.get("stage") and value.get("actor"):
            yield "orchestrator_decision", value
        elif value.get("agent_name") and value.get("action") and "success" in value:
            yield "agent_action", value
        for child in value.values():
            yield from _iter_decisions(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_decisions(child)


def build_run_trajectory(
    *,
    run_id: str,
    doi: str,
    final_state: str,
    success: bool,
    stage_results: dict[str, Any],
) -> RunAuditTrajectory:
    """Project orchestration decisions into a deterministic run audit."""

    events: list[RunAuditEvent] = []
    seen: set[str] = set()

    def append(
        *,
        event_type: str,
        identity: str,
        stage: str,
        actor: str,
        action: str,
        status: str,
        succeeded: bool,
        decision_id: str = "",
        correlation_id: str = "",
        attempt: int | None = None,
        elapsed_seconds: float | None = None,
        reasoning: str = "",
        errors: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        sequence = len(events)
        digest = hashlib.sha256(
            f"{run_id}|{sequence}|{event_type}|{identity}".encode()
        ).hexdigest()[:24]
        events.append(
            RunAuditEvent(
                event_id=digest,
                sequence=sequence,
                event_type=event_type,
                run_id=run_id,
                doi=doi,
                stage=stage,
                actor=actor,
                action=action,
                status=status,
                success=succeeded,
                decision_id=decision_id,
                correlation_id=correlation_id,
                attempt=attempt,
                elapsed_seconds=elapsed_seconds,
                reasoning=reasoning,
                errors=list(errors or []),
                metadata=dict(metadata or {}),
            )
        )

    for result_key, value in stage_results.items():
        if result_key in {"trajectory", "publication_gate_trace"}:
            continue
        if result_key.endswith("_execution_meta") and isinstance(value, dict):
            stage = result_key.removesuffix("_execution_meta")
            for index, attempt_data in enumerate(value.get("attempt_history", []) or [], start=1):
                if not isinstance(attempt_data, dict):
                    continue
                attempt = int(
                    attempt_data.get("attempt_number", attempt_data.get("attempt", index))
                )
                error = attempt_data.get("error")
                append(
                    event_type="action_attempt",
                    identity=f"{stage}:{attempt}",
                    stage=stage,
                    actor="orchestrator",
                    action="execute_stage",
                    status=str(attempt_data.get("status") or ("failure" if error else "success")),
                    succeeded=not bool(error),
                    attempt=attempt,
                    elapsed_seconds=attempt_data.get("elapsed_seconds"),
                    errors=[{"message": str(error)}] if error else [],
                )
        if result_key == "publication_gate_decision" and isinstance(value, dict):
            decision_id = str(value.get("decision_id", ""))
            append(
                event_type="publication_gate",
                identity=decision_id or result_key,
                stage="publication_gate",
                actor="deterministic_harness",
                action=str(value.get("route", "curation_pending")),
                status="allowed" if value.get("allow_materialization") else "blocked",
                succeeded=bool(value.get("allow_materialization")),
                decision_id=decision_id,
                reasoning="; ".join(value.get("blocked_reasons", []) or []),
                metadata={
                    "target_state": value.get("target_state"),
                    "required_human_review": bool(value.get("required_human_review", False)),
                },
            )
        for event_type, decision in _iter_decisions(value):
            decision_id = str(decision.get("decision_id", ""))
            identity = decision_id or ":".join(
                str(decision.get(key, "")) for key in ("agent_name", "action", "trace_id")
            )
            if identity in seen:
                continue
            seen.add(identity)
            status = decision.get("status", "success" if decision.get("success") else "failure")
            status = getattr(status, "value", status)
            append(
                event_type=event_type,
                identity=identity,
                stage=str(decision.get("stage") or result_key),
                actor=str(decision.get("actor") or decision.get("agent_name") or "unknown"),
                action=str(decision.get("action") or decision.get("stage") or result_key),
                status=str(status),
                succeeded=bool(decision.get("success", False)),
                decision_id=decision_id,
                correlation_id=str(decision.get("correlation_id") or ""),
                elapsed_seconds=decision.get("elapsed_seconds"),
                reasoning=str(decision.get("reasoning") or ""),
                errors=[item for item in decision.get("errors", []) or [] if isinstance(item, dict)],
                metadata={"result_key": result_key},
            )

    return RunAuditTrajectory(run_id, doi, final_state, success, events)
