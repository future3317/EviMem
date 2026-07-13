"""Contract tests for the canonical action and gate trajectory."""

from evimem.rl.trajectory import build_run_trajectory


def test_trajectory_records_actions_attempts_and_publication_gate() -> None:
    stage_results = {
        "plan": {
            "decision_id": "decision-plan",
            "stage": "plan",
            "actor": "planner",
            "status": "success",
            "success": True,
            "output": {},
            "fallback_used": False,
            "errors": [],
        },
        "extract_round_1_execution_meta": {
            "attempt_history": [
                {"attempt": 1, "error": "timeout", "timed_out": True, "retryable": True},
                {"attempt": 2, "error": None, "elapsed_seconds": 0.2},
            ]
        },
        "publication_gate_decision": {
            "decision_id": "gate-1",
            "allow_materialization": False,
            "route": "curation_pending",
            "target_state": "curation_pending",
            "blocked_reasons": ["unverified_evidence"],
            "warning_reasons": [],
        },
    }

    trajectory = build_run_trajectory(
        run_id="run-1",
        doi="10.1000/trajectory",
        final_state="curation_pending",
        success=True,
        stage_results=stage_results,
    )

    payload = trajectory.to_dict()
    assert payload["schema_version"] == "evimem.run_audit.v1"
    assert [event["sequence"] for event in payload["events"]] == list(
        range(payload["event_count"])
    )
    assert any(event["event_type"] == "orchestrator_decision" for event in payload["events"])
    attempts = [event for event in payload["events"] if event["event_type"] == "action_attempt"]
    assert [event["attempt"] for event in attempts] == [1, 2]
    gate = next(event for event in payload["events"] if event["event_type"] == "publication_gate")
    assert gate["success"] is False
    assert gate["decision_id"] == "gate-1"
    assert gate["reasoning"] == "unverified_evidence"


def test_trajectory_ids_are_deterministic_and_decisions_are_deduplicated() -> None:
    decision = {
        "decision_id": "same-decision",
        "stage": "verify",
        "actor": "verification_agent",
        "status": "success",
        "success": True,
        "output": {},
        "errors": [],
    }
    stage_results = {"verify_round_1": decision, "verify": decision}

    first = build_run_trajectory(
        run_id="run-2",
        doi="10.1000/deterministic",
        final_state="verified",
        success=True,
        stage_results=stage_results,
    ).to_dict()
    second = build_run_trajectory(
        run_id="run-2",
        doi="10.1000/deterministic",
        final_state="verified",
        success=True,
        stage_results=stage_results,
    ).to_dict()

    assert first == second
    assert first["event_count"] == 1
