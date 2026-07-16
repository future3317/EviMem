"""Tests for the cost-aware DeepSeek SciMem-Update runner.

These tests never contact DeepSeek and never require an API key.
"""

from __future__ import annotations

import json
from pathlib import Path

from evimem.phase1b.ai_adjudication import AdjudicationPacket
from tools import run_deepseek_adjudication as deepseek_runner
from tools.run_deepseek_adjudication import (
    DEFAULT_JUROR_PROTOCOL_VERSION,
    JUROR_PROTOCOL_V2,
    JUROR_PROTOCOL_V3,
    JUROR_PROTOCOL_V4,
    JUROR_PROTOCOL_VERSION,
    JUROR_PROTOCOLS,
    JUROR_SYSTEM,
    ApiCallError,
    CallBudget,
    CallLedger,
    DeepSeekClient,
    ProviderResponse,
    UsageMeter,
    _gate_crossref,
    _load_call_ledger,
    _load_deepseek_api_key,
    _model_packet,
    _needs_second_review,
    _record_is_reusable,
    _run_juror,
    _stable_sample,
    _verify_run,
    main,
)


def make_packet(*, source_dataset: str = "SciREX") -> dict[str, object]:
    external_safe = {
        "id": "scirex:0001" if source_dataset == "SciREX" else "crossref:0001",
        "data": {
            "pair_id": "scirex:0001" if source_dataset == "SciREX" else "crossref:0001",
            "left_claim": "MethodA Metric for Task = 0.1 under {\"dataset\": \"D1\"}",
            "right_claim": "MethodB Metric for Task = 0.2 under {\"dataset\": \"D1\"}",
            "left_source": "source-left",
            "right_source": "source-right",
            "left_evidence_locator": "left:0-10",
            "right_evidence_locator": "right:0-10",
            "source_level_update_type": "none",
            "source_level_update_notice": "none",
        },
        "meta": {
            "left_evidence_checksum": "sha256:" + "a" * 64,
            "right_evidence_checksum": "sha256:" + "b" * 64,
            "source_dataset": source_dataset,
        },
    }
    if source_dataset == "Crossref/Retraction Watch":
        external_safe["data"]["source_level_update_type"] = "retraction"
        external_safe["data"]["source_level_update_notice"] = "source-level metadata only"
    return AdjudicationPacket.from_external_safe_record(
        external_safe, provenance="packet:external_safe"
    ).model_dump(mode="json")


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat_json(self, **kwargs: object) -> ProviderResponse:
        self.calls.append(kwargs)
        return ProviderResponse(
            content={
            "semantic_relation": "COMPATIBLE_DISTINCT",
                "scope_relation": "DIFFERENT_SCOPE",
                "authority_relation": "NOT_APPLICABLE",
                "evidence_sufficiency": "SUFFICIENT",
                "evidence_note": "Left and right use different methods.",
                "uncertainty_note": "",
            },
            model_id="deepseek-v4-pro",
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            request_id="test-request",
        )


def test_model_packet_excludes_source_ids_and_checksums() -> None:
    packet = make_packet()
    model_packet = _model_packet(packet)
    serialized = json.dumps(model_packet, ensure_ascii=False)

    assert "source-left" not in serialized
    assert "evidence_checksum" not in serialized
    assert model_packet["left"]["evidence_locator"] == "left:0-10"


def test_scope_first_protocol_preserves_safety_critical_rules() -> None:
    assert JUROR_PROTOCOL_VERSION == JUROR_PROTOCOL_V2
    assert JUROR_PROTOCOL_V2 in JUROR_PROTOCOLS
    assert "Apply this order internally" in JUROR_PROTOCOLS[JUROR_PROTOCOL_V2]
    assert JUROR_PROTOCOL_V3 in JUROR_PROTOCOLS
    assert DEFAULT_JUROR_PROTOCOL_VERSION == JUROR_PROTOCOL_V4
    assert JUROR_SYSTEM == JUROR_PROTOCOLS[JUROR_PROTOCOL_V4]
    assert "Gate 1, visible claim link" in JUROR_SYSTEM
    assert "scope is the RIGHT claim relative to the LEFT" in JUROR_SYSTEM
    assert "authority_relation must be UNRESOLVED" in JUROR_SYSTEM


def test_key_loader_reads_only_local_dotenv_assignment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    (tmp_path / ".env").write_text(
        "OTHER_SECRET=must-not-be-read\nDEEPSEEK_API_KEY='test-key'\n",
        encoding="utf-8",
    )

    assert _load_deepseek_api_key() == "test-key"


def test_economy_routing_keeps_clear_primary_out_of_second_review() -> None:
    clear = {
        "semantic_relation": "COMPATIBLE_DISTINCT",
        "scope_relation": "DIFFERENT_SCOPE",
        "authority_relation": "NOT_APPLICABLE",
        "evidence_sufficiency": "SUFFICIENT",
    }
    risky = {**clear, "semantic_relation": "CONTRADICTORY"}

    assert not _needs_second_review(clear, "scirex:0001", sample_rate=0)
    assert _needs_second_review(risky, "scirex:0001", sample_rate=0)
    assert _stable_sample("scirex:0001", "second-review", 0) is False
    assert _stable_sample("scirex:0001", "second-review", 1) is True


def test_crossref_is_gated_without_semantic_label() -> None:
    packet = make_packet(source_dataset="Crossref/Retraction Watch")
    gated = _gate_crossref(packet)

    assert gated["status"] == "gated_out_no_claim_level_evidence"
    assert "semantic_relation" not in gated


def test_juror_uses_real_response_and_true_model_id() -> None:
    packet = make_packet()
    client = FakeClient()

    record = _run_juror(
        client,
        packet,
        role="juror-a",
        run_id="test:juror-a",
        model="deepseek-v4-pro",
        max_tokens=160,
    )

    assert record["annotation_provenance"] == "ai_juror"
    assert record["model_id"] == "deepseek-v4-pro"
    assert record["prompt_checksum"] != "sha256:" + "0" * 64
    assert len(client.calls) == 1


def test_deepseek_payload_disables_thinking_without_reasoning_effort() -> None:
    class FakeResponse:
        status_code = 200
        headers = {"x-request-id": "safe-test-request"}

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "model": "deepseek-v4-pro",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ok": true}'},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class FakeSession:
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None

        def post(self, _url: str, *, json: dict[str, object], timeout: int) -> FakeResponse:
            self.payload = json
            assert timeout == 5
            return FakeResponse()

    client = DeepSeekClient(
        "not-a-real-key",
        base_url="https://api.deepseek.com",
        budget=CallBudget(1),
        usage_meter=UsageMeter(),
        json_mode=True,
        thinking_mode="disabled",
        timeout_seconds=5,
    )
    session = FakeSession()
    client._local.session = session  # type: ignore[attr-defined]  # Offline transport injection.

    client.chat_json(
        model="deepseek-v4-pro",
        system="system",
        user="user",
        temperature=0.1,
        max_tokens=128,
    )

    assert session.payload is not None
    assert session.payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in session.payload


def test_deepseek_client_rejects_a_returned_model_mismatch(monkeypatch) -> None:
    class FakeResponse:
        status_code = 200
        headers = {"x-request-id": "safe-test-request"}

        @staticmethod
        def json() -> dict[str, object]:
            return {
                "model": "other-model",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": '{"ok": true}'},
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }

    class FakeSession:
        def post(self, _url: str, **_kwargs: object) -> FakeResponse:
            return FakeResponse()

    client = DeepSeekClient(
        "not-a-real-key",
        base_url="https://api.deepseek.com",
        budget=CallBudget(1),
        usage_meter=UsageMeter(),
        json_mode=True,
        thinking_mode="disabled",
        timeout_seconds=5,
    )
    client._local.session = FakeSession()  # type: ignore[attr-defined]  # Offline transport injection.
    monkeypatch.setattr(deepseek_runner, "MAX_RETRIES", 1)

    try:
        client.chat_json(
            model="deepseek-v4-pro",
            system="system",
            user="user",
            temperature=0.1,
            max_tokens=128,
        )
    except ApiCallError as exc:
        assert "different from the requested" in str(exc)
    else:
        raise AssertionError("a returned non-Pro model must fail closed")


def test_ledger_proves_a_juror_record_and_rejects_unlogged_reuse(tmp_path: Path) -> None:
    packet = make_packet()
    ledger_path = tmp_path / "audit" / "api_call_ledger.jsonl"
    ledger = CallLedger(ledger_path, {})
    record = _run_juror(
        FakeClient(),
        packet,
        role="juror-a",
        run_id="test-run:juror-a",
        model="deepseek-v4-pro",
        max_tokens=160,
        ledger=ledger,
    )

    loaded = _load_call_ledger(ledger_path)
    assert len(loaded[("juror-a", "scirex:0001")]) == 1
    assert _record_is_reusable(
        record,
        packet,
        expected_provenance="ai_juror",
        expected_run_id="test-run:juror-a",
        stage="juror-a",
        ledger_events=loaded[("juror-a", "scirex:0001")],
        thinking_mode="disabled",
    )
    assert not _record_is_reusable(
        record,
        packet,
        expected_provenance="ai_juror",
        expected_run_id="test-run:juror-a",
        stage="juror-a",
        ledger_events=[],
        thinking_mode="disabled",
    )


def test_verify_run_rejects_unlogged_output(tmp_path: Path) -> None:
    packet = make_packet()
    record = _run_juror(
        FakeClient(),
        packet,
        role="juror-a",
        run_id="test-run:juror-a",
        model="deepseek-v4-pro",
        max_tokens=160,
    )
    votes = tmp_path / "run" / "votes"
    votes.mkdir(parents=True)
    (votes / "juror-a.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    valid, report = _verify_run(
        tmp_path / "run",
        {"scirex:0001": AdjudicationPacket.model_validate(packet)},
    )

    assert not valid
    assert report["audit_status"] == "failed"


def test_verify_run_accepts_output_with_matching_ledger(tmp_path: Path) -> None:
    packet = make_packet()
    run = tmp_path / "run"
    ledger = CallLedger(run / "audit" / "api_call_ledger.jsonl", {})
    record = _run_juror(
        FakeClient(),
        packet,
        role="juror-a",
        run_id="test-run:juror-a",
        model="deepseek-v4-pro",
        max_tokens=160,
        ledger=ledger,
    )
    votes = run / "votes"
    votes.mkdir(parents=True)
    (votes / "juror-a.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    valid, report = _verify_run(
        run,
        {"scirex:0001": AdjudicationPacket.model_validate(packet)},
    )

    assert valid
    assert report["ledger_event_count"] == 1


def test_external_runner_rejects_critic_and_judge_label_exposure(
    tmp_path: Path, monkeypatch
) -> None:
    packets_dir = tmp_path / "packets"
    packets_dir.mkdir()
    (packets_dir / "scirex_0001.json").write_text(
        json.dumps(make_packet()), encoding="utf-8"
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert deepseek_runner.main(
        [
            "--packets",
            str(packets_dir),
            "--output",
            str(tmp_path / "run"),
            "--run-id",
            "test-run",
            "--strategy",
            "full",
            "--max-workers",
            "1",
        ]
    ) == 1


def test_primary_only_mode_makes_no_silver_or_downstream_model_calls(
    tmp_path: Path, monkeypatch
) -> None:
    class FakePrimaryClient:
        calls = 0

        def __init__(self, *_args: object, usage_meter: UsageMeter, **_kwargs: object) -> None:
            self.usage_meter = usage_meter

        def chat_json(self, **_kwargs: object) -> ProviderResponse:
            type(self).calls += 1
            self.usage_meter.add(
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            )
            return ProviderResponse(
                content={
                    "semantic_relation": "UNRELATED",
                    "scope_relation": "DIFFERENT_SCOPE",
                    "authority_relation": "NOT_APPLICABLE",
                    "evidence_sufficiency": "SUFFICIENT",
                    "evidence_note": "The methods differ.",
                    "uncertainty_note": "",
                },
                model_id="deepseek-v4-pro",
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                request_id="test-primary-only",
            )

    packets_dir = tmp_path / "packets"
    packets_dir.mkdir()
    (packets_dir / "scirex_0001.json").write_text(
        json.dumps(make_packet()), encoding="utf-8"
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(deepseek_runner, "DeepSeekClient", FakePrimaryClient)

    assert deepseek_runner.main(
        [
            "--packets",
            str(packets_dir),
            "--output",
            str(tmp_path / "run"),
            "--run-id",
            "test-primary-only",
            "--primary-only",
        ]
    ) == 0

    manifest = json.loads(
        (tmp_path / "run" / "reports" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert FakePrimaryClient.calls == 1
    assert manifest["primary_only"] is True
    assert manifest["outputs"]["ai_adjudicated_silver"] == 0
    assert manifest["provenance_audit"]["audit_status"] == "passed"


def test_dry_run_needs_no_api_key(tmp_path: Path, monkeypatch) -> None:
    packets_dir = tmp_path / "packets"
    packets_dir.mkdir()
    packet = make_packet()
    (packets_dir / "scirex_0001.json").write_text(
        json.dumps(packet, ensure_ascii=False), encoding="utf-8"
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert main(["--packets", str(packets_dir), "--output", str(tmp_path / "run"), "--dry-run"]) == 0


def test_resume_rejects_a_changed_juror_protocol(tmp_path: Path, monkeypatch) -> None:
    packets_dir = tmp_path / "packets"
    packets_dir.mkdir()
    (packets_dir / "scirex_0001.json").write_text(
        json.dumps(make_packet()), encoding="utf-8"
    )
    run = tmp_path / "run"
    manifest = run / "reports" / "run_manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"juror_protocol_version": JUROR_PROTOCOL_V2}),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert main(
        [
            "--packets",
            str(packets_dir),
            "--output",
            str(run),
            "--run-id",
            "prior-v2-run",
            "--resume",
            "--primary-only",
            "--juror-protocol",
            JUROR_PROTOCOL_V3,
        ]
    ) == 1
