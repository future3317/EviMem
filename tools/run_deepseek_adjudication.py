"""Cost-aware, fail-closed DeepSeek annotation workflow for SciMem-Update.

This tool calls the OpenAI-compatible DeepSeek API only for DeepSeek V4 Pro
blind-primary candidate generation. It never generates semantic labels with
rules or placeholders, and no external call may receive another model's labels,
critic output, or silver output. Local code may compare completed candidate
exports, but it never selects a label winner.

Crossref/Retraction Watch packets are source-level metadata only.  They are
recorded as *gated out*, not converted into AI labels, because no visible
claim-level evidence exists for a model to adjudicate.

All output directories are runtime artifacts and must remain untracked.  Set a
rotated key in the process environment; this tool never accepts or writes a
key in a command argument, file, report, or manifest.

Example (PowerShell):

    conda run --no-capture-output -n llm python tools/run_deepseek_adjudication.py `
        --packets packets/ --output runs/deepseek-v4-pro-001/ --primary-only

Run ``--dry-run`` first to inspect the call plan without contacting DeepSeek.

Every validated annotation emitted by an authenticated run is linked to one
append-only ledger event.  The ledger holds checksums and aggregate-safe
metadata only: it deliberately does not retain prompts, raw responses,
reasoning text, request IDs, or credentials.  A resume is accepted only when
the existing annotation has a matching ledger event from the same stage/run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from evimem.phase1b.ai_adjudication.schema import (
    AdjudicationPacket,
    CriticReview,
    validate_ai_adjudication_label,
)
from evimem.phase1b.ai_adjudication.validate import load_packets, write_jsonl_records

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_MAX_CALLS = 600
MAX_RETRIES = 3
BACKOFF_SECONDS = 2.0
LEDGER_SCHEMA_VERSION = "deepseek-api-ledger-v1"
JUROR_PROTOCOL_V2 = "phase1b-v4-pro-scope-first-v2"
JUROR_PROTOCOL_V3 = "phase1b-v4-pro-evidence-bound-scope-v3"
JUROR_PROTOCOL_V4 = "phase1b-v4-pro-claim-link-gates-v4"
DEFAULT_JUROR_PROTOCOL_VERSION = JUROR_PROTOCOL_V4
# Backward-compatible name for code that needs to identify the frozen V2 audit.
JUROR_PROTOCOL_VERSION = JUROR_PROTOCOL_V2

LABEL_FIELDS = (
    "semantic_relation",
    "scope_relation",
    "authority_relation",
    "evidence_sufficiency",
)


class ApiCallError(RuntimeError):
    """A safe-to-log DeepSeek request failure without request/response bodies."""


class CallBudgetError(RuntimeError):
    """Raised before an API request would exceed the explicit run budget."""


class LedgerIntegrityError(RuntimeError):
    """Raised when an output record cannot be proven by the local call ledger."""


def _load_deepseek_api_key() -> str | None:
    """Read the key from the process environment or one ignored local .env file.

    Only the single ``DEEPSEEK_API_KEY`` assignment is considered.  The value
    is kept in process memory for the request client and is never logged,
    included in an exception, added to a manifest, or written back to disk.
    """
    environment_value = os.environ.get("DEEPSEEK_API_KEY")
    if environment_value:
        return environment_value

    dotenv_path = Path(".env")
    if not dotenv_path.is_file():
        return None
    try:
        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() != "DEEPSEEK_API_KEY":
                continue
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            return value or None
    except OSError:
        return None
    return None


@dataclass(frozen=True)
class ProviderResponse:
    """Structured API result with aggregate-safe usage metadata."""

    content: dict[str, Any]
    model_id: str
    usage: dict[str, int]
    request_id: str | None


@dataclass
class UsageMeter:
    """Thread-safe aggregate usage accounting; no prompts or responses are kept."""

    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: dict[str, int]) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)

    def as_dict(self) -> dict[str, int]:
        with self._lock:
            return {
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }


@dataclass
class CallBudget:
    """Limits attempted API calls, including retries, before money is spent."""

    maximum: int
    used: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def consume(self) -> None:
        with self._lock:
            if self.used >= self.maximum:
                raise CallBudgetError(
                    f"API call budget exhausted ({self.used}/{self.maximum})"
                )
            self.used += 1


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ledger_checksum(entry: dict[str, Any]) -> str:
    """Checksum a ledger event without recursively including its own digest."""
    unsigned = {key: value for key, value in entry.items() if key != "event_checksum"}
    return _sha256_json(unsigned)


def _annotation_checksum(record: dict[str, Any]) -> str:
    """Stable checksum that links one export record to one API ledger event."""
    return _sha256_json(record)


class CallLedger:
    """Append-only provenance ledger for validated, model-produced annotations.

    Records intentionally contain only hashes for provider-sensitive values.
    This gives an auditor a one-to-one link from an exported annotation to a
    successful API response without persisting chain-of-thought, request text,
    response bodies, credentials, or provider request identifiers.
    """

    def __init__(
        self, path: Path, existing: dict[tuple[str, str], list[dict[str, Any]]]
    ) -> None:
        self.path = path
        self._events = dict(existing)
        self.new_event_count = 0
        self._lock = threading.Lock()

    @property
    def event_count(self) -> int:
        return sum(len(events) for events in self._events.values())

    def get(self, stage: str, task_id: str) -> list[dict[str, Any]]:
        return list(self._events.get((stage, task_id), []))

    def append(
        self,
        *,
        stage: str,
        record: dict[str, Any],
        response: ProviderResponse,
        requested_model: str,
        thinking_mode: str,
        role_run_id: str,
    ) -> None:
        entry: dict[str, Any] = {
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "event_type": "validated_annotation_api_response",
            "stage": stage,
            "task_id": record["task_id"],
            "role_run_id": role_run_id,
            "annotation_provenance": record["annotation_provenance"],
            "packet_checksum": record["packet_checksum"],
            "prompt_checksum": record["prompt_checksum"],
            "annotation_checksum": _annotation_checksum(record),
            "response_content_checksum": _sha256_json(response.content),
            "model_requested": requested_model,
            "model_returned": response.model_id,
            "thinking_mode": thinking_mode,
            "provider_request_id_checksum": (
                _sha256_text(response.request_id) if response.request_id else None
            ),
            "usage": response.usage,
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        entry["event_checksum"] = _ledger_checksum(entry)
        key = (stage, record["task_id"])

        with self._lock:
            previous = self._events.get(key, [])
            if any(
                event.get("annotation_checksum") == entry["annotation_checksum"]
                for event in previous
            ):
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            self._events.setdefault(key, []).append(entry)
            self.new_event_count += 1


def _load_call_ledger(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Load and verify a ledger without reading any raw prompt or response data."""
    if not path.exists():
        return {}

    events: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerIntegrityError(
                    f"invalid ledger JSON at {path}:{line_number}"
                ) from exc
            required = {
                "ledger_schema_version",
                "event_type",
                "stage",
                "task_id",
                "role_run_id",
                "annotation_provenance",
                "packet_checksum",
                "prompt_checksum",
                "annotation_checksum",
                "response_content_checksum",
                "model_requested",
                "model_returned",
                "thinking_mode",
                "provider_request_id_checksum",
                "usage",
                "completed_at_utc",
                "event_checksum",
            }
            if set(event) != required:
                raise LedgerIntegrityError(
                    f"unexpected ledger schema at {path}:{line_number}"
                )
            if event["ledger_schema_version"] != LEDGER_SCHEMA_VERSION:
                raise LedgerIntegrityError(
                    f"unsupported ledger version at {path}:{line_number}"
                )
            if event["event_type"] != "validated_annotation_api_response":
                raise LedgerIntegrityError(
                    f"unexpected ledger event type at {path}:{line_number}"
                )
            if event.get("event_checksum") != _ledger_checksum(event):
                raise LedgerIntegrityError(
                    f"ledger checksum mismatch at {path}:{line_number}"
                )
            key = (event["stage"], event["task_id"])
            if any(
                existing_event["event_checksum"] == event["event_checksum"]
                for existing_event in events.get(key, [])
            ):
                raise LedgerIntegrityError(
                    f"duplicate ledger event checksum at {path}:{line_number}"
                )
            events.setdefault(key, []).append(event)
    return events


class DeepSeekClient:
    """Minimal OpenAI-compatible DeepSeek client that never logs credentials."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str,
        budget: CallBudget,
        usage_meter: UsageMeter,
        json_mode: bool,
        thinking_mode: str,
        timeout_seconds: int,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._budget = budget
        self._usage_meter = usage_meter
        self._json_mode = json_mode
        self._thinking_mode = thinking_mode
        self._timeout_seconds = timeout_seconds
        self._local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                }
            )
            self._local.session = session
        return session

    def chat_json(
        self,
        *,
        model: str,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> ProviderResponse:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            # DeepSeek's OpenAI-compatible API defaults thinking to enabled.
            # Annotation exports require concise final JSON, not preserved CoT.
            "thinking": {"type": self._thinking_mode},
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}

        last_error: ApiCallError | None = None
        for attempt in range(MAX_RETRIES):
            self._budget.consume()
            try:
                response = self._session().post(
                    f"{self._base_url}/chat/completions",
                    json=payload,
                    timeout=self._timeout_seconds,
                )
                if response.status_code >= 400:
                    raise ApiCallError(f"DeepSeek HTTP {response.status_code}")
                data = response.json()
                choices = data.get("choices")
                if not isinstance(choices, list) or not choices:
                    raise ApiCallError("DeepSeek response contains no choices")
                finish_reason = choices[0].get("finish_reason")
                if finish_reason == "length":
                    raise ApiCallError(
                        "DeepSeek response truncated by max_tokens; increase output limit"
                    )
                message = choices[0].get("message", {})
                content = message.get("content")
                if not isinstance(content, str):
                    raise ApiCallError("DeepSeek response content is not text")
                try:
                    parsed = _parse_json_object(content)
                except (json.JSONDecodeError, ApiCallError) as exc:
                    snippet = content[:120].replace("\n", " ")
                    raise ApiCallError(
                        f"DeepSeek response is not valid JSON (content length {len(content)}): {snippet}"
                    ) from exc
                usage = _usage_dict(data.get("usage"))
                returned_model = data.get("model")
                if returned_model != model:
                    raise ApiCallError(
                        "DeepSeek returned a model different from the requested V4 Pro"
                    )
                self._usage_meter.add(usage)
                return ProviderResponse(
                    content=parsed,
                    model_id=returned_model,
                    usage=usage,
                    request_id=response.headers.get("x-request-id"),
                )
            except (requests.RequestException, ValueError, KeyError, ApiCallError) as exc:
                last_error = (
                    exc if isinstance(exc, ApiCallError) else ApiCallError(type(exc).__name__)
                )
                if attempt == MAX_RETRIES - 1:
                    break
                time.sleep(BACKOFF_SECONDS * (2**attempt))
        raise last_error or ApiCallError("DeepSeek request failed")


def _parse_json_object(content: str) -> dict[str, Any]:
    """Accept raw JSON or a fenced JSON object; reject every other shape."""
    candidate = content.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.IGNORECASE)
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ApiCallError("DeepSeek response is not a JSON object")
    return value


def _usage_dict(raw_usage: Any) -> dict[str, int]:
    if not isinstance(raw_usage, dict):
        return {}
    return {
        field: int(raw_usage.get(field, 0) or 0)
        for field in ("prompt_tokens", "completion_tokens", "total_tokens")
    }


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_sample(task_id: str, stage: str, rate: float) -> bool:
    """Deterministic sampling controls cost without selecting semantic labels."""
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    bucket = int(hashlib.sha256(f"{stage}:{task_id}".encode()).hexdigest()[:8], 16)
    return bucket / 0xFFFFFFFF < rate


def _model_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Send only task-visible evidence, not checksums or source identifiers."""
    return {
        "task_id": packet["task_id"],
        "source_dataset": packet["source_dataset"],
        "left": {
            "claim_text": packet["left"]["claim_text"],
            "evidence_locator": packet["left"]["evidence_locator"],
        },
        "right": {
            "claim_text": packet["right"]["claim_text"],
            "evidence_locator": packet["right"]["evidence_locator"],
        },
        "source_level_update_type": packet["source_level_update_type"],
        "source_level_update_notice": packet["source_level_update_notice"],
    }


def _packet_model(packet: dict[str, Any]) -> AdjudicationPacket:
    """Rehydrate a packet before passing it to Pydantic-backed hard-rule checks."""
    return AdjudicationPacket.model_validate(packet)


def _annotation_view(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **{field: record[field] for field in LABEL_FIELDS},
        "evidence_note": record["evidence_note"],
        "uncertainty_note": record["uncertainty_note"],
    }


JUROR_SYSTEM_V2 = """You label one SciMem-Update packet using only visible packet text. Return JSON only; do not emit reasoning steps. Do not add facts, aliases, conditions, dates, certificates, or authority evidence.
Allowed values: semantic_relation={EQUIVALENT, COMPATIBLE_DISTINCT, CONTRADICTORY, UNRELATED, INSUFFICIENT_CONTEXT}; scope_relation={SAME_SCOPE, NARROWER_SCOPE, BROADER_SCOPE, DIFFERENT_SCOPE, UNKNOWN_SCOPE}; authority_relation={NEWER_MORE_AUTHORITATIVE, OLDER_MORE_AUTHORITATIVE, EQUAL_AUTHORITY, UNRESOLVED, NOT_APPLICABLE}; evidence_sufficiency={SUFFICIENT, PARTIAL, INSUFFICIENT}.
Apply this order internally: (1) identify the visible subject, relation/direction, population/material, method, endpoint/metric, conditions/dataset, time, and measurement setting for both claims; (2) decide scope; (3) decide whether propositions can both be true; (4) assess authority only if a same-scope incompatibility makes it relevant; (5) assess evidence sufficiency. A short evidence_note must state only visible facts.
SAME_SCOPE requires every comparison-changing dimension to match. A general claim versus a more specific method, population, condition, dataset, or experiment is normally COMPATIBLE_DISTINCT plus NARROWER_SCOPE, not EQUIVALENT plus SAME_SCOPE. EQUIVALENT is only harmless rewording of the same proposition with no changed scope. CONTRADICTORY requires explicitly incompatible outcomes under the same scope; different mechanisms, modifications, biomarkers, or ambiguous verbs do not become contradictory without visible mutual exclusion. Do not infer that "disrupt" means reduce or reverse an outcome.
For SciREX, different methods require DIFFERENT_SCOPE. If both claims name the same task after "for", they are scientifically related: different method, dataset, metric, or value is normally COMPATIBLE_DISTINCT plus DIFFERENT_SCOPE, never UNRELATED. Use UNRELATED only when the visible claims have no scientifically useful relation (for example, different task and endpoint).
Use NOT_APPLICABLE for compatible, unrelated, or insufficient pairs unless visible evidence makes an authority comparison relevant. Same source wording or a newer date alone is not EQUAL_AUTHORITY or higher authority. A same-scope contradiction requires UNRESOLVED authority without visible claim-level correction. Use INSUFFICIENT only when the visible pair cannot responsibly decide the relation, not merely because the full paper is unavailable.
Never use these operation words in evidence_note or uncertainty_note: ADD, MERGE, LINK, CONFLICT, SUPERSEDE, IGNORE.
Return keys semantic_relation, scope_relation, authority_relation, evidence_sufficiency, evidence_note, uncertainty_note."""

JUROR_SYSTEM_V3 = """Label one SciMem-Update packet using only visible packet text. Return JSON only; do not emit reasoning steps or add facts, aliases, conditions, dates, certificates, or authority evidence.
Allowed values: semantic_relation={EQUIVALENT, COMPATIBLE_DISTINCT, CONTRADICTORY, UNRELATED, INSUFFICIENT_CONTEXT}; scope_relation={SAME_SCOPE, NARROWER_SCOPE, BROADER_SCOPE, DIFFERENT_SCOPE, UNKNOWN_SCOPE}; authority_relation={NEWER_MORE_AUTHORITATIVE, OLDER_MORE_AUTHORITATIVE, EQUAL_AUTHORITY, UNRESOLVED, NOT_APPLICABLE}; evidence_sufficiency={SUFFICIENT, PARTIAL, INSUFFICIENT}.
Decide scope before semantic relation. Scope is directional: B means the RIGHT claim relative to the LEFT claim. RIGHT is NARROWER_SCOPE only if its visible population, cell line, dose, intervention, condition, dataset, or endpoint is a strict subset of LEFT; if RIGHT is the general mechanism and LEFT is the qualified case, use BROADER_SCOPE. SAME_SCOPE requires every comparison-changing dimension to match.
Do not equate an unnamed "intervention", "candidate gene", "these cells", or other unspecified referent with a named entity in the other claim. A mechanism or before/after result is not evidence for a named intervention or outcome unless the right claim visibly states that link. When a missing link blocks a relation or scope judgment, use INSUFFICIENT_CONTEXT/UNKNOWN_SCOPE and PARTIAL or INSUFFICIENT evidence rather than inferred support.
EQUIVALENT is only the same proposition after harmless wording changes. Added mechanism, outcome, condition, or experimental setting is normally COMPATIBLE_DISTINCT. CONTRADICTORY requires explicit incompatible outcomes; a broad claim versus an opposite result in a specific dose, cell type, ICU, or other subset is not SAME_SCOPE. Assess authority only for a same-scope incompatibility; otherwise use NOT_APPLICABLE unless visible provenance establishes relevance.
For SciREX, different methods require DIFFERENT_SCOPE. When both claims name the same task after "for", different method, dataset, metric, or value is normally COMPATIBLE_DISTINCT plus DIFFERENT_SCOPE, never UNRELATED. Use UNRELATED only when visible claims have no scientifically useful relation, such as different tasks and endpoints.
Never use these operation words in evidence_note or uncertainty_note: ADD, MERGE, LINK, CONFLICT, SUPERSEDE, IGNORE.
Return keys semantic_relation, scope_relation, authority_relation, evidence_sufficiency, evidence_note, uncertainty_note."""

JUROR_SYSTEM_V4 = """Label one SciMem-Update packet using only the visible packet text. Return one JSON object only; do not emit reasoning steps. Never add background knowledge, aliases, abbreviations, conditions, dates, certificates, causal links, or authority evidence.
Allowed values: semantic_relation={EQUIVALENT, COMPATIBLE_DISTINCT, CONTRADICTORY, UNRELATED, INSUFFICIENT_CONTEXT}; scope_relation={SAME_SCOPE, NARROWER_SCOPE, BROADER_SCOPE, DIFFERENT_SCOPE, UNKNOWN_SCOPE}; authority_relation={NEWER_MORE_AUTHORITATIVE, OLDER_MORE_AUTHORITATIVE, EQUAL_AUTHORITY, UNRESOLVED, NOT_APPLICABLE}; evidence_sufficiency={SUFFICIENT, PARTIAL, INSUFFICIENT}.

Use these fail-closed gates in order. Gate 1, visible claim link: each claimed entity, intervention, outcome, direction, and condition must be visibly linked in both claims. Do not equate an unnamed programme, intervention, candidate gene, biomarker, cell, method, or pronoun with a named item. Do not use domain knowledge (for example, an alias or a biomarker relationship) to fill a missing link. If a needed link is missing, return INSUFFICIENT_CONTEXT / UNKNOWN_SCOPE / NOT_APPLICABLE and PARTIAL or INSUFFICIENT.

Gate 2, directional scope: scope is the RIGHT claim relative to the LEFT. Ask only: is the visible RIGHT proposition a strict subset of the visible LEFT proposition? If yes use NARROWER_SCOPE. Is the visible LEFT proposition a strict subset of the visible RIGHT proposition? If yes use BROADER_SCOPE. A named dose, cell type, population, experimental setting, route, method, or endpoint added on the RIGHT normally makes RIGHT NARROWER_SCOPE. If neither strict subset relation is visible, use DIFFERENT_SCOPE; use UNKNOWN_SCOPE when Gate 1 prevented the comparison. Never choose BROADER_SCOPE merely because the RIGHT gives more detail.

Gate 3, semantic relation: EQUIVALENT requires harmless rewording of exactly the same proposition. COMPATIBLE_DISTINCT is for compatible propositions with a visibly different or nested scope. CONTRADICTORY requires explicitly incompatible outcomes in the same visible scope. A broad claim and an apparently opposite result in a specific cell type, dose, ICU, or other subset is not SAME_SCOPE. If interpretation of a statistic, mechanism, or missing outcome is needed, do not infer it: use INSUFFICIENT_CONTEXT.

Gate 4, authority and evidence: the packet supplies no claim-level correction, certificate, or curator decision. Thus if (and only if) you choose CONTRADICTORY plus SAME_SCOPE, authority_relation must be UNRESOLVED. Otherwise use NOT_APPLICABLE. Use SUFFICIENT only when the visible pair supports every chosen axis without an inferred link. The evidence_note must point to exact visible wording only; never write an alias or fact absent from the two claims.

Final self-check before emitting JSON: if your note says RIGHT is more specific, scope_relation must be NARROWER_SCOPE; if it says LEFT is more specific, use BROADER_SCOPE. If your conclusion depends on a fact not literally visible, downgrade it under Gate 1. Never use these operation words in evidence_note or uncertainty_note: ADD, MERGE, LINK, CONFLICT, SUPERSEDE, IGNORE.
Return keys semantic_relation, scope_relation, authority_relation, evidence_sufficiency, evidence_note, uncertainty_note."""

JUROR_PROTOCOLS = {
    JUROR_PROTOCOL_V2: JUROR_SYSTEM_V2,
    JUROR_PROTOCOL_V3: JUROR_SYSTEM_V3,
    JUROR_PROTOCOL_V4: JUROR_SYSTEM_V4,
}
# Compatibility alias for callers that import the default active system prompt.
JUROR_SYSTEM = JUROR_PROTOCOLS[DEFAULT_JUROR_PROTOCOL_VERSION]

JUROR_A_USER = """Act as blind juror A. Read this one packet and return the JSON object.
{packet}"""

JUROR_B_USER = """Act as blind juror B. Independently review this one packet. Check population/material, method, endpoint/property, condition, time, and measurement scope before returning the JSON object.
{packet}"""

CRITIC_SYSTEM = """You audit two blinded model annotations against one visible packet. Return JSON only as {"issues": [...]}.
Each issue has axis (semantic|scope|authority|evidence|note), issue_type, evidence_locator_ref copied exactly from the packet, and one short explanation. Find only evidence-supported errors: entity conflation, endpoint swap, scope omission, authority overreach, statistical overreach, or unstated facts. Do not assign a final label.
Never use these operation words in issue explanations: ADD, MERGE, LINK, CONFLICT, SUPERSEDE, IGNORE.
Return an empty issues list if no error is supported."""

CRITIC_USER = """Packet:
{packet}
Juror A:
{juror_a}
Juror B:
{juror_b}"""

JUDGE_SYSTEM = """You adjudicate one SciMem-Update packet using only visible packet evidence plus two blinded annotations and a critic review. Return JSON only.
Never choose a label solely because jurors agree. Do not invent missing facts. For SciREX packets, different model/method names require scope_relation=DIFFERENT_SCOPE even if the task and dataset match. A same-scope contradiction requires UNRESOLVED authority unless visible claim-level correction resolves it.
Never use these operation words in evidence_note or uncertainty_note: ADD, MERGE, LINK, CONFLICT, SUPERSEDE, IGNORE.
semantic_relation must be exactly one of: EQUIVALENT, COMPATIBLE_DISTINCT, CONTRADICTORY, UNRELATED, INSUFFICIENT_CONTEXT. Do not invent values such as NOT_COMPARABLE.
evidence_locator_refs must be a JSON array of one or both exact evidence_locator strings from the packet. Copy them exactly, do not summarize or omit.
Return semantic_relation, scope_relation, authority_relation, evidence_sufficiency, evidence_note, uncertainty_note, requires_higher_tier_ai_review, evidence_locator_refs."""

JUDGE_USER = """Packet:
{packet}
Juror A:
{juror_a}
Juror B:
{juror_b}
Critic:
{critic}"""


def _prompt_checksum(system: str, user: str) -> str:
    return _sha256_json({"system": system, "user": user})


def _juror_record(
    packet: dict[str, Any],
    response: ProviderResponse,
    *,
    run_id: str,
) -> dict[str, Any]:
    content = response.content
    return {
        "task_id": packet["task_id"],
        **{field: content[field] for field in LABEL_FIELDS},
        "evidence_note": content["evidence_note"],
        "uncertainty_note": content["uncertainty_note"],
        "annotation_provenance": "ai_juror",
        "annotator_id": run_id,
        "model_id": response.model_id,
        "packet_checksum": packet["packet_checksum"],
        "schema_version": "phase1b-v3",
        "gold_status": "not_gold",
        "juror_run_id": run_id,
    }


def _record_validated_api_annotation(
    record: dict[str, Any],
    response: ProviderResponse,
    *,
    ledger: CallLedger | None,
    stage: str,
    requested_model: str,
    thinking_mode: str,
    role_run_id: str,
) -> dict[str, Any]:
    """Persist the safe provenance proof only after schema validation succeeds."""
    if ledger is not None:
        ledger.append(
            stage=stage,
            record=record,
            response=response,
            requested_model=requested_model,
            thinking_mode=thinking_mode,
            role_run_id=role_run_id,
        )
    return record


def _run_juror(
    client: DeepSeekClient,
    packet: dict[str, Any],
    *,
    role: str,
    run_id: str,
    model: str,
    max_tokens: int,
    juror_protocol: str = DEFAULT_JUROR_PROTOCOL_VERSION,
    thinking_mode: str = "disabled",
    ledger: CallLedger | None = None,
) -> dict[str, Any]:
    try:
        juror_system = JUROR_PROTOCOLS[juror_protocol]
    except KeyError as exc:
        raise ValueError(f"unknown juror protocol: {juror_protocol}") from exc
    user = (JUROR_A_USER if role == "juror-a" else JUROR_B_USER).format(
        packet=json.dumps(_model_packet(packet), ensure_ascii=False, separators=(",", ":"))
    )
    response = client.chat_json(
        model=model,
        system=juror_system,
        user=user,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    record = _juror_record(packet, response, run_id=run_id)
    record["prompt_checksum"] = _prompt_checksum(juror_system, user)
    validated = dict(validate_ai_adjudication_label(record, packet=_packet_model(packet)))
    return _record_validated_api_annotation(
        validated,
        response,
        ledger=ledger,
        stage=role,
        requested_model=model,
        thinking_mode=thinking_mode,
        role_run_id=run_id,
    )


def _critic_record(
    packet: dict[str, Any],
    juror_a: dict[str, Any],
    juror_b: dict[str, Any],
    response: ProviderResponse,
    *,
    run_id: str,
    prompt_checksum: str,
) -> dict[str, Any]:
    issues = response.content.get("issues")
    if not isinstance(issues, list):
        raise ApiCallError("critic response does not contain an issues list")
    record = {
        "task_id": packet["task_id"],
        "critic_run_id": run_id,
        "juror_run_ids": (juror_a["juror_run_id"], juror_b["juror_run_id"]),
        "packet_checksum": packet["packet_checksum"],
        "issues": issues,
        "model_id": response.model_id,
        "prompt_checksum": prompt_checksum,
        "annotation_provenance": "ai_critic",
        "schema_version": "phase1b-v3",
    }
    validated = CriticReview.model_validate(record).model_dump(mode="json")
    allowed_locators = {packet["left"]["evidence_locator"], packet["right"]["evidence_locator"]}
    if any(issue["evidence_locator_ref"] not in allowed_locators for issue in validated["issues"]):
        raise ApiCallError("critic referenced an evidence locator absent from the packet")
    return validated


def _run_critic(
    client: DeepSeekClient,
    packet: dict[str, Any],
    juror_a: dict[str, Any],
    juror_b: dict[str, Any],
    *,
    run_id: str,
    model: str,
    max_tokens: int,
    thinking_mode: str = "disabled",
    ledger: CallLedger | None = None,
) -> dict[str, Any]:
    user = CRITIC_USER.format(
        packet=json.dumps(_model_packet(packet), ensure_ascii=False, separators=(",", ":")),
        juror_a=json.dumps(_annotation_view(juror_a), ensure_ascii=False, separators=(",", ":")),
        juror_b=json.dumps(_annotation_view(juror_b), ensure_ascii=False, separators=(",", ":")),
    )
    response = client.chat_json(
        model=model,
        system=CRITIC_SYSTEM,
        user=user,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    validated = _critic_record(
        packet,
        juror_a,
        juror_b,
        response,
        run_id=run_id,
        prompt_checksum=_prompt_checksum(CRITIC_SYSTEM, user),
    )
    return _record_validated_api_annotation(
        validated,
        response,
        ledger=ledger,
        stage="critic",
        requested_model=model,
        thinking_mode=thinking_mode,
        role_run_id=run_id,
    )


def _judge_record(
    packet: dict[str, Any],
    juror_a: dict[str, Any],
    juror_b: dict[str, Any],
    critic: dict[str, Any],
    response: ProviderResponse,
    *,
    run_id: str,
    prompt_checksum: str,
) -> dict[str, Any]:
    content = response.content
    record = {
        "task_id": packet["task_id"],
        **{field: content[field] for field in LABEL_FIELDS},
        "evidence_note": content["evidence_note"],
        "uncertainty_note": content["uncertainty_note"],
        "annotation_provenance": "ai_adjudicated_silver",
        "annotator_id": run_id,
        "model_id": response.model_id,
        "prompt_checksum": prompt_checksum,
        "packet_checksum": packet["packet_checksum"],
        "schema_version": "phase1b-v3",
        "gold_status": "not_gold",
        "juror_run_ids": (juror_a["juror_run_id"], juror_b["juror_run_id"]),
        "critic_run_id": critic["critic_run_id"],
        "adjudication_path": "deepseek-primary+deepseek-second+deepseek-critic->deepseek-judge",
        "evidence_locator_refs": tuple(content["evidence_locator_refs"]),
        "requires_higher_tier_ai_review": content["requires_higher_tier_ai_review"],
    }
    validated = dict(validate_ai_adjudication_label(record, packet=_packet_model(packet)))
    allowed_locators = {packet["left"]["evidence_locator"], packet["right"]["evidence_locator"]}
    refs = set(validated["evidence_locator_refs"])
    if not refs or not refs <= allowed_locators:
        raise ApiCallError("judge evidence_locator_refs must be non-empty packet locators")
    return validated


def _run_judge(
    client: DeepSeekClient,
    packet: dict[str, Any],
    juror_a: dict[str, Any],
    juror_b: dict[str, Any],
    critic: dict[str, Any],
    *,
    run_id: str,
    model: str,
    max_tokens: int,
    thinking_mode: str = "disabled",
    ledger: CallLedger | None = None,
) -> dict[str, Any]:
    user = JUDGE_USER.format(
        packet=json.dumps(_model_packet(packet), ensure_ascii=False, separators=(",", ":")),
        juror_a=json.dumps(_annotation_view(juror_a), ensure_ascii=False, separators=(",", ":")),
        juror_b=json.dumps(_annotation_view(juror_b), ensure_ascii=False, separators=(",", ":")),
        critic=json.dumps(critic["issues"], ensure_ascii=False, separators=(",", ":")),
    )
    response = client.chat_json(
        model=model,
        system=JUDGE_SYSTEM,
        user=user,
        temperature=0.0,
        max_tokens=max_tokens,
    )
    validated = _judge_record(
        packet,
        juror_a,
        juror_b,
        critic,
        response,
        run_id=run_id,
        prompt_checksum=_prompt_checksum(JUDGE_SYSTEM, user),
    )
    return _record_validated_api_annotation(
        validated,
        response,
        ledger=ledger,
        stage="judge",
        requested_model=model,
        thinking_mode=thinking_mode,
        role_run_id=run_id,
    )


def _needs_second_review(record: dict[str, Any], task_id: str, sample_rate: float) -> bool:
    """Route risk indicators, not semantic outcomes, to a second blind call."""
    return (
        record["evidence_sufficiency"] != "SUFFICIENT"
        or record["semantic_relation"] in {"CONTRADICTORY", "INSUFFICIENT_CONTEXT"}
        or record["scope_relation"] in {"SAME_SCOPE", "UNKNOWN_SCOPE"}
        or _stable_sample(task_id, "second-review", sample_rate)
    )


def _jurors_disagree(juror_a: dict[str, Any], juror_b: dict[str, Any]) -> bool:
    return any(juror_a[field] != juror_b[field] for field in LABEL_FIELDS)


def _gate_crossref(packet: dict[str, Any]) -> dict[str, str]:
    return {
        "task_id": packet["task_id"],
        "source_dataset": packet["source_dataset"],
        "packet_checksum": packet["packet_checksum"],
        "status": "gated_out_no_claim_level_evidence",
        "reason": "source-level metadata cannot support claim-level AI annotation",
    }


def _load_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            task_id = record.get("task_id")
            if not task_id or task_id in records:
                raise ValueError(f"invalid or duplicate task_id at {path}:{line_number}")
            records[task_id] = record
    return records


def _atomic_write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    ordered = sorted(records, key=lambda record: record["task_id"])
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_jsonl_records(temporary, ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.replace(path)


def _record_is_reusable(
    record: dict[str, Any],
    packet: dict[str, Any],
    *,
    expected_provenance: str,
    expected_run_id: str,
    stage: str,
    ledger_events: list[dict[str, Any]],
    thinking_mode: str,
) -> bool:
    """Allow resume only for a record cryptographically linked to this run."""
    if record.get("annotation_provenance") != expected_provenance:
        return False
    if record.get("packet_checksum") != packet["packet_checksum"]:
        return False
    if not record.get("prompt_checksum") or record["prompt_checksum"] == "sha256:" + "0" * 64:
        return False
    ledger_matches = any(
        event.get("stage") == stage
        and event.get("role_run_id") == expected_run_id
        and event.get("annotation_provenance") == expected_provenance
        and event.get("packet_checksum") == record["packet_checksum"]
        and event.get("prompt_checksum") == record["prompt_checksum"]
        and event.get("annotation_checksum") == _annotation_checksum(record)
        and event.get("thinking_mode") == thinking_mode
        for event in ledger_events
    )
    if not ledger_matches:
        return False
    try:
        if expected_provenance == "ai_juror":
            validated = validate_ai_adjudication_label(record, packet=_packet_model(packet))
            return validated.get("juror_run_id") == expected_run_id
        if expected_provenance == "ai_critic":
            validated = CriticReview.model_validate(record)
            return validated.critic_run_id == expected_run_id
        validated = validate_ai_adjudication_label(record, packet=_packet_model(packet))
        return validated.get("annotator_id") == expected_run_id
    except ValueError:
        return False


def _parallel_calls(
    tasks: list[tuple[str, Callable[[], dict[str, Any]]]],
    *,
    max_workers: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    completed: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call): task_id for task_id, call in tasks}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                completed[task_id] = future.result()
            except Exception as exc:  # noqa: BLE001 - errors are reduced before logging.
                failures.append({"task_id": task_id, "error_type": type(exc).__name__})
    return completed, failures


def _packet_set_checksum(packets: dict[str, Any]) -> str:
    return _sha256_json({task_id: packet.packet_checksum for task_id, packet in sorted(packets.items())})


def _verify_run(
    output: Path,
    packets: dict[str, AdjudicationPacket],
    *,
    thinking_mode: str = "disabled",
) -> tuple[bool, dict[str, Any]]:
    """Verify that every exported annotation is backed by a matching ledger event.

    This is offline and does not need an API key.  It establishes provenance
    linkage and safety-rule validity only; it does not claim semantic accuracy
    or human review.
    """
    ledger_path = output / "audit" / "api_call_ledger.jsonl"
    try:
        ledger = CallLedger(ledger_path, _load_call_ledger(ledger_path))
    except LedgerIntegrityError as exc:
        return False, {"audit_status": "failed", "errors": [str(exc)]}

    stage_specs = (
        ("juror-a", output / "votes" / "juror-a.jsonl", "ai_juror"),
        ("juror-b", output / "votes" / "juror-b.jsonl", "ai_juror"),
        ("critic", output / "reviews" / "critic.jsonl", "ai_critic"),
        ("judge", output / "silver" / "ai_adjudicated_silver.jsonl", "ai_adjudicated_silver"),
    )
    errors: list[str] = []
    error_count = 0

    def add_error(message: str) -> None:
        nonlocal error_count
        error_count += 1
        if len(errors) < 20:
            errors.append(message)

    output_keys: set[tuple[str, str, str]] = set()
    stage_counts: dict[str, int] = {}

    for stage, path, provenance in stage_specs:
        try:
            records = _load_jsonl_index(path)
        except (ValueError, json.JSONDecodeError) as exc:
            add_error(f"cannot read {stage} output: {type(exc).__name__}")
            continue
        stage_counts[stage] = len(records)
        for task_id, record in records.items():
            packet = packets.get(task_id)
            if packet is None:
                add_error(f"{stage}:{task_id} is absent from packet set")
                continue
            role_run_id = (
                record.get("juror_run_id")
                if provenance == "ai_juror"
                else record.get("critic_run_id")
                if provenance == "ai_critic"
                else record.get("annotator_id")
            )
            if not isinstance(role_run_id, str):
                add_error(f"{stage}:{task_id} has no role run identifier")
                continue
            if not _record_is_reusable(
                record,
                packet.model_dump(mode="json"),
                expected_provenance=provenance,
                expected_run_id=role_run_id,
                stage=stage,
                ledger_events=ledger.get(stage, task_id),
                thinking_mode=thinking_mode,
            ):
                add_error(f"{stage}:{task_id} lacks a matching ledger event")
                continue
            output_keys.add((stage, task_id, _annotation_checksum(record)))

    ledger_keys = {
        (stage, task_id, event["annotation_checksum"])
        for (stage, task_id), events in ledger._events.items()
        for event in events
    }
    orphaned_events = ledger_keys - output_keys
    if orphaned_events:
        add_error(f"{len(orphaned_events)} ledger events have no matching current output record")

    result: dict[str, Any] = {
        "audit_status": "passed" if error_count == 0 else "failed",
        "ledger_path": str(ledger_path),
        "ledger_event_count": ledger.event_count,
        "output_counts": stage_counts,
        "orphaned_ledger_event_count": len(orphaned_events),
        "error_count": error_count,
        "errors": errors,
    }
    return error_count == 0, result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=(DEFAULT_MODEL,),
        default=DEFAULT_MODEL,
        help="Only DeepSeek V4 Pro is permitted for external annotation calls.",
    )
    parser.add_argument(
        "--juror-protocol",
        choices=tuple(JUROR_PROTOCOLS),
        default=DEFAULT_JUROR_PROTOCOL_VERSION,
        help="Versioned blind-primary prompt; V2 remains frozen for reproducibility.",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument(
        "--strategy",
        choices=("economy", "full"),
        default="economy",
        help="Legacy planning field; authenticated execution is always --primary-only.",
    )
    parser.add_argument("--second-review-sample-rate", type=float, default=0.10)
    parser.add_argument("--critic-sample-rate", type=float, default=0.05)
    parser.add_argument("--max-api-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    parser.add_argument(
        "--thinking-mode",
        choices=("disabled", "enabled"),
        default="disabled",
        help="DeepSeek thinking toggle. Disabled by default to avoid CoT token spend.",
    )
    parser.add_argument("--juror-max-tokens", type=int, default=768)
    parser.add_argument("--critic-max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=896)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--primary-only",
        action="store_true",
        help=(
            "Run one blind juror pass only. Outputs ai_juror candidates, never "
            "critic or ai_adjudicated_silver records."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verify-run",
        action="store_true",
        help="Audit output records against the append-only API ledger without a key or network.",
    )
    parser.add_argument("--no-json-mode", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.model != DEFAULT_MODEL:
        raise ValueError(f"--model must be {DEFAULT_MODEL}")
    for name in ("second_review_sample_rate", "critic_sample_rate"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.max_api_calls < 1 or args.max_workers < 1 or args.timeout_seconds < 1:
        raise ValueError("API call, worker, and timeout limits must be positive")
    if min(args.juror_max_tokens, args.critic_max_tokens, args.judge_max_tokens) < 32:
        raise ValueError("per-role output token limits must be at least 32")
    if args.resume and not args.run_id:
        raise ValueError("--resume requires an explicit --run-id")
    if not args.primary_only and not (args.dry_run or args.verify_run):
        raise ValueError(
            "external API execution must use --primary-only; compare blind "
            "candidates locally with tools/run_blind_adjudication_gate.py"
        )


def _validate_resume_protocol(output: Path, expected_protocol: str) -> None:
    """Refuse to reuse an API run under a different frozen prompt protocol."""
    manifest_path = output / "reports" / "run_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("resume requires a readable run manifest") from exc
    actual_protocol = manifest.get("juror_protocol_version")
    if actual_protocol != expected_protocol:
        raise ValueError(
            "resume juror protocol does not match the original run: "
            f"expected {expected_protocol}, found {actual_protocol!r}"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _validate_args(args)
        packets = load_packets(args.packets)
        if not packets:
            raise ValueError("no packets found")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_id = args.run_id or f"deepseek-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    eligible = {task_id: packet for task_id, packet in packets.items() if packet.source_dataset != "Crossref/Retraction Watch"}
    gated = {task_id: packet for task_id, packet in packets.items() if task_id not in eligible}
    primary_upper_bound = len(eligible)
    full_upper_bound = primary_upper_bound * 4
    plan = {
        "run_id": run_id,
        "execution_mode": "blind_primary_only",
        "juror_protocol_version": args.juror_protocol,
        "primary_only": args.primary_only,
        "model_requested": args.model,
        "thinking_mode": args.thinking_mode,
        "output_token_limits": {
            "juror": args.juror_max_tokens,
            "critic": args.critic_max_tokens,
            "judge": args.judge_max_tokens,
        },
        "packet_count": len(packets),
        "eligible_packet_count": len(eligible),
        "crossref_gated_out_count": len(gated),
        "primary_calls": primary_upper_bound,
        "legacy_non_primary_call_upper_bound": full_upper_bound,
        "non_primary_api_calls": "prohibited",
        "max_api_calls": args.max_api_calls,
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if args.verify_run:
        verified, report = _verify_run(
            args.output, packets, thinking_mode=args.thinking_mode
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if verified else 1

    if not args.resume and args.output.exists() and any(args.output.iterdir()):
        print(
            "ERROR: --output must be new/empty without --resume; use a new run directory ",
            "or an explicit --resume --run-id with a verifiable ledger.",
            file=sys.stderr,
        )
        return 1
    if args.resume:
        try:
            _validate_resume_protocol(args.output, args.juror_protocol)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    api_key = _load_deepseek_api_key()
    if not api_key:
        print(
            "ERROR: DEEPSEEK_API_KEY is not set in the process environment or local .env.",
            file=sys.stderr,
        )
        return 1

    output = args.output
    votes_dir = output / "votes"
    reviews_dir = output / "reviews"
    silver_dir = output / "silver"
    gates_dir = output / "gates"
    reports_dir = output / "reports"
    primary_path = votes_dir / "juror-a.jsonl"
    second_path = votes_dir / "juror-b.jsonl"
    critic_path = reviews_dir / "critic.jsonl"
    silver_path = silver_dir / "ai_adjudicated_silver.jsonl"
    ledger_path = output / "audit" / "api_call_ledger.jsonl"

    primary_run_id = f"{run_id}:juror-a"
    second_run_id = f"{run_id}:juror-b"
    critic_run_id = f"{run_id}:critic"
    judge_run_id = f"{run_id}:judge"
    try:
        ledger = CallLedger(ledger_path, _load_call_ledger(ledger_path))
        primary_records = _load_jsonl_index(primary_path) if args.resume else {}
        second_records = _load_jsonl_index(second_path) if args.resume else {}
        critic_records = _load_jsonl_index(critic_path) if args.resume else {}
        silver_records = _load_jsonl_index(silver_path) if args.resume else {}
    except (LedgerIntegrityError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: cannot safely resume run: {type(exc).__name__}", file=sys.stderr)
        return 1

    usage_meter = UsageMeter()
    call_budget = CallBudget(args.max_api_calls)
    started_at = datetime.now(UTC)
    client = DeepSeekClient(
        api_key,
        base_url=args.base_url,
        budget=call_budget,
        usage_meter=usage_meter,
        json_mode=not args.no_json_mode,
        thinking_mode=args.thinking_mode,
        timeout_seconds=args.timeout_seconds,
    )
    failures: dict[str, list[dict[str, str]]] = {}

    reusable_primary = {
        task_id: record
        for task_id, record in primary_records.items()
        if task_id in eligible
        and _record_is_reusable(
            record,
            eligible[task_id].model_dump(mode="json"),
            expected_provenance="ai_juror",
            expected_run_id=primary_run_id,
            stage="juror-a",
            ledger_events=ledger.get("juror-a", task_id),
            thinking_mode=args.thinking_mode,
        )
    }
    reused_primary_count = len(reusable_primary)
    primary_records = reusable_primary
    primary_tasks = [
        (
            task_id,
            lambda packet=packet: _run_juror(
                client,
                packet.model_dump(mode="json"),
                role="juror-a",
                run_id=primary_run_id,
                model=args.model,
                max_tokens=args.juror_max_tokens,
                juror_protocol=args.juror_protocol,
                thinking_mode=args.thinking_mode,
                ledger=ledger,
            ),
        )
        for task_id, packet in sorted(eligible.items())
        if task_id not in primary_records
    ]
    primary_new, primary_failures = _parallel_calls(primary_tasks, max_workers=args.max_workers)
    primary_records.update(primary_new)
    failures["juror_a"] = primary_failures
    _atomic_write_jsonl(primary_path, primary_records.values())
    print(f"[progress] juror-a: {len(primary_records)}/{len(eligible)} completed, {len(primary_failures)} failures", flush=True)

    second_task_ids = (
        []
        if args.primary_only
        else [
            task_id
            for task_id, record in sorted(primary_records.items())
            if args.strategy == "full"
            or _needs_second_review(record, task_id, args.second_review_sample_rate)
        ]
    )
    reusable_second = {
        task_id: record
        for task_id, record in second_records.items()
        if task_id in second_task_ids
        and _record_is_reusable(
            record,
            eligible[task_id].model_dump(mode="json"),
            expected_provenance="ai_juror",
            expected_run_id=second_run_id,
            stage="juror-b",
            ledger_events=ledger.get("juror-b", task_id),
            thinking_mode=args.thinking_mode,
        )
    }
    reused_second_count = len(reusable_second)
    second_records = reusable_second
    second_tasks = [
        (
            task_id,
            lambda task_id=task_id: _run_juror(
                client,
                eligible[task_id].model_dump(mode="json"),
                role="juror-b",
                run_id=second_run_id,
                model=args.model,
                max_tokens=args.juror_max_tokens,
                juror_protocol=args.juror_protocol,
                thinking_mode=args.thinking_mode,
                ledger=ledger,
            ),
        )
        for task_id in second_task_ids
        if task_id not in second_records
    ]
    second_new, second_failures = _parallel_calls(second_tasks, max_workers=args.max_workers)
    second_records.update(second_new)
    failures["juror_b"] = second_failures
    _atomic_write_jsonl(second_path, second_records.values())
    print(f"[progress] juror-b: {len(second_records)}/{len(second_task_ids)} completed, {len(second_failures)} failures", flush=True)

    critic_task_ids = [
        task_id
        for task_id in second_task_ids
        if task_id in second_records
        and (
            args.strategy == "full"
            or _jurors_disagree(primary_records[task_id], second_records[task_id])
            or _stable_sample(task_id, "critic", args.critic_sample_rate)
        )
    ]
    reusable_critic = {
        task_id: record
        for task_id, record in critic_records.items()
        if task_id in critic_task_ids
        and _record_is_reusable(
            record,
            eligible[task_id].model_dump(mode="json"),
            expected_provenance="ai_critic",
            expected_run_id=critic_run_id,
            stage="critic",
            ledger_events=ledger.get("critic", task_id),
            thinking_mode=args.thinking_mode,
        )
    }
    reused_critic_count = len(reusable_critic)
    critic_records = reusable_critic
    critic_tasks = [
        (
            task_id,
            lambda task_id=task_id: _run_critic(
                client,
                eligible[task_id].model_dump(mode="json"),
                primary_records[task_id],
                second_records[task_id],
                run_id=critic_run_id,
                model=args.model,
                max_tokens=args.critic_max_tokens,
                thinking_mode=args.thinking_mode,
                ledger=ledger,
            ),
        )
        for task_id in critic_task_ids
        if task_id not in critic_records
    ]
    critic_new, critic_failures = _parallel_calls(critic_tasks, max_workers=args.max_workers)
    critic_records.update(critic_new)
    failures["critic"] = critic_failures
    _atomic_write_jsonl(critic_path, critic_records.values())
    print(f"[progress] critic: {len(critic_records)}/{len(critic_task_ids)} completed, {len(critic_failures)} failures", flush=True)

    judge_task_ids = [
        task_id
        for task_id in critic_task_ids
        if task_id in critic_records
        and (
            args.strategy == "full"
            or _jurors_disagree(primary_records[task_id], second_records[task_id])
            or bool(critic_records[task_id]["issues"])
        )
    ]
    reusable_silver = {
        task_id: record
        for task_id, record in silver_records.items()
        if task_id in judge_task_ids
        and _record_is_reusable(
            record,
            eligible[task_id].model_dump(mode="json"),
            expected_provenance="ai_adjudicated_silver",
            expected_run_id=judge_run_id,
            stage="judge",
            ledger_events=ledger.get("judge", task_id),
            thinking_mode=args.thinking_mode,
        )
    }
    reused_silver_count = len(reusable_silver)
    silver_records = reusable_silver
    judge_tasks = [
        (
            task_id,
            lambda task_id=task_id: _run_judge(
                client,
                eligible[task_id].model_dump(mode="json"),
                primary_records[task_id],
                second_records[task_id],
                critic_records[task_id],
                run_id=judge_run_id,
                model=args.model,
                max_tokens=args.judge_max_tokens,
                thinking_mode=args.thinking_mode,
                ledger=ledger,
            ),
        )
        for task_id in judge_task_ids
        if task_id not in silver_records
    ]
    silver_new, judge_failures = _parallel_calls(judge_tasks, max_workers=args.max_workers)
    silver_records.update(silver_new)
    failures["judge"] = judge_failures
    _atomic_write_jsonl(silver_path, silver_records.values())
    print(f"[progress] judge: {len(silver_records)}/{len(judge_task_ids)} completed, {len(judge_failures)} failures", flush=True)

    _atomic_write_jsonl(gates_dir / "crossref_source_level_only.jsonl", (_gate_crossref(packet.model_dump(mode="json")) for packet in gated.values()))
    review_queue = [
        {
            "task_id": task_id,
            "packet_checksum": packet.packet_checksum,
            "status": (
                "blind_primary_candidate_only"
                if args.primary_only
                else "not_adjudicated_in_economy_strategy"
            ),
            "reason": (
                "primary-only mode intentionally emits blind AI candidates, never silver"
                if args.primary_only
                else "primary candidate retained; no final judge call was authorized by the routing policy"
            ),
        }
        for task_id, packet in eligible.items()
        if task_id not in silver_records
    ]
    _atomic_write_jsonl(reports_dir / "review_queue.jsonl", review_queue)

    manifest = {
        **plan,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(UTC).isoformat(),
        "provider": "DeepSeek",
        "base_url": args.base_url,
        "packet_set_checksum": _packet_set_checksum(packets),
        "call_budget": {"used": call_budget.used, "maximum": call_budget.maximum},
        "usage": usage_meter.as_dict(),
        "call_accounting": {
            "ledger_path": str(ledger_path),
            "ledger_events_total": ledger.event_count,
            "new_validated_annotation_calls": ledger.new_event_count,
            "successful_api_responses": usage_meter.requests,
            "responses_without_validated_annotation": (
                usage_meter.requests - ledger.new_event_count
            ),
            "api_attempts_including_retries": call_budget.used,
            "reused_verified_records": {
                "juror_a": reused_primary_count,
                "juror_b": reused_second_count,
                "critic": reused_critic_count,
                "judge": reused_silver_count,
            },
            "new_records": {
                "juror_a": len(primary_new),
                "juror_b": len(second_new),
                "critic": len(critic_new),
                "judge": len(silver_new),
            },
        },
        "outputs": {
            "primary_candidates": len(primary_records),
            "second_reviews": len(second_records),
            "critic_reviews": len(critic_records),
            "ai_adjudicated_silver": len(silver_records),
            "crossref_gated_out": len(gated),
            "review_queue": len(review_queue),
        },
        "failures": failures,
        "independence_statement": "Juror A and B are separate blinded API calls with no cross-prompt context. They use the same configured model and must not be described as independent-model agreement.",
        "training_status": "not_training_data; not_gold; no automatic publication or memory operation",
    }
    _atomic_write_json(reports_dir / "run_manifest.json", manifest)
    verified, audit_report = _verify_run(
        output, packets, thinking_mode=args.thinking_mode
    )
    manifest["provenance_audit"] = audit_report
    _atomic_write_json(reports_dir / "run_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
