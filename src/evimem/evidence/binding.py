"""Small deterministic evidence-binding cascade for Phase 0."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from evimem.contracts import CandidateObservation, EvidenceRef, SlotStatus
from evimem.domains import DomainPack, DomainValidator, normalize_token

from .store import EvidenceBlockStore


def _normalized_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", value).strip()


def _contains_value(text: str, raw_value: str) -> bool:
    escaped = re.escape(raw_value.strip())
    return bool(escaped and re.search(rf"(?<![\w.]){escaped}(?![\w.])", text, re.IGNORECASE))


@dataclass(frozen=True)
class BindingResult:
    resolved_evidence: tuple[EvidenceRef, ...]
    slot_status: dict[str, SlotStatus]
    slot_evidence: dict[str, tuple[EvidenceRef, ...]]
    binding_method: str
    support_tier: str
    reason_codes: tuple[str, ...]


class EvidenceBinder:
    """Bind claim slots to release blocks without trusting proposer labels."""

    def __init__(self, store: EvidenceBlockStore, domain_pack: DomainPack):
        self.store = store
        self.domain_pack = domain_pack
        self.domain_validator = DomainValidator(domain_pack)

    def bind(
        self,
        candidate: CandidateObservation,
        *,
        evidence_refs: tuple[EvidenceRef, ...] | list[EvidenceRef] | None = None,
    ) -> BindingResult:
        supplied = tuple(evidence_refs or candidate.proposed_evidence)
        if not supplied:
            supplied = self.store.refs_for_doi(
                self._release_id(candidate),
                candidate.doi,
            )
        valid: list[tuple[EvidenceRef, str]] = []
        reasons: list[str] = []
        for ref in supplied:
            try:
                block = self.store.validate_ref(ref)
            except ValueError:
                reasons.append("invalid_evidence_ref")
                continue
            valid.append((ref, str(block.get("text", block.get("content", "")) or "")))

        claim = candidate.claim
        resolved_property = self.domain_validator.resolve_property(claim.property_key)
        aliases = (
            (claim.property_key,)
            if resolved_property is None
            else (
                claim.property_key,
                resolved_property[1].canonical_name,
                *resolved_property[1].aliases,
            )
        )
        requested_slots = ["property", "value", "unit"]
        if claim.material_raw or claim.material_normalized:
            requested_slots.append("material")
        if claim.composition_raw or claim.composition_normalized:
            requested_slots.append("composition")
        if claim.conditions_raw or claim.conditions:
            requested_slots.append("condition")

        slot_evidence: dict[str, tuple[EvidenceRef, ...]] = {}
        for slot in requested_slots:
            matching: list[EvidenceRef] = []
            for ref, text in valid:
                normalized = _normalized_text(text)
                if slot == "property":
                    present = any(normalize_token(alias) in normalize_token(text) for alias in aliases)
                elif slot == "value":
                    present = _contains_value(text, claim.value_raw)
                elif slot == "unit":
                    unit = claim.unit_canonical or claim.unit_raw or ""
                    present = bool(unit and normalize_token(unit) in normalize_token(text))
                elif slot == "material":
                    material = claim.material_normalized or claim.material_raw or ""
                    present = bool(material and _normalized_text(material) in normalized)
                elif slot == "composition":
                    composition = claim.composition_normalized or claim.composition_raw or ""
                    present = bool(composition and _normalized_text(composition) in normalized)
                else:
                    raw_condition = _normalized_text(claim.conditions_raw or "")
                    values = tuple(_normalized_text(str(value)) for value in claim.conditions.values())
                    present = bool(
                        (raw_condition and raw_condition in normalized)
                        or any(value and value in normalized for value in values)
                    )
                if present:
                    matching.append(ref)
            slot_evidence[slot] = tuple(matching)

        slot_status: dict[str, SlotStatus] = {}
        for slot, refs in slot_evidence.items():
            slot_status[slot] = SlotStatus.VERIFIED if refs else SlotStatus.MISSING
        if not valid:
            return BindingResult((), slot_status, slot_evidence, "unbound", "unbound", tuple(dict.fromkeys(reasons or ["no_valid_evidence"])))

        missing = [slot for slot, status in slot_status.items() if status != SlotStatus.VERIFIED]
        if missing:
            reasons.extend(f"unbound_slot:{slot}" for slot in missing)
            resolved = tuple(dict.fromkeys(ref for refs in slot_evidence.values() for ref in refs))
            return BindingResult(resolved, slot_status, slot_evidence, "partial_slot_match", "unbound", tuple(dict.fromkeys(reasons)))

        resolved = tuple(dict.fromkeys(ref for refs in slot_evidence.values() for ref in refs))
        common = set(slot_evidence[requested_slots[0]])
        for slot in requested_slots[1:]:
            common.intersection_update(slot_evidence[slot])
        if common:
            selected = sorted(common, key=lambda ref: ref.block_id)[0]
            method = "exact_quote_match" if selected.quote else "single_block_tuple_match"
            resolved = (selected,)
        else:
            method = "multi_block_slot_match"
        return BindingResult(
            resolved,
            slot_status,
            slot_evidence,
            method,
            "verified_strong",
            tuple(dict.fromkeys(reasons)),
        )

    @staticmethod
    def _release_id(candidate: CandidateObservation) -> str:
        if not candidate.proposed_evidence:
            raise ValueError("release_id is required when candidate has no proposed evidence")
        releases = {ref.release_id for ref in candidate.proposed_evidence}
        if len(releases) != 1:
            raise ValueError("candidate proposed evidence mixes releases")
        return next(iter(releases))

