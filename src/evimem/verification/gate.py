"""Deterministic publication gate; controller/model declarations carry no authority."""

from __future__ import annotations

import re
from dataclasses import dataclass

from evimem.contracts import SlotStatus
from evimem.domains import DomainPack, DomainValidationResult
from evimem.evidence import BindingResult

_PREDICTIVE_PATTERNS = (
    r"\bpredicted\b",
    r"\bprediction\b",
    r"\bmay\s+(?:reach|achieve|exhibit|show)\b",
    r"\bcould\s+(?:reach|achieve|exhibit|show)\b",
    r"\bexpected\s+to\b",
    r"\bin\s+future\b",
    r"\bhypothetical\b",
)


@dataclass(frozen=True)
class GateDecision:
    final_decision: str
    reason_codes: tuple[str, ...]
    predictive_or_hypothetical: bool


class PublicationGate:
    def __init__(self, domain_pack: DomainPack):
        self.domain_pack = domain_pack

    def evaluate(
        self,
        *,
        evidence_text: str,
        publication_requested: bool,
        domain_validation: DomainValidationResult,
        binding: BindingResult,
        conflict_result: str,
    ) -> GateDecision:
        folded = evidence_text.casefold()
        predictive = any(re.search(pattern, folded) for pattern in _PREDICTIVE_PATTERNS)
        hard_reasons: list[str] = []
        review_reasons: list[str] = []
        if predictive:
            hard_reasons.append("predictive_or_hypothetical_claim")

        for pattern in self.domain_pack.false_positive_patterns.values():
            if any(keyword.casefold() in folded for keyword in pattern.keywords):
                target = hard_reasons if pattern.risk_level == "high" else review_reasons
                target.append(pattern.reason_code)

        hard_reasons.extend(domain_validation.reason_codes)
        if not self.domain_pack.publication_policy.strict_materialization_allowed:
            hard_reasons.append("strict_materialization_disabled_for_domain")
        if binding.support_tier in {"unbound", "structured_prompt_support"}:
            hard_reasons.append("insufficient_evidence_support")
        elif binding.support_tier == "ambiguous":
            review_reasons.append("ambiguous_evidence_support")
        if any(status != SlotStatus.VERIFIED for status in binding.slot_status.values()):
            hard_reasons.append("tuple_slots_not_verified")
        if conflict_result in {"unresolved_conflict", "resolvable_conflict"}:
            review_reasons.append("unresolved_conflict")

        hard_reasons = list(dict.fromkeys(hard_reasons))
        review_reasons = list(dict.fromkeys(review_reasons))
        if hard_reasons:
            return GateDecision("reject", tuple(hard_reasons), predictive)
        if review_reasons:
            return GateDecision("review", tuple(review_reasons), predictive)
        if not publication_requested:
            return GateDecision("defer", ("publication_not_requested",), predictive)
        return GateDecision("publish", (), predictive)

