"""Deterministic same-context conflict classification."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.contracts import ScientificClaim, make_candidate_fingerprint


@dataclass(frozen=True)
class ConflictAssessment:
    result: str
    reason_codes: tuple[str, ...] = ()


def _context_signature(claim: ScientificClaim) -> tuple[object, ...]:
    return (
        claim.property_key.casefold(),
        (claim.material_normalized or claim.material_raw or "").casefold(),
        (claim.composition_normalized or claim.composition_raw or "").casefold(),
        claim.sample_id or "",
        tuple(sorted((str(key), str(value)) for key, value in claim.conditions.items())),
        claim.conditions_raw or "",
    )


class ConflictResolver:
    @staticmethod
    def assess(
        claim: ScientificClaim,
        existing_claims: tuple[ScientificClaim, ...] | list[ScientificClaim] = (),
    ) -> ConflictAssessment:
        for existing in existing_claims:
            if make_candidate_fingerprint(existing) == make_candidate_fingerprint(claim):
                return ConflictAssessment("exact_duplicate")
            if _context_signature(existing) == _context_signature(claim):
                return ConflictAssessment("unresolved_conflict", ("same_context_different_value",))
        return ConflictAssessment("pass")

