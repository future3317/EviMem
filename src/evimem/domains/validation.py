"""Deterministic claim validation against a DomainPack."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from evimem.contracts import ScientificClaim

from .models import DomainPack, PropertyDefinition


def normalize_token(value: str | None) -> str:
    if not value:
        return ""
    value = unicodedata.normalize("NFKC", value).casefold()
    value = value.replace("μ", "u").replace("µ", "u").replace("·", "")
    return re.sub(r"[\s_{}()^\-]+", "", value)


@dataclass(frozen=True)
class DomainValidationResult:
    passed: bool
    canonical_property: str | None
    canonical_unit: str | None
    reason_codes: tuple[str, ...]


class DomainValidator:
    def __init__(self, domain_pack: DomainPack):
        self.domain_pack = domain_pack
        candidates: dict[str, set[str]] = {}
        for key, definition in domain_pack.properties.items():
            for alias in (definition.canonical_name, *definition.aliases):
                candidates.setdefault(normalize_token(alias), set()).add(key)
        self._aliases = {
            alias: next(iter(keys)) for alias, keys in candidates.items() if len(keys) == 1
        }
        # Explicit property IDs always win over potentially ambiguous natural-language aliases.
        self._aliases.update({normalize_token(key): key for key in domain_pack.properties})

    def resolve_property(self, value: str) -> tuple[str, PropertyDefinition] | None:
        key = self._aliases.get(normalize_token(value))
        if key is None:
            return None
        return key, self.domain_pack.properties[key]

    def validate(self, claim: ScientificClaim) -> DomainValidationResult:
        reasons: list[str] = []
        resolved = self.resolve_property(claim.property_key)
        if resolved is None:
            return DomainValidationResult(False, None, None, ("unknown_property",))
        property_key, definition = resolved
        if property_key in self.domain_pack.publication_policy.blocked_materialization_properties:
            reasons.append("property_blocked_by_domain_policy")

        canonical_unit = claim.unit_canonical or claim.unit_raw
        if definition.units:
            allowed = {normalize_token(unit): unit for unit in definition.units}
            unit_key = normalize_token(canonical_unit)
            if unit_key not in allowed:
                reasons.append("unit_not_allowed_for_property")
            else:
                canonical_unit = allowed[unit_key]

        if definition.expected_range is not None:
            if claim.value_num is None:
                reasons.append("numeric_value_required")
            elif not definition.expected_range[0] <= claim.value_num <= definition.expected_range[1]:
                reasons.append("value_outside_expected_range")

        for context in definition.required_context:
            if context == "material" and not (claim.material_raw or claim.material_normalized):
                reasons.append("required_material_missing")
            elif context == "condition" and not (claim.conditions_raw or claim.conditions):
                reasons.append("required_condition_missing")
            elif context == "composition" and not (
                claim.composition_raw or claim.composition_normalized
            ):
                reasons.append("required_composition_missing")

        return DomainValidationResult(
            passed=not reasons,
            canonical_property=property_key,
            canonical_unit=canonical_unit,
            reason_codes=tuple(dict.fromkeys(reasons)),
        )

    def required_slots(self, claim: ScientificClaim) -> tuple[str, ...]:
        slots = ["property", "value", "unit"]
        resolved = self.resolve_property(claim.property_key)
        if resolved is not None:
            for context in resolved[1].required_context:
                if context in {"material", "composition", "condition"}:
                    slots.append(context)
        return tuple(dict.fromkeys(slots))
