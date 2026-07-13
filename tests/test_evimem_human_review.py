from __future__ import annotations

from evimem.contracts import CurationBudget, SlotStatus, VerificationSlot
from evimem.controller import StateBuilder
from evimem.human_review import ExpectedValueReviewPolicy, HumanReviewRequest

from .evimem_helpers import candidate, evidence_ref


def _ambiguous_state():
    state = StateBuilder.build(
        candidate=candidate(),
        required_slots=("property", "material"),
        evidence_release_id="release-1",
        domain_pack_id="piezoelectric",
        domain_pack_version="1.3.0",
        domain_pack_hash="policy-hash",
        budget=CurationBudget(tool_calls=1, human_queries=1, wall_clock_seconds=10),
    )
    slots = dict(state.claim_state.slots)
    slots["material"] = VerificationSlot(
        status=SlotStatus.AMBIGUOUS,
        evidence_refs=(evidence_ref(),),
    )
    return state.model_copy(
        update={
            "claim_state": state.claim_state.__class__(**{
                **state.claim_state.model_dump(mode="python"),
                "slots": slots,
                "unresolved_slots": (),
            })
        }
    )


def test_review_policy_requires_trigger_and_positive_value() -> None:
    policy = ExpectedValueReviewPolicy()
    decision = policy.evaluate(
        _ambiguous_state(),
        recovery_probability=0.8,
        verified_record_value=5.0,
        query_cost=1.0,
    )
    assert decision.ask_human
    assert decision.expected_value == 3.0


def test_review_request_pins_release_and_evidence() -> None:
    request = HumanReviewRequest(
        request_id="review-1",
        run_id="run-1",
        candidate_id="candidate-1",
        slot_name="material",
        evidence_release_id="release-1",
        evidence_bundle=(evidence_ref(),),
        ambiguity_reason_codes=("multiple_materials",),
        policy_version="1.3.0",
        policy_hash="policy-hash",
    )
    assert request.evidence_bundle[0].release_id == "release-1"
