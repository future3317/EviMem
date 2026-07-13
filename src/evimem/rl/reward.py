"""Verifier-shaped reward computed only from audited contracts."""

from __future__ import annotations

from dataclasses import dataclass

from evimem.core.contracts import CurationTrajectory, VerificationCertificate


@dataclass(frozen=True)
class RewardConfig:
    verified_slot: float = 1.0
    bound_slot: float = 0.25
    ambiguity_reduction: float = 0.5
    conflict_resolved: float = 1.0
    tool_call_cost: float = 0.05
    token_cost_per_thousand: float = 0.01
    repeated_action_cost: float = 0.20
    human_query_cost: float = 1.0
    publish_verified_strong: float = 5.0
    correct_defer: float = 2.0
    correct_reject: float = 2.0
    rejected_publication_request: float = -3.0


@dataclass(frozen=True)
class RewardBreakdown:
    step_reward: float
    terminal_reward: float
    total_reward: float
    repeated_actions: int


class VerifierShapedReward:
    def __init__(self, config: RewardConfig | None = None):
        self.config = config or RewardConfig()

    def compute(
        self,
        trajectory: CurationTrajectory,
        certificate: VerificationCertificate,
    ) -> RewardBreakdown:
        if trajectory.candidate_id != certificate.candidate_id:
            raise ValueError("trajectory and certificate candidates differ")
        if trajectory.evidence_release_id != certificate.evidence_release_id:
            raise ValueError("trajectory and certificate releases differ")
        config = self.config
        seen: set[tuple[str, str]] = set()
        repeats = 0
        step_reward = 0.0
        for step in trajectory.steps:
            key = (step.action, str(sorted(step.action_args.items())))
            if key in seen:
                repeats += 1
            seen.add(key)
            delta = step.verifier_delta
            step_reward += len(delta.newly_verified_slots) * config.verified_slot
            step_reward += len(delta.newly_bound_slots) * config.bound_slot
            step_reward += delta.ambiguity_reduction * config.ambiguity_reduction
            if delta.conflict_resolution in {"distinct_context", "resolved"}:
                step_reward += config.conflict_resolved
            step_reward -= step.cost.tool_calls * config.tool_call_cost
            step_reward -= step.cost.tokens / 1000.0 * config.token_cost_per_thousand
            step_reward -= step.cost.human_queries * config.human_query_cost
        step_reward -= repeats * config.repeated_action_cost

        if (
            trajectory.terminal_action == "REQUEST_PUBLICATION"
            and certificate.final_decision == "publish"
            and certificate.support_tier == "verified_strong"
        ):
            terminal = config.publish_verified_strong
        elif trajectory.terminal_action == "REQUEST_PUBLICATION":
            terminal = config.rejected_publication_request
        elif (
            trajectory.terminal_action == "DEFER_FOR_REVIEW"
            and certificate.final_decision in {"defer", "review"}
        ):
            terminal = config.correct_defer
        elif (
            trajectory.terminal_action in {"REJECT_CANDIDATE", "STOP_NO_RECORD"}
            and certificate.final_decision == "reject"
        ):
            terminal = config.correct_reject
        else:
            terminal = 0.0
        return RewardBreakdown(step_reward, terminal, step_reward + terminal, repeats)
