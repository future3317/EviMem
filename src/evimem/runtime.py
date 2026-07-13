"""End-to-end EviMem episode runtime without publication write authority."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from evimem.controller import (
    ControllerPolicy,
    ControllerState,
    EpisodeOutcome,
    SequentialCurationEngine,
)
from evimem.core.contracts import CurationTrajectory, VerificationCertificate
from evimem.memory import GovernedMemoryStore, MemoryConsolidator
from evimem.rl import TrajectoryReplayBuffer, VerifierShapedReward


class VerificationHarness(Protocol):
    """External deterministic verifier; implementations may call PGCE."""

    def certify(
        self,
        *,
        outcome: EpisodeOutcome,
    ) -> VerificationCertificate: ...


@dataclass(frozen=True)
class CuratedEpisode:
    outcome: EpisodeOutcome
    certificate: VerificationCertificate
    trajectory: CurationTrajectory
    memory_id: str | None

    @property
    def publication_authorized(self) -> bool:
        """Whether the external gate approved the request, not whether it was committed."""

        return self.certificate.final_decision == "publish"


class EviMemRuntime:
    """Coordinates policy execution, certification, replay and memory admission."""

    def __init__(
        self,
        *,
        engine: SequentialCurationEngine,
        harness: VerificationHarness,
        replay_buffer: TrajectoryReplayBuffer,
        memory_store: GovernedMemoryStore,
        reward: VerifierShapedReward | None = None,
    ) -> None:
        self.engine = engine
        self.harness = harness
        self.replay_buffer = replay_buffer
        self.memory_store = memory_store
        self.reward = reward or VerifierShapedReward()

    def run_candidate(
        self,
        *,
        run_id: str,
        initial_state: ControllerState,
        policy: ControllerPolicy,
        domain: str,
        material_family: str | None = None,
    ) -> CuratedEpisode:
        outcome = self.engine.run(
            run_id=run_id,
            initial_state=initial_state,
            policy=policy,
        )
        certificate = self.harness.certify(outcome=outcome)
        self._validate_certificate(outcome, certificate)
        reward = self.reward.compute(outcome.trajectory, certificate)
        trajectory = outcome.trajectory.model_copy(
            update={
                "final_certificate_id": certificate.certificate_id,
                "total_reward": reward.total_reward,
            }
        )
        self.replay_buffer.append(trajectory)

        consolidation = MemoryConsolidator.consolidate(
            candidate=initial_state.candidate,
            certificate=certificate,
            trajectory=trajectory,
            domain=domain,
            material_family=material_family,
        )
        memory_id = None
        if consolidation.item is not None and consolidation.admitted_to_long_term:
            self.memory_store.admit(consolidation.item, certificate)
            memory_id = consolidation.item.memory_id
        return CuratedEpisode(
            outcome=outcome,
            certificate=certificate,
            trajectory=trajectory,
            memory_id=memory_id,
        )

    @staticmethod
    def _validate_certificate(
        outcome: EpisodeOutcome,
        certificate: VerificationCertificate,
    ) -> None:
        state = outcome.final_state
        if certificate.candidate_id != state.candidate.candidate_id:
            raise ValueError("harness certificate candidate mismatch")
        if certificate.evidence_release_id != state.claim_state.evidence_release_id:
            raise ValueError("harness certificate evidence release mismatch")
        if certificate.domain_pack_id != state.claim_state.domain_pack_id:
            raise ValueError("harness certificate DomainPack mismatch")
        if certificate.domain_pack_version != state.claim_state.domain_pack_version:
            raise ValueError("harness certificate policy version mismatch")
        if certificate.domain_pack_hash != state.claim_state.domain_pack_hash:
            raise ValueError("harness certificate policy hash mismatch")
        if certificate.final_decision == "publish" and not outcome.publication_requested:
            raise ValueError("harness cannot publish without a controller publication request")
