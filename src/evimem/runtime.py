"""End-to-end EviMem episode runtime without publication write authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from evimem.contracts import (
    ActionCost,
    CandidateObservation,
    CurationBudget,
    CurationTrajectory,
    ProposerProvenance,
    ScientificClaim,
    VerificationCertificate,
    make_candidate_fingerprint,
    make_candidate_id,
)
from evimem.contracts.ids import deterministic_id
from evimem.controller import (
    ActionExecutor,
    ActionToolResult,
    ActionType,
    ControllerPolicy,
    ControllerState,
    EpisodeOutcome,
    HeuristicController,
    RegisteredAction,
    SequentialCurationEngine,
    StateBuilder,
)
from evimem.domains import DomainValidator, load_domain_pack
from evimem.evidence import EvidenceBinder, EvidenceBlockStore, EvidenceReleaseManager
from evimem.memory import GovernedMemoryStore, MemoryConsolidator
from evimem.publication import (
    AuditRecordResult,
    PublicationCommitResult,
    PublicationCommitService,
    PublicationStore,
    RejectionAuditStore,
)
from evimem.rl import TrajectoryReplayBuffer, VerifierShapedReward
from evimem.verification import DeterministicActionVerifier, TupleVerifier


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


@dataclass(frozen=True)
class Phase0RunResult:
    candidate: CandidateObservation
    outcome: EpisodeOutcome
    certificate: VerificationCertificate
    trajectory: CurationTrajectory
    publication: PublicationCommitResult | None
    rejection_audit: AuditRecordResult | None
    memory_id: str | None

    @property
    def published(self) -> bool:
        return self.publication is not None and self.certificate.final_decision == "publish"


class DeterministicPhase0Runtime:
    """Real Evidence -> Certificate -> Publish/Reject -> Memory Phase 0 loop.

    The caller supplies a proposed claim. This runtime deliberately does not
    pretend that an LLM proposer or model training has been implemented.
    """

    publication_policy_version = "evimem-phase0-publication.v1"

    def __init__(self, work_dir: str | Path):
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_manager = EvidenceReleaseManager(self.work_dir / "evidence")
        self.evidence_store = EvidenceBlockStore(self.evidence_manager)
        self.publication_store = PublicationStore(self.work_dir / "publication.sqlite")
        self.audit_store = RejectionAuditStore(self.work_dir / "rejection_audit.sqlite")
        self.memory_store = GovernedMemoryStore(self.work_dir / "memory.sqlite")
        self.replay_buffer = TrajectoryReplayBuffer(self.work_dir / "replay.sqlite")
        self.reward = VerifierShapedReward()

    def run_document(
        self,
        *,
        run_id: str,
        doi: str,
        document_text: str,
        proposed_claim: ScientificClaim,
        release_id: str,
        domain_id: str = "piezoelectric",
        source: str = "phase0_fixture",
    ) -> Phase0RunResult:
        document_text = document_text.strip()
        if not document_text:
            raise ValueError("document_text must be non-empty")
        if release_id not in self.evidence_manager.list_releases():
            block_id = "block-" + deterministic_id(
                release_id,
                doi,
                document_text,
                length=24,
                namespace="evidence",
            )
            release = self.evidence_manager.create_release(
                [
                    {
                        "doi": doi,
                        "source": source,
                        "block_id": block_id,
                        "text": document_text,
                        "domain_name": domain_id,
                    }
                ],
                release_id=release_id,
                metadata={"builder": "DeterministicPhase0Runtime"},
            )
        else:
            release = self.evidence_manager.get_release(release_id)
            existing = self.evidence_manager.load_by_doi(release_id, doi)
            if len(existing) != 1 or str(existing[0].get("text", "")).strip() != document_text:
                raise ValueError("immutable release_id already exists with different evidence")

        domain_pack = load_domain_pack(domain_id)
        refs = self.evidence_store.refs_for_doi(release_id, doi)
        fingerprint = make_candidate_fingerprint(proposed_claim)
        candidate_id = make_candidate_id(run_id, 0, fingerprint)
        timestamp = datetime.fromisoformat(str(release.manifest["created_at"]))
        candidate = CandidateObservation(
            candidate_id=candidate_id,
            run_id=run_id,
            doi=doi,
            claim=proposed_claim,
            proposed_evidence=list(refs),
            proposer_provenance=ProposerProvenance(
                provider="external_proposal",
                model="deterministic-phase0-input",
                extraction_schema_version=CandidateObservation.schema_version,
                prompt_hash="not_applicable",
                extraction_timestamp=timestamp,
            ),
        )

        binder = EvidenceBinder(self.evidence_store, domain_pack)
        action_verifier = DeterministicActionVerifier(binder)

        def retrieve(action, state):
            query = str(action.arguments.get("query", proposed_claim.property_key))
            matches = self.evidence_store.search(
                release_id=release_id,
                doi=doi,
                query=query,
            )
            return ActionToolResult(
                payload={"query": query, "result_count": len(matches)},
                evidence_refs=tuple(item.evidence_ref for item in matches),
            )

        executor = ActionExecutor(
            actions={
                ActionType.RETRIEVE_TABLE: RegisteredAction(
                    handler=retrieve,
                    cost=ActionCost(tool_calls=1),
                ),
                ActionType.RETRIEVE_PASSAGE: RegisteredAction(
                    handler=retrieve,
                    cost=ActionCost(tool_calls=1),
                ),
            },
            verifier=action_verifier,
        )
        validator = DomainValidator(domain_pack)
        initial_state = StateBuilder.build(
            candidate=candidate,
            required_slots=validator.required_slots(proposed_claim),
            evidence_release_id=release_id,
            domain_pack_id=domain_pack.domain_id,
            domain_pack_version=domain_pack.version,
            domain_pack_hash=domain_pack.pack_hash,
            budget=CurationBudget(tool_calls=3, wall_clock_seconds=30),
        )
        outcome = SequentialCurationEngine(executor=executor, max_steps=4).run(
            run_id=run_id,
            initial_state=initial_state,
            policy=HeuristicController(),
        )
        verifier = TupleVerifier(
            domain_pack=domain_pack,
            evidence_store=self.evidence_store,
        )
        certificate = verifier.certify(outcome=outcome)
        EviMemRuntime._validate_certificate(outcome, certificate)
        reward = self.reward.compute(outcome.trajectory, certificate)
        trajectory = outcome.trajectory.model_copy(
            update={
                "final_certificate_id": certificate.certificate_id,
                "total_reward": reward.total_reward,
            }
        )
        self.replay_buffer.append(trajectory)

        publication = None
        rejection_audit = None
        if certificate.final_decision == "publish":
            publication = PublicationCommitService(
                self.publication_store,
                policy_version=self.publication_policy_version,
            ).commit(
                doi=doi,
                domain_name=domain_pack.domain_id,
                certificate=certificate,
                idempotency_key=f"{run_id}:{candidate_id}",
            )
        else:
            rejection_audit = self.audit_store.record(certificate)

        consolidation = MemoryConsolidator.consolidate(
            candidate=candidate,
            certificate=certificate,
            trajectory=trajectory,
            domain=domain_pack.domain_id,
            material_family=proposed_claim.material_normalized or proposed_claim.material_raw,
        )
        memory_id = None
        if consolidation.admitted_to_long_term and consolidation.item is not None:
            self.memory_store.admit(consolidation.item, certificate)
            memory_id = consolidation.item.memory_id
        return Phase0RunResult(
            candidate=candidate,
            outcome=outcome,
            certificate=certificate,
            trajectory=trajectory,
            publication=publication,
            rejection_audit=rejection_audit,
            memory_id=memory_id,
        )
