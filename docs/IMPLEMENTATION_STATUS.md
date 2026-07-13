# Methods implementation status

This file distinguishes executable research code from contracts and planned
experiments. A schema or trainer factory alone is not counted as a completed
method.

Status meanings:

- `IMPLEMENTED`: exercised by real deterministic code and tests.
- `PARTIAL`: useful executable behavior exists, but the full Methods claim is not complete.
- `CONTRACT_ONLY`: typed interface/factory exists without the required data or experiment.
- `MISSING`: no implementation is present.

| Methods component | Status | Executable code | Evidence/tests |
|---|---|---|---|
| Immutable EvidenceRelease | IMPLEMENTED | `evimem/evidence/release.py` | checksum, immutability, exact DOI and canonical-ref tests |
| EvidenceRef and typed locators | IMPLEMENTED | `evimem/contracts/evidence.py`, `locators.py` | contract and release tests |
| CandidateObservation | IMPLEMENTED | `evimem/contracts/candidate.py`; consumed by controller/runtime | controller and E2E tests |
| ClaimState | IMPLEMENTED | `evimem/contracts/claim_state.py`, `controller/state_builder.py` | verifier-owned state tests |
| VerificationCertificate | IMPLEMENTED | `verification/tuple_verifier.py` generates it from evidence | both Phase 0 E2E tests; not manually constructed there |
| DomainPack schema/loader/hash | IMPLEMENTED | `evimem/domains/` and packaged configs | all three packs load and validate |
| Evidence block store/retrieval | IMPLEMENTED | `evimem/evidence/store.py` | checksum, quote, locator and E2E retrieval |
| Evidence binding cascade | IMPLEMENTED | `evimem/evidence/binding.py` | exact tuple, slot and multi-block paths |
| Tuple-level verification | IMPLEMENTED | `verification/tuple_verifier.py` | publish/reject E2E |
| Multi-block distributed verification | IMPLEMENTED | slot evidence is resolved across multiple immutable refs in `binding.py` | component path covered indirectly; broader corpus evaluation not run |
| Prediction/hypothesis rejection | IMPLEMENTED | `verification/gate.py` | predictive negative-control E2E |
| Conflict resolution | PARTIAL | deterministic duplicate/same-context conflict classifier in `verification/conflicts.py` | unit behavior available; no longitudinal benchmark yet |
| Publication gate | IMPLEMENTED | `verification/gate.py` | controller publication request is rejected by negative control |
| Atomic idempotent commit | IMPLEMENTED | `publication/commit.py`, `store.py` | publish E2E verifies retry; failure injection verifies full rollback |
| Separate rejection audit | IMPLEMENTED | `publication/audit.py` | reject E2E verifies zero publication and one audit row |
| Warranted-memory admission | IMPLEMENTED | `memory/governed_store.py` | certificate/release/policy/support checks and idempotency tests |
| Memory retrieval | IMPLEMENTED | `memory/retriever.py` | structural/policy ranking tests |
| Memory consolidation | IMPLEMENTED | `memory/consolidation.py` | verified and rejected E2E memory |
| Memory supersession | IMPLEMENTED | `memory/supersession.py` | preservation and audit-chain tests |
| Action controller and masking | IMPLEMENTED | `controller/` | legal-action, budget and terminal tests |
| Deterministic executor | IMPLEMENTED | `controller/executor.py` | controller tests; no publication dependency |
| Trajectory/replay/reward | IMPLEMENTED | `rl/` | integrity and verifier-shaped reward tests |
| Heuristic/no-memory baselines | IMPLEMENTED | `controller/policies.py` | benchmark/controller tests |
| Sequential benchmark contracts | PARTIAL | `benchmark/` | runner and oracle-isolation tests; no paper-scale episodes |
| Oracle trajectory builder | MISSING | none | Phase 1 work |
| Production LLM proposer adapter | MISSING | Phase 0 accepts an externally proposed `ScientificClaim` | no provider integration |
| Human review | PARTIAL | request and expected-value policy in `human_review/` | contract/policy tests; no user-service integration |
| SFT controller | CONTRACT_ONLY | TRL/PEFT factory, dataset codec and CLI in `training/` | no frozen dataset, training run or result |
| GRPO controller | CONTRACT_ONLY | TRL GRPO factory and reward adapter in `training/` | no episodes, model training, seeds or result |
| Continual-learning experiments | MISSING | none | no experiment artifacts |

## Confirmed Phase 0 boundary

The following chain is currently executable:

```text
document -> immutable release -> externally proposed claim -> controller action
-> verifier-owned slot updates -> automatically generated certificate
-> atomic publish or deterministic rejection -> governed memory
```

The publish fixture is `tests/e2e/test_publishable_phase0.py`. The negative
control is `tests/e2e/test_rejected_phase0.py`; its controller requests
publication, but predictive language causes a deterministic rejection, zero
published rows, an audit record and rejected memory.

## Explicitly not claimed

Phase 1 benchmark episodes have not been assembled. The SFT and GRPO modules
are integration interfaces only: there is no training dataset, checkpoint,
multi-seed experiment, learned controller result or completed Phase 2/3 claim.
