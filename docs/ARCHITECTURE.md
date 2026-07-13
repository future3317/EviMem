# EviMem-RL Runtime Architecture

This document describes the implemented EviMem substrate derived from
`METHODS.md`. EviMem learns or selects curation actions, while an external
deterministic verification and publication harness retains exclusive
verification and write authority.

## Safety boundary

```text
CandidateObservation (always unpublished)
        |
        v
ControllerState + Warranted Memory
        |
        v
ControllerPolicy -> CurationAction
        |
        v
ActionExecutor -> deterministic verifier state update
        |
        v
REQUEST_PUBLICATION (request only)
        |
        v
external deterministic harness -> VerificationCertificate
        |
        +-- reject/defer -> audit + governed memory
        |
        `-- publish -> PublicationCommitService (sole durable writer)
```

The controller package does not import any publication store. Terminal
actions are executor-owned and cannot be registered as tool
handlers. This prevents a learned policy or benchmark controller from
acquiring database write access through dependency injection.

## Canonical contracts

The immutable Pydantic contracts live in `src/evimem/core/contracts/`:

- `EvidenceRef`: pins release, document, block, typed locator and SHA-256.
- `CandidateObservation`: proposer output whose only publication status is
  `unpublished`.
- `ClaimState`: verifier-owned slot, conflict and remaining-budget state.
- `VerificationCertificate`: complete deterministic gate result.
- `WarrantedMemoryItem`: evidence-, certificate- and policy-bound memory.
- `CurationTrajectory`: replayable candidate action sequence, including the
  structured rationale codes required to reconstruct exact state hashes.

`ScientificClaim` is the shared claim value object nested by candidates,
certificates and published records. There is no V1/V2 compatibility package.
Mutable binding diagnostics are not accepted as memory or publication
evidence; canonical refs must be created directly from an immutable release.

## Governed memory

`src/evimem/memory/` implements:

- append-only SQLite admission in a database separate from publication data;
- certificate, evidence-release and policy-identity checks;
- verified, rejected, conflict, correction and policy memory types;
- structured retrieval with policy, authority, staleness and conflict terms;
- optional scikit-learn TF-IDF semantic scoring;
- supersession events that preserve the old item and audit chain;
- certificate-driven consolidation with no free-form self-reflection path.

Memory and replay paths are supplied explicitly by the caller. Creating these
files is a runtime action, never an import side effect.

## Controller and replay

`src/evimem/controller/` provides the discrete action enum, fixed episode
state, state builder, standard evidence/memory tools, deterministic baselines,
termination checks and the single executor. Every episode pins one evidence
release and one DomainPack version/hash. State changes to verification slots
come only from the injected deterministic verifier.

`src/evimem/rl/` provides canonical trajectory recording, integrity-checked
SQLite replay and verifier-shaped reward. Reward reads only audited trajectory
deltas and the final certificate. It never reads model self-confidence or
rationale text.

`src/evimem/runtime.py` composes these pieces for one candidate. It records
the certified trajectory and consolidates admissible memory, but intentionally
does not commit publication records.

## Sequential benchmark

`src/evimem/benchmark/` separates policy-visible `BenchmarkEpisode` from
`OracleAnnotation`. Gold evidence is introduced only after inference for
scoring. Metrics cover verified-strong yield, publication-request rejection,
negative-control false publication, tool/token/human cost, trajectory length
and oracle action/evidence accuracy.

The built-in `HeuristicController` and `NoMemoryController` are transparent
non-learning baselines. `TfidfSemanticScorer` is the vector-memory baseline;
the governed retriever is the warranted-memory baseline.

## Training boundary

`src/evimem/training/` implements the Phase 2/3 interface layer without
reimplementing optimization algorithms. It compiles replayable trajectories
into document-split next-action examples, renders only policy-visible state,
decodes strict legal actions, and connects externally certified rewards to
TRL's `SFTTrainer` and `GRPOTrainer` with PEFT LoRA. Training code returns only
a `CurationAction` and does not import publication storage.
