# EviMem architecture

## Authority boundaries

EviMem has two independent governed writes:

1. **Memory admission** accepts a `ScientificMemoryRecord` only when immutable evidence, a complete `VerificationCertificate`, and policy version/hash agree. A learned `MemoryManagerAction` is a request to this gate.
2. **Publication** accepts only a deterministic publish certificate. `publication_requested` is not publication authority, and the memory package never imports the publication writer.

Verification slots are outputs of deterministic evidence binding and tuple verification. Neither retrieval nor the learned manager may edit them.

## Canonical memory

`src/evimem/contracts/memory.py` defines:

- `ScientificClaimRecord`: minimal cross-dataset subject/relation/object/value/unit/condition schema;
- `ScientificMemoryRecord`: claim + evidence + certificate + decision + source/time + policy + origin;
- `AdmissionAction`: `WRITE_VERIFIED`, `WRITE_REJECTED`, `WRITE_CONFLICT`, `EPHEMERAL_ONLY`, `IGNORE`;
- `UpdateOperation`: `ADD`, `MERGE`, `LINK`, `CONFLICT`, `SUPERSEDE`, `IGNORE`;
- `MemoryManagerAction`: the strict, extra-fields-forbidden learned output.

Missing source fields remain null or empty. Adapters may not invent values. Controlled corruptions are marked in `MemoryOrigin.annotation_kind` and cannot masquerade as natural conflicts.

Supersession is an append-only edge. Reading the old record yields an effective `superseded` status and successor lineage; the original serialized payload remains intact.

## Retrieval

`MemoryRetriever` scores every eligible record with explicit components:

```text
semantic + structure + authority + temporal + policy
         - unresolved-conflict risk - superseded penalty
```

`RetrievalQuery.as_of` is mandatory benchmark state (defaulting to current UTC for ordinary use). The store excludes records observed after that time. Results return the complete memory record, including evidence and certificate, rather than an ungrounded text chunk.

TF-IDF is the local sparse baseline. Learned bi-encoders are trained outside the store through maintained Sentence Transformers APIs.

## Typed updates

`TypedMemoryUpdateService` processes a model prediction in this order:

1. load target records;
2. deterministically check operation semantics;
3. run certificate-governed admission;
4. append the new record;
5. append typed lineage/conflict/link edges.

The update gate rejects, among other cases, merging non-identical claims, conflicts without an identical context, links without a related subject/relation, and supersession by older or lower-authority evidence.

## SciMem-Curate benchmark

`BenchmarkEpisode` contains only history, the current document, and a query. `OracleAnnotation` is physically separate and is introduced only during scoring. Episode validation rejects future memories. Missing publication time remains null; undated documents cannot receive timestamped history and are never assigned a fabricated epoch.

The manifest in `configs/datasets.json` records component-level licenses and enforces task-view gates. `ViewSample` keeps `retrieval_view`, `admission_view`, and `update_view` disjoint; `InferenceViewInput` and `OracleViewTarget` are separate strict contracts. The public adapters preserve native labels and exact evidence offsets. They do not infer admission or update gold from claim veracity, QA evidence, relation extraction, or measurement slots. Raw upstream releases are audited from temporary external paths and are not bundled.

Metrics cover tuple/evidence quality, Recall@1/5/10, MRR, nDCG@10, admission precision, typed-update accuracy, conflict resolution, stale-memory errors, unsupported publication, negative-control publication, memory size, and retrieval tokens.

## Deterministic safety substrate

The retained `evidence`, `domains`, `verification`, and `publication` packages provide immutable releases, typed locators, DomainPack validation, evidence binding, conflict checks, certificates, and atomic/idempotent publication. `TupleVerifier.certify` consumes a candidate and immutable evidence directly; it no longer depends on an action trajectory.
