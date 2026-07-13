# EviMem: Evidence-Certified Memory for Continual Scientific Curation

## 1. Research question

Scientific information extraction is usually evaluated one document at a time, while real databases receive a chronological stream in which later evidence can repeat, contradict, or invalidate earlier records. EviMem asks whether an extractor can learn:

1. **Should Write**: which verified, rejected, or conflicting episode belongs in long-term memory;
2. **What to Retrieve**: which prior evidence-certified records help the current document;
3. **How to Update**: whether new evidence should add, merge, link, conflict with, supersede, or ignore prior memory.

The project does not train a materials-domain foundation model. Its main experiments use public scientific datasets; the original 150 DOI materials collection is a final real-world case study only.

## 2. Evidence-certified memory

Each memory is

\[
m_i=(c_i,e_i,z_i,d_i,t_i,v_i,o_i),
\]

where (c) is a structured claim, (e) immutable evidence, (z) a verification certificate, (d) the decision, (t) source/time, (v) policy/schema identity, and (o) dataset/license/annotation origin.

The minimum claim schema is:

```text
subject, relation, object, value, unit, condition, qualifiers
```

Missing fields remain null; converters cannot fabricate them. A reusable record must include evidence, certificate, and policy identity. Free-form self-reflection is never high-authority memory.

Memory states are:

- **verified**: evidence-bound, all required slots verified, policy-valid, and publication-gate accepted;
- **rejected**: evidence-bound deterministic rejection with a stable reason;
- **conflict**: same subject/relation/condition/measurement setting but incompatible conclusions;
- **superseded**: an old record retained with successor lineage after newer authorized evidence.

## 3. Learnable components and hard gates

### 3.1 Admission

The manager predicts one of:

```text
WRITE_VERIFIED | WRITE_REJECTED | WRITE_CONFLICT | EPHEMERAL_ONLY | IGNORE
```

The prediction cannot override certificate checks. The admission gate independently verifies evidence membership, evidence release, policy version/hash, certificate decision, support tier, slot status, constraints, and conflict state.

### 3.2 Retrieval

For query (q) and record (m):

\[
S(q,m)=\alpha S_{sem}+\beta S_{struct}+\gamma S_{auth}+\eta S_{time}
+\kappa S_{policy}-\delta S_{conflict}-\xi S_{stale}.
\]

Retrieval is time-bounded when a source timestamp exists, preventing future evidence leakage. Returned objects include the full claim, evidence, certificate, and decision--not only text. An undated source receives no fabricated timestamp or timestamped history.

### 3.3 Typed update

The manager returns a strict JSON `MemoryManagerAction`:

```json
{
  "admission": "WRITE_REJECTED",
  "update_operation": "CONFLICT",
  "target_memory_ids": ["mem_123"],
  "reason_code": "same_context_incompatible_value"
}
```

The deterministic update gate checks operation semantics before appending records or edges. Supersession requires newer evidence and non-decreasing authority. No record is destructively overwritten.

## 4. SciMem-Curate

Every episode contains policy-visible history, one current document, and a query. Oracle evidence, relevant memories, final record, admission, and update labels are held in a separate object and introduced only after inference.

Phase 1A public-data decisions:

- SciREX filtered document-level N-ary relations may supervise retrieval only;
- the leakage-safe SciFact subset may supervise rationale retrieval only;
- QASPER questions/evidence are local-evaluation-only until licensing is confirmed;
- Evidence Inference is blocked until source-article OA terms are resolved;
- MeasEval is slot extraction only and blocked from training pending an explicit license.

None of these labels supplies admission or update-operation gold. SUPPORT/CONTRADICT is not PUBLISH/REJECT, contradiction is not SUPERSEDE, and missing fields remain null.

Evaluation sources:

- unchanged official in-domain dev/test splits;
- POLYIE and BioRED as held-out OOD tests;
- SciFact-Open as the approximately 500K-corpus scale test;
- the original 150 DOI materials collection as a deterministic-safety case study.

Natural annotations take priority. Allowed controlled corruptions replace an entity, condition, unit, or evidence link and are explicitly marked. They are robustness tests, not natural conflicts.

## 5. Training

The planned sequence is: Stage 1 trains a bi-encoder retriever with contrastive positives and hard negatives; Stage 2 performs supervised QLoRA training of the typed manager after SciMem-Update exists; Stage 3 fixes the proposer and publication gate and compares memory methods under equal token and retrieval budgets. None of these stages was executed in Phase 1A.

The manager objective is a sum of supervised admission, update, type, target, and reason losses. Invalid generation fails closed. Publication remains a deterministic decision outside model parameters.

## 6. Comparisons

Memory comparisons should include No Memory, Full History, BM25/TF-IDF, dense vector memory, summary memory, Mem0, HippoRAG, A-Mem, ReasoningBank, and EviMem where licenses and maintained implementations permit.

Scientific extraction comparisons should reuse official protocols (for example SciBERT, document-level IE, DyGIE++, PURE, and fixed zero/few-shot language-model baselines). The base proposer, input data, token budget, and publication gate remain fixed when attributing gains to memory.

## 7. Metrics

Report:

- record quality: tuple precision/recall/F1, evidence span F1, published-record F1, verified-strong recall, unsupported and negative-control publication rates;
- memory: Recall@1/5/10, MRR, nDCG, admission precision, typed-update accuracy, conflict resolution, repeated-error reduction, stale-memory error, and pollution robustness;
- continual behavior: (F1_{memory}-F1_{no-memory}) over stream position, memory size, retrieval tokens, error propagation, and policy/schema-version invalidation.

## 8. Safety invariants

- A publication request is never publication authority.
- Verification slots change only through deterministic verifier output.
- Long-term memory requires evidence, a certificate, and policy identity.
- Oracle annotations are invisible during inference.
- OOD, scale, and case-study data cannot enter main-model optimization.
- Dataset, paper, checkpoint, and experiment artifacts remain outside the repository.
