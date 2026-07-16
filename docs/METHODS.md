# EviMem: Evidence-Certified Memory for Continual Scientific Curation

> Research-direction note (2026-07): the document-curation system below remains
> an audited implementation history. The planned materials method is DACC,
> described in [WBM_CALIBRATION_CORESET_AMENDMENT.md](WBM_CALIBRATION_CORESET_AMENDMENT.md).
> DACC does not use SciMem-Update labels or model-generated annotations.

## 0. Proposed MatMem method

MatMem takes a frozen materials predictor and a chronological stream of native
formation-energy outcomes. It keeps at most \(K\) protocol-certified residual
cards, selected by a greedy decision-aware coverage objective rather than by
residual magnitude. Retrieval is direct only for matching scientific protocols;
an explicit same-structure-calibrated transport map is required otherwise, and
unsupported transitions abstain. Formation energy remains native; energy above
hull is recomputed against a versioned current hull snapshot. Screening is
stable only when a protocol-stratified calibrated upper bound is below the
threshold. This method is implemented as contracts and deterministic evaluation
machinery only; Matbench Discovery/JARVIS experiments are not yet executed.

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

### 3.3 Hierarchical update assessment and deterministic compilation

The manager returns a strict hierarchical JSON `MemoryManagerAction`. It cannot
emit a compiled update operation:

```json
{
  "admission": "WRITE_CONFLICT",
  "semantic_relation": "CONTRADICTORY",
  "scope_relation": "SAME_SCOPE",
  "authority_relation": "UNRESOLVED",
  "evidence_sufficiency": "SUFFICIENT",
  "target_memory_ids": ["mem_123"],
  "reason_code": "same_context_incompatible_value"
}
```

`UpdateCompiler` combines these labels with certificate eligibility and current
store state to compile `ADD`, `MERGE`, `LINK`, `CONFLICT`, `SUPERSEDE`, or
`IGNORE`. It fails closed on missing targets, insufficient context/evidence, and
inadmissible certificates. Destructive supersession additionally requires all of:

- exact claim-scope identity;
- sufficient pair evidence and an eligible new certificate;
- an actually newer record with strictly higher authority;
- a verified claim-level correction, retraction, or curator-correction source.

A document-level Crossref/Retraction Watch status is source provenance only and
cannot authorize claim-level supersession. No record is destructively overwritten;
superseded records retain append-only lineage.

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

## 5. Phase 1B retrieval validity pilot

The retrieval pilot forms one evidence-memory item per unique aligned evidence
text and source document. Query inputs contain text/entity context only; positive
memory IDs remain in the scorer object. All baselines use the same per-dataset
memory pool, query sequence, top-10 cutoff, and 256-token budget. The pilot uses:

- training: SciREX official train 517 and leakage-safe SciFact train 679;
- primary evaluation: SciREX dev/test 177 and SciFact dev 139;
- internal diagnostic: QASPER leakage-safe dev/test 4,555, never training.

Baselines are TF-IDF, BM25, frozen MiniLM, frozen SPECTER, MiniLM fine-tuned with
`MultipleNegativesRankingLoss` for seeds 13/42/97, dense plus a fail-closed
certificate-aware reranker, and the combined EviMem retrieval score. No OOD or
test record enters optimization. Fine-tuned weights/checkpoints are not committed.

The authorized retrieval views have no certificate or verified/rejected/conflict
memory-type gold. Those stratified recalls and certificate-mismatch rates are
therefore null, not synthetically derived. The certificate-aware reranker leaves
dense scores unchanged and its effectiveness is not estimable in this pilot.
Stale and policy-incompatible rates are reported for the constructed pool; the
pool contains no stale records and only licensed policy-compatible evidence.

Detailed fixed-k, fixed-token-budget, per-dataset, three-seed, and QASPER
diagnostic results are in `reports/phase1b/retrieval_results.json`. They are
validity-pilot measurements, not formal paper results.

## 6. SciMem-Update annotation pilot

The Phase 1B candidate pool contains 360 unlabeled pairs: SciREX 160, SciFact
160, and Crossref/Retraction Watch factual metadata 40. Sampling strata increase
coverage of possible equivalence, related/different scope, contradiction, hard
negative, and document-status cases, but are not labels or gold.

Annotators independently assign `SemanticRelation`, `ScopeRelation`,
`AuthorityRelation`, and `EvidenceSufficiency` using the repository labelbook.
The standard Label Studio UI omits compiled operations. Crossref records preserve
API response checksum, source, update type, timestamp, and DOI relation; all
claim-level effects remain `awaiting_human_evidence_annotation`. Model-only
candidate labels may prioritize targeted review but never satisfy annotation or
adjudication requirements. A visible-text audit of current DeepSeek V4 Pro
holdouts rejected the candidate sets for scientific-relation quality; see
`reports/phase1b/DEEPSEEK_V3_V4_SCIFACT_HOLDOUT_AUDIT.md`. Targeted expert
annotation, adjudication, agreement analysis, evidence recheck, and export
audit must finish before any SciMem-Update gold claim.

## 7. Training status

Stage 1's limited retrieval pilot has been executed as described above. Stage 2
would perform supervised QLoRA only after human-reviewed hierarchical
SciMem-Update labels exist; it has not started. Stage 3 remains planned and will
fix the proposer and publication gate while comparing memory methods under equal
token and retrieval budgets.

The future manager objective supervises admission plus the four hierarchical
axes, target selection, and reason codes. It does not supervise a flat six-way
operation head. Invalid generation fails closed. Operation compilation and
publication remain deterministic decisions outside model parameters.

## 8. Comparisons

Memory comparisons should include No Memory, Full History, BM25/TF-IDF, dense vector memory, summary memory, Mem0, HippoRAG, A-Mem, ReasoningBank, and EviMem where licenses and maintained implementations permit.

Scientific extraction comparisons should reuse official protocols (for example SciBERT, document-level IE, DyGIE++, PURE, and fixed zero/few-shot language-model baselines). The base proposer, input data, token budget, and publication gate remain fixed when attributing gains to memory.

## 9. Metrics

Report:

- record quality: tuple precision/recall/F1, evidence span F1, published-record F1, verified-strong recall, unsupported and negative-control publication rates;
- memory: Recall@1/5/10, MRR, nDCG, admission precision, typed-update accuracy, conflict resolution, repeated-error reduction, stale-memory error, and pollution robustness;
- continual behavior: (F1_{memory}-F1_{no-memory}) over stream position, memory size, retrieval tokens, error propagation, and policy/schema-version invalidation.

## 10. Safety invariants

- A publication request is never publication authority.
- Verification slots change only through deterministic verifier output.
- Long-term memory requires evidence, a certificate, and policy identity.
- Oracle annotations are invisible during inference.
- OOD, scale, and case-study data cannot enter main-model optimization.
- Aggregate audit and pilot metrics may be committed, but raw datasets, API
  responses, weights, checkpoints, and training logs remain outside the repository.
