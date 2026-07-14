# Implementation status

Status is intentionally conservative. Phase 1A is frozen at
`64ecedd0341cb5199df33ad5f05a0d4a45a9429c` with tag
`phase1a-data-validity-audit`. Phase 1B has executed a retrieval validity pilot
and built an unlabeled annotation candidate pool; it has not produced formal
paper results or SciMem-Update gold.

## Phase 1B gates

| Deliverable | Status | Evidence |
|---|---|---|
| Retrieval pilot implemented | Yes | `tools/run_phase1b_retrieval_pilot.py`; fixed-k/fixed-token evaluator tests |
| Retrieval training executed | Yes, retriever only | SciREX train 517 + leakage-safe SciFact train 679; MiniLM bi-encoder, 3 seeds |
| Retrieval pilot results | Generated, pilot-only | `reports/phase1b/retrieval_results.json` and `.md` |
| Annotation candidate pool built | Yes, 360 unlabeled pairs | SciREX 160, SciFact 160, Crossref/Retraction Watch 40 |
| Standard annotation format | Yes | Label Studio JSONL + `annotation/label_studio_config.xml`; no custom front end |
| Human annotation completed | **No** | Candidate tasks are explicitly `unlabeled` and `candidate_is_gold=false` |
| Update manager trained | **No** | No QLoRA or manager optimization was run |
| SciMem-Update gold available | **No** | Hierarchical labels require double annotation and adjudication |
| Formal paper results generated | **No** | Retrieval output is explicitly scoped as a validity pilot |

## System components

| Component | Status | Evidence |
|---|---|---|
| Unified cross-domain memory schema | Implemented | `contracts/memory.py`; schema/governance tests |
| Embedded evidence + certificate + policy warrant | Implemented | record validation and admission-gate tests |
| Verified/rejected/conflict admission | Implemented | `memory/governed_store.py` |
| Hierarchical manager labels | Implemented | semantic, scope, authority, and evidence-sufficiency enums and validation tests |
| Deterministic `UpdateCompiler` | Implemented | truth-table, illegal-supersede, authority/evidence, and source-level mapping tests |
| Destructive supersession boundary | Implemented fail-closed | same scope, eligible certificate, newer/higher authority, sufficient evidence, and verified claim-level correction are all required |
| Certificate-aware structured retriever | Implemented | production scorer plus Phase 1B fail-closed reranker pilot |
| TF-IDF / BM25 / frozen dense / scientific dense | Executed in pilot | identical query/memory pools and budgets |
| Fine-tuned dense retriever | Executed in pilot | 3 seeds; no checkpoint committed |
| Supervised update manager | Interface only | hierarchical output codec; no update gold or QLoRA training |
| Component-level public dataset manifest | Implemented in Phase 1A | official LICENSE files take precedence over Hugging Face metadata |
| Separate retrieval/admission/update views | Implemented | oracle payload physically separated from model input |
| Annotation protocol | Implemented | labelbook, guidelines, disagreement taxonomy, Label Studio import/export |

## Public-data readiness

“Training ready” applies only to the named task view and leakage-safe official
train subset. It does not authorize admission/update training.

| Dataset | Adapter/audit | License | Phase 1B use | Training ready | Human annotation required |
|---|---|---|---|---|---|
| SciREX | Passed for 694 official filtered retrieval relations; 1,454 table-missing relations remain separately excluded | Apache-2.0 confirmed | 517 train, 177 evaluation; 160 unlabeled update candidates | Retrieval only | Yes for all admission/update labels |
| SciFact | Raw claim split leaks by document; quarantine subset passes | annotations CC-BY-4.0, abstracts ODC-By-1.0, code Apache-2.0 confirmed | 679 train, 139 evaluation; 160 unlabeled update candidates | Retrieval only | Yes, especially for same-scope conflict evidence |
| QASPER | 29 evidence conversions rejected; query-family quarantine passes afterward | dataset/source-text status remains ambiguous | 4,555 dev/test records, internal diagnostic only | **No** | Yes for admission/update; license confirmation also required |
| Evidence Inference 2.0 | Fixture alignment passed; full article audit incomplete | annotations/code MIT; article text blocked without per-document OA terms | Excluded | **No** | Yes plus per-document license review |
| MeasEval | Slot extraction only | blocked: no explicit dataset license | Excluded | **No** | License confirmation required |
| POLYIE | OOD protocol only | Apache-2.0 confirmed | Excluded from pilot training | **No; OOD only** | Semantic audit before OOD evaluation |
| BioRED | Not audited | dataset-specific license blocked | Excluded | **No** | License confirmation required |

## Retrieval validity pilot

The primary aggregate contains SciREX 177 and SciFact 139 evaluation queries.
QASPER 4,555 remains an internal diagnostic and never enters optimization.
Fine-tuning uses one epoch of `MultipleNegativesRankingLoss` from the same
MiniLM initialization for seeds 13, 42, and 97.

| Baseline | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| TF-IDF | 0.3608 | 0.6677 | 0.7310 | 0.4949 | 0.5476 |
| BM25 | 0.1835 | 0.3196 | 0.3544 | 0.2505 | 0.2668 |
| Frozen dense | 0.5443 | 0.9272 | 0.9620 | 0.7030 | 0.7654 |
| Frozen scientific dense | 0.3987 | 0.7658 | 0.8797 | 0.5621 | 0.6335 |
| Fine-tuned dense, seed 13 | 0.6139 | 0.9494 | 0.9747 | 0.7541 | 0.8079 |
| Fine-tuned dense, seed 42 | 0.6234 | 0.9462 | 0.9778 | 0.7602 | 0.8133 |
| Fine-tuned dense, seed 97 | 0.6139 | 0.9399 | 0.9715 | 0.7544 | 0.8070 |
| Dense + certificate-aware reranker | 0.6139 | 0.9494 | 0.9747 | 0.7541 | 0.8079 |
| EviMem full retrieval score | 0.6076 | 0.9367 | 0.9684 | 0.7454 | 0.7991 |

Three-seed mean ± sample standard deviation is Recall@1
`0.6171 ± 0.0055`, Recall@5 `0.9451 ± 0.0048`, Recall@10
`0.9747 ± 0.0032`, MRR `0.7562 ± 0.0034`, and nDCG@10
`0.8094 ± 0.0034`.

Certificate-aware effectiveness is not estimable: authorized retrieval views do
not contain certificate or memory-type gold. The fail-closed reranker therefore
left the reference dense ranking unchanged (all primary deltas 0.0). Verified,
rejected, conflict, and certificate-mismatch metrics are reported as null rather
than fabricated. Fixed-token-budget and per-dataset results are in the JSON
report.

## Annotation pilot readiness

The 360-pair pool contains 160 SciREX pairs, 160 SciFact claim/evidence pairs,
and 40 source-level correction/retraction metadata pairs. Eighty SciFact
CONTRADICT candidates and one SciREX same-scope heuristic pair may contain true
conflicts, but these sampling strata are not gold. All 40 Crossref/Retraction
Watch pairs contain only source-level status; claim-level effect remains
`awaiting_human_evidence_annotation`.

The repository is technically ready to start a double-annotated human pilot:
licenses for selected fields pass, evidence locators/checksums are present,
Label Studio import/config files exist, and compiled operations are hidden.
SciMem-Update remains unavailable until annotation, adjudication, agreement, and
post-export alignment checks pass.
