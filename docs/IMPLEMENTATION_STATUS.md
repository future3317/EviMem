# Implementation status

Status is intentionally conservative: code contracts and data audits are not experimental results. Phase 1 is **not complete** and no model training or paper-result generation was performed in Phase 1A.

## System components

| Component | Status | Evidence |
|---|---|---|
| Unified cross-domain memory schema | Implemented | `contracts/memory.py`; schema/governance tests |
| Embedded evidence + certificate + policy warrant | Implemented | record validation and admission gate tests |
| Verified/rejected/conflict admission | Implemented | `memory/governed_store.py` |
| Append-only supersession lineage | Implemented | `memory/update.py`; supersession tests |
| Typed ADD/MERGE/LINK/CONFLICT/SUPERSEDE/IGNORE gate | Implemented | semantic precondition tests |
| Certificate-aware structured retriever | Implemented baseline | weighted retrieval and as-of leakage test |
| No Memory / Full History / TF-IDF / BM25 / dense baselines | Implemented interfaces | common chronological store boundary; no paper results |
| Learned dense retriever | Interface only | no trained weights or results |
| Supervised memory manager | Interface only | no QLoRA, training, or generated results |
| Component-level public dataset manifest | Implemented for Phase 1A | `configs/datasets.json`; official LICENSE files take precedence over HF metadata |
| Separate retrieval/admission/update views | Implemented | `benchmark/views.py`; oracle payload is physically separate from model input |
| Semantic/alignment/leakage audit generator | Implemented | `benchmark/audit.py`; `tools/run_phase1a_audit.py` |

## Public-data readiness

“Training ready” below applies only to the named task view and the leakage-safe official-train subset. It does not authorize admission/update training.

| Dataset | Data adapter implemented | Semantic audit | License | Training ready | Synthetic-derived labels | Human annotation required |
|---|---|---|---|---|---|---|
| SciREX | Yes, official v1 JSONL | Passed for 694 filtered retrieval relations; 1,454 table-missing relations are separately excluded | Confirmed Apache-2.0; official LICENSE SHA-256 recorded | Retrieval only: 517 train samples | Yes: deterministic relation-to-query/evidence projection, explicitly marked | Yes for every admission/update operation |
| QASPER | Yes, official v0.3 | Not passed for training: 29 annotations rejected for non-round-tripping evidence and 995 samples quarantined for cross-split query families | Dataset/source-text CC-BY-4.0 claim remains ambiguous because no official dataset LICENSE checksum was recovered; baseline code Apache-2.0 confirmed | No; 4,555 leakage-safe dev/test retrieval samples are local-evaluation-only | No operation labels; evidence is native | Yes for admission/update labels |
| SciFact | Yes, official release | Raw official claim split fails document isolation (110 documents; 477 samples); quarantine-derived subset passes | Confirmed: annotations CC-BY-4.0, abstracts ODC-By-1.0, code Apache-2.0 | Retrieval only: 679 train samples after quarantine; 139 dev samples for evaluation | No operation labels; rationale spans are native | Yes for admission/update, especially CONFLICT/SUPERSEDE |
| Evidence Inference 2.0 | Normalized adapter plus pinned real fixture; full-release adapter audit incomplete | Fixture alignment passed; full release not passed | Annotations/code MIT confirmed; source articles blocked unless per-document OA terms are resolved | No | No operation labels | Yes; also per-document license review |
| MeasEval | Yes, official TSV/text | Slot extraction alignment passed for 1,663 annotation sets; no EviMem data view is produced | Blocked: repository has no explicit dataset/root code license | No; local fixture tests only until license clarification | No operation labels | Yes for licensing and any admission/update task |
| POLYIE | External OOD protocol only | Adapter semantics not audited in Phase 1A | Apache-2.0 confirmed at pinned commit; official checksum recorded | No; OOD only | N/A | Semantic audit before OOD evaluation |
| BioRED | External OOD protocol only | Not audited in Phase 1A | Blocked pending dataset-specific license | No | N/A | License confirmation |

## Safety conclusions

- SUPPORT/CONTRADICT labels are retained as claim-veracity targets and never converted to PUBLISH/REJECT, admission actions, or SUPERSEDE.
- QASPER evidence is retrieval/evidence supervision only.
- MeasEval is slot extraction only and cannot enter training while blocked.
- Missing material, condition, time, certificate, evidence, value, and unit fields remain null/absent; no 1970 timestamp is injected.
- Every converted record is marked `native`, `deterministic_derived`, or `controlled_corruption`; Phase 1A generated no controlled-corruption records.
- No public dataset supplies trustworthy natural gold for ADD, MERGE, LINK, CONFLICT, SUPERSEDE, or IGNORE under EviMem policy semantics.
- A human-reviewed SciMem-Update set is required before admission/update training or evaluation can be claimed.

Not yet completed: full Evidence Inference per-document OA resolution, explicit QASPER dataset LICENSE checksum, MeasEval licensing, POLYIE/BioRED component audits, human SciMem-Update annotation, learned-model training, multi-seed experiments, OOD runs, scale runs, or paper tables.
