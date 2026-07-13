# Phase 1A: Public Dataset Validity and License Audit

Audit date: 2026-07-14 (Asia/Shanghai). No training, QLoRA, retriever fine-tuning, or paper-result generation was performed. Phase 1 is **not complete**.

## Decision summary

| Dataset / view | Converted | Quarantined or rejected | Immediately trainable | Local evaluation | Decision |
|---|---:|---:|---:|---:|---|
| SciREX retrieval | 694 | 1,454 table-missing relations separately excluded; 0 split leaks | 517 train | 177 dev/test | Allowed for retrieval only |
| SciFact retrieval | 1,295 | 477 samples from 110 cross-split source documents | 679 train | 139 dev | Allowed only after deterministic quarantine |
| QASPER retrieval | 7,964 | 995 family-leak samples quarantined; 29 annotations rejected | 0 | 4,555 dev/test | Local evaluation only; license remains ambiguous |
| Evidence Inference retrieval | 1 pinned real fixture | Full release and per-article OA audit incomplete | 0 | 1 fixture audit only | Block full dataset |
| MeasEval slot extraction | 1,663 | 0 alignment failures | 0 | Fixture tests only | Block dataset until explicit license |
| POLYIE | Adapter not audited | N/A | 0 | 0 | Apache-2.0 confirmed; OOD protocol only |
| BioRED | Not audited | N/A | 0 | 0 | Block pending dataset-specific license |

Counts above are converted annotation/task samples, not document counts. “Immediately trainable” means only a retrieval-view record from the official train split that has confirmed required component licenses, exact evidence alignment, conservative semantics, and no document/query-family split leakage. It does not mean admission or update training is ready.

## License findings

- SciREX official commit `7daad660...` has Apache-2.0 LICENSE SHA-256 `1eb85fc9...e9c8c6`.
- SciFact official `LICENSE.md` SHA-256 `a2532535...d13377fc` separately assigns CC-BY-4.0 to claims/evidence annotations, ODC-By-1.0 to abstracts, and Apache-2.0 to code.
- QASPER v0.3 source archives were pinned and checksummed, and baseline code is Apache-2.0. The dataset is described as CC-BY-4.0, but no official dataset LICENSE file checksum was recovered; annotations/source text therefore remain ambiguous and training is fail-closed.
- Evidence Inference annotations/code use MIT at commit `a661e8c...`, LICENSE SHA-256 `f76b1ceb...fc77089`. Article text needs per-document PMC OA-license resolution.
- MeasEval commit `1fa738b...` has no root dataset/code LICENSE; all components remain blocked despite its README referencing CC-BY source articles.
- Hugging Face metadata was not used to override an official LICENSE or missing license file.

## Semantic and leakage findings

- SciREX official filtered evaluation protocol leaves 694 relations whose four entity types have text mentions; 1,454 relations with at least one table-only/missing mention are marked separately and excluded from retrieval training.
- SciFact's official claim-level train/dev split reuses 110 source abstracts across splits. All 477 affected samples are quarantined; the retained subset is 679 train and 139 dev rationale-retrieval samples.
- QASPER has no source-document overlap, but exact normalized question families cross splits. Quarantining those families leaves 2,414 train, 1,488 dev, and 3,067 test annotations; training is still blocked by component licensing and 29 evidence non-round-trip rejections.
- Evidence locators round-trip exactly for every accepted conversion. Rejected QASPER annotations are retained in `rejected_conversion_samples.jsonl`, not silently repaired.
- Oracle evidence/targets and model inputs have different strict Pydantic contracts; model input objects cannot contain oracle fields.

## Memory-operation gold

| Operation | Natural gold in audited public data | Reason |
|---|---|---|
| ADD | No | A supported extraction is not a policy-authorized memory admission. |
| MERGE | No | No audited dataset annotates memory-identity consolidation. |
| LINK | No | Citation/evidence relevance does not specify EviMem memory links. |
| CONFLICT | No | Claim contradiction is not a governed memory-conflict action. |
| SUPERSEDE | No | No audited dataset establishes temporal replacement under matched scope and policy. |
| IGNORE | No | Missing/negative evidence is not an EviMem ignore decision. |

All operation distributions therefore report `NO_UPDATE_GOLD`; no SUPPORT/REFUTE label is promoted to PUBLISH/REJECT, and no contradiction is promoted to SUPERSEDE.

## Required next dataset

Yes: a human-reviewed **SciMem-Update** set is required. It should annotate candidate/history pairs with scope-matched ADD/MERGE/LINK/CONFLICT/SUPERSEDE/IGNORE decisions, evidence anchors, temporal ordering, policy identity, and explicit abstention/uncertainty. Controlled corruptions may be added only as separately marked stress tests and must never be reported as natural gold.
