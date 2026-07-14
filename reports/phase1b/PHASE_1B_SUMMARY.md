# Phase 1B Summary: Retrieval Validity Pilot + SciMem-Update Protocol

This is a pilot report, not a formal paper result. No QLoRA memory manager was
trained, no compiled update gold was generated, and no copyrighted full text or
model checkpoint is committed.

## 1. Actual retrieval results

Primary fixed-k aggregate over SciREX 177 + SciFact 139:

| Baseline | R@1 | R@5 | R@10 | MRR | nDCG@10 |
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

At the fixed 256-token budget, the seed-42 dense model has Recall@1/5/10
`0.6234 / 0.9462 / 0.9715`, MRR `0.7588`, and nDCG@10 `0.8114`.
The EviMem score has `0.6076 / 0.9367 / 0.9684`, MRR `0.7442`, and
nDCG@10 `0.7991`. Complete per-dataset and QASPER diagnostic results are in
`retrieval_results.json`.

QASPER was never used for training. Among its 4,555 diagnostic tasks, 3,699
contain aligned retrieval gold. The EviMem score obtains fixed-k Recall@10
`0.4196` and MRR `0.2474`; this remains an internal diagnostic because QASPER's
dataset license gate is not confirmed.

## 2. Three-seed variance

Fine-tuned dense mean ± sample standard deviation:

- Recall@1: `0.6171 ± 0.0055`
- Recall@5: `0.9451 ± 0.0048`
- Recall@10: `0.9747 ± 0.0032`
- MRR: `0.7562 ± 0.0034`
- nDCG@10: `0.8094 ± 0.0034`

## 3. Certificate-aware reranking

Effectiveness is not estimable in this pilot. The authorized retrieval views
have no certificate or verified/rejected/conflict memory-type gold. The
fail-closed reranker therefore leaves the seed-13 dense ranking unchanged; all
primary metric deltas are `0.0`. Typed Recall@1/5/10 and certificate-mismatch
rate are null rather than synthesized.

## 4. Candidate source distribution

The unlabeled pool contains 360 pairs:

- SciREX train: 160
- SciFact leakage-safe train: 160
- Crossref/Retraction Watch factual metadata: 40

Crossref-derived records comprise 19 retractions, 19 corrections, and 2 errata;
34 are publisher-sourced and 6 are Retraction Watch-sourced. Both API response
checksums are pinned in `data_manifests/scimem_update_pilot_manifest.json`.

## 5. Pairs that may contain true conflict

`scifact:0080` through `scifact:0159` are sampled from native CONTRADICT
rationales, and `scirex:0000` is the single structured same-scope/different-value
heuristic pair found under the deterministic sampler. These 81 pairs are only
conflict candidates. SUPPORT/CONTRADICT is not update gold, and human annotators
must still establish semantic contradiction, same scope, and sufficient evidence.

## 6. Source-level correction/retraction only

All 40 `crossref:*` pairs are document/source-level status records. Their exact
IDs are listed in `candidate_distribution.json`. None is claim-level
supersession evidence; every record remains
`awaiting_human_evidence_annotation`.

## 7. Human annotation pilot readiness

The project is technically ready to launch a double-annotated pilot:

- selected source components pass the candidate-pool license gate;
- 360 unique task IDs pass split leakage checks;
- evidence locators and checksums are included;
- Label Studio import and XML configuration are present;
- the labelbook, missing-information policy, and disagreement taxonomy are present;
- compiled operations are absent from annotator-visible tasks.

It is not ready to claim SciMem-Update gold or train the update manager. Human
annotation, adjudication, agreement analysis, export validation, and evidence
alignment re-audit must finish first.
