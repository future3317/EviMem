# Planned supervised training protocol

Phase 1A performed no training. Phase 1B executed only the bounded retriever
validity pilot described below; it did not run QLoRA, train an update manager,
produce SciMem-Update gold, or generate formal paper results. EviMem may train a
component only after its selected view passes the component-license, semantic,
evidence-alignment, and leakage gates.

## Stage 1: retriever

The Phase 1B pilot trained a Sentence Transformers MiniLM bi-encoder with
contrastive supervision derived from natural retrieval annotations:

- positives: supporting evidence, relevant relations, and correct-context history;
- hard negatives: same subject/different relation, same relation/different condition, same value/different subject, and superseded/conflict memories.

`RetrievalTrainingExample` stores only explicit positive and hard-negative memory IDs. Positive and negative sets must be disjoint. Official test examples are rejected from optimization by `require_official_training_splits`.

The executed pilot used SciREX 517 plus SciFact 679, one epoch,
`MultipleNegativesRankingLoss`, and seeds 13/42/97. It reported Recall@1/5/10,
MRR, nDCG@10, failure rates, and selected tokens under both fixed-k and a
256-token budget. Every baseline used the same query and memory pool. No
checkpoint was committed. See `reports/phase1b/retrieval_results.json`.

## Stage 2: memory manager

Supervise a 3B--7B instruct model with QLoRA only after a human-reviewed update
dataset exists. This stage has not started. Each future example contains:

```text
current certified record
+ retrieved certified memories
-> hierarchical MemoryManagerAction
```

The completion is exactly one JSON object with admission, semantic relation,
scope relation, authority relation, evidence sufficiency, target IDs, and a
reason code. It contains no `update_operation`. Invalid output fails closed to
`EPHEMERAL_ONLY + INSUFFICIENT_CONTEXT + UNKNOWN_SCOPE + NOT_APPLICABLE +
INSUFFICIENT`. `UpdateCompiler` alone derives the memory operation after
certificate and store-state checks.

`ManagerTrainingExample.prompt_record()` produces prompt/completion fields. `build_lora_config` and `build_manager_training_args` use PEFT and Transformers rather than a local optimizer implementation.

## Stage 3: memory-conditioned curation

Freeze the same proposer, token budget, current evidence, and deterministic publication gate. Change only the memory method. Retrieved records are organized as verified precedents, known failure patterns, possible conflicts, and required checks; they are not copied as facts.

Primary evidence should come from natural annotations and publication order. Controlled entity/condition/unit/evidence corruptions are stress tests and must retain `annotation_kind=controlled_corruption`.

## Data separation

- retrieval training candidates: SciREX official train (517 filtered relations) and the SciFact leakage-safe train subset (679 rationale samples);
- local evaluation only: QASPER leakage-safe dev/test until its official dataset LICENSE is checksummed;
- blocked: Evidence Inference article text without per-document OA terms, MeasEval, and BioRED;
- admission/update training: none; the audited public datasets have no valid natural operation gold and the 360-pair Phase 1B pool remains unlabeled;
- validation/test: retain official membership and quarantine cross-split documents/query families;
- OOD only: POLYIE zero-shot and BioRED;
- scale only: SciFact-Open;
- case study only: the original 150 DOI materials collection.

Run the license/split audit before any data job:

```powershell
conda run --no-capture-output -n llm evimem-data audit-licenses
```

Do not start optimization based on the manifest audit alone. The selected view must also have a passing `evidence_alignment.json` and a leakage-safe subset in `leakage_report.json`. Ambiguous or blocked components are never overridden by Hugging Face metadata.

Do not commit downloaded data, model weights, checkpoints, logs, or generated episodes.
