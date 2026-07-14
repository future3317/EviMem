# Maintainer contract

- Use `conda run --no-capture-output -n llm ...` for Python, pytest and Ruff.
- Do not add datasets, downloaded papers, checkpoints or experiment outputs to
  the repository.
- The learned controller may emit only structured `CurationAction` values.
- `REQUEST_PUBLICATION` is a request, never publication authority.
- Verification slots may be changed only by deterministic verifier output.
- Memory admission requires evidence, certificate and policy identity.
- Oracle/gold benchmark annotations must remain invisible during inference.
- Prefer maintained libraries such as Pydantic, scikit-learn,
  Transformers/PEFT/TRL and Accelerate over local reimplementations.
- Do not introduce legacy EviPGCE compatibility adapters.

## SciMem-Update annotation safety

- External annotators and external models may receive only
  `annotation/scimem_update_pilot_external_safe.jsonl`, never the original
  unlabeled export or a file containing `sampling_stratum_not_gold`,
  `candidate_is_gold`, native benchmark labels, compiled operations, or prior
  annotator outputs.
- Keep blinded annotators isolated from one another. A third blind annotator
  must not select its own tasks by reading earlier labels; use a prepared,
  label-free task pack or annotate the whole permitted source subset.
- A model label is always `model_generated`, `model_review`, or another
  explicitly machine provenance. Never call it human-reviewed, adjudicated
  human evidence, SciMem-Update gold, or a training target merely because
  multiple models agree.
- Do not claim independent model agreement when outputs were produced by shared
  deterministic rules, keyword matching, overlap heuristics, or a common
  classifier. Such artifacts are heuristic baselines, not blind annotations.
- Never write a memory operation, compiled operation, or operation-like label
  into an annotation export or annotation note. The `UpdateCompiler` remains
  the only operation compiler.
- Missing entities, aliases, conditions, dates, certificates, authority, or
  evidence remain missing. Do not infer them from dataset-native labels,
  sampling strata, surrounding files, or domain convention.

## Annotation import gates

- Reject an annotation export if it contains `sampling_stratum`,
  `candidate_is_gold`, `native_support`, `native_contradict`, an operation
  field, or an operation label in a task/annotation note.
- Crossref/Retraction Watch records are source-level metadata only. Without
  separately visible claim-level evidence, they must be
  `INSUFFICIENT_CONTEXT / UNKNOWN_SCOPE / UNRESOLVED / INSUFFICIENT`; do not
  infer `EQUIVALENT`, `SAME_SCOPE`, `NEWER_MORE_AUTHORITATIVE`, or claim-level
  correction/supersession from a DOI relation or timestamp.
- For `CONTRADICTORY + SAME_SCOPE`, authority must be assessed. Use
  `UNRESOLVED` when no explicit claim-level correction, curator decision, or
  stronger certificate resolves it; `NOT_APPLICABLE` is invalid for that case.
- Different SciREX methods are different scope under the current labelbook,
  even when task, dataset, and metric match.
- Preserve quarantined contaminated outputs for audit traceability, but do not
  stage, train on, evaluate against, or re-export them. Do not overwrite their
  provenance with a repaired output; write a new versioned file instead.
