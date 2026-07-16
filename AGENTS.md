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

## External-model API safety

- API keys are process secrets. Read a rotated key only from an environment
  variable such as `DEEPSEEK_API_KEY`; never accept it as a CLI option or write
  it to code, prompts, manifests, reports, fixtures, Git, or terminal output.
- An API model may see only a minimal view derived from a safe packet: task ID,
  source dataset, two claim texts, evidence locators, and visible source-level
  notice. It must not receive source identifiers, checksums, raw exports,
  sampling metadata, prior labels, or quarantined files.
- API retries, malformed output, rate limits, or exhausted budget must leave a
  task unannotated and queued for review. Never write a rule-derived fallback
  as `ai_juror`, `ai_critic`, or `ai_adjudicated_silver`.
- When the same model serves two blinded calls, call them separate blinded API
  calls, not independent-model agreement. Record the true returned model ID,
  a non-zero prompt checksum, packet-set checksum, and aggregate token usage.
- DeepSeek API calls must explicitly set `thinking: {"type": "disabled"}`
  unless the user specifically authorizes thinking-mode cost. Never retain
  `reasoning_content` or chain-of-thought in artifacts.
- Every exported API annotation must have a matching append-only per-call
  ledger event with its stage, task/packet/prompt/annotation checksums,
  requested and returned model IDs, hashed provider request ID, timestamp, and
  usage. A `--resume` may reuse only an output proven by that ledger; manifests
  must report newly called versus reused records separately.
- External API execution is blind-primary-only. Never send one model's labels,
  notes, critic output, or silver output to another external model. Compare
  blind candidate exports locally with `tools/run_blind_adjudication_gate.py`.
  The gate never selects a label winner: disagreement, non-sufficient evidence,
  same-scope contradiction, or insufficient model diversity must remain in the
  high-risk review queue. Even unanimous candidates are not gold, training
  data, or final AI adjudication.
