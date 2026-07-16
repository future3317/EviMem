---
name: evimem-ai-adjudication
description: Fail-closed blind-candidate annotation workflow for SciMem-Update Phase 1B. Activate when the user invokes /evimem-ai-adjudication with modes packet, juror, critic, judge, normalize, or validate, or when asked about blind juror review, candidate routing, or canonical annotation validation for SciMem-Update.
argument-hint: <mode> [mode-specific args]
allowed-tools: [Read, Glob, Grep, Bash, Write, Edit]
---

# evimem-ai-adjudication

This skill implements the SciMem-Update Phase 1B AI-adjudicated silver-label
workflow. It generates machine-provenance silver labels only; it does **not**
create human annotation, SciMem-Update gold, memory operations, or training
data.

## Important operational rule

Do **not** run `juror`, `critic`, or `judge` inside the full EviMem-RL
repository. Each juror must operate inside a minimal sandbox that contains only
the safe packets (`annotation/scimem_update_pilot_external_safe.jsonl` derived)
and no old model outputs, unlabeled exports, sampling strata, or native labels.
Independence comes from **different Claude sessions in different sandboxes with
frozen outputs**, not from multiple votes in the same session.

Current Claude Code 2.1.118 supports loading a plugin temporarily with
`--plugin-dir`:

```bash
claude --plugin-dir E:\CODE\EviMem-RL\.claude\plugins\evimem-ai-adjudication
```

## Usage

```
/evimem-ai-adjudication <mode> [args...]
```

Modes:

- `packet` — Generate minimal, auditable packets from `annotation/scimem_update_pilot_external_safe.jsonl`.
- `juror` — Review packets one at a time and produce a four-axis juror annotation JSONL.
- `critic` — Review packets plus two frozen juror outputs and produce a critic review JSONL.
- `judge` — Review packets, two juror outputs, and a critic review to produce an `ai_adjudicated_silver` JSONL.
- `normalize` — Losslessly normalize Label Studio nested exports or canonical JSONL.
- `validate` — Fail-closed validation of canonical JSONL against the schema and hard rules.
- `deepseek-api` — Run the V4-Pro-only DeepSeek blind-candidate workflow on safe packets only.

## General safety rules (apply to every mode)

1. The only permitted external-model input is `annotation/scimem_update_pilot_external_safe.jsonl`. Never expose `annotation/scimem_update_pilot_unlabeled.jsonl`, sampling strata, `candidate_is_gold`, native benchmark labels, old model outputs, old candidate sets, old audit reports, or compiled operations.
2. Crossref/Retraction Watch pairs are source-level metadata only. Without separately visible claim-level correction evidence, the four axes must be `INSUFFICIENT_CONTEXT / UNKNOWN_SCOPE / UNRESOLVED / INSUFFICIENT`.
3. `CONTRADICTORY + SAME_SCOPE` requires an authority assessment. If no explicit claim-level correction, curator decision, or stronger certificate resolves it, `authority_relation` must be `UNRESOLVED`; `NOT_APPLICABLE` is invalid for this combination.
4. Different SciREX methods are `DIFFERENT_SCOPE`, even when task, dataset, and metric match.
5. Never fabricate missing entity aliases, scope dimensions, conditions, dates, certificates, authority, or evidence.
6. All provenance must be machine: `ai_juror`, `ai_critic`, `ai_adjudicated_silver`, `packet:external_safe`, or similar. Never use `human-reviewed`, `adjudicated human evidence`, `gold`, or `SciMem-Update gold`.
7. Never output operation fields or labels: `ADD`, `MERGE`, `LINK`, `CONFLICT`, `SUPERSEDE`, `IGNORE`, `compiled_operation`, `update_operation`, `operation`, or `memory_operation`.
8. Each label must be based only on the visible evidence in the current packet. Do not use dataset strata, batch patterns, previous task labels, or hidden files to infer labels.
9. Temporary reasoning may be written locally, but the final exported record must keep only a short, evidence-locatable note. Do not preserve chain-of-thought in the exported artifact.

## Mode: deepseek-api

Use `tools/run_deepseek_adjudication.py` when an API model is authorized for
machine-provenance candidate generation. The runner permits only
`deepseek-v4-pro` and reads its credential only from `DEEPSEEK_API_KEY`. Never place a
credential in a command, file, report, prompt, or Git configuration.

The runner accepts only a packet directory. It sends DeepSeek a minimal model
view that excludes source document IDs and checksums, retains the true returned
model ID and a per-call prompt checksum, and writes all output under ignored
`runs/` paths. It explicitly disables DeepSeek thinking mode and never retains
reasoning content. Each validated annotation is linked to an append-only
checksum-only API ledger; do not accept a resumed record without its matching
ledger proof. It never auto-falls back to a rule-derived semantic label.

Run a no-cost plan first:

```powershell
conda run --no-capture-output -n llm python tools/run_deepseek_adjudication.py `
  --packets packets --output runs/deepseek-v4-pro-pass-a --primary-only --dry-run
```

Then, after setting a rotated key in the inherited process environment or the
gitignored local `.env`, remove `--dry-run`. The runner reads only the
`DEEPSEEK_API_KEY` assignment and never exports it. After completion, run the
offline ledger audit:

```powershell
conda run --no-capture-output -n llm python tools/run_deepseek_adjudication.py `
  --packets packets --output runs/<run-id> --verify-run
```

Authenticated API execution must include `--primary-only`. Crossref/Retraction
Watch records are put in a source-level gate file without a model call.
Primary-only results are candidates, not adjudicated silver. Never pass one
model's labels or notes to another external model. Compare two or more blind
candidate exports locally with `tools/run_blind_adjudication_gate.py`; its gate
never selects a label winner and routes disagreement, non-sufficient evidence,
same-scope contradiction, or insufficient model diversity to high-risk review.

## Mode: blind-gate

Run this mode only on the local machine after two or more blind candidate
exports. The gate reads candidate files locally; it never sends their labels or
notes to another model, and it does not write a selected four-axis label, silver
label, or memory operation. Under the V4-Pro-only policy, the two exports must
come from separate blind API calls with distinct `juror_run_id` values; this is
a stability check, never independent-model agreement.

```powershell
conda run --no-capture-output -n llm python tools/run_blind_adjudication_gate.py `
  --packets runs/review/packets `
  --candidate runs/deepseek-v4-pro-pass-a/votes/juror-a.jsonl `
  --candidate runs/deepseek-v4-pro-pass-b/votes/juror-a.jsonl `
  --same-model-repeat --output runs/blind-gate
```

With `--same-model-repeat`, a fully consistent, sufficient-evidence result is
only `same_model_repeat_consistent_candidate`: it is not multi-model consensus,
silver, gold, a training target, or a paper result. Disagreement on any axis,
non-sufficient evidence, same-scope contradiction, or a reused blind run
requires review.

## Mode: packet

Generate minimal packets from the external-safe file.

```
/evimem-ai-adjudication packet --input annotation/scimem_update_pilot_external_safe.jsonl --output packets/
```

Behavior:

1. Run `python tools/run_ai_adjudication.py packet --input <input> --output <output>`.
2. Verify that each generated packet contains only `task_id`, `source_dataset`, `left`, `right`, `source_level_update_type`, `source_level_update_notice`, `packet_provenance`, and `packet_checksum`.
3. Confirm that no packet contains `sampling_stratum_not_gold`, `candidate_is_gold`, native labels, or operation fields.
4. Packets may be written as one file per task (`--output packets/`) or as a single JSONL file (`--output packets.jsonl --jsonl`).

## Mode: juror (run in an isolated sandbox)

Review packets one at a time and produce a four-axis annotation JSONL.

```
/evimem-ai-adjudication juror --run-id juror-a --input packets/ --draft juror-a-draft.jsonl --output votes/juror-a.jsonl
```

Behavior:

1. Confirm you are in a sandbox that contains **only** the packet files under `packets/`. If you can see other juror outputs, the unlabeled export, or old model files, stop and ask the user to fix the sandbox.
2. Read only the packets in `--input`. Do not read other juror outputs, model outputs, or hidden files.
3. For each packet:
   - Write an observation/evidence matrix. Map each scope dimension (population/material, method, endpoint/property, conditions, time, measurement setting) to the exact evidence locator in the packet.
   - Assign the four axes using only the visible evidence:
     - `semantic_relation`
     - `scope_relation`
     - `authority_relation`
     - `evidence_sufficiency`
   - Do not use keywords, regex, word-overlap heuristics, or rule scripts to generate labels. Background knowledge may help you understand terminology, but it must not supply facts absent from the packet.
4. Compile all records into the output JSONL at `--output`. Each record must include the canonical fields plus `juror_run_id`.
5. Write the reviewed records first to a draft JSONL, then compile only that
   draft. The CLI deliberately cannot generate placeholder labels:

   ```
   python tools/run_ai_adjudication.py juror --run-id juror-a --input packets/ --draft juror-a-draft.jsonl --output votes/juror-a.jsonl
   ```
6. Run `python tools/run_ai_adjudication.py validate --input <output> --packets <input>` to verify.
7. If validation fails, revise the offending record. Never ask the tool to auto-fix the label.

## Mode: critic (run in a read-only sandbox)

**External-model prohibition:** an external API model must not run this mode,
because it would receive prior annotator labels. Use `blind-gate` for model
candidate routing. This mode is retained only for a local human review process.

Review packets plus two frozen juror outputs and identify errors.

```
/evimem-ai-adjudication critic --input packets/ --juror-a votes/juror-a.jsonl --juror-b votes/juror-b.jsonl --draft critic-draft.jsonl --output reviews/critic.jsonl
```

Behavior:

1. Confirm the sandbox contains only the packets and the two frozen juror outputs. Treat both juror outputs as read-only.
2. For each packet:
   - Read the packet and the two juror outputs.
   - Do not compile an operation.
   - Per axis, look for:
     - Entity conflation or alias invention.
     - Endpoint or property swapping.
     - Scope omissions (missing population, method, condition, time).
     - Authority overreach (newer date alone treated as higher authority; Crossref treated as claim-level correction).
     - Statistical overreach (generalizing beyond the visible population or conditions).
     - Unstated facts smuggled into `evidence_note` or `uncertainty_note`.
   - Every issue must reference an evidence locator from the packet (`left_evidence_locator` or `right_evidence_locator`).
3. Write the critic reviews to a draft JSONL, then compile only that draft:

```
python tools/run_ai_adjudication.py critic --input packets/ --juror-a votes/juror-a.jsonl --juror-b votes/juror-b.jsonl --draft critic_draft.jsonl --output reviews/critic.jsonl
```
4. Run `python tools/run_ai_adjudication.py validate --input <output> --packets <input>`.

## Mode: judge (run in a separate sandbox)

**External-model prohibition:** an external API model must not run this mode,
because it would receive prior annotator labels. Use `blind-gate` for model
candidate routing. This mode is retained only for a local human review process.

Produce an `ai_adjudicated_silver` label from packet + jurors + critic.

```
/evimem-ai-adjudication judge --input packets/ --juror-a votes/juror-a.jsonl --juror-b votes/juror-b.jsonl --critic reviews/critic.jsonl --draft judge-draft.jsonl --output silver/ai_adjudicated_silver.jsonl
```

Behavior:

1. Confirm the sandbox contains only the packets, the two frozen juror outputs, and the frozen critic review.
2. For each packet:
   - Read the packet, both juror outputs, and the critic review.
   - Accept a final label only if every field has visible evidence support in the packet.
   - If evidence is insufficient, downgrade to `PARTIAL` or `INSUFFICIENT`, or set `requires_higher_tier_ai_review: true`.
   - Do **not** accept a conclusion merely because two jurors agree. Majority vote is not evidence.
   - For Crossref/Retraction Watch pairs without claim-level evidence, the output must remain conservative (`INSUFFICIENT_CONTEXT / UNKNOWN_SCOPE / UNRESOLVED / INSUFFICIENT`).
3. Write the judge labels to a draft JSONL, then compile only that draft:

```
python tools/run_ai_adjudication.py judge --input packets/ --juror-a votes/juror-a.jsonl --juror-b votes/juror-b.jsonl --critic reviews/critic.jsonl --draft judge_draft.jsonl --output silver/ai_adjudicated_silver.jsonl
```
4. Run `python tools/run_ai_adjudication.py validate --input <output> --packets <input>`.

## Mode: normalize

Losslessly normalize an annotation export.

```
/evimem-ai-adjudication normalize --input annotation/export.jsonl --output annotation/canonical.jsonl
/evimem-ai-adjudication normalize --input annotation/label_studio_export.jsonl --output annotation/canonical.jsonl --label-studio
```

Behavior:

1. Run `python tools/run_ai_adjudication.py normalize --input <input> --output <output> [--label-studio]`.
2. Ensure the transformation preserves the original file checksum (`original_checksum`), original provenance, and transformation version. Do not change semantic labels.
3. If normalizing from Label Studio nested format, verify that `data` and `annotations[0].result` are flattened into the canonical schema without altering choice values or note text.

## Mode: validate

Fail-closed validation of canonical JSONL.

```
/evimem-ai-adjudication validate --input silver/ai_adjudicated_silver.jsonl --packets packets/
```

Behavior:

1. Run `python tools/run_ai_adjudication.py validate --input <input> [--packets <dir>] [--label-studio]`.
2. The validator only rejects; it never auto-fixes labels or emits operations.
3. Report the summary (record count, unique task IDs, source path). If validation fails, report the first violation and stop.

## Canonical record schema

Every canonical annotation record must contain:

```json
{
  "task_id": "...",
  "semantic_relation": "...",
  "scope_relation": "...",
  "authority_relation": "...",
  "evidence_sufficiency": "...",
  "evidence_note": "...",
  "uncertainty_note": "...",
  "annotation_provenance": "...",
  "annotator_id": "...",
  "model_id": "...",
  "prompt_checksum": "...",
  "packet_checksum": "...",
  "schema_version": "phase1b-v3",
  "gold_status": "not_gold"
}
```

Judge output additionally contains:

```json
{
  "juror_run_ids": ["...", "..."],
  "critic_run_id": "...",
  "adjudication_path": "...",
  "evidence_locator_refs": ["..."],
  "requires_higher_tier_ai_review": false
}
```

## Output provenance

- `annotation_provenance` must be one of: `ai_juror`, `ai_critic`, `ai_adjudicated_silver`, `packet:external_safe`, `normalize:label_studio`, `normalize:canonical`.
- `annotator_id` should identify the model run (e.g., `claude-opus-4-7-juror-a`).
- `model_id` should identify the model family/version.
- `prompt_checksum` should be a SHA-256 of the prompt/context used.
- `packet_checksum` must match the packet being annotated.

## Disallowed outputs

Never include:

- `sampling_stratum_not_gold`, `candidate_is_gold`, `native_support`, `native_contradict`.
- `compiled_operation`, `update_operation`, `operation`, `memory_operation`.
- Operation labels `ADD`, `MERGE`, `LINK`, `CONFLICT`, `SUPERSEDE`, `IGNORE` in notes.
- Claims that the output is human-reviewed, gold, or a training target.
