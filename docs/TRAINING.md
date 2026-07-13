# Controller Training

EviMem uses maintained Hugging Face libraries for optimization and keeps its
own code limited to scientific-curation contracts, action safety and verifier
reward integration.

## Stage A: next-action imitation

Each JSONL row is an `OracleActionExample`. Its prompt is constructed only
from `ControllerState` and the legal action mask. The target action appears
only in the assistant completion. Split examples with
`split_examples_by_document`; never split individual steps from one DOI across
train, validation and test.

```powershell
conda run --no-capture-output -n llm evimem-train sft `
  --model Qwen/Qwen2.5-1.5B-Instruct `
  --train-jsonl data/train.oracle.jsonl `
  --eval-jsonl data/validation.oracle.jsonl `
  --output-dir checkpoints/controller-sft
```

The trainer uses TRL `SFTTrainer` with completion-only loss and PEFT LoRA.

## Stage B: verifier-shaped GRPO

GRPO prompt rows omit expert actions and all gold annotations. A separate
reward JSONL contains `CertifiedRewardRecord` objects:

```json
{
  "example_id": "oracle-step-...",
  "candidate_id": "candidate-...",
  "evidence_release_id": "release-...",
  "domain_pack_id": "piezoelectric",
  "domain_pack_version": "1.3.0",
  "domain_pack_hash": "sha256-...",
  "verifier_version": "harness-1",
  "action_rewards": {
    "{\"arguments\":{\"query\":\"d33\"},\"type\":\"RETRIEVE_TABLE\"}": 1.25
  },
  "artifact_hash": "sha256:..."
}
```

The reward artifact must be produced by a deterministic verifier rollout,
not by an LLM judge or self-confidence score. It is captured by the reward
callback and is never serialized into the model prompt.
Use `CertifiedRewardRecord.create(...)` to bind and hash these fields; do not
construct reward records by manually copying IDs.

```powershell
conda run --no-capture-output -n llm evimem-train grpo `
  --model checkpoints/controller-sft `
  --train-jsonl data/train.oracle.jsonl `
  --rewards-jsonl data/train.certified_rewards.jsonl `
  --output-dir checkpoints/controller-grpo `
  --num-generations 4
```

TRL performs group-relative normalization (`scale_rewards="group"`) and the
clipped GRPO objective. The EviMem executor remains the final action mask, and
invalid or unavailable actions receive deterministic penalties.

## Single-GPU defaults

- LoRA rank 16, alpha 32;
- per-device batch size 1;
- gradient accumulation 8;
- bf16 and gradient checkpointing enabled;
- four GRPO generations per state;
- maximum completion length 256 tokens;
- no automatic reporting or upload.

Run three explicit seeds for paper results. Do not select a seed after looking
at the test set, and do not publish checkpoints or metrics as scientific
results without recording the evidence release and DomainPack identities.
