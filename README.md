# EviMem

**Evidence-Certified Memory for Continual Scientific Curation**

EviMem studies three learnable decisions over a chronological scientific-document stream:

```text
Should Write -> What to Retrieve -> How to Update
```

The research object is long-term scientific memory, not a domain-pretrained language model or a free-acting agent. Every reusable memory record contains a structured claim, immutable evidence references, a complete verification certificate, a typed decision, document time/source, and policy identity. A learned manager can propose only hierarchical semantic/scope/authority/evidence labels; deterministic gates retain authority over memory admission, operation compilation, verification slots, supersession, and publication.

## Implemented boundary

```text
public dataset annotations / current evidence
                    |
                    v
        ScientificMemoryRecord candidate
                    |
          +---------+---------+
          |                   |
          v                   v
 supervised admission   certified retrieval
 + relation labels      (semantic + structure
          |               + authority + time
          |               - conflict - stale)
          +---------+---------+
                    |
                    v
         deterministic memory gate
                    |
                    v
       append-only memory + lineage

current immutable evidence -> deterministic verifier -> publication gate
```

The package includes:

- cross-domain `ScientificMemoryRecord` and `ScientificClaimRecord` contracts;
- verified, rejected, conflict, and superseded memory semantics;
- hard certificate/evidence/policy admission checks;
- hierarchical update labels and a deterministic compiler for `ADD`, `MERGE`, `LINK`, `CONFLICT`, `SUPERSEDE`, and `IGNORE`;
- time-safe evidence-certified retrieval that returns full records and warrants;
- SciMem-Curate episode/oracle separation and public-dataset converters;
- an audited dataset manifest that prevents OOD and 150-DOI case-study data from entering training;
- an executed three-seed retrieval validity pilot and structured memory-manager interfaces;
- deterministic evidence binding, tuple verification, publication gating, and atomic publication storage.

The repository intentionally contains no downloaded datasets, papers, model checkpoints, or raw training logs. It contains aggregate audit/pilot reports, but does not claim formal paper results, a trained update manager, or SciMem-Update gold.

## Environment and installation

Python 3.11 or newer is required. The maintained development environment is the Conda environment named `llm`:

```powershell
conda create -n llm python=3.11 -y
conda run --no-capture-output -n llm python -m pip install -e ".[dev,semantic]"
```

The optional `train` extra installs Transformers, PEFT, TRL-adjacent data tooling, and Accelerate interfaces, but installing it does not authorize or start training:

```powershell
conda run --no-capture-output -n llm python -m pip install -e ".[dev,semantic,train]"
```

## Phase 1B status

Phase 1A is frozen at `64ecedd0341cb5199df33ad5f05a0d4a45a9429c`
with tag `phase1a-data-validity-audit`. Phase 1B has:

- run a retrieval-only pilot on SciREX 517 train / 177 evaluation and the
  leakage-safe SciFact 679 train / 139 evaluation split;
- run QASPER 4,555 only as an internal diagnostic, never as training data;
- compared TF-IDF, BM25, two frozen dense encoders, three fine-tuned seeds,
  certificate-aware reranking, and the EviMem score under identical pools,
  top-k, and token budgets;
- built 360 **unlabeled** SciMem-Update candidates and a standard Label Studio
  configuration.

### Annotation-safety status

The pilot's generated local external annotation export is
`annotation/scimem_update_pilot_external_safe.jsonl` (not versioned in Git).
It removes `sampling_stratum_not_gold` and `candidate_is_gold` while retaining
the minimum source and evidence provenance required for review. The original
unlabeled pool is an internal artifact and must not be supplied to external
annotators or external models.

Model-produced annotations are currently **debugging and planning artifacts
only**. They are not human-reviewed labels, SciMem-Update gold, update-manager
training data, or paper results. Earlier candidate outputs that exposed native
sampling strata are quarantined locally. Current independent-model and
protocol-recheck outputs remain unmerged until they pass the annotation QA gates described in the
[annotation guidelines](docs/SCIMEM_UPDATE_ANNOTATION_GUIDELINES.md). In
particular, source-level Crossref/Retraction Watch metadata cannot establish a
claim-level relation, authority comparison, or supersession.

The primary fine-tuned three-seed mean is Recall@10 `0.9747 ± 0.0032` and MRR
`0.7562 ± 0.0034`. These are retrieval-validity pilot measurements, not formal
paper main results. Certificate-aware gain is not estimable because the licensed
retrieval views contain no certificate/memory-type gold; the fail-closed reranker
does not invent it. See [retrieval results](reports/phase1b/retrieval_results.md)
and [implementation status](docs/IMPLEMENTATION_STATUS.md).

Human annotation has not started. No model consensus is human consensus, the
update manager was not trained, and no QLoRA run was performed. See the
[annotation guidelines](docs/SCIMEM_UPDATE_ANNOTATION_GUIDELINES.md) and
[labelbook](docs/SCIMEM_UPDATE_LABELBOOK.md).

## Data protocol

Phase 1A's fail-closed manifest authorizes only the leakage-safe retrieval subsets of SciREX and SciFact for the Phase 1B retriever pilot. It does not authorize admission/update training. QASPER is local-evaluation-only pending an official dataset LICENSE checksum; Evidence Inference source articles require per-document OA resolution; MeasEval and BioRED are blocked. POLYIE remains OOD-only, SciFact-Open remains scale-only, and the 150 DOI collection is never optimization data.

Review and update [configs/datasets.json](configs/datasets.json) before acquiring any data:

```powershell
conda run --no-capture-output -n llm evimem-data audit-licenses
```

The manifest records component-level terms for annotations, source text, code, and derived artifacts. A view is fail-closed unless every required component is `confirmed`, has an official license checksum, and permits training. License readiness alone does not override semantic, evidence-alignment, or leakage failures; see [the Phase 1A report](reports/data_audit/PHASE_1A_SUMMARY.md).

The deterministic audit runner reads pinned upstream releases from paths outside this repository and writes only checksums and aggregate reports:

```powershell
conda run --no-capture-output -n llm python tools/run_phase1a_audit.py --help
```

Phase 1B runners likewise read releases and Crossref responses outside the
repository. They commit no full text or model weights:

```powershell
conda run --no-capture-output -n llm python tools/run_phase1b_candidate_pool.py --help
conda run --no-capture-output -n llm python tools/run_phase1b_retrieval_pilot.py --help
```

## Development

```powershell
conda run --no-capture-output -n llm python -m pytest -q
conda run --no-capture-output -n llm ruff check .
```

See [Methods](docs/METHODS.md), [Architecture](docs/ARCHITECTURE.md), [Supervised training](docs/TRAINING.md), and [implementation status](docs/IMPLEMENTATION_STATUS.md).
