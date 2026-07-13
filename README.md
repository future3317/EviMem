# EviMem

**Evidence-Certified Memory for Continual Scientific Curation**

EviMem studies three learnable decisions over a chronological scientific-document stream:

```text
Should Write -> What to Retrieve -> How to Update
```

The research object is long-term scientific memory, not a domain-pretrained language model or a free-acting agent. Every reusable memory record contains a structured claim, immutable evidence references, a complete verification certificate, a typed decision, document time/source, and policy identity. A learned manager can propose only a structured admission/update action; deterministic gates retain authority over memory admission, verification slots, supersession, and publication.

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
 and typed update       (semantic + structure
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
- typed `ADD`, `MERGE`, `LINK`, `CONFLICT`, `SUPERSEDE`, and `IGNORE` updates;
- time-safe evidence-certified retrieval that returns full records and warrants;
- SciMem-Curate episode/oracle separation and public-dataset converters;
- an audited dataset manifest that prevents OOD and 150-DOI case-study data from entering training;
- supervised retriever and structured memory-manager interfaces;
- deterministic evidence binding, tuple verification, publication gating, and atomic publication storage.

The repository intentionally contains no datasets, papers, checkpoints, or experiment outputs. It also does not claim completed model training or paper results.

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

## Data protocol

Phase 1A authorizes no training. The current audited candidates are only the leakage-safe retrieval subsets of SciREX and SciFact. QASPER is local-evaluation-only pending an official dataset LICENSE checksum; Evidence Inference source articles require per-document OA resolution; MeasEval and BioRED are blocked. POLYIE remains OOD-only, SciFact-Open remains scale-only, and the 150 DOI collection is never optimization data.

Review and update [configs/datasets.json](configs/datasets.json) before acquiring any data:

```powershell
conda run --no-capture-output -n llm evimem-data audit-licenses
```

The manifest records component-level terms for annotations, source text, code, and derived artifacts. A view is fail-closed unless every required component is `confirmed`, has an official license checksum, and permits training. License readiness alone does not override semantic, evidence-alignment, or leakage failures; see [the Phase 1A report](reports/data_audit/PHASE_1A_SUMMARY.md).

The deterministic audit runner reads pinned upstream releases from paths outside this repository and writes only checksums and aggregate reports:

```powershell
conda run --no-capture-output -n llm python tools/run_phase1a_audit.py --help
```

## Development

```powershell
conda run --no-capture-output -n llm python -m pytest -q
conda run --no-capture-output -n llm ruff check .
```

See [Methods](docs/METHODS.md), [Architecture](docs/ARCHITECTURE.md), [Supervised training](docs/TRAINING.md), and [implementation status](docs/IMPLEMENTATION_STATUS.md).
