# EviMem-RL

EviMem-RL is a clean research codebase for evidence-warranted memory and
constrained sequential control in scientific literature curation. It was
extracted from the larger EviPGCE project so the new paper can evolve without
legacy pipeline APIs, historical datasets, or experiment artifacts.

## Independence

This repository is runtime-independent from Piepaper/EviPGCE. Production code
contains no imports, hard-coded paths, symbolic links or package metadata that
refer to the old repository. External extraction, verification and publication
systems integrate through typed protocols and artifacts rather than source-code
imports. `tests/test_project_isolation.py` enforces this boundary.

Required third-party dependencies are Pydantic, pandas, PyArrow and structlog.
The `semantic`, `train` and `dev` extras declare scikit-learn,
Transformers/Datasets/Accelerate/PEFT/TRL, and test tooling respectively.

The core safety boundary is simple:

```text
controller policy -> structured action -> deterministic executor
                  -> external verifier certificate
                  -> audited reward/replay/memory
```

A policy may request publication, but it cannot authorize or commit it.
Publication remains the responsibility of an external deterministic harness.

## Included

- immutable evidence, candidate, claim-state, certificate, memory and
  trajectory contracts;
- discrete controller actions and a deterministic executor;
- evidence-warranted append-only memory and structured retrieval;
- verifier-shaped reward and integrity-checked replay;
- sequential benchmark contracts with oracle isolation;
- human-review request and expected-value policy contracts;
- an end-to-end episode runtime with no publication database dependency.

## Not included

- source papers, datasets, downloaded files or historical experiment results;
- EviPGCE extraction and publication implementations;
- compatibility adapters for the old project;
- a handwritten imitation-learning or GRPO implementation.

Training should use the optional maintained Transformers, PEFT and TRL
dependencies after benchmark episodes and oracle trajectories are frozen.
The implemented SFT/GRPO interfaces and command lines are documented in
[docs/TRAINING.md](docs/TRAINING.md).

## Development

Use the requested Conda environment:

```powershell
conda run --no-capture-output -n llm python -m pytest -q
conda run --no-capture-output -n llm ruff check .
```

The complete research proposal is in [docs/METHODS.md](docs/METHODS.md), and
the implemented software boundary is documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).
