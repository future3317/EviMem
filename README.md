# EviMem-RL

This repository contains the reproducibility artifact for *Globally
Adjudicated Active Search: Revocable Discoveries, Greedy Search, and Planning Headroom*.
The paper studies active search in which a query reveals an immediate target
energy, while the selected candidate's discovery label is adjudicated on the
complete latent pool.

## Paper-facing result

The central object is complete-pool terminal utility, not a new universally
superior planner. The artifact implements and audits three conceptual policy
levels:

- posterior target-margin, a history-local proxy objective;
- Delta-Hull, the complete-pool greedy final-label baseline;
- Delta-Hull-anchored lookahead, used to test whether planning adds terminal
  value after the objective is aligned.

On the frozen cross-fitted MatPES development roster, objective alignment
changes low-budget acquisition and reduces selected-history revocation. The
tested lookahead solvers change many actions but add little terminal utility
beyond Delta-Hull. MatPES is development evidence; MAD-1.5 is an
atomization-energy protocol-shift stress test. Neither is an untouched
formation-energy holdout or a deployment benchmark.

## Reproduction entry points

The paper figures are rendered by the scripts in `tools/`. Generated figures,
datasets, oracle vaults, checkpoints, traces, and summaries remain outside
Git. Use the `llm` Conda environment:

```powershell
conda run --no-capture-output -n llm pytest -q
conda run --no-capture-output -n llm ruff check src tests tools
```

The live paper-facing analysis scripts are:

- `tools/render_delayed_search_paper_figures.py`: graphical problem statement
  and controlled mechanism figures;
- `tools/render_delayed_label_theory_figures.py`: objective and exact-certificate
  figures;
- `tools/render_e32_direct_lookahead_figure.py`: solver-comparison figure;
- `tools/render_followup_manuscript_figures.py`: E32/MAD supplementary curves;
- `tools/analyze_exact_certificate_audit.py`: posterior-only synthetic audit.

## Code architecture

- `src/matmem/protocol_closed_loop.py` isolates legal policy state, persisted
  actions, append-only reveals, the oracle vault, and causal hull evaluation.
- `src/matmem/posterior.py` and `src/matmem/transport.py` implement the frozen
  target-protocol working posterior.
- `src/matmem/hull_geometry.py` contains fixed-composition hull primitives and
  membership checks.
- `src/matmem/protocol_acquisition.py` contains the source, Delta-Hull,
  rollout, and IC-SARR policies.
- `src/matmem/policy_registry.py` and `src/matmem/configs/` hold registered
  policy identities and defaults.

Read the current paper and experiment contract in
[`docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md`](docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md)
before changing a method or launching an experiment. The complete scientific
audit trail, including stopped exploratory lines, remains in
[`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md); the documentation map
is [`docs/PROTOCOL_INDEX.md`](docs/PROTOCOL_INDEX.md).

## Scientific safeguards

Policy-facing code never sees unrevealed target energies or final labels.
Every paid reveal is retained in the immutable archive. Exact chemical-system
units and stoichiometry are preserved, and online metrics are kept separate
from evaluator-only `D`, `F`, and `T` metrics. Any future state-compression
proposal must reduce exactly to full-history use in the homogeneous
zero-transport-cost null regime.
