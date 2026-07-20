# matmem

`matmem` studies oracle-isolated, protocol-aware closed-loop materials
discovery. Every paid target-protocol outcome remains in an append-only archive;
the live research question is how a calibrated multi-fidelity belief can improve
target-protocol convex-hull discovery over direct low-fidelity ranking.

## Current status

There is no paper-level positive result yet.

- CAW-Joint, DACC, P3C, AKSC and the JARVIS--MP certificate experiments are
  stopped or historical lines. Their evidence dispositions and recovery points
  live in [the experiment ledger](docs/EXPERIMENT_LEDGER.md); their retired
  runners are not compatibility dependencies of the live package.
- The MatPES 2025.2 audit finds 385,890 exact same-configuration PBE--r2SCAN
  pairs, including 84,532 pairs with formation energies on both sides. Exact
  chemical systems and original Materials Project parents, rather than the
  upstream row split, define development partitions.
- A causal-hull audit found that the first MatPES runner normalized composition
  but retained an unnormalized cell total energy. Those earlier closed-loop
  traces are invalid after their first reveal. Candidates and oracle outcomes
  now preserve stoichiometric atom count; only policy-facing per-atom
  compositions are normalized.
- The coarse transport posterior used a system-balanced linear discrepancy plus
  one random system intercept. On eight development systems, 88 disjoint fit
  systems did not make its hull policy competitive with source margin.
- The current development model is a hierarchical PBE--r2SCAN discrepancy
  posterior: a system-balanced global delta model, an exact-system random
  intercept, and a system-local Matérn-5/2 residual process. The global mean
  uses only PBE observables and normalized element fractions; the local kernel
  uses a frozen 64-dimensional CHGNet-0.3.0 source-structure representation.
  Every revealed target outcome conditions the posterior.
- The resulting method is **Delta-Hull Active Search**: propagate the joint
  PBE--r2SCAN posterior through the complete target-protocol convex hull, then
  select the candidate with the largest posterior probability of belonging to
  that final hull. This is a delayed structured-label problem: the oracle
  reveals a continuous r2SCAN energy, while the query's value depends jointly
  on every pool energy and is known only through the final hull. The rule is
  the exact one-query Bayes action for equal query costs; variable-cost ratio
  scoring is deliberately rejected.
- On 24 hash-selected development systems at budget six, the 1024-point nested
  Sobol implementation obtains `3.7083` oracle-final confirmations per system
  versus `3.4583` for source margin. The paired exact-system difference is
  `+0.2500`, with a deterministic bootstrap 95% interval `[+0.0417,+0.5000]`
  (6 wins, 17 ties, 1 loss). It captures six of the 19 confirmations left
  between source margin and the finite-pool oracle ceiling. Causal discoveries
  remain tied at `4.3333`, and wall time rises from `1.99` to `22.13` seconds
  per system. MC512 and MC1024 give the same system-level discovery effect;
  their first actions agree in 23/24 systems, complete six-step traces in
  21/24, and individual rounds in 134/144. This is improved effect-level
  numerical stability, not exact trace convergence or a paper-level GO. The
  next gate is replication on a fresh exact-system split with MC1024 frozen.

No opened JARVIS--MP evaluation system is reused for current development, and
no MatPES result above is a sealed confirmatory evaluation.

## Live architecture

- `src/matmem/protocol_closed_loop.py`: typed candidates, observable policy
  state, append-only action/reveal records, oracle vault, composition-dependent
  causal hull, and selected-action-only reveal execution.
- `src/matmem/protocol_knowledge_gradient.py`: system-balanced transport,
  hierarchical local discrepancy posterior, nested scrambled-Sobol final-hull
  integration, Delta-Hull Active Search, two-step knowledge-gradient and
  continuous hull-risk diagnostics.
- `src/matmem/protocol_policy_worker.py`: oracle-free source, ridge, CHIC and
  protocol-hull policies in a persistent subprocess.
- `src/matmem/matpes_data.py`: protocol-neutral MatPES identities and strict
  same-configuration checks.
- `tools/audit_matpes_protocol_pairs.py`: complete PBE--r2SCAN pairing and
  parent/split audit.
- `tools/build_matpes_protocol_task.py`: composition-aware task construction
  with a separate target oracle vault.
- `tools/run_matpes_protocol_closed_loop_exploratory.py`: action-driven MatPES
  development runner and post-trace causal/final evaluator.
- `docs/EXPERIMENT_LEDGER.md`: authoritative history for valid, invalidated,
  incomplete and stopped evidence.
- `docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md`: no-deletion, null-regime and
  decision-alignment boundary for paper-facing methods.

Datasets, checkpoints, oracle vaults, event logs and experiment outputs remain
outside Git. The repository contains source, tests and small provenance notes
only.

## Environment

Python, pytest and Ruff use the `llm` Conda environment locally:

```powershell
conda run --no-capture-output -n llm python -m pip install -e ".[dev]"
conda run --no-capture-output -n llm pytest -q
conda run --no-capture-output -n llm ruff check src tests tools
```

Real-data runs execute on the laboratory server with explicit one-thread BLAS
limits so wall-time comparisons are reproducible. A representative development
command is:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
python tools/run_matpes_protocol_closed_loop_exploratory.py \
  --task /path/to/matpes-task.json \
  --development-vault /path/to/development-oracle-vault.json \
  --output /path/outside/git/result.json \
  --transport-family hierarchical_matern52_frozen_structure \
  --posterior-sample-count 128 \
  --policies source_margin ridge_margin delta_hull_active_search
```

## Research rule

Improve the measured failure cause, not the score count. A new paper-facing
method must keep all outcomes, use exact chemical-system units, preserve
stoichiometry, isolate the oracle, reduce to direct full-history use in the
homogeneous zero-cost null, and report causal discovery separately from
oracle-final confirmation and invalidation. Composition-aware posterior gains
must replicate across systems before any new acquisition module is added.
