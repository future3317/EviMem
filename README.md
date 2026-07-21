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
  subsequent outcome-independent repartition reserves 48 systems (16 binary,
  16 ternary and 16 higher-order) and refits transport on the other 276. On
  this panel Delta-Hull obtains `3.6250` oracle-final confirmations versus
  `3.5625` for source margin: paired difference `+0.0625`, bootstrap 95% CI
  `[-0.1042,+0.2292]`, exact two-sided sign-flip `p=0.6291` (10 wins, 31 ties,
  7 losses). The development signal therefore does not replicate strongly
  enough for a superiority claim.
- The main cause is limited decision headroom, not a constant selector.
  Delta-Hull and source margin agree on only 75/288 round actions, but source
  already reaches the budget ceiling on 24/48 systems. It leaves 35 total
  confirmations and Delta-Hull recovers a net three (8.57%). Delta-Hull is
  better than the two posterior-margin baselines, but source margin remains the
  strongest comparator.
- Cached fixed-composition hull propagation and separation of offline
  posterior-hull diagnostics reduce the 48-system run to about 9.5 minutes.
  The cached backend exactly matches all six Delta-Hull actions on the three
  real MC1024 traces completed by the original pymatgen reference run, in
  addition to the binary/ternary/duplicate-composition property tests.
- The live continuation is now **Source-Rollout Delta-Hull**, not another
  myopic score. For every candidate first action, it samples a complete target
  energy vector, updates the simulated causal hull with that sampled outcome,
  and uses the deployed source-margin policy for every remaining budget step.
  It deviates from source only when eight paired scrambled-Sobol blocks give a
  positive one-sided numerical-integration lower bound. This is a
  posterior-relative rollout improvement mechanism, not a real-distribution
  safety guarantee.
- The 276 transport-fit systems are assigned outcome-independently to six
  cross-fit folds of 46 systems each; every fold contains 12 binary, 19
  ternary and 15 higher-order systems. The 48 opened systems are excluded.
  Existing opened traces are used only for horizon attribution: five of the
  seven final Delta-Hull losses become persistent only in rounds 5 or 6,
  consistent with a finite-horizon mismatch but not sufficient to prove it.
  On the first 46-system out-of-fold budget-six panel at MC1024,
  Source-Rollout improves over source by `+0.1739` confirmations/system (11
  wins, 30 ties, 5 losses), with bootstrap 95% interval
  `[-0.0217,+0.3696]`. On the 22 systems where source leaves headroom, the
  difference is `+0.5455`. MC512/1024 agree on 45/46 system-level effects and
  41/46 first actions, but only 31/46 complete traces and 220/276 individual
  rounds. The mechanism remains promising development evidence; folds 1--5
  are paused because action-level numerical convergence is not yet adequate.

No opened JARVIS--MP evaluation system is reused for current development. The
48 MatPES systems are now opened and must not be used to tune Delta-Hull.

## Live architecture

- `src/matmem/protocol_closed_loop.py`: typed candidates, observable policy
  state, append-only action/reveal records, oracle vault, composition-dependent
  causal hull, and selected-action-only reveal execution.
- `src/matmem/protocol_knowledge_gradient.py`: system-balanced transport,
  hierarchical local discrepancy posterior, nested scrambled-Sobol final-hull
  integration, exact cached causal-hull envelopes, myopic Delta-Hull,
  full-budget Source-Rollout, two-step knowledge-gradient and continuous
  hull-risk diagnostics.
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
- `tools/build_matpes_confirmatory_task.py`: outcome-independent fresh split
  builder that excludes every development system and parent configuration.
- `tools/freeze_matpes_transport_model.py`: immutable transport artifact with
  optimizer metadata and fit-system checksums.
- `tools/audit_matpes_fixed_hull_parity.py`: fail-closed action/sample parity
  audit for the optional cached lower-hull backend.
- `tools/audit_matpes_sobol_seed_stability.py`: independent-scramble seed
  diagnostic for development systems.
- `tools/build_matpes_source_rollout_crossfit.py`: outcome-independent six-fold
  development plan excluding all opened MatPES evaluation systems.
- `tools/diagnose_matpes_horizon_mismatch.py`: attribution-only prefix analysis
  of the already opened myopic/source traces.
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
