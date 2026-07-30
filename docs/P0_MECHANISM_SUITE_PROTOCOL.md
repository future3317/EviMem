# P0 mechanism-suite protocol

**Status:** locked before implementation or new outcomes (2026-07-30).

## Scope and scientific status

This protocol creates a complete **retrospective, cross-fitted MatPES
development/mechanism suite** and an independent synthetic exact-DP mechanism
suite.  It does not create an external evaluation.  All 324 eligible MatPES
chemical systems have previously been opened in development or historical
experiments; no subset may be renamed a holdout, validation set, sealed set,
or confirmatory evidence.  The primary MAD-1.5 curve (E43) and its stopped
five-policy ladder (E45) are not rerun, extended, or used here.

The frozen IC-SARR policy is an object of comparison, not tuning.  This work
must not change its posterior, score, target, source continuation, scrambled
Sobol streams, two-stage numerical gate, threshold, fallback, seed, or
MC1024/MC8192 integration budgets.  Comparator variants are isolated policy
identities and cannot replace or silently alter IC-SARR.

## Fixed MatPES task and execution identity

| Item | Frozen value |
|---|---|
| Task SHA-256 | `f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df` |
| Oracle-vault SHA-256 | `a272d3a2ce6286443ae6fce35726a688751a37284e3df362c5d1f70e2fcb9952` |
| Five-fold manifest SHA-256 | `a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa` |
| Query systems | the existing five cross-fit folds, exact chemical-system grain |
| Fit/query rule | each fold fits transport only on the complement of its query fold |
| Budgets | exact unit-cost `B=1,2,3,4,5,6`; same system/fold at every budget |
| Hull rule | frozen fixed-composition backend; same tolerance, duplicate and reference rules as E23/E43 |
| Transport posterior | frozen hierarchical Matern-5/2 frozen-structure posterior |
| IC-SARR randomization | seed `20270720`, stage one 1024 samples/16 scrambled-Sobol blocks, stage two 8192 samples/16 blocks |
| Runner identity | commit `c460e1de0d06f3e75fb496546ff9e2a1d878df2d` plus checksums of every touched module |

All new raw outputs are written only below
`E:\DATA\EviMem-RL\analysis\p0_mechanism_suite_20260730\`.  Each output
must include the task, vault, manifest, code, configuration and output hashes.
No artifact is committed to Git.

## Predeclared policies and comparison ladder

Every MatPES budget/fold output contains every policy below; a policy may not
be omitted after its results are known.

| Identity | Role | Joint covariance | source continuation | common worlds | numerical gate |
|---|---|---:|---:|---:|---:|
| `random` | deterministic hash-random reference | -- | -- | -- | -- |
| `source_margin` | strong source-protocol baseline | -- | -- | -- | -- |
| `posterior_mean_target_margin` | target posterior-mean margin baseline | yes | -- | -- | -- |
| `ridge_margin` | non-hull learned-margin baseline | -- | -- | -- | -- |
| `ridge_uncertainty` | non-hull uncertainty baseline | -- | -- | -- | -- |
| `delta_hull_active_search` | greedy one-step complete-pool action | yes | -- | common | -- |
| `ungated_source_rollout` | rollout without numerical fallback | yes | yes | common | no |
| `source_rollout_delta_hull` | stage-one simultaneous-gated SARR | yes | yes | common | stage one |
| `diagonal_ic_sarr` | covariance ablation | diagonal only | yes | common | stages one and two |
| `independent_mc_ic_sarr` | independent IID-MC world ablation | yes | yes | no | stages one and two |
| `independent_confirmation_source_rollout` | frozen IC-SARR | yes | yes | common | stages one and two |

`random` is run at the five fixed seeds `20270721` through `20270725`; its
reported system value is the within-system mean over these five predeclared
paths.  All other policies use `20270720`.  No old/stopped method (WBM,
DACC/P3C/AKSC, CHIC, or any Dual-Horizon policy) is included.

`posterior_mean_target_margin` means the candidate's posterior mean target
formation energy minus the competing causal hull at its composition, selecting
the smallest value with the canonical pair-ID tie break.  It is not an alias
for `ridge_margin`.

## Required estimands and analysis

For every policy, budget and exact system, preserve the full action/reveal
trace and report:

- provisional causal discoveries `D`, selected-history confirmations `F`,
  complete-pool confirmations `T`, `D-F`, and `F-T`;
- wall seconds/system and incremental `T` per incremental second against
  source margin;
- finite-pool ceiling and recovered headroom;
- action disagreement with source margin; for gate-bearing policies, accepted
  deviations, rejected positive screens, stage-one and stage-two diagnostics;
- predicted first-action paired rollout advantage and post hoc actual
  counterfactual `T` and `F` advantage at every IC-SARR decision state.

The primary comparison ladder is Source margin -> greedy final -> ungated
rollout -> IC-SARR.  Diagonal and independent-world variants are component
ablations, not alternative primary methods.  The exact chemical system is the
only statistical unit.  At each budget and for predeclared curve summaries,
report paired system-bootstrap 95% intervals (20,000 resamples, seed
`20260730`) and 100,000-draw sign-flip/randomization p-values (seed
`20260731`).  Report all budgets; neither a favorable budget nor a favorable
endpoint may be selected after outcomes open.

Curve summaries use the equally weighted AUC across `B=1..6`.  No new
cost-aware utility is introduced: runtime is reported as a coequal tradeoff,
and this development suite cannot overturn E43's frozen MAD cost result.

## Rollout-advantage calibration

For every legal IC-SARR decision state, record the stage-one paired predicted
advantage of the final selected non-source action versus the source action.  A
counterfactual replay uses the true complete target pool while retaining the
same selected first action and frozen source continuation, yielding paired
actual `T` and `F` differences.  The summary must include:

- predicted-advantage deciles versus mean actual `T` and `F` advantage;
- Spearman correlation (and its finite-state sample count);
- accepted versus rejected/screened state strata;
- false-positive accepted-deviation rate: accepted non-source states with
  actual complete-pool `T` advantage less than or equal to zero.

These diagnostics assess posterior-model value calibration, not the numerical
gate's integration guarantee and not a new efficacy endpoint.

## Independent exact-DP synthetic suite

The suite is generated independently of MatPES, MAD, WBM and all oracle
vaults.  Before generation, the implementation fixes 1,000 instances, seed
`20260730`, pool size uniformly in `5..10`, query budget uniformly in `1..4`,
and equal query costs.  Each instance has a finite discrete latent world,
source signal strength drawn from `{0.0, 0.5, 1.0}`, energy-correlation
strength drawn from `{0.0, 0.5, 1.0}`, delayed-label coupling drawn from
`{0.0, 0.5, 1.0}`, posterior-noise scale drawn from `{0.02, 0.05, 0.10}`, and
number of competing facets drawn from `1..3`.  The generated parameter
record, world probabilities and complete energy table are saved under the
P0 data directory before aggregate outcomes are inspected.

The policies are exact belief-state DP, source margin, greedy final, exact
source-continuation rollout, and sampled frozen-logic IC-SARR.  The synthetic
IC comparator uses a separately registered 128-sample stage-one screen and
512-sample stage-two confirmation, each in sixteen blocks, because it samples
the finite discrete world posterior rather than the material Gaussian model.
The suite
reports the full distribution of `V_DP - V_policy`, the fraction that rollout
equals DP, the fractions rollout beats source and greedy, and sampled IC-SARR
agreement with the exact rollout first action.  It also reports every
predeclared factor cell, including cells with no gap.  It establishes only
controlled mechanism behavior and no materials or deployment claim.

## Completion and reporting rule

The protocol is complete only after every policy, fold, budget, random seed,
and all 1,000 synthetic instances have either completed or have an explicit
failure artifact.  A failed variant cannot be silently removed.  Until then,
no manuscript claim, table, or figure is changed and no incomplete cell is
used for inference.  The final paper-facing interpretation remains limited to
mechanism evidence unless a genuinely new, pre-frozen external task is created
outside this protocol.
