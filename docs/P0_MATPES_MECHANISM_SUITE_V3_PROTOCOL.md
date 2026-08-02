# P0 MatPES mechanism suite v3 protocol

**Experiment identity:** `P0-MATPES-V3-20260802`  
**Status:** frozen execution protocol; retrospective cross-fitted development/mechanism evidence only  
**Frozen code commit:** `9e9c2c0025b6ec23ec0eeea0b9b2299769905112`  
**Base scientific protocol:** `docs/P0_MECHANISM_SUITE_PROTOCOL.md`

## Purpose and claim level

This run completes the already registered MatPES baseline, finite-horizon and
component-mechanism suite. It is intended to strengthen the positive story
around delayed full-pool acquisition, source-anchored nonmyopic rollout and
the numerical-confirmation mechanism. It does not create a holdout, change
the method, or upgrade the 324-system corpus to confirmatory evidence.

All 324 eligible MatPES systems are development or historical systems. The
statistical unit remains the exact chemical system, and every summary must
retain the retrospective cross-fit label.

## Frozen scientific inputs

| Item | Frozen value |
|---|---|
| Task | `f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df` |
| Oracle vault | `a272d3a2ce6286443ae6fce35726a688751a37284e3df362c5d1f70e2fcb9952` |
| Five-fold manifest | `a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa` |
| Query folds | the five frozen exact-system cross-fit folds |
| Budgets | core curve `B=1..6`; component audit `B=6` |
| Hull backend | `fixed_composition` |
| Transport | `hierarchical_matern52_frozen_structure` |
| IC-SARR seed | `20270720` |
| IC-SARR integration | stage one `1024/16` blocks; stage two `8192/16` blocks |
| Random reference seeds | `20270721..20270725` |

The policy roster, estimands, inference seeds, synthetic exact-DP suite and
completion rule are inherited verbatim from the base protocol. No score,
posterior, MC count, Sobol stream, gate, threshold, fallback, budget,
manifest or source-margin rule may be changed after execution begins.

## New execution identity and output isolation

The complete v3 raw output root is:

`/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_v3_20260802/`

All raw JSON, logs, failure markers and summaries remain outside Git. The
incomplete v1 and v2 roots are excluded completely:

- `p0_mechanism_suite_20260730/`
- `p0_mechanism_suite_v2_20260730/`

The launcher may retain the historical `p0v2` filename stem for schema
compatibility, but a v3 summary may read only files below the new v3 root.
Missing or failed cells are errors; they may not be skipped, inferred or
pooled with an older root.

The remote project environment is `equivcompiler` because the configured
remote Conda installation does not provide the repository's `llm` environment.
The exact initialization and command are recorded in the run log.

## Execution order

1. Run one-system, one-budget preflight using the frozen task, fold and
   backend, writing only to a v3 `preflight/` subdirectory.
2. Verify output schema, policy action/reveal boundary, task/vault/manifest
   hashes, and absence of P0 processes before starting the main schedule.
3. Run the complete core curve: five folds × six budgets, with the three
   registered curve policies.
4. Run the complete B=6 baseline/component audit: five folds, the full
   registered roster, and all five random reference seeds.
5. Run the registered complete 1,000-instance synthetic exact-DP suite if it
   is absent from the v3 root; never reuse the older synthetic output as a
   partial substitute for a failed material cell.
6. Summarize only after every registered unit is complete and verify all
   manuscript-facing tables/figures against the resulting hashes.

No manuscript text or figure is changed before the complete v3 schedule has
an explicit completion record.

