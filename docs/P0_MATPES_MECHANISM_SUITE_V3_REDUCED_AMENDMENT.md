# P0 MatPES suite v3 reduced-ablation amendment

**Experiment identity:** `P0-MATPES-V3-REDUCED-20260802`  
**Status:** authorized scope reduction; no change to method or core curve  
**Frozen code baseline:** `70e0393`  
**Parent protocol:** `docs/P0_MATPES_MECHANISM_SUITE_V3_PROTOCOL.md`

## Reason for the amendment

The full B=6 roster with five random seeds is a useful exhaustive audit, but
it is not necessary for the central paper story once the core budget curve
and the independent exact-DP suite are complete. The full roster would spend
most of its time on low-priority reference variants and repeated random
paths. The amendment therefore reduces only the ablation allocation; it does
not change IC-SARR, the posterior, the hull backend, the seeds of retained
policies, the estimands, or the statistical unit.

The already-running v3 core curve remains unchanged and continues under its
original output root. The full v3 ablation roster is not started and is
explicitly cancelled by this amendment.

## Reduced B=6 mechanism roster

Run the following six policies on the five frozen cross-fit folds at exact
budget `B=6`:

1. `source_margin` — strong source-protocol baseline;
2. `delta_hull_active_search` — greedy one-step complete-pool action;
3. `ungated_source_rollout` — source continuation without numerical gate;
4. `source_rollout_delta_hull` — stage-one simultaneous-gated rollout;
5. `diagonal_ic_sarr` — diagonal-covariance ablation with the two-stage gate;
6. `independent_confirmation_source_rollout` — frozen IC-SARR.

This roster isolates the three paper-relevant mechanisms:

- objective change: source margin to greedy final-hull action;
- finite-horizon planning: greedy to ungated/source-continuation rollout;
- numerical confirmation and joint-world modeling: ungated/stage-one/diagonal
  variants to IC-SARR.

The reduced suite does not support the phrase “complete baseline suite.”
Posterior-mean, ridge-uncertainty and multi-seed random references remain
optional appendix diagnostics only and are not run in this phase.

## Execution and reporting

The reduced raw output root is:

`/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_reduced_v1_20260802/`

Outputs remain outside Git. Every fold must complete all six policies, with
the same task, vault, cross-fit manifest, `fixed_composition` backend,
hierarchical Matern-5/2 transport, and frozen IC-SARR integration settings as
the parent protocol. Missing or failed cells are not summarized.

The reduced suite is reported as a targeted mechanism ablation, not as a
confirmatory evaluation and not as a holdout. The paper may use it to explain
where the observed acquisition signal comes from, while the core curve and
synthetic exact-DP suite carry the main finite-horizon story.

