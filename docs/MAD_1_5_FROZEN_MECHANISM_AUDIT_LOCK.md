# MAD-1.5 frozen mechanism-audit lock

Status: secondary analysis on an already opened public protocol-shift panel.
This is not a new confirmatory evaluation, a new primary endpoint, or authority
to tune IC-SARR, the MAD task, or any cost weight.

## Purpose

The primary MAD curve is frozen in `MAD_1_5_CURVE_PROTOCOL_LOCK.md` and its
claim remains unchanged: IC-SARR has a small oracle-final AUC signal, no
final-causal superiority, and negative preregistered cost-aware utility.  The
present audit asks mechanism questions only:

1. Is the observed ordering compatible with posterior mean correction,
   one-step full-pool targeting, or finite-horizon source continuation?
2. How much remaining finite-pool headroom is present before each policy acts?
3. Does a descriptive gain--time frontier change the frozen primary cost
   conclusion? (It must not.)

The 96 systems have already been opened.  All outputs from this document are
secondary, descriptive mechanism evidence and must never be relabeled as a
holdout, a new external confirmation, or IC-SARR superiority.

## Frozen inputs and implementation

- Task: `E:\DATA\MAD-1.5-v1\mad15_task_v1.json`
- Vault: `E:\DATA\MAD-1.5-v1\mad15_vault_v1.json`
- Query manifest: `E:\DATA\MAD-1.5-v1\mad15_curve_manifest_v1.json`
  (the same 96 exact systems and six recorded folds as E42--E43)
- Baseline repository commit before this audit: `c460e1de0d06f3e75fb496546ff9e2a1d878df2d`
- Runner: `tools/run_matpes_protocol_closed_loop_exploratory.py`
- Policy worker: `src/matmem/protocol_policy_worker.py`
- Hull backend: `fixed_composition`
- Transport family: `ridge_random_intercept`, ridge penalty `1.0`, prior
  standard deviation `0.1`, boundary temperature `0.05 eV/atom`
- Policy seed: `20270720`; statistical seed, if a summary is produced:
  `20260730`
- Posterior sample count: `64`; fantasy count: `3`
- Exact budgets: `B=1,2,3,4,5,6`; every budget reuses the same manifest.

No parameter in this list may be changed after any output is read.  The
primary E43 source-margin versus IC-SARR outputs remain untouched and are not
overwritten.

## Originally registered ladder (superseded after budget one)

The runner already implements the following policies.  This audit does not
add or modify a policy.

| Mechanism role | Canonical runner policy |
| --- | --- |
| Source baseline | `source_margin` |
| Ridge posterior mean-margin | `ridge_margin` |
| One-step full-pool objective | `delta_hull_active_search` |
| Source-continuation rollout (SARR) | `source_rollout_delta_hull` |
| Independent-confirmation rollout (IC-SARR) | `independent_confirmation_source_rollout` |

For each budget, all five policies run against the same task, vault, manifest,
transport fit and tie break.  Outputs must be written outside Git with a new
identity:

```powershell
conda run --no-capture-output -n llm python tools/run_matpes_protocol_closed_loop_exploratory.py `
  --task E:\DATA\MAD-1.5-v1\mad15_task_v1.json `
  --development-vault E:\DATA\MAD-1.5-v1\mad15_vault_v1.json `
  --query-manifest E:\DATA\MAD-1.5-v1\mad15_curve_manifest_v1.json `
  --query-budget B --maximum-budget 6 --minimum-candidates 8 `
  --seed 20270720 --posterior-sample-count 64 --fantasy-count 3 `
  --hull-backend fixed_composition --transport-family ridge_random_intercept `
  --policies source_margin ridge_margin delta_hull_active_search `
    source_rollout_delta_hull independent_confirmation_source_rollout `
  --split development `
  --output E:\DATA\MAD-1.5-v1\mad15_mechanism_ladder_BB_20260730.json
```

Replace both `B` tokens with the same integer.  Before any curve-level table,
verify each output's task, vault, manifest, exact budget, policy list and
implementation checksums.  Do not compare a newly rerun source or IC trace to
E43 as a timing result; E43 remains the primary pre-registered cost record.

## Numerical, headroom, and cost rules

The MC audit is a separate action-state check.  It may reevaluate only a
predeclared hash sample of IC-SARR/source disagreement states at MC512, MC1024
and MC8192 with the same posterior, seed family and candidate IDs.  It reports
action agreement, score ordering and retained-sign rate; it does not alter the
independent confirmation gate or select a new MC count.

Headroom strata use pre-query quantities only: candidate count, composition
count, chemical-system order, source/target discrepancy, source-margin density
and oracle recoverable headroom.  Oracle headroom is descriptive and cannot
define a new policy or filter the primary estimate.

Cost--gain plots may report incremental oracle confirmation against wall time,
incremental confirmations per second, Pareto dominance and break-even cost
weights.  The E43 coefficient `0.10` and its negative utility are immutable;
no descriptive break-even value may be promoted to a favorable utility claim.

## Stopping and interpretation

If a policy cannot be reproduced from the existing canonical implementation,
the ladder stops at that policy rather than introducing an adapter or new
hyperparameter.  If the complete ladder exceeds the operational budget, retain
only complete outputs and report the remainder as incomplete.  No incomplete
file supplies a metric or ranking.

Regardless of outcome, this audit cannot establish final-causal superiority,
cost-aware superiority, runtime advantage, bounded-memory superiority, or
universal external generalization.

## Execution status and stop decision

The `B=1` batch completed on 2026-07-30 and is recorded as E45 in the
experiment ledger. It wrote 480 complete traces (96 systems by five policies)
to a new external output identity. Budgets `B=2..6` have not been started and
are now **stopped**, not deferred: completing them would create new degrees of
freedom on an already opened panel without strengthening the paper's central
evidence. The completed batch is retained only as a horizon-one-collapse sanity
check: one-step Delta-Hull, SARR and IC-SARR tie when no continuation horizon
remains. It is appendix/ledger scope, not a primary result, a curve point, or a
new positive claim.
