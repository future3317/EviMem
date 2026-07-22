# Dual-Horizon SARR

`constrained_dual_horizon_source_rollout` is an independent policy built on
Source-Rollout SARR. It is a diagnostic and development method; it does not
change the definitions or frozen parameters of IC-SARR.

For every legal first action `x`, the policy uses common scrambled-Sobol
posterior samples and the source-margin continuation for the remaining budget.
The same simulated selected set is scored at two horizons:

* **Terminal horizon T**: selected candidates that are stable in the complete
  sampled target-protocol hull containing every visible candidate.
* **Selected-history horizon F** (also called campaign-final causal horizon):
  selected candidates that are stable in the hull containing only the current
  revealed phase records and the outcomes selected along that simulated
  continuation. Unselected sampled outcomes are never inserted into this hull.
  This is not the online discovery horizon D.

Let `Delta_T(x)` and `Delta_F(x)` be paired advantages over the current
source-margin action. A non-source action is admissible only if its simultaneous
one-sided lower bounds satisfy `LB_T(x) > 0` and `LB_F(x) >= 0`. The selected
action maximizes terminal posterior mean among admissible actions. If no action
passes both gates, the source-margin action is returned. There is no
reward-mixing weight and no outcome-selected posterior coreset.

All actual outcomes remain append-only in the protocol archive; the policy sees
only the reveal history available before the action. The implementation uses
`2 * max(n - 1, 1)` Bonferroni comparisons for the two objective families and
the same block partition for both horizons. Unequal query costs are rejected,
as in the existing Source-Rollout implementation.

The two endpoint lower bounds use a single familywise numerical error budget:
the implementation passes `2 * max(n - 1, 1)` to each endpoint bound, so the
per-candidate alpha is allocated over both endpoints and all non-source
actions. This controls only scrambled-Sobol integration error, not posterior
calibration, oracle-truth advantage or realized campaign non-inferiority.

The first run is a fold-0 development pilot. Fold 0 was used for earlier
Source-Rollout development, so it is not external confirmation. Report
terminal confirmations, causal confirmations, causal-time announcements,
within-campaign revocations, unqueried-competitor invalidations, action regret,
runtime and dual-gate fallback rate. A terminal gain without a causal gate is
intentionally rejected and is not evidence for the method.
