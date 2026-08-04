# Delayed-label evaluator sensitivity audit v1

**Status:** registered evaluator-only audit, 2026-08-04. This audit does not
rerun a policy, refit a posterior, alter an action, or open a new evaluation
system. It re-evaluates the already recorded E32-A selected sets against the
frozen development task and oracle vault. Raw and derived outputs remain
outside Git.

## Purpose

Measure whether the terminal complete-pool label and the direct policy
ordering are sensitive to the numerical hull tolerance or to omitted,
initially unqueried competitors. The primary E32 estimand and all policy
actions remain unchanged.

## Frozen inputs

- E32-A five-fold, B=1,...,6 selected sets and policy roster;
- E32 task, oracle vault, reference phases, duplicate-composition convention,
  and `fixed_composition` backend;
- all seven E32 policies; no policy is executed again.

## Evaluator conditions

1. Numerical formation/hull tolerance: `1e-8`, `1e-10`, `1e-11`, and `1e-12`.
2. Nested candidate pools: deterministic SHA-256-ranked 70%, 85%, and 100%
   of the candidate pool, with the union of all compared policies' selected
   IDs protected so that no recorded action is removed.
3. Competitor-removal stress: retain 90% and 80% of initially unqueried
   candidates using a system- and budget-specific deterministic seed; all
   selected IDs remain protected.

For every condition, report policy terminal `T`, change from the frozen full
pool, selected-label flip rate, mean retained candidate count, direct
Delta-Hull contrasts, and whether the ordering by mean `T` changes. Because
selected IDs are protected and actions are not recomputed, these are
evaluator/pool-sensitivity results, not action-robustness or external
validation results.

## Interpretation rule

Stable labels support the narrower statement that the observed development
mechanism is not numerically or visibly-pool fragile under these evaluator
perturbations. Any change in ordering is reported as sensitivity; it is not
used to select a preferred policy or to tune the manuscript claim.
