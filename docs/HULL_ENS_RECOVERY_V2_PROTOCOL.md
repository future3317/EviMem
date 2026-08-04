# Hull-ENS P0 recovery v2: registered execution amendment

**Status:** registered engineering recovery, 2026-08-04. Raw outputs remain
outside Git. This amendment does not overwrite or merge the incomplete
`hull_ens_p0_v1_20260804` root.

## Why a recovery is needed

The first E48 P0 launcher completed all five `B=1` units, then failed in every
`B>=2` unit at the same `Co-Fe-Li-O` system while evaluating `hull_ens`.
Read-only traces show an empty `hull_ens` event file at that point. The
launcher exposed a 1,800-second option, but the runner routed that timeout
only to the older source-rollout policy list, so Hull-ENS actually used the
30-second default. A new single-system smoke reproduced this routing bug and
is retained as a separate incomplete diagnostic root. The remaining cost is
localized to repeated four-component fixed-hull evaluations, not to an oracle,
posterior, task, or reveal-boundary mismatch.

## Action-preserving implementation amendment

The acquisition objective and all numerical samples are unchanged. The runner
now applies the registered long selection timeout to Hull-ENS and Safe
Hull-ENS. The implementation also evaluates independent first-action
candidates in a bounded `ThreadPoolExecutor`. The default remains serial
(`hull_candidate_workers=1`) and the existing parity tests compare serial and
parallel scores and selected actions. The recovery uses
`hull_candidate_workers=4` and four independent closed-loop units at a time.
This reduces resource oversubscription relative to the failed 20-unit
launcher while preserving the registered posterior, fantasy count, seed,
fixed-composition backend, tie-breaking, and policy roster.

The concurrency is only an execution optimization: each candidate receives
the same conditional Gaussian samples, each fixed-hull call uses the same
cached composition template, and no candidate is pruned or replaced by a
proxy. The recovery identity records the worker count and source hashes.

## Recovery identity

- new root: `/home/workspace/lrh/DATA/EviMem-RL/analysis/hull_ens_p0_v2_recovery_20260804`;
- policies: `source_margin`, `delta_hull_active_search`, `hull_ens`,
  `safe_hull_ens`;
- folds: five outcome-independent folds; budgets `B=1..6`;
- method seed: `20270804`;
- posterior worlds: `128`; fantasy worlds: `8`;
- candidate-level workers: `4`; independent fold-budget units: `4`;
- hull backend: `fixed_composition`;
- transport family: `hierarchical_matern52_frozen_structure`.

The recovery may only be summarized if all `5 x 6` outputs are complete and
the summary verifies roster, system counts, task/vault/cross-fit/protocol/code
hashes, B=1 parity, direct paired D/F/T contrasts, wall time, and safe-gate
diagnostics. If it fails again, it remains incomplete and no manuscript claim
is updated.
