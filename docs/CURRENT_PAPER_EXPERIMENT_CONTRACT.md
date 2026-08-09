# Current paper and experiment contract

**Status (2026-08-09): active.** This is the short operational contract for
paper-facing work. `EXPERIMENT_LEDGER.md` remains the immutable provenance
record, but its historical method-specific stopping rules do not define the
current paper unless they protect an opened dataset, oracle boundary, or
invalidated result.

## Paper identity

The paper studies active search when a query reveals an immediate observation
but utility is assigned later by a global latent adjudicator. In the materials
instance, querying reveals a target-protocol energy and the complete target
pool determines whether a queried candidate belongs to the final convex hull.

The contribution order is:

1. formulate complete-pool, globally adjudicated utility;
2. characterize when repeated greedy can fail and when it is sufficient; and
3. compare objective alignment with additional planning complexity under a
   shared posterior and evaluator.

Delta-Hull is deliberately simple: it is the one-step Bayes action for final
hull membership. It is a baseline induced by the objective, not a complicated
new planner. The primary empirical claim is that, in the studied MatPES
regime, aligning the acquisition objective produces the measurable low-budget
gain, while the tested lookahead solvers change many actions but add little
terminal utility beyond Delta-Hull.

## Paper-facing policy ladder

The main text uses only three conceptual layers:

1. posterior target-margin greedy: matched-posterior objective comparator;
2. Delta-Hull: greedy complete-pool utility; and
3. Delta-Hull-anchored lookahead: planning beyond the aligned greedy policy.

Source-margin, SARR, IC-SARR, Hull-ENS, selective gates, and numerical wrappers
are historical or appendix diagnostics. They do not define the paper's method
identity and should not be expanded into a solver zoo.

## E52: authorized validation campaign

E52 is a new, independent identity authorized on 2026-08-09. It does not
overwrite, pool with, or retroactively relabel E32, E49, E50, or E51.

### E52-A: matched two-step numerical audit

`protocol_hull_knowledge_gradient` and Delta-Hull-anchored rollout estimate the
same exact two-step Bellman value. They differ in numerical integration, not
in planning class. Freeze a small outcome-independent development roster and
compare their actions, centered Q-values, and planning headroom at B=2. A
larger Hull-KG curve is authorized only if this audit finds disagreement beyond
the frozen numerical tolerance; Hull-KG is not a new headline method.

### E52-B: final-membership calibration

At every pre-reveal decision state, policy code may record candidate IDs and
posterior final-hull membership probabilities. Complete-pool labels may be
joined only after the trace has finished, on the evaluator side. Report Brier
score, clipped Bernoulli NLL, a reliability diagram, and ranking performance,
with uncertainty or resampling clustered by exact chemical system.

### E52-C: acquisition under pool shift

Construct nested 70%, 85%, and 100% visible query pools by an
outcome-independent stable hash over the protocol identity, exact chemical
system, and pair ID. Reduce only the current query fold; retain every row from
the other 184 development systems used for transport fitting. Rerun acquisition
without protecting selected IDs. This isolates query-pool robustness from a
change in posterior fitting and differs from the evaluator-only E49 audit.

### E52-D: external formation-energy validation gate

An external claim requires systems and pair IDs disjoint from every opened
MatPES development roster, with policy and analysis frozen before target
outcomes are opened. The 2026-08-09 raw-release rebuild yielded exactly the
same 324 systems and 10,236 pairs as the existing task, so it supplies zero
external candidates and must not be opened or reported as validation. E52-D is
blocked pending a genuinely disjoint formation-energy source or release; this
does not block E52-A--C.

### E52-E: clean-room rerun and secondary MatPES confirmation

Rebuild the current idea from a fresh execution boundary. The 230 systems in
prior folds 1--5 form a five-fold development universe; each fold fits only on
the other 184 development systems. The complementary 94 systems are excluded
from all E52 development fitting, calibration, baseline selection, thresholds,
and debugging. After E52-A--C and the analysis procedure are frozen, refit once
on all 230 development systems and run the 94-system panel once.

This execution is holdout-aware and free of fit overlap under E52. It is not
epistemically untouched: the 94 systems are exactly 46 earlier fold-0 systems
plus 48 previously opened repartition systems. Report it as a **secondary
held-out MatPES rerun under the frozen current protocol**, not as an untouched
or external confirmation panel.

## Invariants that remain binding

- Policy-facing code never receives oracle target energies or final labels.
- Every legal reveal remains in the immutable archive and conditions the
  registered posterior; no outcome-selected deletion or coreset is allowed.
- Composition-dependent hull transitions use the registered phase records and
  evaluator.
- Existing opened systems remain development evidence. A new split of those
  systems is not an external holdout.
- Raw datasets, vaults, checkpoints, and experiment outputs remain outside
  Git. Counts, identities, protocol hashes, and summary hashes are audited
  before any manuscript claim is added.

## What may now be simplified

Historical state-compression gates, SARR/IC-SARR numerical-gate rules,
Hull-ENS recovery details, runtime replays, and selective-planning screens no
longer govern current method development. They remain available in the ledger
or their historical protocol documents for provenance. Paper-facing code and
documentation should prefer one canonical name per policy, one shared runner
path, and the smallest diagnostic schema needed for E52.

The stopped 90-unit E52 launcher is not resumed. The reduced development run
contains 15 B=6 objective units (three query-pool fractions by five folds),
from which B=1..6 target-margin and Delta-Hull prefixes are derived, plus one
10-system B=2 numerical-equivalence unit. Final-membership calibration uses
all legal candidate-state predictions on the 100% Delta-Hull trajectories;
selected-action calibration is secondary.
