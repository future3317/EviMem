# E55 Numerical Convergence and Rollout-Curve Design

## Objective

E55 adds two paper-facing checks without changing any existing policy identity,
posterior model, E32/E53/E54 trajectory, or headline estimand:

1. numerical convergence of Delta-Hull and the matched CAL-style hull-entropy
   control; and
2. a complete `B=2,...,6` comparison of repeated Delta-Hull with the existing
   Delta-Hull-anchored two-step rollout.

The first item requires new development-only trajectories. The second item is a
read-only reanalysis of the already completed E32 five-fold 230-system artifacts.

## Evidence boundaries

- All systems are from the opened MatPES development universe.
- No secondary 94-system panel is opened or rerun.
- No target energy, final label, prior policy outcome, or E54 effect is used to
  choose the convergence roster.
- Raw tasks, vaults, manifests, trajectories, logs, and summaries remain outside
  Git.
- E32 and E54 artifacts are immutable inputs. E55 writes only to a new external
  root and refuses to overwrite an output or reuse a failure marker.

## Frozen inputs

E55 uses the E52 100% visible-pool task, vault, and five-fold development
cross-fit manifest. It retains the hierarchical Matérn-5/2 frozen-structure
transport family, fixed-composition hull backend, exact reveal boundary,
reference phases, duplicate convention, immutable-ID tie breaking, and seed
`20260810`.

The expected task, vault, and cross-fit hashes are recorded in every unit
manifest and verified by the summarizer. Each development fold contains 46
query systems and 184 fit systems.

## Delta-Hull convergence panel

Run one `B=6` Delta-Hull-only closed-loop trajectory for every development fold
at posterior sample counts

`M = {64, 128, 256, 512, 1024}`.

The `M=1024` setting is the numerical reference. For every lower setting, report:

- per-decision selected-action agreement with `M=1024` on matched observed
  states until trajectories diverge;
- first-action agreement per system;
- Spearman rank correlation and centered-score error on matched candidate score
  vectors when diagnostics expose a common candidate roster;
- exact-system mean terminal `T`, paired difference from the `M=1024` reference,
  and a system bootstrap 95% interval; and
- runtime median and IQR.

The convergence result is a numerical sensitivity analysis, not a new policy
comparison. Existing E53 `M=1024` trajectories are not silently pooled because
E55 has its own frozen seed and one-policy execution identity.

## CAL-style convergence panel

Select 15 transport-supported systems, three from each original cross-fitting
fold. Selection uses only the task's public candidate count, the cross-fit
assignment, chemical-system strings, and fit-element coverage:

1. within a fold, order systems by candidate count and divide them into three
   deterministic rank bins using integer terciles;
2. within each bin, retain systems whose elements all remain represented in the
   original 184-system fit roster;
3. select the lowest SHA-256 rank of
   `release_id || e55-cal-convergence-v1 || fold_index || bin || system`.

The resulting roster covers low, middle, and high candidate-count complexity in
every fold without using target outcomes. A derived sub-cross-fit manifest keeps
the original 184 fit systems and only the selected three query systems.

For every selected fold roster, run CAL-style-only `B=6` trajectories on the
Cartesian grid

`M = {100, 200, 400}` and `K = {5, 10, 20}`.

All runs use relative ridge `1e-10`, `hull_candidate_workers=8`, the tested
runtime-plan geometry cache, and a 21,600-second per-selection timeout. The
`M=400, K=20` setting is the numerical reference. Report selected-action
agreement, matched score/rank stability where available, terminal `T`, and
runtime across the 15 fixed systems. The formal E54 230-system result remains the
frozen `M=200, K=10` estimand; E55 does not replace or rescale it.

## Existing rollout curve

Read the complete E32 amendment-A artifacts only. Verify the recorded task,
vault, cross-fit identity, five folds, 230 unique systems, policy roster,
posterior count `128`, fantasy count `3`, and complete independent executions at
budgets `B=1,...,6`.

For `B=2,...,6`, compare `delta_hull_anchored_rollout` directly with
`delta_hull_active_search` and report:

- absolute mean terminal `T` for both policies;
- paired rollout-minus-Delta-Hull mean, system-bootstrap 95% interval, and the
  existing paired sign-randomization p-value;
- systems with any action disagreement and mean per-system disagreement rate;
- runtime median, IQR, and matched difference; and
- the trapezoidal integrated `B=2,...,6` terminal-utility effect.

No E32 trajectory is regenerated. If the immutable inputs do not contain a
required metric, the summary records it as unavailable rather than inventing a
replacement execution.

## Implementation

Add one roster builder, one E55 launcher, and one E55 summarizer under `tools/`,
with focused tests under `tests/`. Reuse the unified secure closed-loop runner and
existing paired-inference helpers. Do not add a second acquisition
implementation or modify policy code unless a failing parity test demonstrates
that the existing diagnostics are insufficient.

The launcher supports explicit stages (`delta`, `cal`) and writes a unit
manifest before execution. Units may run in parallel, but each output is
write-once and each failure marker is terminal for that root.

## Acceptance criteria

- roster construction is deterministic, outcome-independent, fold-balanced,
  fit-element-supported, and preserves each fold's original fit roster;
- the launcher emits exactly 25 Delta-Hull units and 45 CAL-style units with the
  frozen settings;
- focused tests, Ruff, and the relevant secure-runner tests pass;
- all completed artifacts pass hash, roster, count, policy, budget, backend,
  seed, and diagnostic audits;
- no existing E32/E53/E54 file is modified or overwritten; and
- manuscript changes are made only after the E55 summary and E32 read-only
  summary both pass.
