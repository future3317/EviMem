# E32 delayed-label objective and lookahead follow-up

**Status:** amended development/mechanism experiment, authorized 2026-08-03.
This identity is separate from E31 and must not overwrite or pool with any
previous MatPES or MAD output. Amendment A was recorded after the scratch
pilot showed that unrestricted six-step nested rollout is computationally
infeasible on the shared CPU server.

## Scientific question

E31 showed that IC-SARR improves complete-pool `T` relative to source margin,
but does not beat Delta-Hull or ungated source rollout. E32 tests the missing
attribution and mechanism questions:

1. Under one frozen posterior, does changing the acquisition target from a
   source/current-hull proxy to complete-pool hull membership change actions?
2. Does a rollout with repeated conditional Delta-Hull continuation improve
   over the repeated-greedy Delta-Hull baseline?
3. Are rollout--Delta-Hull action differences concentrated in posterior
   rank-unstable and cross-candidate-coupled states?

The new policy identity is `delta_hull_anchored_rollout`. It is a repeated
two-step rollout: it conditions the joint Gaussian posterior only on the
simulated reveals in a fantasy branch, then recomputes the myopic Delta-Hull
action for the next step. It never reads an unrevealed oracle outcome. The
existing `source_rollout_delta_hull` policy remains an unchanged comparator.
The frozen IC-SARR and diagonal-covariance comparators are taken from the
separately registered E31/P0 outputs, because their 1024/8192 integration
settings must not be silently changed by the E32 runtime amendment. The
two-step truncation is explicit; it is not presented as an exact full-horizon
solver.

## Frozen task and split

- MatPES task: the E31 all-eligible development task, SHA-256
  `f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df`.
- Oracle vault: E31 development vault, SHA-256
  `a272d3a2ce6286443ae6fce35726a688751a37284e3df362c5d1f70e2fcb9952`.
- Five-fold manifest: SHA-256
  `a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa`.
- Query unit: exact chemical system; all systems are opened development
  systems, not a holdout.
- Fit/query rule: transport fits only on the complement of each query fold.
- Hull backend: `fixed_composition`; all reference, duplicate and tolerance
  rules remain those of E31.

## Frozen policies and settings

Every policy in one output shares the same fit fold, query systems, seed,
posterior and tie break:

```text
source_margin
posterior_mean_target_margin
posterior_current_hull_probability
delta_hull_active_search
ungated_source_rollout
source_rollout_delta_hull
delta_hull_anchored_rollout
```

The amended E32-A settings are seed `20270720`, hierarchical Matern-5/2
frozen-structure transport, ridge penalty `1.0`, boundary temperature `0.05
eV/atom`, common posterior world count `128`, conditional continuation count
`16`, posterior-only rank diagnostic with 32 base samples, 8 conditional
samples, and two registered observations per candidate, and budgets
`B=1,...,6`. The stopped unrestricted pilot has a distinct output identity
and cannot be pooled with E32-A.

## Posterior-only rank/coupling diagnostics

At every Delta-Hull-anchored decision state, before any new oracle outcome is
opened, record:

- posterior final-hull membership probabilities and top-two rank margin;
- rank-switch probability after conditioning on each candidate's sampled
  posterior observation;
- cross-candidate influence
  `c_h(x -> y) = E[|p(y | O_x) - p(y)| | h]` using the registered conditional
  posterior samples;
- `kappa_h = max_x sum_{y != x} c_h(x -> y)` and its candidate-count
  normalized version.

The high-coupling strata are defined from these posterior-only quantities
before reading the realized evaluation fields. No realized `D`, `F`, `T`,
action regret or wall time is used to select a stratum.

## Required reports

For every policy and budget report exact-system paired `D`, `F`, `T`, `D-F`,
`F-T`, wall time, action disagreement and headroom. For direct comparisons
report paired bootstrap intervals, sign-flip p-values, win/tie/loss, and
incremental confirmations per second. For the Delta-Hull-anchored policy
also report action disagreement and realized `T` difference against
Delta-Hull by rank-margin and coupling quartile.

The analysis must additionally include the MAD-1.5 direct comparator only if
all registered policies can be run on the existing frozen MAD task without
changing its manifest, posterior, seed, cost definition or hull proxy. MAD
remains an atomization-energy protocol-shift proxy, not a formation-energy
holdout.

Hull-tolerance and candidate-pool sensitivity are separate endpoint analyses;
they must not be mixed into the primary E32 estimand. Query/planning cost is
reported as two separate quantities; no new cost weight is fitted from E32.

## Interpretation gate

- If Delta-Hull-anchored rollout has no direct gain over Delta-Hull, the paper
  remains an objective/theory/mechanism study with an empirical greedy-
  sufficiency result.
- If the direct gain is positive only in a posterior-only high-coupling or
  low-rank-margin stratum, report a mechanism interaction rather than overall
  solver superiority.
- No E32 result can be called external validation, deployment superiority,
  universal generalization, or an untouched holdout.

Raw outputs remain outside Git under a new `E32` analysis root. Manuscript
changes are forbidden until every registered output and failure artifact has
been audited.
