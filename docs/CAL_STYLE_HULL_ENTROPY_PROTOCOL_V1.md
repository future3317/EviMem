# E54 CAL-style hull-entropy campaign protocol

Status: registered for execution on 2026-08-10.

This protocol adds a matched-posterior CAL-style baseline to the current
objective-first paper experiment. The policy scores a query by expected
reduction in the joint Gaussian entropy of the sampled complete-pool hull
vector. It is not an independent GP reproduction, a marginal Bernoulli
membership-entropy surrogate, or a new planning method.

## Frozen settings

- policy roster: `posterior_mean_target_margin`, `delta_hull_active_search`,
  `cal_style_hull_entropy`;
- one secure closed-loop `B=6` trajectory per frozen cross-fit fold, with
  `B=1..6` results derived from trajectory prefixes;
- five development folds covering the 230-system development universe;
- posterior samples `m=200` (the implementation may increase this to
  `max(200, c+32)` when the visible unique-composition grid has size `c>=200`);
- Gaussian fantasies `K=10`, relative covariance ridge `1e-10`, and
  `fixed_composition` hull backend;
- identical current joint PBE--r2SCAN posterior, reveal boundary, reference
  phases, duplicate convention, costs, and lexicographic immutable-ID tie
  breaking as the existing unified runner.

The visible entropy coordinates are unique reduced compositions from the
current query pool. Reference-only compositions remain hull inputs but do not
become entropy coordinates. Policy-facing code receives no oracle target
outcomes or final labels.

## Audit and analysis

The development summary must verify task, vault, and cross-fit identities;
policy roster; fold disjointness; fit/query disjointness; complete B=6
trajectories; zero failure markers; and six CAL diagnostics per system. It
reports exact-system paired sign-randomization contrasts, B=1..6 prefix
utilities, trapezoidal integrated effects, and CAL runtime/grid/sample-setting
metadata.

Only after the development code and summary audit pass may the explicitly
authorized 94-system rerun be executed once. It is a secondary held-out
MatPES rerun under the frozen protocol. Because those systems have prior
exposure documented in the experiment ledger, it must not be called external,
untouched, pristine, or independent-dataset validation.

Raw task, vault, traces, logs, and summaries remain outside Git.
