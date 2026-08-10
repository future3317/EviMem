# CAL-style Hull-Entropy Acquisition

## Goal

Add a matched-posterior CAL-style baseline for the current MatPES unified
experiment. The baseline minimizes expected joint entropy of the random
completed-pool hull function, while reusing the existing posterior, hull
backend, reveal boundary, deterministic tie-breaking, and D/F/T evaluator.

## Scientific contract

- This is a CAL-style implementation, not an exact reproduction of an
  independent GP implementation from prior work.
- The acquisition object is the joint entropy of the sampled hull vector, not
  a sum of marginal membership entropies.
- The policy uses the current joint PBE--r2SCAN posterior and does not refit
  after fantasy observations; fantasies use exact Gaussian conditioning.
- The entropy grid is frozen before policy execution and contains only unique
  reduced compositions from the visible candidate pool. Reference-only
  compositions remain in every hull construction but are excluded from the
  entropy vector.
- Default formal settings are posterior hull samples `m=200`, or
  `max(200, c_max + 32)` when any grid has at least 200 compositions, fantasy
  count `K=10`, and relative covariance ridge `1e-10`.
- Common random numbers are reused across candidates and fantasies wherever
  dimensions permit; all seeds and numerical settings are recorded in the
  result trace.
- Policy-facing code sees only posterior state, source features, and legal
  revealed observations. Oracle target outcomes and final labels remain on the
  evaluator side.

## Architecture

1. `protocol_acquisition.py` receives a focused `protocol_hull_entropy`
   function and immutable result model. It constructs the candidate grid,
   samples current posterior worlds, evaluates the existing complete-pool hull
   backend, conditions the Gaussian posterior for each candidate, and computes
   expected post-query joint entropy by `slogdet` with a fixed relative ridge.
2. `protocol_policy_worker.py` registers `cal_style_hull_entropy` as an active
   policy and serializes its scores, selected action, entropy diagnostics, and
   numerical settings into the existing decision-round schema.
3. A dedicated CAL runner/summary path extends the existing unified MatPES
   campaign rather than creating a parallel experiment contract. It runs the
   policy on the frozen 230-system development roster and the authorized
   secondary 94-system rerun using the existing folds, B=6 prefixes, and D/F/T
   evaluator. Existing authoritative comparator traces are read-only inputs.

## Numerical behavior

- `slogdet` must return a positive finite sign; otherwise the implementation
  fails loudly rather than substituting a marginal or diagonal entropy.
- Covariance matrices are symmetrized before eigendecomposition and use the
  fixed relative ridge based on the current hull-vector covariance scale.
- Zero or near-zero observation variance yields zero expected information gain
  for that candidate and leaves the conditional state unchanged.
- Candidate IDs are the only tie-break key, sorted lexicographically through
  the existing stable action helper.
- The implementation records current entropy, conditional entropy means,
  information gains, grid count, ridge, and runtime per state for audit.

## Testing and experiment outputs

Unit tests cover grid construction, Gaussian conditioning, entropy finiteness,
ridge sensitivity, zero-variance behavior, permutation invariance, and
deterministic replay. Integration tests cover worker serialization, policy
registration, evaluator isolation, and a small unified campaign fixture.

The formal run emits CAL traces, per-budget T summaries, Delta-Hull-minus-CAL
paired contrasts with the existing system-level uncertainty procedure, curve
area, per-state cost, and grid/sample setting summaries. Raw outputs remain
outside Git. No manuscript patch is generated until task/vault/cross-fit
hashes, roster counts, policy roster, and summary consistency have passed.
