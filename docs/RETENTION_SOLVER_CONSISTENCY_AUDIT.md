# Retention solver consistency audit

Status: post-freeze audit. The immutable freeze is tag
`caw-method-no-go-2026-07-15`.

## Finding

Equation 13 optimizes over every legal set:

\[
\arg\min_{M'\subseteq M\cup\{m_t\},\ |M'|\le K}\Phi(M').
\]

The frozen `BruteForceRetentionSolver` did **not** implement this objective. It
enumerated all subsets of cardinality exactly `K`. Because streaming admission
contains at most `K+1` candidates, this reduced to `K+1` one-eviction choices.
It was exhaustive only within the exact-size-`K` neighborhood.

No equivalence proof is available. Interval intersection makes the potential
non-monotone in set inclusion: adding a witness can create an empty intersection
and a conflict penalty. Consequently the optimum may contain fewer than `K`
witnesses and may require simultaneous removal of multiple old witnesses.

## Constructed counterexample

The regression test uses `K=2`, two old ambiguous witnesses, and one new
certifying witness. The new interval conflicts separately with both old
intervals. Every size-two choice has positive risk or conflict, while the
singleton containing only the new witness has zero potential.

Observed behavior:

- frozen exact-size-`K` solver: test fails and retains both old witnesses;
- corrected all-legal-subsets solver: retains only the new witness and evicts
  both old witnesses;
- full repository verification after correction: `165 passed`; Ruff passes.

The test is
`test_exact_retention_can_remove_two_old_witnesses_after_conflicting_admission`
in `tests/test_matmem_boundary.py`.

## Corrected solver and complexity

The post-freeze reference solver enumerates sizes `0,1,...,K` and all
combinations at each size. Equal-potential ties prefer the larger set and then
lexicographic witness IDs; this tie rule does not change the primary objective.

For streaming candidate count `K+1`, the number of feasible subsets is

\[
\sum_{j=0}^{K}\binom{K+1}{j}=2^{K+1}-1.
\]

One potential evaluation scans `O(nK)` query--witness pairs. With `n`
candidates and two hypothetical scenarios, direct retention-aware acquisition
is therefore `O(n^2 K 2^K)` per round. The exhaustive solver is an auditable
small-`K` reference, not a scalable WBM implementation. Any future one-swap or
cached solver must either change the mathematical neighborhood explicitly or
prove exact equivalence and pass randomized plus constructed equivalence tests.
