# Campaign-level constrained rollout (proposal)

The current Dual-Horizon policy is stopped as a development diagnostic. Its
constraint is enforced at every observed belief state, which is a conservative
sufficient condition for a campaign-level objective but is not equivalent to
one. A future method must therefore optimize complete campaign policies rather
than keep tuning the local `LB_F` gate.

Let \(\pi_0\) be source-margin and let \(\Pi\) be a finite, frozen family of
complete rollout policies. For a posterior draw \(\omega\), simulate the
entire remaining campaign and compute:

* \(F^\pi(\omega)\): selected-history-hull confirmations at campaign end;
* \(T^\pi(\omega)\): complete-pool final-hull confirmations at campaign end.

The decision rule is

\[
\hat\pi = \arg\max_{\pi\in\Pi}\;\mathbb E[T^\pi]
\quad\text{s.t.}\quad
\mathbb E[F^\pi] \geq \mathbb E[F^{\pi_0}].
\]

The source policy is always in \(\Pi\), so the rule is fail-closed. Candidate
policies must be frozen before opening a new holdout. A minimal first family is
source plus one-shot first-action interventions followed by source
continuation; a richer family may contain predeclared multi-action rollout
policies, but it must not be generated adaptively from holdout outcomes.

For each policy, use common random numbers and independent block means for both
endpoints. Select policies only when a simultaneous lower bound for
\(F^\pi-F^{\pi_0}\) is non-negative, with multiplicity over all candidate
policies and both endpoints. This is an integration-error statement only. It
does not imply posterior calibration, oracle-truth superiority or a causal-time
guarantee. Actual execution still reveals exactly the selected action and
appends every outcome to the archive.

This proposal requires fresh development systems. It must not be tuned on the
opened 230-system IC-SARR history, and it must not be merged into the current
Dual-Horizon policy or used for a holdout claim before a new implementation and
pilot pass their own parity tests.
