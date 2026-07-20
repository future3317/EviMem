# Decision-Sufficient Scientific State

**Status (2026-07-20): complete v4 implementation, authoritative method-level
NO-GO.** The all-outcome state, fail-closed protocol activation,
source-environment transport and robust hull-decision certificate are
implemented and pass replay, no-deletion, self-removal and interval-soundness
tests. The real JARVIS--MP v1, v3 and fresh v4 gates are all NO-GO. V4's
certificate is sound on simultaneous-interval inliers, but coverage is below
its frozen evaluation gate and the method is dominated by simpler source
baselines. P3C remains stopped and AKSC remains unauthorized for WBM.

The organizing principle is **Decision--Inference--Systems Alignment**. A
bounded scientific-state claim must simultaneously preserve the registered
decision, justify its inference under shift and adaptive use, and reduce a
measured end-to-end cost. Posterior similarity, predictive accuracy, and
component-level compression are not substitutes for these three gates.

## 1. Problem correction

The immutable archive already retains every legally revealed scientific
outcome. The research question is therefore not which paid observations should
be deleted, or even which residual cards should continue to count. It is:

> What is the least costly observable, protocol-valid state that preserves the
> scientific decisions available from the complete archive?

Let

- `A_t` be the complete immutable archive through round `t`;
- `U_t` be the currently visible candidate pool;
- `H_t` be the composition-dependent causal hull;
- `C_t` be protocol, provenance, fidelity and transport state; and
- `ACTIONS_t` be the legal action set.

An online scientific state is

\[
Z_t=\Psi(A_t,U_t,H_t,C_t).
\]

For a downstream loss `ell`, let `a_A^ell` be a Bayes-optimal legal action when
the decision rule can use the complete state and let `a_Z^ell` be optimal when
it can use only `Z_t`. The operational distortion for a registered loss class
`L` is

\[
\delta_{\mathcal L}(Z_t;A_t)=
\sup_{\ell\in\mathcal L}
\mathbb E\!\left[
  \ell(a_Z^\ell,Y)-\ell(a_A^\ell,Y)
  \mid A_t,U_t,H_t,C_t
\right].
\]

The scientific-state problem is

\[
\min_{\Psi}\;\operatorname{Cost}(Z_t)
\quad\text{subject to}\quad
\delta_{\mathcal L}(Z_t;A_t)\le\epsilon,
\]

plus observability, causal ordering, protocol compatibility, provenance
certification and update-budget constraints. `Cost` must be an end-to-end
quantity (latency, memory, access, certification, communication or attention),
not the runtime of a component chosen after seeing results.

This definition preserves decisions rather than an arbitrary posterior. A
posterior approximation may be one construction of `Z_t`, but posterior
fidelity is neither the definition nor sufficient evidence of decision
fidelity under model mismatch.

## 2. Three conditional results

### 2.1 No-forgetting principle

Suppose `Z=T(A)` is a measurable function of the full information state and
there is no cost for observing or processing `A`. If complete-state decision
rules are allowed to ignore information, then for every loss,

\[
\inf_{d(A)}\mathbb E[\ell(d(A),Y)]
\le
\inf_{g(Z)}\mathbb E[\ell(g(Z),Y)].
\]

**Reason.** Every compressed-state rule `g(T(A))` is also an admissible
complete-state rule. The complete-state rule class is therefore a superset.

This information-ordering statement does not require a correct parametric
model. Correct inference matters when replacing the Bayes-optimal rules above
with a particular learned algorithm: a misspecified algorithm can use more
data badly. Consequently, deleting outcomes to repair a misspecified GP is a
robust-inference intervention, not evidence that forgetting is intrinsically
valuable.

### 2.2 Decision-sufficiency bound

Let `Q_A(a)` and `Q_Z(a)` be complete-state and compressed-state action values
for the same legal action set, with larger values preferred. If

\[
\sup_{a\in\mathcal A_t}|Q_A(a)-Q_Z(a)|\le\epsilon,
\]

and `a_A` and `a_Z` maximize their respective values, then

\[
Q_A(a_A)-Q_A(a_Z)\le 2\epsilon.
\]

**Proof.** Insert and subtract `Q_Z(a_A)` and `Q_Z(a_Z)`. The first and third
terms are at most `epsilon`, and optimality of `a_Z` makes the middle term
non-positive. This is a one-step result. A multistep claim requires separately
registered transition-consistency assumptions and is not currently made.

### 2.3 Research-relevance condition

If the component targeted by a compression method is fraction `f` of measured
end-to-end time and all other costs remain unchanged, eliminating that
component entirely yields at most

\[
S_{\max}=\frac{1}{1-f}.
\]

This is an Amdahl upper bound, not a quality theorem. A compute-compression
paper needs a preregistered material value of `f` before implementing the
compressor. In the WBM B40 replay, the largest measured GP share was `0.6888%`,
so the ideal bound was `1.00694x`; the current workload fails this gate.

## 3. When active scientific state is nontrivial

At least one of the following must be present and measured:

1. inference, retrieval or access is an end-to-end bottleneck;
2. archived evidence is not directly compatible with the target protocol;
3. the data distribution, scientific target or action set shifts;
4. inference is materially misspecified and requires explicit robust or
   calibration treatment;
5. privacy, certification, communication or human-attention constraints limit
   legal use.

The current single-protocol WBM setup is a null regime for items 1 and 2. It
does exhibit item 4: the frozen GP is strongly underdispersed. That defect must
be addressed as calibration or robust inference, not hidden inside an
outcome-dependent retention policy.

## 4. Material instantiation: all-outcome state and future hull certificates

The implemented first layer is not another residual posterior coreset. For a
frozen outcome-independent feature map, every direct or certified transported
outcome updates a fixed-dimensional linear--Gaussian state

\[
\Lambda_t=\Lambda_{t-1}+\phi_t\phi_t^\top/\sigma_t^2,
\qquad
\eta_t=\eta_{t-1}+\phi_t r_t/\sigma_t^2.
\]

No capacity, similarity or one-swap interface is allowed to remove an accepted
outcome contribution. Incompatible or over-radius source observations remain
in the immutable archive but abstain from influencing the target-protocol
state. Same-order streaming and archive replay are exactly equal in tests.

This establishes representation compression for the registered linear model,
not decision sufficiency under model misspecification. A conformal transport
radius is an error certificate, not automatically a Gaussian standard
deviation; any working likelihood that uses it as such remains a modeling
assumption and cannot inherit distribution-free coverage.

The implemented second layer is **Certified Hull-Decision State**. `active`
means that archived evidence has a certified, protocol-valid influence for the
current target, possibly through a directed transport map with explicit
uncertainty. The archive remains complete.

The state should preserve only what can change a registered hull decision:

- possible hull-support phases and unresolved facets;
- stable, unstable and abstain decisions for legal candidates;
- an `epsilon`-optimal next-action set;
- protocol-transport uncertainty capable of flipping either decision; and
- certificates tying every influence to observable provenance.

Two archive states are decision-equivalent at tolerance `epsilon` only if they
induce the same registered stable/unstable/abstain decisions and compatible
`epsilon`-optimal action sets. The robust LP now supplies a computable
selective certificate for these decisions. The v4 evaluation does not
establish that this state is superior: its point policy is worse than naive
source reuse and its 90% simultaneous-system gate misses by one of 56
supported systems.

The required null behavior is exact:

> If all evidence uses the target protocol, transport error is zero and there
> is no binding state cost, Certified Hull-Decision State must reduce to full
> history.

This makes the negative WBM result expected rather than something to tune away.

## 5. What the stopped methods established

| Iteration | Surrogate that was tested | Failure boundary |
|---|---|---|
| DACC | Singleton facility gain represents a joint active GP | Singleton utility is not joint posterior decision utility. |
| P3C | A proper projection of a reference GP preserves causal decisions | Posterior fidelity is not truth fidelity or decision sufficiency; outcome-dependent contribution deletion adds selection mismatch. |
| AKSC proposal | The full GP operator is worth approximating and is the bottleneck | WBM's GP share fails the end-to-end relevance gate; the reference is also severely underdispersed. |

These are not three incomplete attempts at the same selector. They eliminate
three increasingly strong posterior-surrogate hypotheses and motivate direct
decision sufficiency.

## 6. Real multi-protocol gate and current stopping result

The JARVIS--MP task now supplies real protocol heterogeneity: 1,658
same-material, structure-matched OptB88vdW--MP/GGA pairs, split by exact
chemical system with target outcomes behind an oracle vault. The v1 global
affine certificate passed calibration but violated its radius on 45.3202% of
evaluation pairs. Although paired transport was strongly predictive, the
rank-16 all-outcome state did not improve hull cost or regret. A v3
composition-aware transport fit on 23 systems produced a clustered radius of
0.177264 eV/atom on ten disjoint radius systems, exceeding the frozen 0.15
ceiling. Its 12 fresh evaluation systems were never opened.

Accordingly, no new selector and no threshold relaxation is authorized. Before
another positive method evaluation:

1. obtain richer public protocol metadata/calibration or predefine a
   structure/environment-conditional transport on new systems;
2. freeze protocol identity, directed transport, transport uncertainty and
   fail-closed compatibility rules without evaluation outcomes;
3. define hull decisions, abstention, action values and `epsilon` before
   evaluation;
4. verify the homogeneous zero-transport null pair is exactly full-history
   equivalent round by round;
5. demonstrate that protocol-aware activation improves registered decision
   loss over both naive pooling and target-protocol-only evidence, without
   using outcome deletion as calibration; and
6. measure the access/certification/compute constraint that makes the state
   problem nontrivial.

The task exists; the currently tested certificates do not support safe reuse.
The correct state is therefore target-only abstention for unsupported transport
and full history in homogeneous WBM. A richer transport must be calibrated on
new systems and pass its frozen gate before any fresh evaluation is opened.

That authorized continuation has now been completed once. V4 excluded all 45
v1/v3 systems, calibrated on 210 new exact systems and evaluated 72 further
systems only after its decision-level gate passed. The robust certificate made
zero errors on simultaneous-interval inliers, but the interval event held on
only 50/56 supported evaluation systems versus the frozen 90% requirement.
More importantly, environment transport plus all-outcome correction was
significantly worse than naive source-as-target for both hull error and action
regret. Those opened systems are development-closed. Another representation or
threshold adjustment on the same task is not authorized.

## 7. Scope and literature boundary

Decision-sufficient representations motivate preserving downstream decisions
rather than all predictive information. Information sufficiency can be more
conservative than operational sufficiency, and global decision sufficiency can
be computationally hard; neither fact proves that the proposed material state
is small. Cross-database DFT comparisons and multi-fidelity materials learning
support the existence of protocol-dependent labels, but do not validate a
particular transport map. CAL supports treating convex-hull uncertainty as a
global functional, but is not evidence for this state-compression hypothesis.

The paper must cite these sources for motivation only and keep the current
empirical claim at NO-GO until the gate above is passed.
