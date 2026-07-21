# Decision-Sufficient Scientific State

**Status (2026-07-21): v4, CHIC and myopic Delta-Hull are negative; a
source-relative full-budget MatPES rollout is under development with no
paper-level positive result yet.** The all-outcome state, fail-closed protocol activation,
source-environment transport and robust hull-decision certificate are
implemented and pass replay, no-deletion, self-removal and interval-soundness
tests. The real JARVIS--MP v1, v3 and fresh v4 gates are all NO-GO. V4's
certificate is sound on simultaneous-interval inliers, but coverage is below
its frozen evaluation gate and the method is dominated by simpler source
baselines. P3C remains stopped and AKSC remains unauthorized for WBM. CHIC did
not delete archive contributions, but its JARVIS development task did not show
a training-state or acquisition advantage. The active MatPES continuation
changes the posterior and protocol task, not archive retention.

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

## 7. Active continuation: CHIC

CHIC changes two assumptions that invalidated the previous development loop.
First, the selected action is persisted and becomes the only legal oracle
reveal; no fixed hash trajectory substitutes for deployment. Second, the
capacity applies to an expensive gradient update, not to whether an observed
outcome remains scientific evidence.

For model parameters `theta`, candidate margin `m_x` and the legal competing
hull LP weights `lambda*`, CHIC uses

\[
\nabla_\theta m_x=
\nabla_\theta f_\theta(x)-
\sum_j\lambda_j^*\nabla_\theta f_\theta(x_j).
\]

Joint non-negative gradient matching approximates the full-history update in a
metric oriented by the pool hull-decision gradient. A smooth one-step bound
then controls the downstream loss deviation by the update-direction error. The
bound is conditional; it does not prove that the current data have informative
hull gradients or that the selected update beats a strong source policy.

The first fixed-trace diagnostic confirms non-degenerate selections but finds
no advantage over diversity or hard-example selection. The first true
eight-system closed loop finds that pure influence is nearly the same as ridge
uncertainty. A causal two-step lookahead improves action regret from `0.182061`
to `0.145319`, but the simple source-margin policy remains better at `0.107554`
and wins six of eight systems. The current JARVIS--MP task therefore lacks a
positive CHIC signal. The next valid change is a dataset/task change to a real
paired PBE--r2SCAN workload with enough updates to make gradient selection
nontrivial, not another weight on the same eight systems.

## 8. Active continuation: hierarchical protocol discrepancy

The MatPES PBE--r2SCAN task changes the failed JARVIS assumption in two ways.
It supplies exact same-configuration pairs at much larger scale, and it uses a
true action-driven reveal loop. A data audit identifies 385,890 exact pairs;
84,532 have formation energies for both protocols. Upstream row splits are not
independent, so exact chemical systems and original Materials Project parents
remain the development units.

Let (E_S(x)) and (E_T(x)) be source- and target-protocol formation energies
for one shared configuration. The current working posterior is

\[
E_T(x)=E_S(x)+\phi(x)^\top\beta+b_s+g_s(\tilde\phi(x))+\epsilon_x,
\]

where (s) is the exact chemical system, (b_s\sim\mathcal N(0,\tau^2)),
(g_s\sim\mathcal{GP}(0,\sigma_g^2 k_{5/2})), and
(epsilon_x\sim\mathcal N(0,\sigma_n^2)). The global delta mean is fitted
with equal total weight per fit system. Matérn length, signal and nugget scales
maximize system-macro marginal likelihood on disjoint fit systems. The
observable representation contains PBE quantities and normalized element
fractions; it contains no r2SCAN outcome. Every revealed target outcome in the
current system conditions the joint posterior.

The live method is **Delta-Hull Active Search**. For one remaining equal-cost
query and utility equal to oracle-final phase confirmation, its Bayes action is

\[
a_t\in\arg\max_{x\in U_t}
\Pr\!\left[x\in H_T^\star(U_t\cup H_t)\mid D_t\right].
\]

This follows immediately by linearity of expectation: the conditional expected
one-query reward for action `x` is its final-hull membership probability. It is
not a weighted blend of uncertainty and margin. Unequal query costs require a
separate knapsack or Lagrangian objective; dividing by cost is not silently
claimed to solve the finite-budget problem. The implementation therefore fails
closed on unequal costs. CAL instead values global hull-uncertainty reduction,
and nonmyopic multifidelity active search values future budget allocation. A
two-step diagnostic is retained, but it has not improved the earlier panel and
is not the live contribution.

The observation and label must remain distinct. Querying `x` reveals the
continuous target energy `E_T(x)`, not its final-stability label. The latter is
a delayed structured event determined by all target energies in the visible
pool. Pointwise phase-field active learning, including BALPI's CALPHAD
classification and level-set formulations, therefore does not implement this
same estimand. For a query set `S`, the registered terminal utility is

\[
R(S,E_T)=\sum_{x\in S}\mathbf 1\{x\in H_T^\star(E_T)\}.
\]

The exact finite-horizon Bellman recursion is well-defined but not claimed
tractable. The implemented acquisition is its `b=1` specialization; repeated
greedy use at budget six is an empirical policy, not a horizon-optimality
theorem.

The expanded 24-system development panel at budget six shows a clearer but
still non-confirmatory signal. With 1024 nested scrambled-Sobol draws,
Delta-Hull obtains `3.7083` oracle-final confirmations per system versus
`3.4583` for source margin. The paired exact-system difference is `+0.2500`
with deterministic bootstrap 95% interval `[+0.0417,+0.5000]`: six systems
win, seventeen tie and one loses. Source margin leaves 19 confirmations below
the finite-pool oracle ceiling, of which Delta-Hull recovers six. Causal
discoveries are tied at `4.3333`; invalidation against the oracle pool falls
from `0.8750` to `0.6250`, while wall time rises from `1.99` to `22.13` seconds
per system. Thus the mechanism is more precise final-hull targeting, not more
transient causal discoveries or cheaper inference.

The added cost is not a large-GP effect at the present pool sizes. The current
reference implementation propagates each joint target-energy sample through a
fresh composition-dependent `PhaseDiagram`. At budget six and MC1024 that is
6,144 phase-diagram constructions per system. A one-system diagnostic profile
at commit `090b4cb` assigns essentially all worker cumulative time to this
final-hull propagation, while posterior conditioning and Gaussian sampling are
negligible at that scale. The profiler inflates Python-heavy absolute time, so
only the bottleneck attribution is retained; scientific timing remains the
unprofiled closed-loop measurement above.

The higher integration level resolves the effect-level concern without proving
exact trace convergence. MC512 and MC1024 have the same `+0.2500` discovery
difference, interval and win/tie/loss counts. They agree on the first action in
23/24 systems, the complete six-step trace in 21/24 and 134/144 individual
rounds. The remaining three trace changes do not alter any system's discovery
difference. MC1024 is therefore frozen for a fresh-split replication; no more
posterior or acquisition tuning is authorized on these 24 systems. Current
evidence remains development-only until that replication succeeds.

The subsequent outcome-independent repartition reserves 48 exact systems and
refits transport on the other 276. Delta-Hull obtains `3.6250` oracle-final
confirmations/system versus `3.5625` for source margin: paired `+0.0625`, 95%
CI `[-0.1042,+0.2292]`, exact two-sided `p=0.6291`. It differs from source on
213/288 round actions, so the failure is not a degenerate selector. Instead,
source margin already reaches the finite-pool budget ceiling in 24/48 systems;
only 35 confirmations of total headroom remain and Delta-Hull recovers a net
three. This closes Delta-Hull as a superiority method on the opened MatPES
task. Further score, posterior, support or seed changes on these systems are
not authorized as independent evidence.

The next method changes the measured horizon assumption rather than the score.
For remaining budget `b`, **Source-Rollout Delta-Hull** evaluates every legal
first action under a complete target-energy posterior sample and then executes
the deployed source-margin policy for the other `b-1` simulated queries. Every
sampled reveal is inserted into its composition-dependent simulated causal
hull. Terminal reward is the number of simulated queries that belong to that
sample's complete final target hull. Because the source action is itself in
the first-action set, exact posterior expectation gives the standard
model-relative rollout inequality

\[
\max_x Q_b^{\pi_{\rm src}}(x\mid O_t)
\ge Q_b^{\pi_{\rm src}}(x_{\rm src}\mid O_t)
=V_b^{\pi_{\rm src}}(O_t).
\]

This is not a guarantee under posterior misspecification. The implementation
uses sixteen common-random-number scrambled-Sobol blocks and falls back to the
source action unless the Bonferroni-simultaneous one-sided
numerical-integration lower bound is positive for a candidate. The correction
controls the family of non-source comparisons, while the bound itself controls
only integration noise.
The 276 former fit systems are partitioned outcome-independently into six
cross-fit development folds; all 48 opened systems remain excluded from method
development. On the first 46-system fold at budget six, the effect changes
from `+0.1522` at MC512 to `+0.1739` confirmations/system at MC1024. The latter
has 11 wins, 30 ties and 5 losses, bootstrap 95% interval
`[-0.0217,+0.3696]`, and exact two-sided sign-flip `p=0.1351`. On the 22
systems where source leaves budget-feasible headroom, the effect is `+0.5455`
with interval `[+0.2273,+0.8636]`. Binary, ternary and higher-order stratum
means are respectively `+0.0833`, `+0.2105` and `+0.2000`. MC512 and MC1024
agree on 45/46 system-level effects and 41/46 first actions, but only 31/46
complete traces and 220/276 individual actions. The outcome signal is stable
enough to reject simple effect collapse, but action-level numerical convergence
is not established. Folds 1--5 are therefore paused; no positive claim or
cross-fold expansion is authorized before resolving integration stability
without changing the posterior or acquisition objective.

The report-motivated continuation is implemented separately as
`conformal_source_rollout_delta_hull`. It calibrates an exact-system
nonconformity score

\[
S_s=\max_{t,x}[\widehat A_{s,t}(x)-A_{s,t}(x)]_+
\]

and permits at most one source-relative deviation when the RQMC-adjusted
advantage exceeds the frozen finite-sample radius. The runner explicitly
records whether that deviation has been used and executes source margin
thereafter. This is development infrastructure, not a new result: no
calibration artifact or evaluation run exists, and it cannot be used to
change the pending SARR fold-0 gate or support a positive claim.

## 9. Scope and literature boundary

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
