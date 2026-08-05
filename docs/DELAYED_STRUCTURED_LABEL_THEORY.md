# Delayed Structured Labels: current theory and manuscript position

Status: current manuscript theory note, 2026-08-04. This document records the
paper-facing formulation and claim boundary. It does not authorize a new
experiment, a new data split, or any change to the frozen MatPES/MAD policy,
posterior, gate, hull backend, seeds, or estimands.

The current manuscript is **Globally Adjudicated Active Search for Convex-Hull
Discovery**. The paper's central
contribution is the problem and its mechanism theory; convex-hull acquisition
is the materials instance. The primary solver is ungated source-anchored
rollout (SARR), Delta-Hull is the materials greedy structured-label baseline,
and IC-SARR is an optional numerical screen documented in the appendix.

## Manuscript integration checkpoint

The evidence-integrated manuscript is committed in the paper repository as
`9f65ac5`. The manuscript keeps the delayed-label motivation, the exact Bellman
objective, the greedy failure/sufficiency theory, the MatPES budget curve and
D/F/T waterfall, and integrates the E32-A objective/lookahead follow-up and
MAD direct-mechanism curve. The rollout schematic,
controlled stress heatmaps, policy ablations, posterior/hull implementation
specification, and numerical-gate audit are in the appendix; references precede
the appendix and the main text occupies nine pages before references.

This packaging pass adds one evaluator-only sensitivity audit and does not
change any policy, posterior, gate, manifest, budget, hull backend, or frozen
result. The audit re-evaluates the 1,380 frozen E32-A selected sets over four
hull tolerances and protected nested/competitor-removed pools. It finds zero
selected-label flips across the tolerance scan and small, bounded changes under
pool perturbation, but it does not recompute actions. A formation-energy holdout,
an instrumented signed-information replay, and unrestricted continuation remain
unperformed limitations rather than implied evidence.

## General problem

For a finite candidate pool (C), latent complete world (Z), query
observation (O_x(Z)), and structured label (Y_x(Z)=g_x(Z_C)), the terminal
utility is

\[
R(S,Z)=\sum_{x\in S}Y_x(Z).
\]

The label is delayed when (O_x) does not determine (Y_x) and observations
of one candidate can change the posterior label of another. In the materials
instance, (O_x=E_x) is a target-protocol energy and (Y_x) is complete-pool
convex-hull membership. The online policy never sees unrevealed energies or
labels; complete-pool adjudication is an evaluation utility, not online
feedback.

## Theory results in the manuscript

1. **Repeated greedy has no uniform approximation ratio.** For any
   (eta>0), use a hidden state (Z\sim\mathrm{Unif}\{1,\ldots,K\}),
   budget two, an information-only candidate (s) with (O_s=Z,Y_s=0),
   two constant-observation decoys with label probability (2/K), and
   specialists (g_k) with (Y_{g_k}=1\{Z=k\}). Repeated greedy selects the
   decoys and obtains (4/K), while querying (s) and then (g_Z) obtains one.
   Taking (K>4/\eta) makes the ratio smaller than (eta).

2. **Adaptive submodularity fails in general.** With (Z\sim
   \mathrm{Bernoulli}(1/2)), let (O_a=Z,Y_a=0), and (O_b) be constant
   with (Y_b=Z). The marginal of selecting (b) is (1/2) before querying
   (a), but one after observing (O_a=1). The conditional marginal
   increases, violating adaptive diminishing returns even though terminal
   utility is additive.

3. **Greedy is optimal in stable regimes and near-optimal under weak
   coupling.** If every legal observation continuation preserves the
   posterior ranking of remaining candidates, repeated greedy is Bayes-optimal
   by backward induction. If every continuation changes each remaining label
   probability by at most (epsilon), the manuscript proves the bound
   (V_n^\star-V_n^{\mathrm{greedy}}\le 2n\epsilon). Thus exact
   pointwise labels are the (epsilon=0) null, while small cross-candidate
   posterior movement gives a quantitative greedy-sufficiency condition.

4. **Delayed information has a strict value criterion.** With two queries
   remaining and greedy final-label continuation,

   \[
   \mathcal I_h(x)=
   \mathbb E_{O_x|h}\left[\max_{y\ne x}p_{h,x,O_x}(y)\right]
   -\max_{y\ne x}p_h(y)\ge 0.
   \]

   It is strictly positive when positive-probability observation events place
   the conditional posterior vectors in different, non-tied maximizer regions
   with a positive maximizing gap.

5. **Mechanism values can be separated.** With (J/D) denoting joint versus
   diagonal covariance, (A/N) adaptive versus open-loop planning, and
   \(* / \mathrm{src}) unrestricted versus source continuation, the paper
   uses the exact order-specific identity

   \[
   V^{J,A,*}-V^{D,N,\mathrm{src}}
   =\underbrace{(V^{J,A,*}-V^{D,A,*})}_{\text{joint covariance}}
   +\underbrace{(V^{D,A,*}-V^{D,N,*})}_{\text{adaptivity}}
   +\underbrace{(V^{D,N,*}-V^{D,N,\mathrm{src}})}_{\text{continuation}}.
   \]

   This is a bookkeeping decomposition, not a license to attribute a
   source-relative bar to one component. Interactions depend on the chosen
   order; the existing direct paired ablations measure only selected slices.

6. **Top-two exchange is value-degenerate.** Let `g_1` and `g_2` be the two
   largest current final-label probabilities and let `I_h(x)` be the delayed
   information value in Result 4. For the two-step objective,

   \[
   Q_2(g_2\mid h)-Q_2(g_1\mid h)=I_h(g_2)-I_h(g_1).
   \]

   The immediate terms are the same pair `p_h(g_1)+p_h(g_2)` under either
   order. Therefore an action switch between the top two candidates is not
   evidence of positive planning value: a small information-estimation error
   can change the action while leaving terminal utility unchanged. For a
   lower-ranked candidate `x`, the direct probability deficit must be offset by
   its information advantage, making the rank gap an explicit hurdle rather
   than a generic disagreement statistic.

7. **Observed planning gain has three distinct sources.** Let `J_P` denote
   evaluator value, `J_q` working-posterior value, `pi_g` Delta-Hull,
   `pi_q^star` the posterior-optimal planning policy, and `hat pi` the
   implemented Monte-Carlo policy. Then

   \[
   \begin{aligned}
   J_P(\hat\pi)-J_P(\pi_g)
   ={}&[J_q(\pi_q^\star)-J_q(\pi_g)]
   -[J_q(\pi_q^\star)-J_q(\hat\pi)]\\
   &+[(J_P-J_q)(\hat\pi)-(J_P-J_q)(\pi_g)].
   \end{aligned}
   \]

   The terms are, respectively, structural planning headroom, solver/
   Monte-Carlo regret, and differential posterior model error. An incremental
   compute price `lambda(C(hat pi)-C(pi_g))` is a separate declared penalty;
   it is not a target-query cost and does not create a cost-aware claim. This
   identity explains why many action changes can coexist with little realized
   gain, and motivates a selective gate rather than an always-on rollout.

## Connection to frozen evidence

- The exact 1,000-instance suite is consistent with the theory: source rollout
  equals exact DP on 84.8% of instances and beats source margin on 75.8%, but
  does not exceed greedy final selection under its registered generator.
- The same-posterior objective-efficiency evaluator shows Delta-Hull minus
  posterior-mean target-margin differences of `+0.0826`, `+0.0870`, `+0.0826`,
  `+0.0696`, `+0.0217`, and `-0.0043` at `B=1..6`.  The exact-system AUC
  contrast is `+0.3000`, 95% CI `[+0.0522,+0.5587]`, sign-flip `p=0.0247`,
  with 60/114/56 wins/ties/losses.  This is early-budget efficiency evidence,
  not a final-budget superiority result.
- On 230 MatPES development systems at (B=6), IC-SARR has a source-relative
  (+0.1696\) complete-pool-(T) signal, but direct paired contrasts are
  (+0.0043) versus Delta-Hull and (-0.0130) versus ungated SARR, both with
  intervals crossing zero. IC-SARR and Delta-Hull tie on terminal (T) for
  187/230 systems.
- MAD-1.5 has a complete-pool proxy AUC difference of (+0.2500), but only
  (+0.0260) selected-history (F) AUC, (+8.5739) wall-time AUC, and
  (-0.6074) fixed cost utility. It is a protocol-shift mechanism panel
  using an atomization-energy hull proxy, not formation-energy confirmation.

The empirical interpretation is therefore: delayed full-pool adjudication is a
distinct active-search objective; many observed systems are greedy-sufficient
at the measured terminal utility; and lookahead value is a coupling-regime
question. The new evaluator-only E32 trace audit shows that the model-relative
rollout Q-gap predicts action changes (Spearman rho 0.816), but realized
terminal-T gain is only weakly related to that gap (rho 0.108), while selected-
action rank-switch is not predictive. The frozen artifact does not identify the
exact I_h(x) term because it stores absolute rather than signed conditional
hull probabilities; a new instrumented posterior replay would be required.
The MatPES ties do not prove posterior rank stability, and no solver or gate
superiority claim is supported.

The next registered method-development line, `E51`, treats the two results
above as an audit target. It estimates top-two information/headroom with an
independent inner stream and invokes Delta-Hull-anchored lookahead only when a
predeclared lower bound exceeds model and compute penalties. Its first stage is
finite-world only; no selective material curve is implied by this theory note.

The completed E51 finite-world audit is consistent with this separation:
exact Hull-ENS reaches exact finite-world DP in all four small headroom strata,
while sampled/double-sampled Hull-ENS shows small regret in the low-headroom
strata. In a separate 40-system outer synthetic replay, nested selective
gating invoked exact-HENS lookahead on 2.083% of states and retained all
observed full-planner gain. These are implementation/mechanism checks with
small synthetic panels; the unregistered maximum exact-versus-sampled
discrepancy prevents promoting them to a material selective result.

## Forbidden upgrades

- Do not call the 230/324 MatPES corpus an untouched holdout.
- Do not call MAD-1.5 a formation-energy thermodynamic hull or a pristine
  external holdout.
- Do not call the source-relative (T) signal final-causal, deployment, cost-
  aware, runtime, universal-generalization, or IC-SARR superiority.
- Do not treat the theoretical counterexamples as material experiments.
- Do not retune opened MatPES/MAD systems to estimate the theoretical terms.

The manuscript implementation is recorded at paper commit `9f65ac5`; raw and
derived experiment artifacts remain outside Git under `E:\DATA` and the
registered remote roots. The code repository's result manifest points to this
paper commit.
