# Delayed Structured Labels: current theory and manuscript position

Status: current manuscript theory note, 2026-08-03. This document records the
paper-facing formulation and claim boundary. It does not authorize a new
experiment, a new data split, or any change to the frozen MatPES/MAD policy,
posterior, gate, hull backend, seeds, or estimands.

The current manuscript is **Active Search with Delayed Structured Labels:
Theory and Mechanisms for Durable Convex-Hull Discovery**. The paper's central
contribution is the problem and its mechanism theory; convex-hull acquisition
is the materials instance. The primary solver is ungated source-anchored
rollout (SARR), Delta-Hull is the materials greedy structured-label baseline,
and IC-SARR is an optional numerical screen documented in the appendix.

## Manuscript integration checkpoint

The evidence-integrated manuscript is committed in the paper repository as
`5c2b8ca`. Its main text keeps the delayed-label motivation, the exact Bellman
objective, the greedy failure/sufficiency theory, the MatPES budget curve and
D/F/T waterfall, and the MAD protocol-shift curve. The rollout schematic,
controlled stress heatmaps, policy ablations, posterior/hull implementation
specification, and numerical-gate audit are in the appendix; references precede
the appendix and the main text occupies nine pages before references.

This packaging pass changed exposition and provenance only. It ran no new
scientific experiment and did not change any policy, posterior, gate, manifest,
budget, hull backend, or frozen result. The paper now explicitly records the
unperformed same-posterior objective sensitivity suite, hull tolerance and
duplicate-rule sensitivity, candidate-pool perturbation, formation-energy
holdout, and unrestricted-continuation comparison as limitations rather than
implied evidence.

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
   probability by at most (epsilon), the manuscript proves the conservative
   bound (V_n^\star-V_n^{\mathrm{greedy}}\le 4n\epsilon). Thus exact
   pointwise labels are the (epsilon=0) null, while small cross-candidate
   posterior movement gives a quantitative greedy-sufficiency condition.

4. **Delayed information has a strict value criterion.** With two queries
   remaining and greedy final-label continuation,

   \[
   \mathcal I_h(x)=
   \mathbb E_{O_x|h}\left[\max_{y\ne x}p_{h,x,O_x}(y)\right]
   -\max_{y\ne x}p_h(y)\ge 0.
   \]

   It is strictly positive when the observation of (x) changes the future
   posterior maximizer with nonzero probability and the resulting posterior
   means are not tied.

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

## Connection to frozen evidence

- The exact 1,000-instance suite is consistent with the theory: source rollout
  equals exact DP on 84.8% of instances and beats source margin on 75.8%, but
  does not exceed greedy final selection under its registered generator.
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
question. The MatPES ties do not prove posterior rank stability, and no solver
or gate superiority claim is supported.

## Forbidden upgrades

- Do not call the 230/324 MatPES corpus an untouched holdout.
- Do not call MAD-1.5 a formation-energy thermodynamic hull or a pristine
  external holdout.
- Do not call the source-relative (T) signal final-causal, deployment, cost-
  aware, runtime, universal-generalization, or IC-SARR superiority.
- Do not treat the theoretical counterexamples as material experiments.
- Do not retune opened MatPES/MAD systems to estimate the theoretical terms.

The manuscript implementation is recorded at paper commit `5c2b8ca`; raw and
derived experiment artifacts remain outside Git under `E:\DATA` and the
registered remote roots. The code repository's result manifest points to this
paper commit.
