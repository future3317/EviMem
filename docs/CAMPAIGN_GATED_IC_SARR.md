# Campaign-Gated IC-SARR

Status: implemented as an independent development API; no real-system result
has been opened and no holdout run is authorized by this implementation alone.

## Motivation

The stopped Dual-Horizon policy imposed two lower bounds on every deviation:

\[
LB_T(s_t,a_t)>0,\qquad LB_F(s_t,a_t)\ge 0.
\]

The attribution experiment showed that this local condition is not equivalent
to the desired campaign constraint. A temporary selected-history loss can be
recovered by later actions, but the local gate rejects that trajectory before
the compensation can occur.

The campaign-level estimand is instead

\[
\max_{\pi\in\Pi} \mathbb E[T^\pi\mid S_0]
\quad\text{s.t.}\quad
\mathbb E[F^\pi-F^{\pi_{\rm src}}\mid S_0]\ge0,
\]

where `T` is full-pool terminal confirmation reward and `F` is the final
selected-history confirmation reward. This is posterior-relative and controls
numerical integration error only; it is not an oracle-truth safety guarantee.

## First restricted implementation

The initial policy class is fixed before any experiment:

\[
\Pi=\{\pi_{\rm src},\pi_{\rm IC}\},
\]

where `pi_src` is source-margin continuation and `pi_IC` is the already frozen
IC-SARR policy (including its stage-one/stage-two seeds and continuation).
At the initial system state, the implementation:

1. samples complete target-energy vectors from the current posterior;
2. simulates both complete six-round adaptive campaigns under the same sampled
   energy world;
3. updates each policy's posterior after each simulated reveal, exactly as in
   the live protocol state;
4. computes paired campaign differences `d_T` and `d_F` from the same world;
5. forms simultaneous outer scrambled-Sobol lower bounds for the two paired
   means (Bonferroni comparison count two);
6. selects IC-SARR once iff `L_T > 0` and `L_F >= 0`, otherwise locks the whole
   system to source-margin.

The IC-SARR inner RQMC stream is independent of the outer energy-world stream.
No outcome is deleted: simulated reveals only condition the posterior for the
next simulated state, while the complete scientific archive remains the live
protocol source of truth.

The API is `matmem.campaign_gated_ic_sarr`. It is intentionally not wired into
the policy subprocess yet: a new development tranche must first validate this
restricted two-policy gate and its runtime on disjoint systems. The full Pareto
frontier and residual-budget Bellman formulations remain theory proposals, not
implemented algorithms.

## Required evaluation

On new development systems, report selection probability, paired oracle
campaign differences for `T` and `F`, posterior-vs-oracle sign/ranking
diagnostics, and compute cost. Do not tune the lower-bound threshold, add
chemistry fallbacks, or replace the gate with the posterior point selector on
the opened attribution systems. A positive result requires system-clustered
evidence and cannot be inferred from the first-action regret diagnostic.
