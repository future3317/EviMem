# Selective Delta-Hull planning: registered protocol v1

**Status:** registered method-development line, 2026-08-04. Raw experiment
outputs remain outside Git. This protocol is a new identity and does not
overwrite, merge, or relabel E32-A, P0-v3, MAD-1.5, or the Hull-ENS roots.

## Scientific question

When a delayed structured-label task has little usable planning headroom, can
the policy identify that fact from the posterior and fall back to the
complete-pool greedy rule instead of paying for an always-on rollout?

The changed failed assumption is therefore **not** the posterior, label, or
reveal model. It is the assumption that a source-relative rollout should be
invoked for every decision. The new policy is Delta-Hull-relative: Delta-Hull
is the default, and lookahead is called only when a predeclared lower bound on
posterior planning headroom exceeds fixed model and compute penalties.

## Frozen scientific boundary

The following remain unchanged:

- the target observation and delayed complete-pool label;
- the oracle vault, reveal boundary, and append-only scientific archive;
- the all-outcome posterior update semantics;
- the candidate and reference-phase definitions;
- the fixed-composition hull backend and deterministic ID tie-breaking;
- the task, folds, budgets, and existing E32/P0/MAD seeds and manifests.

No state-compression or outcome-selected posterior coreset is introduced. Every
revealed outcome remains in the archive and conditions the posterior.

## Policy and gate

For a state `h`, let `g(h)` be Delta-Hull and let `Q_2(x|h)` be the two-step
complete-pool value with an independent inner posterior stream. The measured
headroom is

\[
 H(h)=\max_x Q_2(x\mid h)-Q_2(g(h)\mid h).
\]

The selective policy computes a posterior-only estimate

\[
 \widehat G(h)=\widehat H(h)-c_{\alpha}\,\widehat{se}(H)
 -\tau_{\rm model}-\tau_{\rm compute}.
\]

It executes the Delta-Hull action when `G <= 0`; otherwise it invokes the
registered Delta-Hull-anchored rollout. `tau_model`, `tau_compute`, and the
critical value are fixed before opening an evaluated system or learned only
with nested training folds. No realized final label, complete-pool outcome,
or post-hoc action result may enter the gate.

The two-step information value is recorded separately for each candidate:

\[
 I_h(x)=\mathbb E_{O_x\mid h}\left[\max_{y\ne x}p_{h,x,O_x}(y)\right]
 -\max_{y\ne x}p_h(y).
\]

The implementation stores signed conditional changes, conditional
membership probabilities, rank gaps, and maximizer-switch probabilities. These
are diagnostics, not evaluator labels.

## Required implementation audits

Before any material selective curve, the code must pass:

1. top-two exchange parity: for the current top two candidates,
   `Q_2(g2)-Q_2(g1) = I(g2)-I(g1)`;
2. the planning-value identity separating structural headroom, posterior
   solver/Monte-Carlo regret, differential model error, and incremental
   planning cost;
3. exact finite-world Hull-ENS, sampled Hull-ENS, and independent
   double-sampled Hull-ENS comparisons;
4. observation measurability: continuation actions depend only on the revealed
   observation branch, never on unrevealed world coordinates;
5. deterministic tie-breaking and exact Delta-Hull parity at horizon one.

## Staged experiment authorization

### Stage A: finite-world audit

Use only the registered four-world delayed-label benchmark, disjoint from all
materials data and vaults. Compute exact belief-state DP, repeated
Delta-Hull/greedy, exact Hull-ENS, sampled Hull-ENS, and double-sampled
Hull-ENS. Report paired action agreement and value regret. Stratify instances by

\[
 H^\star=V_{\rm DP}-V_{\rm greedy}
\]

using the frozen bins `H*=0`, `0<H*<=0.02`, `0.02<H*<=0.10`, and `H*>0.10`.
The generator, acceptance rule, count, seed, and quotas must be frozen before
the summary is read. If the high-headroom bins cannot be populated without
outcome-dependent selection, report them as unavailable rather than relaxing
the rule after inspection.

The initial 4,000-candidate capacity preflight was incomplete because the
`0<H*<=0.02` bin contained only 13 candidates for the frozen quota of 32; it
produced no selected records or summary. E51-A increased only the candidate-
pool capacity to 16,000, retaining the same seed, exact-DP bin rule, quota,
and policy/evaluator code; it still found only 28 candidates and also produced
no summary. E51-A2 attempted 32,000 candidates but timed out during exact-DP
pre-screening before producing selected records. E51-A3 is the final
lightweight audit recovery: it uses the same 16,000-candidate pool and seed
with quota 16 per bin. The smaller, explicitly reported quota is for mechanism
auditing only; it does not support a performance-generalization claim. If a
bin is unavailable, the staged audit remains incomplete and no material run
is authorized. The preflights are not merged into the recovery result.

### Stage B: selective synthetic replay

Only if Stage A passes the implementation audits, evaluate the selective gate
on 80 newly generated finite-world instances: the first 40 are an inner
calibration set and the last 40 are held out for one outer evaluation. The
candidate model-penalty grid is frozen as
`{0, .002, .005, .01, .02, .05, .10}`; compute penalty is fixed at zero in this
synthetic audit. Select the largest penalty satisfying both inner invocation
rate `<=0.25` and retained full-planner gain `>=0.80` when such a candidate
exists; otherwise select the candidate with the largest inner selective gain,
breaking ties by larger penalty. Apply the selected penalty once to the outer
systems. Report invocation rate, value relative to Delta-Hull, retained
fraction of full-rollout value, and computation separately. This stage is
mechanism evidence, not material or deployment evidence.

### Stage C: material decision

No MatPES or MAD selective run is authorized by this document until Stages A/B
are complete, the exact/sampled audit agrees within its registered tolerance,
and a new material execution identity freezes all penalties and nested folds.
If a material run is later authorized, its primary comparison is selective
Delta-Hull versus Delta-Hull and always-on Delta-Hull-anchored rollout, with
direct paired D/F/T and wall-time contrasts. It may not be described as a
holdout, formation-energy validation, final-causal, cost-aware, deployment,
runtime, or universal-superiority result.

Stage A/B were completed as of 2026-08-05, but the largest per-instance
exact-versus-double-sampled value discrepancy in Stage A was `0.016556` and
no tolerance for that maximum had been frozen before the audit. Therefore the
registered Stage C decision is **no material selective run**. Selective
planning remains a synthetic mechanism result and implementation proposal;
opening MatPES or MAD would require a new protocol that predefines the
numerical tolerance, material penalties, nested folds, and direct contrasts.

## Claim boundary

A negative result is scientifically useful: it can show that the tested
posterior/task lies in a greedy-sufficient regime or that approximate planning
regret/model error consumes the available headroom. A positive result supports
only the measured delayed-label planning mechanism under the registered task
and posterior. It does not establish a generally safe gate or posterior
correctness.
