# IC-SARR numerical-gate plan

**Status:** proposal only, authorized by the completed SARR MC8192 numerical
opportunity-cost audit. This file does not authorize a performance claim,
evaluation-system access, or a change to `source_rollout_delta_hull`.

## Question

The fold-0 audit confirms two distinct facts: accepted SARR actions are stable
at high precision, while the simultaneous Bonferroni fallback rejects many
actions with positive high-precision posterior-model-relative opportunity
cost. The narrow question is whether an independent confirmation stage can
resolve this numerical conservatism without using fold-0 outcomes to tune a
threshold.

## Candidate method: Independent-Confirmation SARR (IC-SARR)

IC-SARR is a separately named policy, not a modification of SARR. At a
pre-reveal state it does the following.

1. Run the frozen SARR screen: MC1024, sixteen scrambled-Sobol blocks and its
   existing 95% Bonferroni-simultaneous paired lower bounds.
2. If SARR accepts a non-source action, execute exactly that action. If the
   source is selected and every non-source mean advantage is non-positive,
   execute source.
3. Otherwise select the unique stage-one candidate with greatest positive mean
   advantage (pair ID breaks ties).
4. On an **independent** set of MC8192 scrambled-Sobol blocks, estimate only
   that preselected candidate's paired advantage over source and form its 95%
   one-comparison lower bound. Execute the candidate iff this lower bound is
   strictly positive; otherwise execute source.

The stage-two randomization must use a disjoint fixed seed stream. Conditional
on the stage-one screen, it is a single preselected comparison; it must never
reuse stage-one blocks or search over additional candidates. This is an
integration-noise control, not a posterior-calibration or oracle-final safety
guarantee.

## Frozen development protocol

- Fold 0, its 196 audit states, all 48 opened systems, and all sealed
  evaluation systems are excluded from configuration and effect estimation.
- The source continuation, target posterior, terminal reward, equal-cost
  requirement, causal hull, oracle vault and selected-action-only reveal rule
  remain identical to SARR.
- The policy name, `1024/8192` sample counts, sixteen blocks at each stage,
  confidence level, independent seed derivation, tie break and stage-two
  one-comparison rule are fixed before any unused-fold trace is inspected.
- First run a one-system implementation/parity test that checks source-only
  fallback, accepted-SARR identity, independent stage-two seeds and no
  access to unrevealed outcomes. It is not an effect result.
- Only then run a whole unused development fold once. Report the same causal
  and oracle-final metrics as SARR, action count, stage-two invocation rate,
  online time and exact-system clustered uncertainty. Do not stop early or
  select systems by observed gain.

## Decision boundary

IC-SARR is a numerical-gate diagnostic. It may only become a paper-facing
method candidate if it first preserves the existing source/reveal invariants
and then replicates on multiple unused cross-fit folds. A fold-level
oracle-final improvement or a higher posterior rollout score alone is not a
GO decision.
