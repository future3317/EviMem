# IC-SARR numerical-gate plan

**Status:** five-fold development replication complete; no external evaluation
has run. Authorized by the completed SARR MC8192 numerical opportunity-cost
audit, this file does not authorize a general performance claim,
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

## Implementation record (2026-07-22)

`independent_confirmation_source_rollout` is now a distinct worker policy.
The worker fixes stage one at MC1024 and stage two at MC8192; its generic
Monte-Carlo CLI setting is intentionally ignored for this method. Stage two
derives its seed as `stage_one_seed + 1,000,000,007`, whereas stage one uses
the existing `stage_one_seed + 104729 * block` streams for blocks 0--15. The
two streams are therefore disjoint by construction. The stage-two evaluator
constructs only the source and the single screen-selected first-action rollout;
it does not re-rank candidates.

The policy emits the stage-one selection, screened candidate, stage-two use,
paired mean and lower bound, both seed derivations and both sample counts in
the pre-reveal diagnostics. Unit tests fail if an accepted SARR action changes,
if no positive stage-one candidate still triggers stage two, if the second
stream is not independent, or if a non-positive/positive one-comparison lower
bound respectively does/does not fall back to/deviate from source. These are
implementation gates only. The next permitted physical run remains a single
unused-development-system parity preflight; no fold-level comparison has been
opened.

### Completed preflight (2026-07-22)

The deterministic first system of unused fold 1, `Ag-F-Li`, passed the
implementation preflight under the v6 task checksum
`f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df` and
cross-fit manifest checksum
`a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa`.
The append-only six-round trace has exact selected-action/reveal parity; three
rounds invoked stage two, all reported the frozen MC1024/MC8192 counts and no
stage-two seed equalled its stage-one seed. The atomic preflight record is
`sha256:9843b8e72f6b11644f884746f6d76a577363048d9b07a1ac537dab3bf66ff243`.
It contains no evaluator metrics and records
`evaluation_systems_accessed=false`. This clears only the implementation gate;
it is not a per-system effect result and does not authorize a method change.

## Decision boundary

IC-SARR preserves the existing source/reveal invariants and has now replicated
the registered terminal oracle-final metric over five unused development folds:
`+0.161` confirmations/system, 95% system-bootstrap interval
`[+0.083,+0.239]`, and 50/162/18 system wins/ties/losses. The full immutable
artifact inventory, secondary metrics and scope boundary are in
`docs/IC_SARR_FIVE_FOLD_RESULTS.md`.

This clears the method's **development replication** gate. It is not an
external evaluation or a general deployment guarantee: final causal
confirmation is not improved reliably, and the implementation is slower. Any
future method change must reserve a new disjoint evaluation partition; an
action-parity-preserving phase-diagram performance optimization is allowed as
an engineering task but cannot be reported as a changed policy.
