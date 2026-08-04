# Delayed-label experiment triage

This memo fixes the evidence priority after the manuscript was reframed as an
objective/theory/mechanism paper. It is a scope control document: a proposed
experiment must change a paper claim to justify additional compute.

## Keep and finish

1. **E32-A objective/lookahead suite.** This is the necessary same-posterior
   comparison of source-derived targeting, current/final-hull greedy actions,
   source continuation, and Delta-Hull-anchored continuation. It directly
   tests whether the complete-pool objective changes acquisition and whether
   lookahead adds value after the strong greedy baseline is available.
2. **MAD15 direct mechanism curve.** This is the necessary direct comparison on
   the already frozen 96-system protocol-shift panel. It reports Delta-Hull,
   ungated rollout, and IC-SARR against one another across `B=1..6`, rather than
   inferring attribution from source-relative bars.

Both registered suites are now complete and summarized. The E32-A follow-up
shows that target-margin and Delta-Hull targeting explain most of the
source-relative gain; a Delta-Hull-anchored two-step rollout adds a modest
development-level terminal-$T$ increment at the largest budget, while the
posterior-only coupling/rank strata do not yet identify a monotone interaction.
The MAD direct curve leaves all three solver contrasts small with paired
intervals crossing zero. These results close the current experimental stage;
they do not authorize another full campaign.

Both are development/mechanism evidence. MAD remains an atomization-energy
hull proxy and cannot be called a formation-energy holdout or deployment test.

## E50 -- objective-efficiency evaluator analysis

The completed E32-A outputs were reprocessed without rerunning a policy.  The
contrast is Delta-Hull minus posterior-mean target-margin greedy under the same
target posterior, worlds, tie-breaking, budgets, and hull evaluator.  The
system-level paired differences are `+0.0826`, `+0.0870`, `+0.0826`, `+0.0696`,
`+0.0217`, and `-0.0043` for `B=1..6`; the AUC contrast is `+0.3000` with
95% CI `[+0.0522,+0.5587]`, sign-flip `p=0.0247`, and 60/114/56
wins/ties/losses.  Bootstrap resampling is by exact chemical system and retains
all budgets within each draw.

The derived artifact is outside Git at
`E:\DATA\EviMem-RL\analysis\delayed_label_objective_efficiency_e50_20260804.json`,
SHA-256
`db4799d6770bd7eae1d50196ff38bb48bc5944e45661536b199523296ea1cc99`.
This result supports an early-budget objective-efficiency claim; it does not
support a final-budget solver-superiority claim or authorize a new policy run.

## Use lightweight post-processing only when needed

- Paired contrasts, rank-margin/coupling strata, and action disagreement are
  derived from the E32 traces and do not require another policy run.
- Hull-tolerance and competitor-removal sensitivity has now been completed as
  `DELAYED_LABEL_EVALUATOR_SENSITIVITY_PROTOCOL_V1`: zero selected-label flips
  across `1e-8`--`1e-12`, and small bounded changes under protected nested and
  unqueried-competitor-removal pools. It reuses frozen selections and the
  oracle vault only; it does not change actions, posterior, policies, or the
  primary estimand, and therefore does not establish action robustness.
- Cost reporting should separate planning time from target-query cost using
  existing timing records; no cost weight is to be fitted from the new runs.

## Do not add to this project stage

- Do not search for or manufacture an external formation-energy holdout. The
  opened MatPES systems cannot be relabeled as holdout, and MAD is not the same
  label definition.
- Do not repeat the 1,000-instance exact-DP suite or the controlled grid; they
  already establish the theoretical mechanism boundary.
- Do not launch a new instrumentation run solely for hull-membership
  calibration before checking whether the frozen decision traces contain the
  required probabilities. Existing weak rollout calibration remains a
  limitation, not a reason to change the method.
- Do not add runtime-scaling or GPU work as a paper contribution. The existing
  action-preserving cache audit is engineering evidence only.

## Stop rule

After E32-A and MAD15 direct summaries, update the manuscript once. If a
proposed additional audit cannot alter the objective-vs-solver interpretation,
report it as future work rather than spending another full campaign.
