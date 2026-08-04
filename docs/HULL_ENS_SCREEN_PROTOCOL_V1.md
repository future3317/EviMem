# Hull-ENS feasibility screen v1

**Status:** registered screening run, 2026-08-04. Raw outputs remain outside
Git. This run is not a replacement for the high-precision E48 P0 and cannot
support a manuscript superiority claim by itself.

## Purpose

The full E48 numerical setting (128 posterior worlds × 8 conditional fantasy
worlds) requires about 206 seconds for one difficult four-component system even
after bounded candidate-level parallelism. Before spending the corresponding
multi-hour budget, this screen asks whether Hull-ENS changes actions or direct
terminal outcomes at all under the same task and objective.

## Frozen scientific components

The task, vault, cross-fit folds, target-energy posterior family, seed,
candidate ordering, fixed-composition hull backend, policies, reveal boundary,
and terminal D/F/T metrics are inherited unchanged from
`docs/HULL_ENS_PROTOCOL_V1.md`. Only the numerical integration budget is
reduced to 16 posterior worlds and 2 conditional fantasy worlds. Candidate
level parallelism is 4 and independent units are scheduled with 8 workers.
The lower MC setting is explicitly a feasibility screen, not a calibrated
confidence calculation.

## Roster and output rule

The screen has five outcome-independent folds, budgets `B=1..6`, and the same
four-policy roster: `source_margin`, `delta_hull_active_search`, `hull_ens`,
and `safe_hull_ens`. It uses a new external root:

`/home/workspace/lrh/DATA/EviMem-RL/analysis/hull_ens_screen_v1_20260804`

Only a complete 30/30 roster with no failure markers may be summarized. The
summary must report direct paired action and D/F/T contrasts, wall time, and
safe-gate rates. Screen results are used solely to decide whether a later
128×8 confirmation is computationally justified; they are not copied into the
main manuscript as evidence of superiority.
