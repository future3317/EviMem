# CAL hull runtime-plan design

## Goal

Remove repeated invariant geometry construction from the CAL-style hull-entropy
score while preserving the exact fixed-composition objective, random samples,
and deterministic action ranking.

## Scope and invariants

The optimization applies only when `protocol_hull_entropy` receives the frozen
fixed-composition backend. It must not change the task, vault, cross-fit split,
seed, posterior MC count, fantasy count, query pool, policy roster, hull
semantics, or evaluator. The current fold5 recovery remains untouched; any
execution using this implementation receives a fresh output identity.

## Design

`protocol_hull_entropy` builds one private CAL runtime plan for its immutable
selection state. The plan stores the validated fixed template, the ordered
query-index groups for each evaluation composition, and the causal hull
envelope. `_cal_hull_values` consumes that plan for every posterior or
conditional world, so it only reduces duplicate-composition energies and
evaluates the already-derived envelope. Calls without a plan retain the
existing validation and construction path.

Candidate-level threading remains unchanged. The plan is immutable and shared
read-only across candidates, so no result depends on task scheduling.

## Verification

Tests must first demonstrate that the new plan API is absent, then establish
bitwise equality of fixed-backend hull vectors with and without the plan and
exact CAL score/action equality between serial and parallel candidates.
Relevant protocol and campaign tests plus Ruff must pass. A new remote run is
allowed only after those checks and is kept distinct from incomplete artifacts.
