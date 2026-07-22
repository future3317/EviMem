# Canonical experiment plan

The active MatPES development artifact is the v6 task with frozen CHGNet
source embeddings, its v5 oracle vault, and the fold-0 cross-fit transport
model. Their remote paths and SHA-256 values are recorded in
`manifests/matpes-canonical-development-v1.json`.

The old v2--v5 task snapshots, the historical 48-system task, redundant
transport files and completed exploratory/engineering outputs were moved to
`DATA/EviMem-RL/archive/superseded-20260722/`. They remain recoverable but are
not active inputs.

Until the method is frozen, experiments are limited to unit/parity tests, small
canonical smoke pilots using source-margin and IC-SARR, and independent
implementation diagnostics for campaign-level constrained rollout.

No retraining is required: CHGNet is frozen and transport fitting is a small
system-level ridge/kernel fit. After a method freeze, rebuild a fresh task
manifest from the canonical source, reserve a new holdout, and run the complete
policy, ablation and baseline matrix once.

The first canonical rebuild pilot used two fold-0 systems, budget four,
fixed-composition hulls and 1024-stage IC-SARR integration. It is a smoke
check, not a method result or external confirmation.
