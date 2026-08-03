# MAD-1.5 direct full-pool mechanism comparison

**Experiment identity:** `MAD15-DIRECT-MECHANISM-20260803`  
**Status:** registered task-level development experiment; not a holdout

This amendment answers the missing direct-comparison question on the already
frozen MAD-1.5 protocol-shift task: does targeting the complete-pool hull
objective help against source margin, and does nonmyopic continuation add
anything beyond the one-step Delta-Hull action? It does not reopen, retune or
relabel the 96-system panel.

## Frozen inputs and semantics

- Task: `E:\DATA\MAD-1.5-v1\mad15_task_v1.json`, SHA-256
  `b229e21273754bca832c9f8aa3168cbb62f12425b2ef6e55f99188d71653de59`.
- Oracle vault: `E:\DATA\MAD-1.5-v1\mad15_vault_v1.json`, SHA-256
  `e35025caf2c442373340dd89d0761c3a3b94c184cc0c0f7245aa96474d30ab4f`.
- Query manifest: `E:\DATA\MAD-1.5-v1\mad15_curve_manifest_v1.json`, SHA-256
  `d713ecd2f442ac0bdf2b2fb6acfbc233be32c637c23385a1539acebc99c7f7ae`.
- Query set: the same 96 exact chemical systems as the frozen curve; fit set:
  the same 1,788 disjoint systems.
- Hull meaning: atomization-energy convex-hull proxy relative to isolated-atom
  references. It is not a solid-state formation-energy hull.
- Backend: `fixed_composition`; transport family: `ridge_random_intercept`;
  ridge penalty `1.0`; prior standard deviation `0.1`; boundary temperature
  `0.05 eV/atom`; seed `20270720`; posterior count `64`; fantasy count `3`.
- Minimum candidate count: `8`, matching the frozen MAD curve.
- Budgets: `B=1,...,6`, with the existing `B=0` source baseline retained only
  for curve AUC construction.

The policy roster is fixed for every budget:

```text
source_margin
delta_hull_active_search
ungated_source_rollout
independent_confirmation_source_rollout
```

No score, posterior, MC count, gate, threshold, fallback, tie-break, manifest,
cost weight or hull rule may change after the first output is opened. Raw
outputs and the derived summary remain outside Git under a new
`E:\DATA\MAD-1.5-v1\mad15_direct_mechanism_20260803\` root.

## Required comparisons

For every budget and exact chemical system report D, F, T and wall time. The
derived summary must include direct paired bootstrap 95% intervals,
sign-flip/randomization p-values, win/tie/loss counts, and confirmation/time
ratios for:

1. Delta-Hull minus source margin;
2. ungated source rollout minus Delta-Hull;
3. IC-SARR minus ungated source rollout;
4. IC-SARR minus Delta-Hull.

The result is a protocol-shift mechanism comparison only. A positive T or AUC
does not establish final-causal, cost-aware, deployment, universal or external
formation-energy superiority.

## Reproducible execution shape

Run the existing closed-loop runner six times with the same task, vault,
manifest and settings, changing only `--query-budget B` and the unique output
path:

```text
--policies source_margin delta_hull_active_search ungated_source_rollout independent_confirmation_source_rollout
```

Summarize only after all six outputs complete with
`tools/summarize_mad15_direct_mechanism.py`. Refuse missing or failed budget
cells and refuse overwriting an existing derived summary.
