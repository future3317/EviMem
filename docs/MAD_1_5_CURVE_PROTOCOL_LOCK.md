# MAD-1.5 fixed-budget acquisition-curve protocol lock

Status: frozen before the next curve run; this is a development-task protocol,
not a MatPES holdout or confirmatory evaluation.

## Frozen inputs

- Task: `E:\DATA\MAD-1.5-v1\mad15_task_v1.json`
  - SHA-256: `b229e21273754bca832c9f8aa3168cbb62f12425b2ef6e55f99188d71653de59`
- Oracle vault: `E:\DATA\MAD-1.5-v1\mad15_vault_v1.json`
  - SHA-256: `e35025caf2c442373340dd89d0761c3a3b94c184cc0c0f7245aa96474d30ab4f`
- Query manifest: `E:\DATA\MAD-1.5-v1\mad15_curve_manifest_v1.json`
  - SHA-256: `d713ecd2f442ac0bdf2b2fb6acfbc233be32c637c23385a1539acebc99c7f7ae`
- Implementation freeze: Git commit `1621159`
- Selected systems: 96 exact chemical systems; all 96 are outside the union
  of the previously opened MAD probe and source-curve panels.
- Opened-system union excluded by the manifest: 96 systems.
- Manifest folds: 6 precomputed folds with counts `17, 17, 16, 16, 16, 14`.
  The primary curve uses the complete 96-system manifest set; the folds are
  retained for a separately registered cross-fit analysis and must not be
  recomputed from target outcomes.
- Fit rule for the primary curve: all development systems in the task except
  the 96 query systems, giving 1,788 transport-fit systems. No query outcome
  may enter the transport fit.

The manifest selection uses only task-visible system identity, candidate count,
composition count, release ID and the recorded opened-system lists. It does
not read target energies, hull labels or post-query metrics when assigning
systems or folds.

## Policies and numerical settings

The only comparison is frozen `source_margin` versus
`independent_confirmation_source_rollout` (IC-SARR). For every budget:

- hull backend: `fixed_composition`;
- transport family: `ridge_random_intercept`;
- ridge penalty: `1.0`;
- prior standard deviation: `0.1`;
- boundary temperature: `0.05 eV/atom`;
- posterior sample count: `64`;
- fantasy count: `3`;
- policy seed: `20270720`;
- common statistical seed: `20260730`;
- all revealed outcomes remain in the append-only archive;
- tie-breaking is the existing immutable candidate-ID rule.

No score, posterior temperature, MC count, threshold, seed, rollout rule,
kernel, or cost weight may be changed after any curve output is read.

## Curve and estimands

The single system manifest is reused unchanged for `B=0,1,2,3,4,5,6`.
`B=0` is a synthetic zero-query row with zero confirmations and zero policy
time; no oracle outcome is opened for it. For `B=1..6`, the runner uses the
explicit `--query-budget B` override. This avoids the old candidate-count
fallback and guarantees that every system receives the registered budget.

For each budget and policy, report at exact-system grain:

1. oracle-final confirmations;
2. final-causal confirmations;
3. wall time per system;
4. incremental confirmations per incremental second;
5. source-relative recovered headroom, defined as
   `mean(IC - source) / mean(ceiling - source)` over systems with positive
   source headroom.

The primary curve-level estimands are trapezoidal AUC over the fixed grid
`B=0..6` for oracle-final and final-causal confirmations. For every budget and
for both AUC estimands, report paired exact-system bootstrap 95% intervals,
Monte Carlo sign-flip randomization `p` values, and win/tie/loss counts. The
bootstrap uses 20,000 replicates and the sign-flip calculation uses 100,000
replicates with the frozen common statistical seed.

Cost is co-primary through the pre-registered scalar utility

`U_AUC = delta oracle-final AUC - 0.10 confirmations/(second*system) * delta wall-time AUC`.

The coefficient `0.10` is fixed before this curve is run. It must not be
selected or changed based on the earlier `+2.752 s/system` probe. The output
must still report the unscalarized confirmation and time curves; a positive
confirmation AUC with negative `U_AUC` is not a cost-aware superiority result.

## Reproducible command

For each `B` in `1..6`, run the following command with the final output path
changed to a unique file outside Git:

```powershell
conda run --no-capture-output -n llm python tools/run_matpes_protocol_closed_loop_exploratory.py `
  --task E:\DATA\MAD-1.5-v1\mad15_task_v1.json `
  --development-vault E:\DATA\MAD-1.5-v1\mad15_vault_v1.json `
  --query-manifest E:\DATA\MAD-1.5-v1\mad15_curve_manifest_v1.json `
  --query-budget B `
  --maximum-budget 6 `
  --minimum-candidates 8 `
  --seed 20270720 `
  --posterior-sample-count 64 `
  --fantasy-count 3 `
  --hull-backend fixed_composition `
  --transport-family ridge_random_intercept `
  --policies source_margin independent_confirmation_source_rollout `
  --split development `
  --output E:\DATA\MAD-1.5-v1\mad15_curve_BB_20260730.json
```

Replace both `B` occurrences with the same integer. After all six outputs
complete, summarize them with:

```powershell
conda run --no-capture-output -n llm python tools/summarize_mad15_curve.py `
  --manifest E:\DATA\MAD-1.5-v1\mad15_curve_manifest_v1.json `
  --input E:\DATA\MAD-1.5-v1\mad15_curve_B1_20260730.json `
  --input E:\DATA\MAD-1.5-v1\mad15_curve_B2_20260730.json `
  --input E:\DATA\MAD-1.5-v1\mad15_curve_B3_20260730.json `
  --input E:\DATA\MAD-1.5-v1\mad15_curve_B4_20260730.json `
  --input E:\DATA\MAD-1.5-v1\mad15_curve_B5_20260730.json `
  --input E:\DATA\MAD-1.5-v1\mad15_curve_B6_20260730.json `
  --output E:\DATA\MAD-1.5-v1\mad15_curve_summary_20260803_recomputed.json
```

The six runner outputs and the corrected summary remain outside Git. The
original `mad15_curve_summary_20260730.json` is retained as a superseded
artifact because its paired-AUC aggregation omitted the `B=0\rightarrow1`
trapezoid; the runner inputs and all per-budget values are unchanged. A result from this
protocol remains a MAD-1.5 task-level mechanism result. It cannot be called an
independent MatPES holdout, a universal discovery claim, or evidence for
decision-sufficient state compression.

## Completed curve result

The frozen `B=0..6` curve completed on the 96-system manifest. The summary is
outside Git at
`E:\DATA\MAD-1.5-v1\mad15_curve_summary_20260803_recomputed.json`; all six runner inputs
passed the manifest, task, vault, transport and exact-budget checks.

The prior summary at
`E:\DATA\MAD-1.5-v1\mad15_curve_summary_20260730.json` is superseded solely
because its paired-AUC aggregation omitted the `B=0\rightarrow1` trapezoid.
The corrected summary SHA-256 is
`96bc094c87e121b4509cc8523899f0cf115443a19f1540cbf0450a2729e187ca`.

The oracle-final AUC was `13.9479` for IC-SARR and `13.6979` for source
margin: paired difference `+0.2500`, system-bootstrap 95% interval
`[+0.0521,+0.4583]`, sign-flip `p=0.0193`, with 18 wins, 73 ties and 5
losses. The final-causal AUC difference was only `+0.0260`, interval
`[-0.0313,+0.0833]`, sign-flip `p=0.5320`, with 4 wins, 90 ties and 2
losses. Wall-time AUC increased by `+8.5739 s*budget/system`.

Under the frozen cost utility, `U_AUC = +0.2500 - 0.10*8.5739`, the result
is `-0.6074`. Therefore this curve supports a small oracle-acquisition
mechanism signal, but fails the cost-aware superiority criterion and does not
show a final-causal gain. It remains development-only MAD task evidence.
