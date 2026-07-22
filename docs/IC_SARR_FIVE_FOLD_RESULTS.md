# IC-SARR five-fold development replication

**Status:** completed development replication, not an external confirmatory
evaluation (2026-07-22).

## Frozen comparison

This report summarizes five previously unused, outcome-independently assigned
cross-fit folds of 46 exact chemical systems each (230 systems total). The
comparison is the fixed `independent_confirmation_source_rollout` policy
(IC-SARR) against `source_margin`, with budget six, equal unit query costs,
fixed-composition causal hulls, the hierarchical frozen-structure transport
posterior, and seed `20270720`. Each IC-SARR decision uses the registered
MC1024 sixteen-block simultaneous SARR screen; a positive-but-unresolved
fallback receives the independent MC8192 one-comparison confirmation gate.

The task, oracle vault and cross-fit manifest checksums are identical in every
output, and every output records `evaluation_systems_accessed=false`:

```text
task:     f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df
vault:    a272d3a2ce6286443ae6fce35726a688751a37284e3df362c5d1f70e2fcb9952
manifest: a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa
```

Raw artifacts remain outside Git under
`/home/workspace/lrh/DATA/EviMem-RL/outputs/exploratory/`:

| Fold | SHA256 |
|---:|---|
| 1 | `24c88cb3bf1c711560800ea6e2ec828a39e932b4429549d30be9381efde342eb` |
| 2 | `c67c586d7f69b45d47048cae704e04b52a3191ef9a02c4f243d8ff030cd6fb42` |
| 3 | `60da35e5db599d07b382b05430916f9721ef7ebd641c77690fd6944c1a7b7fde` |
| 4 | `c35d8cff4167fc11bc36b25a2f1721753a73e5199eb527e4a6919fcd853e5b1b` |
| 5 | `705554d91905739b1ada2a39702ca5a9471a80f9cabafb61b10c36ca52d7be89` |

## System-level paired results

Differences are IC-SARR minus source margin. Intervals are a deterministic
system-resampling bootstrap with seed `20260722`; candidates and rounds are
never treated as independent replicates.

| Metric | F1 | F2 | F3 | F4 | F5 | All 230 systems |
|---|---:|---:|---:|---:|---:|---:|
| Oracle-pool confirmations / system | +0.174 | +0.196 | +0.196 | +0.130 | +0.109 | **+0.161** |
| Oracle win / tie / loss | 10/34/2 | 11/31/4 | 12/29/5 | 9/34/3 | 8/34/4 | **50/162/18** |
| Final causal confirmations / system | +0.043 | +0.065 | +0.022 | -0.065 | +0.000 | +0.013 |
| Causal discoveries / system | +0.478 | +0.152 | +0.391 | +0.413 | +0.174 | +0.322 |
| Action regret (eV/atom) | +0.168 | +0.121 | +0.167 | +0.089 | +0.129 | +0.135 |
| Additional wall time / system | +16.81 s | +15.58 s | +13.43 s | +46.15 s | +19.79 s | +22.35 s |

For the primary terminal metric, the combined 95% system-bootstrap interval is
`[+0.083, +0.239]`. It is positive in every fold; folds 4 and 5 individually
have intervals crossing zero, as expected at 46-system scale. Among systems
whose terminal result changes, IC-SARR wins 50 of 68 (73.5%); 162 systems tie
because source margin often reaches the finite-pool ceiling.

## Terminal-adjudication decomposition

The three hull counts are separate estimands, not interchangeable names for
"discovery."  Let `D` be causal-time announcements, `F` be confirmations
that remain on the selected-history hull after the campaign, and `T` be
queried members of the complete oracle-pool hull.  With the common hull
tolerance and duplicate-composition rule, every trace satisfies `T <= F <= D`.
The runner now enforces that order and reports the two resulting failure modes
directly: within-campaign revocation `D - F` and unqueried-competitor
invalidation `F - T`.

| System-macro count | Source margin | IC-SARR | IC-SARR - source |
|---|---:|---:|---:|
| Causal-time announcements `D` | 4.322 | 4.643 | +0.322 |
| Within-campaign retained confirmations `F` | 4.083 | 4.096 | +0.013 |
| Full-pool adjudicated confirmations `T` | 3.622 | 3.783 | +0.161 |
| Within-campaign revocations `D-F` | 0.239 | 0.548 | +0.309 |
| Unqueried-competitor invalidations `F-T` | 0.461 | 0.313 | -0.148 |

Thus the terminal gain is not a causal-time confirmation gain. IC-SARR makes
more provisional online announcements, most of whose incremental amount is
subsequently revoked inside the same campaign, while reducing the number of
final-causal survivors that fail complete-pool adjudication. The correct
paper-facing claim remains only an improvement in the last row's complement:
the budget buys more **full-pool adjudicated confirmations**.

For descriptive context, the system-macro causal-retention rate `F/D` is
0.930 for source margin and 0.859 for IC-SARR; the system-macro oracle-validity
rate `T/F` is 0.874 and 0.905, respectively. These ratios are undefined for a
system with a zero denominator and are not used as a primary objective.

IC-SARR exercised the independent numerical gate: across 1,380 query rounds,
327 actions were accepted by the simultaneous stage-one SARR screen, 332
positive-but-unresolved states entered stage two, and 224 stage-two comparisons
passed their independent lower-bound gate. Sixty-six rounds used a transparent
source fallback because the transport model lacked element support; those
rounds did not expose IC-SARR numerical diagnostics and retain the observable
source-margin fallback.

## Interpretation and boundary

The replicated claim is narrow: under this MatPES PBE--r2SCAN development
task, IC-SARR improves the **oracle-final terminal confirmation count** over
source margin. It does not establish a general improvement in causal-time
confirmation: that metric is near zero overall and its combined interval
crosses zero. Nor does it improve myopic action regret; the positive regret is
consistent with explicitly sacrificing a short-term source-margin action for a
better posterior-model terminal rollout.

The present implementation is materially slower because phase-diagram
construction dominates the MC rollout. Wall times also vary with shared-server
load, so this report does not make a stable hardware-speed claim. Any future
performance rewrite must first pass action-, hull- and reveal-parity tests
against these frozen traces; it must not change the posterior, terminal reward,
source continuation, or IC-SARR gate while being described as an optimization.

## Feedback-driven implementation audit

The cached fixed-composition backend was retested after the metric update on
2026-07-22. A read-only one-system audit (`Co-F-Li-O`, task/vault checksums
above, budget two, MC32) obtained identical Delta-Hull selected-action traces
and all 32 sampled final-hull membership vectors against the pymatgen backend.
The measured wall time was 2.312 seconds with cached geometry versus 5.197
seconds with pymatgen; this is an engineering observation, not a policy result.
The audit manifest is outside Git at
`/home/workspace/lrh/DATA/EviMem-RL/outputs/audits/ic-sarr-feedback-fixed-backend-parity-v1/`.

The same audit run exercised the IC-SARR runner once on an already-opened
development regression fixture (`Ag-S`) with the fixed backend. Its purpose
was only to confirm that the full reveal boundary and the new `D,F,T` metrics
execute together; it is not compared with a newly selected method and is not
included in any effect estimate. The source and IC-SARR traces respectively
gave `(D,F,T)=(4,1,1)` and `(2,1,0)`, illustrating why a one-system rerun must
not be treated as scientific evidence.

This is a five-fold development replication, not a sealed external evaluation
or a claim of real DFT deployment benefit. The manuscript may report the
result as cross-fitted real-data evidence only with these scope limitations.
