# Fixed-GP ceiling audit before AKSC

See `docs/EXPERIMENT_LEDGER.md` for the full DACC--P3C--AKSC chronology,
external result hashes and superseded/invalid evidence labels.

**Status (2026-07-20): P3C remains NO-GO; AKSC implementation is not yet
authorized.** This audit uses the two immutable, initial-structure-correct WBM
panels without changing acquisition, GP parameters, outcomes, or metrics.

The proposed next line is **Archive-Conditioned Kernel Sketch with Sequential
Calibration (AKSC)**:

> outcome-independent basis selection + all-outcome posterior sufficient
> statistics + explicitly scoped calibration.

This is not a renamed P3C variant. P3C selects which observed outcomes remain
in a size-
\(K\) GP. AKSC would retain every revealed residual in the posterior natural
parameters and compress only the function representation and linear-algebra
state.

## P0 provenance closure

The two historical data-contract failures are now type-level constraints:

- `StructureStage.INITIAL` / `RELAXED` and typed WBM `org` / `opt` sources;
- every policy query, witness, SOAP record, and secure-process payload carries
  one exact `StructureArtifactIdentity`;
- relaxed/`opt` structures cannot instantiate policy-visible features;
- `WBMOracleRecord`, the oracle vault, and `CorrectedPhaseEntry` accept only
  `OracleEnergySource.FROZEN_PARITY_CORRECTED`;
- changing only `opt` leaves initial identity and recovered SOAP input
  unchanged;
- changing an unrevealed oracle energy leaves the pre-reveal state checksum,
  action checksum, and selected action unchanged.

The external SOAP/P1 gate schema now uses `structure_stage=initial` and
`causal_available_before_query=true`. Old string-schema gates fail closed and
must be regenerated; they are not silently upgraded.

## Frozen diagnostic inputs

The source panels remain outside Git:

```text
E:\DATA\EviMem-RL\outputs\exploratory\wbm-p3c-16sys-b8-k2-initial-org-v3
E:\DATA\EviMem-RL\outputs\exploratory\wbm-p3c-next16-b8-k2-initial-org-v1
```

The evaluator-only ceiling output is:

```text
E:\DATA\EviMem-RL\outputs\diagnostics\wbm-fixed-gp-ceiling-32sys-b8-k2-v3.json
```

It covers 32 exact systems, 683 candidates, matched frozen actions,
\(B=8,K=2\), 4,096 evaluator-side subset evaluations, and the GP configuration
frozen on disjoint calibration systems. Exact systems are the statistical
units.

## Ceiling results

All differences below are loss differences; negative is better.

| Comparison | CRPS | Brier | log loss | RMSE (eV/atom) | Gaussian NLL |
|---|---:|---:|---:|---:|---:|
| full history minus prior | -0.003716 | -0.060721 | -0.139793 | -0.004590 | +2.300940 |
| GPV \(K=2\) minus full history | -0.001086 | +0.003542 | +0.000401 | -0.001622 | -0.705233 |
| full history minus per-metric oracle best-\(K=2\) | +0.005279 | +0.011346 | +0.060982 | +0.007385 | +2.554469 |

The full-history versus prior effects have 95% system-bootstrap intervals:

- CRPS: `[-0.006820, -0.000639]`;
- Brier: `[-0.081801, -0.040523]`;
- log loss: `[-0.215129, -0.054498]`;
- RMSE: `[-0.009310, +0.000496]`;
- Gaussian NLL: `[+1.127162, +3.723633]`.

Thus the fixed GP has useful mean/probability headroom over its prior, but its
posterior variance is badly misspecified: adding all history improves several
losses while making Gaussian NLL decisively worse. GPV \(K=2\) has lower NLL
than full history by `-0.705233` with interval
`[-1.236071, -0.276490]`. Full history is therefore not a clean universal
reference for compression.

The per-metric oracle best-\(K\) result is an evaluator-only ceiling that uses
hidden labels and selects a different subset for each metric. It proves that
small-subset headroom exists, not that an online selector can attain it. The
current summary saturates 336 stored probabilities, so reconstructed thresholds
make the prior/oracle Brier and log-loss values approximate. Recorded
full-history-versus-GPV comparisons and all residual metrics remain exact.
The evaluator schema has since advanced to `wbm-fixed-gp-ceiling-v2`: every
per-query record must carry `residual_threshold_ev_per_atom`, and legacy
summaries without it fail closed instead of reconstructing a threshold from a
clipped probability. The 32-system values above remain the immutable v1
diagnostic until the same frozen traces are regenerated under v2.

## Kernel representation diagnosis

Regularized effective dimension is

\[
d_{\mathrm{eff}}(\lambda)=
\operatorname{tr}\!\left[K(K+\lambda I)^{-1}\right],\qquad
\lambda=\sigma_n^2+\text{jitter}.
\]

Across 32 systems its mean is `12.424` (95% interval
`[11.356, 13.525]`), median `12.125`; every system exceeds both 2 and 4.
Consequently \(K=2\) is an intentionally severe memory bottleneck, not a
kernel-approximation budget justified by effective dimension.

Residual--kernel alignment is weak and heterogeneous:

- centered kernel-target alignment: mean `0.161`;
- kernel Moran autocorrelation: mean `0.0008`, interval crossing zero;
- nearest-neighbor sign agreement: `0.774`, but this must be compared with the
  residual-sign imbalance baseline; the diagnostic reports the above-chance
  value separately;
- full-history leave-one-out RMSE: `0.0628` eV/atom;
- full-history leave-one-out Gaussian NLL: mean `5.362`, median `0.954`.

The near-zero Moran statistic and very poor LOO NLL support a frozen
SOAP--Matérn/noise mismatch. They do not show that kernels are useless: the
nonzero target alignment and full-history improvements show some signal. The
correct conclusion is that representation compression and probabilistic
calibration must be evaluated separately.

A direct 32-system LOO dispersion audit confirms overconfidence rather than
merely inferring it from NLL. Mean squared standardized residual is `16.90`
(95% system-bootstrap interval `[10.60, 24.31]`). Central 50%, 80%, and 90%
interval coverage is only `34.2%`, `52.6%`, and `62.4%`, respectively. The
diagnostic reads residuals and posterior moments directly and does not invert
stored stable probabilities.

## Compute ceiling

`FixedKernelResidualGP` is lazy: numerical factorization occurs during
`prediction_seconds`, not `posterior_fit_seconds`. After correcting this timing
semantics, full-history posterior update plus prediction is only `0.259%` of
the diagnostic trace wall time on average at \(B=8\). The median standalone
cost of all eight prefix Cholesky factorizations is about 50 microseconds per
system. Phase-diagram work, subprocess boundaries, and evaluation dominate.

A dedicated oracle-blind long-archive panel subsequently selected the three
longest eligible exact systems: Fe--S (46), Fe--Zr (44), and Ni--S (44). A
checkpointed full-history B40 replay gives maximum GP numerical fractions of
`0.738%`, `0.721%`, `0.776%`, and `0.689%` at B8, B12, B24, and B40 across the
three systems. The preregistered Amdahl gate required `9.09%` for even a 10%
ideal end-to-end speedup. It therefore fails decisively: perfect removal of GP
work could provide at most `1.00694x` speedup. See
`docs/WBM_LONG_ARCHIVE_COMPUTE_GATE_2026-07-20.md`.

## Scientific decision

The four-way attribution is now:

| Source | Judgment |
|---|---|
| P3C numerical degeneration | low after failure-capable tests |
| historical data/provenance contract failures | severe, now closed at P0 |
| insufficient systems for the \(B=8,K=2\) P3C effect | low; 32 systems are enough to stop the line |
| insufficient trajectory length for compute scaling | resolved experimentally; B40 compute relevance fails |
| fixed GP / SOAP / noise-model misspecification | high |
| P3C truth-objective mismatch | high |
| theory-claim mismatch | high |

Do not patch or enlarge P3C. The longer-archive gate has failed, so AKSC is not
authorized as this paper's WBM main method. It may only be reconsidered in a
genuinely long-archive setting that first passes the same compute-relevance
gate. The exact-threshold ceiling rerun remains a data-quality cleanup task, not
a mechanism for reversing the research decision. Probability calibration, if
studied independently, must still separate remaining-pool cross-system
calibration from queried-sequence online guarantees and interval coverage.
