# P3C failure attribution: code, data, method, and theory

**Status (2026-07-20).** This is an exploratory technical diagnosis, not a
paper-level GO result. The method, GP parameters, acquisition trace, causal-hull
evaluator, and `B=8,K=2` cell were not tuned during this analysis.

## Technical summary

P3C is not failing because its selector is numerically constant or because all
candidate subsets reuse one GP posterior. Failure-capable tests and real WBM
traces exclude those implementation collapses. A real data-environment defect
was found: the runner read the parity manifest for predictions but recomputed
WBM oracle energies with modern pymatgen. Fourteen Au--Te candidates differed
from the frozen 2023.5.10 parity environment by 0.105--0.241 eV/atom and three
stable/phase labels changed. The runner now reconstructs every executed phase
entry from the frozen parity corrected formation energy and hard-fails on
missing coverage.

After that repair, a second data-semantic defect was found: policy-facing SOAP
had been built from the DFT-relaxed CSE structure. The official WBM compiler
maps raw `org` to `Key.init_struct` and `opt` to `Key.final_struct`; all 334
frozen structure hashes changed when SOAP was rebuilt from `org`. The earlier
energy-corrected result is therefore retained but is not claim-grade evidence.

Two disjoint, initial-structure-correct panels now cover 32 exact chemical
systems and 683 candidates. In the first panel P3C-Log has essentially zero
Brier advantage and worse CRPS/RMSE; in the fresh next-16 confirmation panel it
is worse on every mean metric, including significantly worse Gaussian NLL. The
descriptive 32-system aggregate has Brier difference `-0.000003`, while CRPS,
log loss, RMSE, NLL, and compute all favor GP variance. The leading explanation
is now method/theory mismatch, not code degeneration or WBM data scarcity.

## What was measured

- Statistical unit: exact chemical system.
- Systems: 8 binary and 8 ternary systems; no quaternary-or-higher claim.
- Candidates: all 334 IDs in the frozen exact-system manifest.
- Matched action trace: frozen acquisition, `B=8`, `K=2`.
- Comparator: GP-variance one-swap under the same fixed GP.
- Oracle source: `parity_corrected_formation_energy_ev_per_atom` from the
  merged 2023.5.10/modern parity audit.
- Primary evaluator: round-averaged prequential causal metrics, macro-averaged
  over systems.

The two structure-correct immutable results are outside Git at:

```text
E:\DATA\EviMem-RL\outputs\exploratory\wbm-p3c-16sys-b8-k2-initial-org-v3
E:\DATA\EviMem-RL\outputs\exploratory\wbm-p3c-next16-b8-k2-initial-org-v1
```

The preceding relaxed-structure result remains immutable evidence of the data
audit, but it must not be used for a closed-loop scientific claim.

## Structure-correct 16-system replication

| P3C minus GP variance | Mean | 95% system bootstrap | Wins / losses | Exact sign-flip p |
|---|---:|---:|---:|---:|
| P3C-Log CRPS | +0.001840 | [-0.001244, +0.005762] | 8 / 6 (2 ties) | 0.413 |
| P3C-Log Brier | -0.000096 | [-0.009454, +0.009146] | 8 / 6 (2 ties) | 0.985 |
| P3C-Log log loss | -0.005240 | [-0.029336, +0.019005] | 9 / 5 (2 ties) | 0.685 |
| P3C-Log RMSE (eV/atom) | +0.002536 | [-0.002158, +0.009719] | 8 / 6 (2 ties) | 0.703 |
| P3C-Log Gaussian NLL | +0.017587 | [-0.207467, +0.270279] | 8 / 6 (2 ties) | 0.896 |
| P3C-Brier CRPS | +0.001789 | [-0.001295, +0.005753] | 8 / 6 (2 ties) | 0.428 |
| P3C-Brier Brier | -0.001068 | [-0.010754, +0.008598] | 8 / 6 (2 ties) | 0.840 |
| P3C-Brier log loss | -0.007880 | [-0.032208, +0.017007] | 10 / 4 (2 ties) | 0.556 |

Negative differences are improvements. Mean wall time was 191.6 s for GP
variance, 212.6 s for P3C-Log, and 210.7 s for P3C-Brier. No current P3C
variant forms a demonstrated calibration--compute Pareto improvement.

## Fresh next-16 confirmation and 32-system aggregate

The next panel uses hash ranks 9--16 in each binary/ternary stratum, contains
349 candidates, and is disjoint from the first panel. No parameter, method, or
metric was changed. P3C-Log minus GP variance is:

| Metric | Fresh next-16 mean [95% CI] | Descriptive pooled 32-system mean [95% CI] |
|---|---:|---:|
| CRPS | +0.001081 [-0.000221, +0.002520] | +0.001461 [-0.000226, +0.003538] |
| Brier | +0.000090 [-0.002210, +0.002809] | -0.000003 [-0.004802, +0.004722] |
| log loss | +0.016463 [-0.008902, +0.059045] | +0.005611 [-0.014179, +0.031187] |
| RMSE (eV/atom) | +0.000085 [-0.002157, +0.002013] | +0.001310 [-0.001381, +0.005120] |
| Gaussian NLL | +0.275486 [+0.020473, +0.631204] | +0.146537 [-0.032742, +0.365553] |

Fresh-panel wall time is 184.6 s for GP variance and 200.4 s for P3C-Log.
The pooled interval is a descriptive system bootstrap, not a retroactively
preregistered confirmatory test. It is nevertheless decisive for method
development: the original Brier/log signal does not replicate.

## Cause attribution

| Possible cause | Current judgment | Evidence |
|---|---|---|
| Numerically degenerate selector | Low | Every real P3C round has non-zero candidate objective range; constructed tests require distinct posterior mean/std vectors, manual optimum agreement, and P3C/GPV subset disagreement. |
| Incorrect oracle-energy environment | Confirmed, fixed | 14 Au--Te energy mismatches and 3 label flips were found; v2 uses parity energies directly and records the parity SHA. |
| Post-DFT geometry leakage | Confirmed, fixed | The old SOAP path read relaxed CSE structures. The new path requires the official `org` field; all selected hashes, cache SHA, parity identity, and runner gate are rebound and failure-tested. |
| Insufficient system-level power | Secondary | There are now 32 disjoint systems. Intervals still permit small effects, but the fresh panel reverses the original probability-metric direction and pooled Brier is essentially zero. More systems cannot rescue a practically meaningful effect if the estimand remains this small. |
| Reference posterior misspecification | High | Proper divergence is proper only relative to the GP reference. The reference has probability headroom, but that does not establish calibration against causal truth. |
| Outcome-dependent post-selection | Medium--high, descriptive | P3C-Log retains cards with mean absolute residual 0.0720 eV/atom versus 0.0461 for evicted cards; the descriptive retention regression has in-sample AUC 0.718. This is a real P3C defect, but selection conditioning is not the preferred repair because the immutable archive still contains every exact residual. |
| Irreversible online search | Secondary | Archive search changes the internal optimum, but causal losses improve only slightly and inconsistently. It is not the main observed bottleneck. |
| Objective/metric mismatch | High | P3C shows weak Brier/log-loss improvement while CRPS and RMSE do not improve. Gaussian-posterior fidelity is not a universal proxy for boundary calibration. |

## WBM has statistical breadth but short trajectories

The frozen exact-system manifest records 801 eligible binary systems and 1,687
eligible ternary systems after the calibration split, all with `N_s >= 16`.
The corrected experiment used only eight systems from each stratum because the
previous grid capped selection at eight per stratum, not because WBM lacks
additional usable systems. No quaternary-or-higher system reaches the frozen
minimum size.

Consequently, another dataset is not required to estimate an exact-system
`B=8,K=2` calibration effect. However, more independent systems do not lengthen
one archive. The corrected panels contain at most a few dozen candidates per
system, and the current full-history posterior occupies only about 0.26% of
wall time at `B=8`. WBM is therefore currently a correctness/calibration
benchmark, not evidence for long-archive compute scaling. A 128-system P3C run
would only estimate the already non-replicating effect more precisely; it is
not the next experiment.

## Historical relaxed-feature decomposition (diagnostic only)

The following decomposition was computed before the `org` correction. It is
retained to explain how the method behaved internally, not as evidence of WBM
closed-loop performance. It must be rerun before any of its numerical values
are cited as a structure-correct result.

For P3C-Log, the union reference has mean headroom over GP variance of
`0.003419` Brier and `0.010579` log loss. It has positive system-level headroom
in 11/16 and 12/16 systems, respectively. Projection recovers 90.6% and 91.3%
of positive headroom when weighted by headroom. Thus the implementation does
solve its stated local projection problem reasonably well.

The remaining gap is scientific alignment. Archive-reference/archive-search
improves Brier from `0.103517` to `0.102228`, but union-reference/archive-search
changes it only to `0.103359`. For log loss, archive search under the union
reference is worse (`0.346246` versus `0.344345`). Reactivation therefore does
not supply a robust causal-metric explanation for current performance.

## The theoretical break

P3C deploys `P(f | D_M)` after choosing `M=S(D_Z)` using observed residuals.
Selection conditioning, `P(f | D_M,S(D_Z)=M)`, would recover only a coarse
encoding of discarded outcomes. It is not the complete target here: the
immutable archive still stores every exact residual, so the statistically most
complete object is `P(f | D_Z)`. The direct repair is therefore not a more
complicated selection-conditioned GP. It is to stop deleting outcome
contributions and compress only the posterior representation and linear-algebra
state.

In addition, minimizing a proper divergence `D(Q || P_M)` guarantees closeness
to `Q`, not to the unknown causal outcome distribution. The corrected WBM result
is consistent with this distinction: reference probability headroom exists and
is mostly recovered, yet cross-system superiority remains unsupported.

## Literature-aligned stronger route

The next method should separate two jobs that P3C currently mixes. The working
name is **Archive-Conditioned Kernel Sketch with Sequential Calibration
(AKSC)**:

1. **Posterior compression.** Use an outcome-independent kernel dictionary,
   ridge-leverage/Nyström sketch, posterior-variance sampling, or a variational
   sparse-GP objective. Every revealed residual updates low-dimensional
   posterior sufficient statistics; the dictionary bounds function
   representation, not which paid outcomes continue to count. This directly
   targets preservation of the kernel operator, posterior covariance, or ELBO
   and avoids residual-dependent deletion.
2. **Sequential calibration.** Keep revealed residuals in the immutable archive
   and calibrate probabilities or intervals with an explicit online or
   non-exchangeable calibration layer. A bounded kernel dictionary may reduce
   computation, but it should not silently define which outcomes count for
   calibration.

Relevant anchors are Calandriello et al., *Gaussian Process Optimization with
Adaptive Sketching: Scalable and No Regret* ([arXiv:1903.05594](https://arxiv.org/abs/1903.05594));
Calandriello et al., *Distributed Adaptive Sampling for Kernel Matrix
Approximation* (SQUEAK; [arXiv:1803.10172](https://arxiv.org/abs/1803.10172));
Ketenci et al., *A Coreset-based, Tempered Variational Posterior for Accurate
and Scalable Stochastic Gaussian Process Inference*
([arXiv:2311.01409](https://arxiv.org/abs/2311.01409)); Farinhas et al.,
*Non-Exchangeable Conformal Risk Control*
([arXiv:2310.01262](https://arxiv.org/abs/2310.01262)); and Bhatnagar et al.,
*Improved Online Conformal Prediction via Strongly Adaptive Online Learning*
([arXiv:2302.07869](https://arxiv.org/abs/2302.07869)). For probability rather
than interval calibration, Gupta and Ramdas, *Online Platt Scaling with
Calibeating* ([arXiv:2305.00070](https://arxiv.org/abs/2305.00070)), is a more
direct anchor than conformal prediction. These works do not prove the proposed
combined architecture, but they show where defensible guarantees currently
live: kernel/operator preservation for compression, online calibration for
probabilities, and conformal methods for coverage or monotone risk control.
Deshpande, Marx, and Kuleshov, *Online Calibrated and Conformal Prediction
Improves Bayesian Optimization*
([arXiv:2112.04620](https://arxiv.org/abs/2112.04620)), is the closest direct
bridge to our sequential setting: it explicitly separates a potentially
misspecified predictive model from online calibration under action-dependent,
non-i.i.d. observations. It still does not justify using an
outcome-selected coreset as if its selection event were ignorable.

## Recommended next experiment

The fresh-panel replication has answered the first question: the small
Brier/log-loss signal does not survive. Stop developing P3C as the main method
and do not spend a 128-system run merely to estimate a near-zero effect more
precisely.

The fixed-GP ceiling audit is reported separately in
`docs/AKSC_CEILING_DIAGNOSTIC_2026-07-20.md`. It finds mean effective dimension
12.42 (all 32 systems above 4), useful full-history Brier/log/CRPS headroom over
the prior, but decisively worse full-history Gaussian NLL and near-zero kernel
Moran autocorrelation. Thus the current GP is neither a clean oracle reference
nor a calibrated probabilistic ceiling.

This recommendation was conditional. The subsequent checkpointed B40 gate
failed: GP numerical work reached only 0.689% of the real round pipeline, far
below the 9.09% Amdahl threshold. Therefore there is no authorized next AKSC
implementation for this WBM paper. The complete closure and artifact inventory
are in `docs/EXPERIMENT_LEDGER.md`.

If a genuinely different, long-archive workload first passes the same gate,
the next implementation there should be a separately named, selection-safe baseline:
an outcome-independent ridge-leverage/Nyström or adaptive kernel dictionary for
GP compression, followed by an explicit online probability/interval calibration
layer that sees the immutable revealed archive. Validate the compression layer
first on posterior mean, covariance, CRPS, and compute under the same 32-system
matched-action protocol; only then add online Platt/calibeating or
non-exchangeable conformal calibration. Do not relabel P3C evidence as evidence
for this new architecture.
