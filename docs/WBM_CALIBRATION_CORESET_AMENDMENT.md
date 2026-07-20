# P3C: Proper Posterior-Projection Coreset amendment

For current method status and the complete DACC--P3C--AKSC experiment chain,
read `docs/EXPERIMENT_LEDGER.md` first. This file specifies the P3C iteration;
it does not authorize further P3C tuning.

**Status (2026-07-19): P3C P0, the eight-system matched-action P1 diagnostic,
and the frozen reference/path/selection follow-up pass their execution gates;
paper-level evidence remains NO-GO.**
The former facility DACC and joint-self-risk objectives are retained only as
diagnostic comparators. Their paused grid outputs are immutable and must not be
continued or relabeled as P3C evidence.

**Replication update (2026-07-20).** The earlier SOAP cache used DFT-relaxed
structures and is not valid closed-loop evidence. After rebuilding from the
official `org` initial structures, two disjoint panels (32 exact systems, 683
candidates) show no P3C Brier advantage and worse mean CRPS, log loss, RMSE,
Gaussian NLL, and runtime than GP variance. The implementation is non-degenerate,
but the P3C main hypothesis is now a method-level NO-GO; details are in
`P3C_FAILURE_ATTRIBUTION_2026-07-20.md`.

## Why the method changed again

The corrected CAW-Joint study did not beat the best independent acquisition and
retention baseline. The subsequent facility DACC exposed a deeper problem: it
minimized each candidate posterior's own asymmetric Bayes risk. A realized
residual can therefore be rewarded merely for making the posterior more
extreme, even when it moves in the wrong direction relative to all legally
available evidence. Positive-part clipping also assigns zero value to
counter-evidence that correctly returns an overconfident posterior toward the
decision boundary.

> Can a size-K working set faithfully compress a fixed, legally observable
> full-evidence posterior under a proper scoring rule while preserving its
> asymmetric stability decision?

Survival-conditioned acquisition is secondary. Its failure must not be hidden
or used to invalidate a successful calibration-compression result.

## State order and sole WBM path

Every round now has one order:

1. record the selected action;
2. reveal one oracle outcome;
3. update the composition-dependent causal hull;
4. build the observable future query pool under the updated hull;
5. update the calibration coreset;
6. refit the residual posterior before the next decision.

`src/matmem/wbm_secure.py` remains the sole real-WBM evaluator. Obsolete
scalar-min synthetic hull code and the older in-process evaluator were removed;
there is no fallback WBM vault, loader, hull reviser, or formal runner.

## Residual posterior

`FixedKernelResidualGP` uses a fixed Matérn-5/2 or RBF kernel over normalized
SOAP-like embeddings. Kernel, length scale, signal and noise parameters must be
frozen using isolated calibration chemical systems. Evaluation systems never
optimize a hyperparameter. Protocol-incompatible witnesses are omitted rather
than silently reusing their residuals.

The three variance terms have different semantics. The signal kernel models the
correlated residual field. `noise_std_ev_per_atom` is a frozen independent
per-candidate model-discrepancy term and remains in the predictive distribution
at an unseen candidate through `WhiteKernel`. A protocol-transfer radius is
attached only to the transported training observation through the GP `alpha`;
`jitter` is numerical stabilization only. Thus transfer uncertainty is not
silently charged again at prediction time. This posterior is a predictive
residual distribution, not a latent-noise-free function posterior.

## Fixed-reference posterior projection

After the new card has been legally revealed and the causal hull updated, set

```text
Z_t = M_{t-1} union {m_t}
Q_t = P(. | Z_t)
P_{t,M} = P(. | M),  M subset Z_t, |M| <= K.
```

Every candidate is compared with the same fixed `Q_t`. P3C selects

```text
M_t* = argmin_M D_S(Q_t || P_{t,M}),
```

where `D_S` is induced by a proper scoring rule. The implementation provides
Bernoulli Brier divergence, Bernoulli log divergence, Gaussian KL, and a
deterministic threshold-weighted CRPS divergence. Reference-posterior decision
difficulty supplies a candidate-independent query weight.

For asymmetric false-stable and false-unstable costs, a candidate probability
is evaluated under the fixed reference probability. Its decision regret is the
excess reference expected cost over the reference Bayes action. Optional
decision-regret and log-divergence constraints are therefore fixed-reference
constraints, not self-confidence bonuses.

At saturation, `PosteriorProjectionOneSwapPlanner` exactly evaluates rejection
and all `K` drop-one subsets of `Z_t`. `ExactArchivePosteriorProjectionPlanner`
enumerates every archive subset up to capacity and reports the online
irreversibility gap, archive reactivation, and retained-residual selection bias.
For `B<=12,K<=4` this is at most 794 subsets.

### What proper projection does and does not imply

Strict propriety is relative to the observable reference posterior, not to the
unknown causal outcome distribution. Let `q` be the reference stability
probability, `p` the projected probability, and `y` the causal label. For Brier
loss,

```text
(p-y)^2 - (q-y)^2 = (p-q)^2 + 2 (p-q)(q-y).
```

P3C minimizes the non-negative projection term, while the reference-alignment
cross term can have either sign. A close projection of a misspecified reference
can therefore be worse against the outcome. The same issue holds for log loss:

```text
ell(p,y) - ell(q,y)
  = KL(Bern(q) || Bern(p)) + (y-q) [logit(q)-logit(p)].
```

There is a second, distinct statistical limitation. P3C observes the realized
residuals in `Z_t` before choosing `M_t = S(Z_t)`, but deployment conditions the
GP only on the retained cards. It uses `P(f | D_M)` rather than the selective
distribution `P(f | D_M, S(D_Z)=M)`. The identity of the evicted card therefore
contains outcome-dependent information that the deployed posterior ignores.
This is not an oracle leak or implementation bug; it is an uncorrected selective-
inference effect. GP variance is much less exposed because its selector depends
on kernel geometry rather than observed residual values.

## Frozen legacy self-risk comparator

The former DACC posterior induces a stable probability and asymmetric self-risk

```text
R_t(u) = min(c_FS [1-p_t(stable|u)], c_FU p_t(stable|u)).
```

For one witness m, define

```text
G_t(u,m) = w_t(u) [R_t(u) - R_t(u|m)]_+,
F_t(M)   = sum_u max_{m in M} G_t(u,m).
```

The weighted prior baseline uses the same units:

```text
R^w_0,t = sum_u w_t(u) R_t(u | empty).
```

The earlier implementation returned an unweighted baseline while reporting a
weighted gain. That did not change facility-location selection, but it made
`baseline - objective` and any normalized diagnostic dimensionally invalid.
The implementation and a failure-capable test now enforce the weighted form.

The baseline posterior and boundary weight are fixed for all witnesses in the
round. A witness cannot become valuable merely because a competitor was added.
Incompatible witnesses have zero gain. `CalibrationUtilityMatrix` stores this
immutable matrix and directly evaluates values and marginal gains.

Because all gains are non-negative, F is normalized, monotone and submodular.
Greedy selection from a complete archive therefore has the usual 1-1/e
cardinality guarantee. Streaming admission is a different, exact statement:
when the current set has K cards, the union with one new card has K+1 possible
size-K subsets. Comparing rejection with each one-swap is exact over this
union, not globally optimal over the full archive. `min_admission_gain` permits
the active set to remain below capacity when a new card is redundant.

These statements remain mathematically correct, but the objective is no longer
paper-facing because it can prefer wrong-direction extreme posteriors. Its
reported quantity is now named `facility_proxy_risk`; it is not the selected
set's actual joint GP risk.

## Objective-fidelity diagnostic

Facility location composes singleton gains and is not the actual risk of a GP
conditioned jointly on a set. The diagnostic comparator therefore evaluates

```text
J_t(M) = sum_u w_t(u) R_t(u | M)
```

for rejection and every legal one-swap subset in exactly the same streaming
neighborhood. `JointPosteriorRiskOneSwap` chooses the minimum `J_t` subset. It
has no submodular or global-optimality guarantee and is not silently substituted
for DACC. Every round records both objectives, Spearman correlation, selector
agreement and DACC regret relative to the minimum neighborhood `J_t`.

## Paused survival-conditioned diagnostic

A base policy first proposes L candidates. For each proposed candidate x, S
residual fantasies are sampled from the current bounded posterior. Each fantasy
is passed to the same streaming preview and receives only compression-surviving
utility:

```text
a_t(x) = a_base(x)
       + beta E_r[F_t(PreviewAdmit(M_t, m(x,r))) - F_t(M_t)].
```

The current unnormalized implementation is retained only for negative-result
reproduction and is disabled by default in the engineering runner. Its base
uncertainty score and pool-summed utility have different scales, its fantasy
call index is not a true round index after saturation, and the subprocess view
cannot reconstruct a composition-dependent fantasy hull. No search over beta,
fantasy count or proposal size is authorized. A future one-time gate, if run,
must use calibration-frozen beta and the normalized form

```text
a_t(x) = a_base(x) + beta / C(x) * E[Delta F_t(x)] / (R^w_0,t + epsilon).
```

Rejected and redundant fantasies receive zero bonus. With `beta=0`, the code
returns the base ranking object verbatim; this remains a tested emulation gate.
The discarded two-scenario CAW heuristic is available only from the frozen
historical tag recorded in `RESEARCH_ITERATION_HISTORY.md`.

## Complexity

For one shared WBM protocol, each online candidate uses a GP fit with at most
`K` witnesses and the fixed reference uses `K+1`; direct exact drop-one is
`O(n K^4)` with repeated small fits. This is acceptable for `K<=4`, but the
planned optimized implementation precomputes `K_ZZ` and `K_UZ` once and uses
small Cholesky downdates. Archive-exact evaluation intentionally enumerates at
most 794 subsets and is a diagnostic, not a free online operation.

## Failure-capable tests

The current suite checks:

- hull update precedes retention in the secure WBM runner;
- utility shape, non-negativity, monotonicity and diminishing returns;
- one-swap streaming output matches exhaustive optimization over the K+1 union;
- weighted baseline risk uses the same query weights and units as gains;
- the legacy single-witness helper reuses the general one-card GP path, with
  randomized equivalence regression cases;
- a wrong-direction extreme posterior can beat the correct direction under
  legacy self-risk but cannot beat it under proper Brier/log divergence;
- Brier, log, Gaussian-KL and threshold-weighted-CRPS P3C variants select the exact
  optimum in the same drop-one union;
- archive P3C enumerates every subset up to capacity;
- formal grid execution verifies the registered config SHA, internal GP-config
  SHA, disjoint calibration IDs and calibration-freeze manifest SHA;
- joint-posterior-risk one-swap matches manual random-instance enumeration;
- GP-variance one-swap matches manual random-instance neighborhood enumeration;
- facility and joint diagnostics score the identical candidate subsets;
- a below-threshold marginal gain is rejected;
- protocol-incompatible cards do not enter the GP posterior;
- posterior samples are seed-deterministic;
- different one-swap candidate sets produce numerically distinct real-GP mean
  and variance vectors and a non-zero P3C objective range in a constructed case;
- P3C and GP variance select different subsets in a frozen constructed case,
  excluding an accidental shared-objective implementation;
- predictive discrepancy, per-card protocol-transfer uncertainty and numerical
  jitter remain separate variance terms;
- zero survival weight returns the exact base ranking;
- a redundant fantasy receives zero bonus and is never admitted;
- WBM action serialization, oracle non-interference and three-hull isolation
  remain intact.
- reference and projected losses satisfy `L_P-L_G=(L_P-L_Q)-(L_G-L_Q)`;
- all four reference/search combinations are evaluated independently;
- Gaussian-NLL mean and variance Shapley terms add exactly to total NLL change;
- the selection audit detects residual-dependent retention in a constructed case;
- archive diagnostic timings are excluded from online retention time;
- exact-emulation equality ignores non-deterministic timing measurements.

Exact test counts and retired synthetic-smoke values are intentionally not part
of the live method specification. Their conclusions and recovery points are in
`RESEARCH_ITERATION_HISTORY.md`.

## Remaining gates

The detailed gate resolution and first real-data diagnostics are recorded in
`WBM_ENGINEERING_P1_P15_AND_PILOT.md`. Local research use, historical-replay
P1 and frozen-pool engineering support now pass. Remaining claim-grade work is:

1. Complete final human publication/redistribution sign-off.
2. Retain the explicit fixed-historical-pipeline scope; do not claim exact
   reproduction of an unavailable independent official corrected summary.
3. Complete relaxed-structure canonicalization, cell-invariant matching,
   prototype clustering and WBM-MP overlap audit.
4. Freeze claim-grade system count from a declared precision target.
5. Freeze any P3C constraint thresholds only on isolated calibration
   systems. The existing GP parameters are loaded from the immutable disjoint
   calibration manifest; every formal trace records that manifest SHA.
6. The matched-action P1 objective diagnostic is complete: 72/72 ledgers over
   eight exact systems at `B=8,K=2` passed strict action parity and numerical
   validation. P3C-Log improved mean NLL relative to GP variance, but its CRPS,
   Brier and log-loss confidence intervals crossed zero. P3C-Gaussian-KL was
   significantly worse on Brier and log loss. No P3C variant established the
   frozen multi-metric GO condition.
7. Union-reference divergence, reference decision regret, archive gap,
   reactivation and residual-selection bias are now reported. Online
   irreversibility is observable (16--22 reactivation rounds depending on the
   P3C score), while the theory-fixed zero-regret constraint was inactive and
   selected exactly the same sets as unconstrained P3C-twCRPS in all 64 rounds.
8. The exact-system grid, trace-prefix reuse, prequential causal metrics and
   system-clustered bootstrap are implemented. The frozen oracle-blind manifest
   contains 16 systems and 334 candidates. On 2026-07-19 the isolated
   calibration run froze the stated GP configuration and set both causal Brier
   and log-loss non-inferiority margins to zero because GP-variance did not
   show a positive paired-bootstrap lower-bound improvement over FIFO. The
   subsequent grid was manually paused after one complete physical group and a
   partial second group; no paused partial output may enter an inference or
   future resumed comparison.

## Frozen reference/path/selection diagnosis

The authoritative follow-up is the external `v5` result:

```text
E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-reference-path-selection-b8-k2-v5\summary.json
SHA-256 0d25f251a1d1ede6dc2b63c5e2ed7c8782fde716f984b45d9c93060ea4b2f9b3

E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-reference-path-selection-b8-k2-v5\p1-decomposition-analysis.json
SHA-256 149ed9562d5a6c6d550c550f99711542df2733c58236cc3f7daf802a9461ef1d
```

All 48 runs (eight exact systems by six strategies), 1,920 factorial evaluator
snapshots, and 168 P3C-Log selection records are complete. Matched frozen-
acquisition action parity passes and the deterministic analysis reports no
quality issue. Recomputing the analysis from the frozen summary reproduces the
file exactly. Failed `v2` and `v3` directories contain no valid summary and are
excluded. `v4` has identical scientific traces, active sets, hull checksums,
selections, and losses, but its lazy-GP timing definition is incomplete; only
`v5` is citable.

For P3C-Log, the union reference has positive system-macro headroom over GP
variance on causal Brier and log loss in 6/8 systems, with positive headroom in
28/64 and 30/64 rounds. Mean headroom is `0.000665` and `0.002414` respectively.
This passes Gate A, but the signal is intermittent rather than roundwise.
Headroom-weighted online recovery is `0.715` for Brier and `0.672` for log loss.
Under the stronger archive-reference/archive-search diagnostic, recovery falls
to `0.484` and `0.428`, and median positive-headroom recovery is zero.

The full causal-loss factorial is:

| Variant | Brier UO | Brier UA | Brier AO | Brier AA | log UO | log UA | log AO | log AA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3C-Log | 0.076162 | 0.076024 | 0.076482 | 0.075625 | 0.261011 | 0.260714 | 0.262068 | 0.261074 |
| P3C-Brier | 0.077517 | 0.077096 | 0.076741 | 0.075776 | 0.264602 | 0.263524 | 0.262942 | 0.262268 |
| GP variance | 0.075365 | 0.075365 | 0.075365 | 0.075365 | 0.258965 | 0.258965 | 0.258965 | 0.258965 |

`U/A` in the first position denotes union/archive reference and in the second
position online/archive subset search. Archive search under the same union
reference improves P3C-Log by only `-0.000138` Brier and `-0.000298` log loss;
exact sign-flip p-values are `0.65625` and `0.90625`. Removing the largest
contributing system reverses both mean effects. Gate C therefore does not
establish predictive value for archive reactivation.

P3C-Log minus GP variance has mean NLL difference `-0.248628`, but its median is
only `-0.019457`, the exact sign-flip p-value is `0.21875`, and the wins/ties/
losses count is 4/1/3. Fe--Y supplies 50.91% of gross improvement and the top
three systems supply 98.17%; the mean after removing Fe--Y is `-0.129183`.
Symmetric counterfactual Shapley attribution assigns `-0.088527` to posterior
mean and `-0.160101` to posterior variance, with maximum additivity error
`5.6e-17`. The aggregate NLL signal is therefore variance-driven and
influential-system-sensitive, not a stable cross-system advantage.

The descriptive P3C-Log retention regression has in-sample ROC-AUC `0.8073`.
Standardized coefficients for absolute residual and residual sign are only
`+0.0127` and `-0.0158`, while reference-mean and reference-variance influence
are `+0.8418` and `+1.0096`. P3C-Brier shows the same qualitative pattern.
Retention is outcome-dependent through posterior influence, but the audit does
not show simple raw extreme-residual chasing and does not prove that selective
misspecification causes the probability-loss gap.

Gate D passes as a measurement gate: union-reference fit, online projection,
archive-reference fit, archive subset enumeration, prequential evaluation, and
hull update are separately timed, and archive diagnostics are excluded from
online retention time. It is not a calibration--compute Pareto result.

The resulting decision is unchanged. Gaussian-KL is frozen as a negative
variant; twCRPS remains diagnostic-only; its decision-safe constraint was
inactive. Gate A and Gate D pass, Gate B remains descriptive without a post-hoc
cutoff, and Gate C does not show a robust causal-metric advantage. P3C solves a
well-defined posterior-compression problem, but the data do not establish that
this is the correct proxy for causal stability calibration.

The P1 result validates execution and exposes a real archive/online gap, but it
does not validate superiority: only eight systems and one `B,K` point were
tested, and P3C's probability metrics do not satisfy the frozen zero-margin
non-inferiority gate. Work is paused before any new multi-point grid. A future
restart requires an explicit frozen decision on whether the weak CRPS/NLL signal
justifies the predeclared non-single-point grid; kernel, acquisition, hull,
fantasy, and evaluation-system-dependent parameter changes remain forbidden.
Survival and MADE remain outside this gate.
