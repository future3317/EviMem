# Decision-aware calibration coreset amendment

**Status (2026-07-16): engineering P1/P1.5 pass; preliminary matched-trace
mechanism signal; claim-grade evidence remains NO-GO.** This amendment does not alter the frozen WBM
pool, cost protocol, or execution gates. It replaces CAW-Joint as the proposed
research hypothesis; the tag `caw-method-no-go-2026-07-15` remains immutable.

## Why the method changed

The corrected CAW-Joint study did not beat the best independent acquisition and
retention baseline. Adding scenarios or boundary heuristics would optimize a
stress test rather than answer a durable scientific question. The new primary
question is narrower and falsifiable:

> Can a size-K decision-aware calibration coreset create a non-dominated
> calibration--compute trade-off relative to simple bounded selectors and full
> history?

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

`src/evimem/matmem/wbm_secure.py` remains the sole real-WBM evaluator. Obsolete
scalar-min synthetic hull code and the older in-process evaluator were removed;
there is no fallback WBM vault, loader, hull reviser, or formal runner.

## Residual posterior

`FixedKernelResidualGP` uses a fixed Matérn-5/2 or RBF kernel over normalized
SOAP-like embeddings. Kernel, length scale, signal and noise parameters must be
frozen using isolated calibration chemical systems. Evaluation systems never
optimize a hyperparameter. Protocol-incompatible witnesses are omitted rather
than silently reusing their residuals.

For query u, the posterior induces a stable probability and asymmetric Bayes
risk

```text
R_t(u) = min(c_FS [1-p_t(stable|u)], c_FU p_t(stable|u)).
```

## Fixed calibration utility

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

For one shared WBM protocol, bounded GP fitting is O(K^3). The batched
single-witness utility matrix is O(nK) kernel work after embeddings are cached.
Survival reranking is O(L S n K) in the direct implementation. These are
working-set computation costs; the immutable archive remains available for
auditing and offline baselines.

## Failure-capable tests

The current suite checks:

- hull update precedes retention in the secure WBM runner;
- utility shape, non-negativity, monotonicity and diminishing returns;
- one-swap streaming output matches exhaustive optimization over the K+1 union;
- weighted baseline risk uses the same query weights and units as gains;
- joint-posterior-risk one-swap matches manual random-instance enumeration;
- GP-variance one-swap matches manual random-instance neighborhood enumeration;
- facility and joint diagnostics score the identical candidate subsets;
- a below-threshold marginal gain is rejected;
- protocol-incompatible cards do not enter the GP posterior;
- posterior samples are seed-deterministic;
- zero survival weight returns the exact base ranking;
- a redundant fantasy receives zero bonus and is never admitted;
- WBM action serialization, oracle non-interference and three-hull isolation
  remain intact.

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
5. Freeze GP hyperparameters and the new method configuration only on isolated
   calibration chemical systems.
6. Replace the engineering-only values in `configs/wbm_calibration.json` with
   calibration-system estimates before any claim-grade comparison. The
   allow-listed WBM subprocess has been exercised only by the small engineering
   pilots described in the companion result document.
7. The GP-variance one-swap baseline is implemented and has passed the first
   matched-action cell. Run only a small frozen capacity grid before deciding
   whether a claim-grade matrix is justified. Survival-conditioned acquisition
   is paused after negative B4/K4 and B8/K2 engineering results.
8. The exact-system grid, trace-prefix reuse, prequential causal metrics and
   system-clustered bootstrap are implemented. The frozen oracle-blind manifest
   contains 16 systems and 334 candidates. Paper-level inference remains
   blocked until non-inferiority margins are frozen outside evaluation systems.

The primary real-data GO condition is a non-single-point region that improves
over FIFO, diversity and GP variance at equal K and is strictly non-dominated
on the calibration--compute Pareto frontier. Survival-conditioned acquisition
is not part of the primary GO decision.
If diversity matches calibration more cheaply, or the advantage appears only
under one synthetic recurrence pattern, the proposed paper remains NO-GO.
