# Decision-aware calibration coreset amendment

**Status (2026-07-16): implementation/mechanism GO; real-WBM evidence
NO-GO until P1 and P1.5 pass.** This amendment does not alter the frozen WBM
pool, cost protocol, or execution gates. It replaces CAW-Joint as the proposed
research hypothesis; the tag `caw-method-no-go-2026-07-15` remains immutable.

## Why the method changed

The corrected CAW-Joint study did not beat the best independent acquisition and
retention baseline. Adding scenarios or boundary heuristics would optimize a
stress test rather than answer a durable scientific question. The new primary
question is narrower and falsifiable:

> Can a size-K decision-aware calibration coreset preserve full-history
> residual calibration at lower online cost than full-history GP updates?

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

`src/evimem/matmem/wbm_secure.py` remains the sole real-WBM evaluator. The
generic scalar-min transition is now named `SyntheticMinHullEngine` and is
restricted to synthetic controls. No second WBM vault, loader, hull reviser, or
in-process formal runner was introduced.

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

## Survival-conditioned acquisition

A base policy first proposes L candidates. For each proposed candidate x, S
residual fantasies are sampled from the current bounded posterior. Each fantasy
is passed to the same streaming preview and receives only compression-surviving
utility:

```text
a_t(x) = a_base(x)
       + beta E_r[F_t(PreviewAdmit(M_t, m(x,r))) - F_t(M_t)].
```

Rejected and redundant fantasies receive zero bonus. With `beta=0`, the code
returns the base ranking object verbatim; this is a tested exact-emulation gate.
The old two-scenario heuristic is explicitly named
`LegacyTwoScenarioAcquisition` and is not the proposed method.

## Complexity

For one shared WBM protocol, bounded GP fitting is O(K^3). The batched
single-witness utility matrix is O(nK) kernel work after embeddings are cached.
Survival reranking is O(L S n K) in the direct implementation. These are
working-set computation costs; the immutable archive remains available for
auditing and offline baselines.

## Failure-capable tests

The current suite checks:

- hull update precedes retention in both synthetic and secure WBM runners;
- utility shape, non-negativity, monotonicity and diminishing returns;
- one-swap streaming output matches exhaustive optimization over the K+1 union;
- a below-threshold marginal gain is rejected;
- protocol-incompatible cards do not enter the GP posterior;
- posterior samples are seed-deterministic;
- zero survival weight returns the exact base ranking;
- a redundant fantasy receives zero bonus and is never admitted;
- WBM action serialization, oracle non-interference and three-hull isolation
  remain intact.

The full repository currently passes 198 tests. Test count is engineering
evidence only, not WBM scientific evidence.

## Synthetic calibration-compression smoke

Command:

```powershell
conda run --no-capture-output -n llm python `
  tools/run_calibration_coreset_smoke.py `
  --seeds 3 --candidates 64 --capacity 4
```

This deterministic smoke writes no artifact and is not a materials result.
Mean values across three paired seeds were:

| Scenario | Strategy | Brier | NLL | Online seconds |
|---|---|---:|---:|---:|
| recurrence | full history | 0.01394 | 0.04057 | 1.198 |
| recurrence | FIFO K=4 | 0.01944 | 0.05616 | 0.592 |
| recurrence | diversity K=4 | 0.01394 | 0.04057 | 0.701 |
| recurrence | decision-aware K=4 | 0.01394 | 0.04057 | 0.948 |
| IID negative control | full history | 0.33897 | 1.39685 | 1.309 |
| IID negative control | FIFO K=4 | 0.40831 | 2.13133 | 0.622 |
| IID negative control | diversity K=4 | 0.41338 | 2.29240 | 0.798 |
| IID negative control | decision-aware K=4 | 0.31424 | 1.20559 | 0.947 |

Interpretation is deliberately limited. In recurrence, decision-aware K=4
matches full-history calibration and is about 21% faster, but diversity also
matches it and is faster than the decision-aware method. The IID behavior warns
that the frozen GP can overfit full history; it is not evidence of a universal
coreset advantage. The smoke therefore supports continued WBM testing, not a
paper claim.

## Remaining gates

1. Finish human license/redistribution decisions.
2. Resolve or formally scope the unavailable independent candidate-level
   MP2020-corrected WBM summary.
3. Complete relaxed-structure canonicalization, cell-invariant matching,
   prototype clustering and WBM-MP overlap audit.
4. Pass P1 numerical/identity parity and the frozen P1.5 support audit.
5. Freeze GP hyperparameters and the new method configuration only on isolated
   calibration chemical systems.
6. Replace the engineering-only values in `configs/wbm_calibration.json` with
   calibration-system estimates and set execution authorization only after an
   independent gate review. The allow-listed WBM subprocess implementation is
   present, but has not been used for a comparative run.
7. Run the preregistered minimal matrix only after all preceding gates pass.

The primary real-data GO condition is non-single-point dominance over FIFO,
diversity and reservoir at equal K while approaching full-history Brier/NLL at
lower measured online cost. Survival-conditioned acquisition is an additional
GO only if it improves the same base policy over multiple pools and K values.
If diversity matches calibration more cheaply, or the advantage appears only
under one synthetic recurrence pattern, the proposed paper remains NO-GO.
