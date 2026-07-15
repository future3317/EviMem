# WBM pilot execution feasibility table

**Status:** frozen before WBM adapter implementation or outcome access. This
table governs the 12 pools of 256 candidates defined in the preregistration.
It prevents slow or unfavorable cells from being removed after inspection.

## Execution units and deterministic resource contract

Three units are deliberately distinct:

1. **Policy execution** creates an oracle-blind action/hull/discovery trace.
   Activation price and cache timing never alter acquisition and therefore do
   not create another policy execution.
2. **Physical cost measurement** deterministically replays a frozen trace under
   one access implementation and one cold/warm cache state. It cannot select a
   different action.
3. **Economic-price/utility recomputation** is offline arithmetic over a trace
   and measured cost vector for one price and \((\alpha,\beta,\gamma)\); it does
   not invoke the policy, oracle, retrieval system, or predictor.

An atomic policy or physical run has a hard limit of 4 wall-clock hours, 32 GiB
resident memory, 8 CPU threads, one fixed GPU allocation, and the frozen batch
size/software manifest. Reaching either limit yields `resource-censored`:
terminate cleanly, retain the partial trace and resource log, and report the
cell. Do not replace it, change \(B\) or \(K\), increase resources, or substitute
a neighboring result.

The first pilot has no approximate retention solver. Any later approximation
must be preregistered and named separately (for example, `CAW-Joint-Approx`);
it cannot be merged with exact CAW-Joint or used to fill a censored exact cell.

## Exact CAW-Joint enumeration

For a streaming set containing at most \(K+1\) witnesses, the exact solver
tests every legal subset of size at most \(K\):

\[
S_K=2^{K+1}-1.
\]

With pool size \(N=256\), the preregistered upper count of subset-potential
evaluations for one run is

\[
E(B,K)=S_K\,B(2N-B+2).
\]

This is an evaluation count, not a runtime estimate; each potential itself
scans query-witness pairs. Exact CAW-Joint is therefore eligible only for
\(K\in\{0,1,2\}\). Values at larger \(K\) document why those cells are excluded
before outcomes, rather than being post-hoc omissions.

| \(B\) | \(K=0\) | \(K=1\) | \(K=2\) | \(K=4\) | \(K=8\) | \(K=16\) |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 12,225 | 36,675 | 85,575 | 378,975 | 6,246,975 | 1,602,342,975 |
| 50 | 23,200 | 69,600 | 162,400 | 719,200 | 11,855,200 | 3,040,847,200 |
| 100 | 41,400 | 124,200 | 289,800 | 1,283,400 | 21,155,400 | 5,426,339,400 |
| 200 | 62,800 | 188,400 | 439,600 | 1,946,800 | 32,090,800 | 8,231,258,800 |

## Cell eligibility

Every action label is evaluated offline at activation prices `0`, `p50`, and
`p90`; prices do not create execution cells. Cold/warm states create physical
replays, not new action traces. `Eligible` means the canonical trace must be
attempted under the fixed resource cap; it does not guarantee completion.

| Policy family | \(K=0\) | \(K=1\) | \(K=2\) | \(K=4\) | \(K=8\) | \(K=16\) | \(K=\infty\) |
|---|---|---|---|---|---|---|---|
| Uncertainty x 4 access strategies | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible |
| CAL-style hull-aware GP x 4 access strategies | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible |
| Compatible residual x 4 access strategies | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible |
| Seeded random x 4 access strategies | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible | Eligible |
| Exact CAW-Joint | Eligible | Eligible | Eligible | Not computationally eligible | Not computationally eligible | Not computationally eligible | Undefined / not eligible |

For the four factorized policy families, `full-history` ignores finite \(K\) in
its evidence view, so it executes once per `(acquisition,pool,B,seed)` and all
finite-\(K\) labels reference that checksum. Free and costed query-specific
top-\(K\) share one trace when cost is not in acquisition. At \(K=\infty\),
formally identical views reference one canonical trace after checksum equality.
Seeded random executes once per `(pool,B,seed)` because it is residual-blind;
access/\(K\) labels are deterministic state-and-cost replays. A failed checksum
invalidates reuse and stops execution for an asymmetry audit.

## Aggregate execution and reuse table

The nominal un-reused matrix contains 24,192 deterministic labels and 161,280
random labels before CAW-Joint. The frozen reuse rules above reduce executions
without deleting any reported label. The utility grid has
\(3\times4\times4=48\) weight triples; every access-cost label is evaluated for
two cache states and three frozen prices.

| Strategy class | Independent action traces | Physical cost replays | Offline utility cells | Frozen reuse rule |
|---|---:|---:|---:|---|
| Three deterministic acquisitions x evidence policies | 2,160 | 4,320 | 622,080 | per acquisition: persistent \(7K\) + top-\(K\) \(7K\) + one full-history trace |
| Matched on-demand reconstruction | 0 | 2,016 | 290,304 | aliases the 1,008 persistent traces; replay changes access costs only |
| Seeded random, 20 seeds | 960 | 28,800 | 4,147,200 | one action trace per pool/budget/seed; replay 15 distinct evidence views x two cache states |
| Exact CAW-Joint, \(K=0,1,2\) | 144 | 288 | 41,472 | one trace per pool/budget/eligible \(K\); prices offline |
| **Total** | **3,264** | **35,424** | **5,101,056** | no outcome-dependent reuse |

Here `physical cost replays` includes both cold and warm cache. Matched
reconstruction is required for the three primary deterministic acquisitions;
random remains an evidence-blind cost diagnostic. Utility rows are stored in
columnar form and generated from checksummed trace/cost inputs.

## Aggregate resource envelope

The estimates below are conservative engineering ranges for \(N=256\). After
the five infrastructure gates, calibration/engineering-system p90 timings are
inserted into the fixed counts above as a feasibility check; evaluation-pool
outcomes are not accessed and the hard limits cannot be raised.

| Work class | Expected CPU hours | Expected GPU hours | Expected disk/log |
|---|---:|---:|---:|
| Predictor/SOAP/index engineering cache | 50--200 | 0--24 | at most 250 GiB cache; 10 GiB logs |
| 3,120 non-CAW action traces | 100--400 | 0 | 25 GiB traces/logs |
| 144 exact CAW-Joint traces | 150--576 | 0 | 5 GiB traces/logs |
| 35,424 physical cost replays | 150--600 | 0 | 40 GiB summaries/logs |
| 5,101,056 offline utility rows | 5--20 | 0 | 10 GiB columnar output |
| **Expected total** | **455--1,796** | **0--24** | **at most 340 GiB including caches** |

The global hard envelope is 4,096 CPU-hours, 128 GPU-hours, 500 GiB total disk,
150 GiB logs, and 14 calendar days. Before any comparative matrix run, stop if
engineering p90 projection exceeds any global limit. During execution, schedule
canonical cells by SHA-256 of the complete cell key; when a global limit is
reached, stop and label every unstarted cell `global-resource-censored` in that
fixed order. Retain completed and partial negative results. Do not expand the
envelope, preferentially run favorable cells, or fill censored cells with an
approximation.

## Pre-execution gates

No policy cell may run until all five gates pass:

1. license, release, checksum, canonicalization, and pool-selection audit;
2. MP-only initial causal hull construction with no WBM outcome exposure;
3. oracle-isolated single-candidate WBM reveal;
4. frozen prediction lookup and deterministic SOAP-cache identity tests;
5. zero-cost persistent/on-demand exact-emulation traces with zero mismatches.

Failure of a gate blocks the matrix. It is not a resource-censored cell and
cannot be bypassed by running a smaller budget or capacity.
