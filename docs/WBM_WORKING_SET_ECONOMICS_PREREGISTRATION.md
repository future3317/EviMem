# WBM working-set economics pilot: preregistered design

Status: design only; no WBM adapter, download, or experiment is authorized by
this document. Target length: at most two manuscript pages.

## Question and utility

The experiment asks whether a persistent certified witness working set is
economically necessary when every DFT result remains in an archive. It does not
optimize the current two-scenario heuristic. For policy `pi`, report the full
cost vector and the preregistered utility

\[
J_{\alpha,\beta,\gamma}(\pi)=D_{final}(\pi)
-\alpha C_{oracle}(\pi)-\beta C_{activate}(\pi)
-\gamma C_{online}(\pi).
\]

`D_final` is final-hull-confirmed discovery count. Costs use two parallel units:

- **oracle/DFT:** one DFT-equivalent unit per WBM energy reveal; a secondary
  atom-count/system-size proxy is reported only if WBM metadata supports it;
- **archive retrieval:** measured milliseconds, bytes read, and records scanned;
- **re-certification:** measured milliseconds per identity, protocol, and hull-
  version check, including cache misses;
- **protocol transport:** measured milliseconds per map/calibration lookup;
  rejected transport remains rejected and receives no residual;
- **scanning/indexing:** distance evaluations, CPU/GPU milliseconds, and peak
  working memory per round;
- **latency:** end-to-end wall-clock seconds and CPU/GPU-hours on one frozen
  hardware/software manifest.

Primary analysis reports the Pareto frontier rather than one convenient set of
weights. Utility weights and latency caps are frozen before evaluation using
only a disjoint 5% engineering/calibration slice. Retrieval/activation settings
are `free=0`, measured median (`p50`), and measured high-but-observed (`p90`);
costs above `p90` are stress tests and cannot establish GO.

## WBM fixed-pool protocol

Before data access, record license decisions, exact WBM/Materials Project
release IDs and checksums, canonical structure identity rules, and excluded
records. The initial hull contains only the frozen MP phase snapshot; no WBM
oracle energy or final-hull label is visible online. Each pool is a fixed set of
WBM structures with observable composition, structure, protocol, frozen model
prediction, and representation. The policy adaptively selects a candidate;
only that candidate's WBM formation energy is revealed and appended to the
archive. The same-system causal hull is then recomputed. Input serialization
order is permuted as an invariance audit, not treated as the discovery order.

Use multiple pools stratified by chemical-system complexity and pool size,
frozen before outcomes are inspected. Pools with insufficient candidates or no
discoveries are retained and reported, not silently replaced. Exact duplicates,
MP overlap, and canonical-group leakage are removed by declared rules.

## Frozen predictor and representation

Primary predictor: the official WBM prediction artifact from a permissively
licensed frozen CHGNet checkpoint, with release/checksum fixed during the audit;
no retraining or selection on pilot outcomes. Replication uses a second official
Matbench Discovery predictor only if its checkpoint and WBM predictions are
reproducible. Structure similarity uses deterministic periodic SOAP, decoupled
from the predictor: cutoff 5 A, `n_max=8`, `l_max=6`, fixed species vocabulary,
and normalized vectors. If these exact artifacts fail the audit, execution is
blocked and the design is amended before seeing results.

## Matrix and policies

Run paired cells

\[
B\in\{25,50,100,200\},\quad
K\in\{0,1,2,4,8,16,\infty\},\quad
c_{act}\in\{0,p50,p90\}.
\]

Every policy receives identical pools, MP hull, predictor, representation,
oracle budget, and revealed history. Required comparisons are:

1. free on-demand full-archive retrieval/scanning;
2. costed on-demand top-`K` retrieval;
3. persistent certified working set;
4. full-history scanning with measured cost;
5. uncertainty acquisition;
6. convex-hull-aware active learning (CAL);
7. protocol-compatible kNN;
8. seeded random;
9. frozen CAW-Joint, without adding scenarios or tuning its heuristic.

The working-set strategy and acquisition rule are factorially separated where
the interfaces permit it. Free on-demand retrieval may emulate any persistent
set; therefore an unexplained information/discovery advantage for persistence
in the zero-cost cell is an implementation-asymmetry alarm, not positive
evidence.

## Pairing, uncertainty, and metrics

Use at least 12 independently defined WBM pools. Deterministic policies use the
same pool directly; stochastic policies use 20 shared seeds per pool. Report
paired policy differences with pool-clustered 95% bootstrap confidence
intervals and unaggregated pool results. No seed or pool may be dropped after
outcomes are observed.

Primary metrics are query-time causal discoveries, final-hull-confirmed
discoveries, invalidated provisional discoveries, unstable/false-stable calls,
DFT-equivalent cost, activation/retrieval/transport/scanning cost, end-to-end
latency, peak memory, and net utility. Also report action disagreement and the
fraction of archived records retrieved or activated.

## Preregistered GO / NO-GO

**Sanity gate:** in the free-retrieval cell, persistence must not obtain an
unexplained discovery advantage over an on-demand policy allowed to emulate its
active set. Any advantage larger than 1 final discovery per 100 queries, or a
paired 95% interval excluding zero, pauses interpretation for an asymmetry audit.

**GO requires all of the following:**

- final discoveries are non-inferior to free on-demand retrieval (paired lower
  95% bound above -0.5 discoveries per 100 queries);
- net utility exceeds the best costed on-demand, CAL, uncertainty, compatible-
  kNN, and other independent acquisition baseline with paired lower 95% bound
  above zero;
- the advantage occurs at measured `p50` or `p90` costs, for at least two
  adjacent `K` values, two oracle budgets, and at least 60% of WBM pools;
- the effect survives final-hull confirmation and is not driven by increased
  provisional invalidation or one chemical-system stratum.

**NO-GO** if uncertainty, CAL, free/costed on-demand retrieval, or another
independent method dominates; if benefit occurs only above observed `p90`
retrieval cost; if it is confined to one hyperparameter/pool; or if causal gains
disappear under the final hull. Only after GO may a MADE experiment be designed.
