# WBM working-set economics pilot: preregistered design

**Status.** Design only. This document does not authorize a WBM adapter,
download, policy run, or MADE experiment. The frozen method-level NO-GO result
remains immutable. Execution may begin only after the data/license audit and
the exact-emulation gate below pass. The cell-level resource contract is in
[`WBM_EXECUTION_FEASIBILITY_TABLE.md`](WBM_EXECUTION_FEASIBILITY_TABLE.md).

## Decision question and cost units

The pilot asks whether keeping a persistent, certified working set has positive
economic value when every revealed DFT result remains in an immutable archive.
It does not tune CAW-Joint or add heuristic scenarios. For policy \(\pi\),

\[
J_{\alpha,\beta,\gamma}(\pi)=D_{\mathrm{final}}(\pi)
-\alpha C_{\mathrm{oracle}}(\pi)
-\beta C_{\mathrm{activate}}(\pi)
-\gamma C_{\mathrm{online}}(\pi).
\]

\(D_{\mathrm{final}}\) is the final-hull-confirmed discovery count.
\(C_{\mathrm{oracle}}\) is WBM energy reveals in DFT-equivalent units.
\(C_{\mathrm{activate}}\) is archive retrieval plus protocol transport, first
certification, and hull-version re-certification. \(C_{\mathrm{online}}\) is
per-round scanning, indexing, serialization, and synchronization. Raw records
scanned, bytes, CPU/GPU time, wall time, and peak memory are also reported.
Non-oracle costs are expressed both in raw seconds and in units normalized by
the corresponding engineering/calibration-system p50. Cold- and warm-cache
normalizers and price schedules are separate and frozen before evaluation.

The utility grid is reported in full, never selected from calibration results:

\[
\alpha\in\{0.01,0.03,0.1\},\qquad
\beta,\gamma\in\{0,0.01,0.03,0.1\}.
\]

Primary summaries are the Pareto frontier, pairwise dominance regions, and
break-even retrieval cost. Component-wise zero, p50, and p90 prices use only
engineering/calibration-system operation timings. Evaluation-pool timings are
reported outcomes and never redefine a price. Values above the frozen p90 are
labelled stress tests and cannot establish GO.

## Leakage-safe fixed pools

Before outcome access, record WBM/Materials Project release IDs, licenses,
checksums, canonicalization, duplicate/MP-overlap exclusions, and the hardware
and software manifest. Group canonical WBM candidates by exact chemical system.
Sort systems by SHA-256 of `release_id || chemical_system` and assign the first
`ceil(0.05 * number_of_systems)` complete systems to calibration; calibration
systems never enter evaluation.

Eligible evaluation systems have at least 256 canonical candidates. Split them
into 2-element, 3-element, and at-least-4-element systems. Compute candidate-
count median separately inside each of those three strata; counts less than or
equal to the median are `small`, and counts above it are `large`, giving six
observable strata. Select the
two systems with the smallest SHA-256 value of `release_id || chemical_system`
in each stratum. Within each selected system, choose exactly 256 candidates by
hash-ranked canonical structure ID. This yields 12 disjoint-system pools
without consulting energies, hull labels, predictor error, or discovery yield.
If a stratum has fewer than two eligible systems, execution stops for a visible
amendment before any evaluation outcome is revealed; pools are not replaced
after inspection. A hash tie is broken by canonical chemical-system string and
then release ID; a structure-hash tie is broken by canonical structure ID and
then immutable source-row ID.

The initial causal hull is built only from the frozen MP snapshot for that
chemical system. Online state exposes composition, structure, protocol, frozen
prediction, SOAP vector, and already revealed history. Selecting a candidate
reveals only its WBM formation energy; the same-system causal hull is then
versioned and recomputed. Final WBM outcomes are evaluation-only. Hash-based
input permutations are an invariance audit, never an oracle ordering.

Pool differences use paired estimates and chemical-system-clustered 95%
bootstrap intervals. If structure prototypes or candidate families connect
systems, the primary uncertainty becomes a hierarchical connected-family /
chemical-system clustered bootstrap; pools are not treated as IID replicates.
Stochastic policies use 20 preregistered shared seeds per pool.

## Frozen predictor, representation, and parameters

The primary predictor is one permissively licensed official WBM prediction
artifact from a frozen CHGNet checkpoint, fixed by release and checksum; there
is no WBM-pilot refitting or model selection. Periodic SOAP is cached with
cutoff 5 Angstrom, `n_max=8`, `l_max=6`, a frozen species vocabulary, and
normalized vectors. The reproducibility baseline is named **CAL-style
hull-aware GP**, not CAL: it does not claim a faithful reproduction of a
specific published CAL implementation. It uses a Matern-5/2 Gaussian process
on normalized SOAP:
residuals are standardized using calibration systems; amplitude, length scale,
and noise are selected by calibration negative log likelihood from
`{0.25,0.5,1} x {0.1,0.3,1} x {1e-6,1e-4,1e-2}`, with lexicographic ties, and
the confidence multiplier is fixed at 1.96. Renaming it to CAL would require a
pre-outcome amendment identifying a verifiable repository, commit, objective,
and complete configuration.

| Quantity | Frozen definition | Provenance |
|---|---|---|
| \(L\) | empirical 0.95 quantile of \(|r_i-r_j|/\max(d_{ij},10^{-8})\) over compatible cross-structure pairs | isolated 5% calibration systems |
| \(q_c\) | finite-sample corrected 0.90 quantile of \([|r_i-r_j|-Ld_{ij}-u_{i\to j}]_+\), by frozen protocol stratum | isolated 5% calibration systems |
| \(I_t^0\) | symmetric radius: finite-sample corrected 0.90 quantile of \(|E_{WBM}-\hat E^0|\) | isolated 5% calibration systems |
| \(\sigma\) | 0.05 eV/atom | external boundary-scale setting |
| \(\omega_0\) | 0.05 | frozen synthetic setting |
| \(c_{\mathrm{FS}},c_{\mathrm{FU}}\) | 5, 1 | external asymmetric decision preference |
| \(\lambda_{\mathrm{disc}},\lambda_{\mathrm{info}}\) | 5, 1 | frozen synthetic CAW-Joint setting |
| \(q_x^-,q_x^+\) | deterministic interval geometry; conflict is 0.5/0.5 | heuristic, not a probability calibration |
| scenario margin | 0.01 eV/atom | frozen synthetic CAW-Joint setting |
| compatible-residual score | prior strength 0.1; reward 5; false-stable/false-unstable cost 1/1; exploration 0.5; temperature 0.04 eV/atom; minimum cosine similarity 0.05 | frozen synthetic settings |

Protocol strata are fixed by the audited protocol identity. A stratum with
fewer than 30 residuals or 30 compatible cross-structure pairs is unsupported;
it is not pooled post hoc. A finite-sample corrected target quantile \(p\) is
the order statistic at index \(\min(n,\lceil(n+1)p\rceil)\). For a nonempty
interval \([l,u]\),
\(q_x^-=\operatorname{clip}((\tau-l)/(u-l),0,1)\) and \(q_x^+=1-q_x^-\);
point intervals use their deterministic side of \(\tau\), and conflicts use
0.5/0.5. These weights have no probabilistic coverage interpretation.

The parameter manifest, seeds, and all tie-breaks are serialized before
evaluation. Evaluation-pool outcomes may not affect them. Inadequate calibration
support blocks the affected method. For provenance only, earlier synthetic
constants were \(L=0.08\), \(q_c=0.01\), and prior half-width 0.15 eV/atom;
they are not WBM estimates. Calibration intervals are empirical inputs and do
not imply simultaneous coverage over adaptive working sets.

## Acquisition-by-access factorial design and estimands

The primary persistent selector is deterministic, acquisition-independent
certified FIFO with capacity \(K\). Query-specific compatible top-\(K\)
retrieval uses the same action trace for its free and costed labels whenever
cost is absent from the acquisition score; only offline accounting differs.
Full-history scanning exposes every compatible archived witness. Each
acquisition consumes the same typed evidence view and oracle-blind state.

| Acquisition policy | Persistent certified | Free on-demand archive | Costed on-demand top-\(K\) | Full-history scan |
|---|---:|---:|---:|---:|
| Uncertainty | required | required | required | required |
| CAL-style hull-aware GP | required | required | required | required |
| Compatible residual | required | required | required | required |
| Seeded random (cost diagnostic) | required | required | required | required |

All required labels cover \(B\in\{25,50,100,200\}\),
\(K\in\{0,1,2,4,8,16,\infty\}\), and activation price
\(\{0,p50,p90\}\), but price changes never trigger policy re-execution. If an
acquisition mathematically ignores residual evidence, one canonical action
trace is reused across access labels after checksum validation, while physical
costs are replayed for each distinct access implementation. Full-history uses
one canonical trace across finite \(K\); formally identical \(K=\infty\) views
are also checksum aliases. Any impossible pairing must be amended before
outcomes and is excluded from causal claims about persistence.
CAW-Joint is an additional, separately resource-gated method and cannot supply
the core causal evidence for persistence.

Two estimands are reported separately:

**A. Matched access economics.** Persistent certified FIFO is paired with an
on-demand reconstruction of the same ordered active set, using identical
selector, history, RNG, tie-breaks, and acquisition. Their action, active-set,
hull, and discovery traces must be identical. The zero-price emulator is the
parity gate; cold/warm measured emulators and frozen p50/p90 prices estimate
only retrieval, certification, cache, maintenance, latency, memory, and utility
cost differences. This is the only estimand that supports a pure persistence /
access-mechanism claim.

**B. Evidence-selection/access-policy comparison.** Persistent FIFO,
query-specific compatible top-\(K\), full history, and any separately
preregistered selector may choose different evidence and actions. Differences
in discovery or utility are composite evidence-selection/access-policy effects,
never a causal effect of persistence alone.

## Exact-emulation implementation gate

The zero-cost on-demand **exact emulator** is a parity harness, not the
query-specific free-retrieval policy. Given the same acquisition,
revealed history, RNG state, tie-breaks, and active-set selector, it reconstructs
from the archive exactly the set that persistent execution carries forward.
Every round must match field-for-field (and by canonical trace checksum) on:

1. selected query/action;
2. ordered active witness IDs and certificate/hull-version identities;
3. causal-hull snapshot ID/checksum and transition;
4. causal, provisional, invalidated, and final-confirmation metric state.

The gate allows **zero mismatches** over every pool, seed, budget, and eligible
\(K\). The measured/costed emulator replays the same trace with instrumentation;
it may differ only in cost fields. Any behavioral mismatch stops all policy
experiments and triggers an implementation-asymmetry audit. Statistical
tolerances apply only to genuinely different policies.

## Cost measurement contract

Cold-cache and warm-cache runs are separate. Each atomic cell builds its own
index: charge the build once to the cell total and report the per-round
amortization as build time divided by \(B\). Warm-cache timing starts after that
build, but its economic total still includes the amortized build. Time admission,
archive lookup, bytes/records scanned, first certification, protocol transport,
and re-certification after every hull-version change separately. On-demand uses
a deterministic LRU cache of capacity \(K\); hits still incur and record lookup
cost. \(K=\infty\) means the complete eligible archive.

For each cache state, component-wise p50/p90 are nearest-rank empirical
quantiles measured only on engineering/calibration systems; sort by
`(duration, operation_id)` and use index \(\lceil pn\rceil\). Archive
materialization I/O, cache misses, admission, first certification, protocol
transport, and hull-version re-certification enter \(C_{\mathrm{activate}}\).
Index construction is charged once to \(C_{\mathrm{online}}\); index lookup,
witness scanning, cache bookkeeping, serialization, and synchronization also
enter \(C_{\mathrm{online}}\). No operation appears in both totals.

Conflict, abstention, and rejected transport are charged for all scanning,
I/O, transport checks, and certification performed before rejection. End-to-end
latency includes disk I/O, serialization, CPU/GPU synchronization, and cache
management. Free retrieval changes only the economic price, not the measured
latency. Every strategy uses the same machine, 8 CPU threads, fixed GPU,
batching, process isolation, and warm-up schedule; background workload is
disallowed and the manifest is retained. All 48 utility-weight combinations
and all three frozen price schedules are recomputed offline and reported; none
is selected after observing a favorable result.

## Metrics and preregistered decision

Report query-time causal discoveries, final confirmations, invalidated
discoveries, conflicts/abstentions, oracle calls, every cost component,
end-to-end latency, peak memory, action disagreement, and archive fraction
retrieved/activated. Keep per-pool results and paired policy differences.

**Access-GO (estimand A)** requires exact behavioral parity and a positive
persistent-minus-reconstruction net-utility difference with paired clustered
95% lower bound above zero. It must span two adjacent utility-weight points,
two adjacent \(K\), two budgets, and at least 8 of 12 pools at a frozen p50 or
p90 cold/warm price, and remain positive under every leave-one-system-or-family-
out analysis. Discovery difference must be exactly zero.

**Evidence-policy GO (estimand B)** requires persistent FIFO to be non-inferior
to query-specific top-\(K\) and full history in final discoveries (paired 95%
lower bound above -0.5 discoveries per 100 queries) and to have positive net
utility against the best matched acquisition/access-policy baseline. The same
multi-weight, adjacent-\(K\), two-budget, 8-of-12-pool, final-hull, and leave-one-
cluster robustness rules apply. This establishes only a composite strategy
advantage. A pure working-set claim requires Access-GO; evidence-policy
superiority alone is insufficient.

**NO-GO** holds if benefit appears only above frozen p90 price, in one
cell/pool/family, disappears on the final hull, or uncertainty, CAL-style GP, or
free/costed on-demand access dominates. An unexplained exact-emulation mismatch
is a hard stop, not a scientific result. Only Access-GO plus a non-dominated
estimand-B result authorizes design of a MADE experiment.
