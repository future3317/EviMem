# WBM working-set economics pilot: preregistered design

**Status.** Frozen design plus P0 engineering implementation. This document
does not authorize a comparative WBM policy run or MADE experiment. The frozen
method-level NO-GO remains immutable. P1 numerical/identity parity and P1.5
informativeness must pass before the minimal P2 matrix. The cell-level resource
contract is in
[`WBM_EXECUTION_FEASIBILITY_TABLE.md`](WBM_EXECUTION_FEASIBILITY_TABLE.md).

## Amendment v2: execution boundary and three-hull evaluation

This amendment supersedes any implementation in which an unqueried candidate
object carries an oracle card. P0 now requires a policy subprocess that
receives only a serialized `PolicyState` and returns an opaque
`query_id`. The evaluator must persist the action and pre-reveal checksum before
calling the vault. A reveal produces two disjoint objects: a
`RevealedObservation` eligible for archive/working-set use and a
`CorrectedPhaseEntry` eligible only for the hull reviser. The policy process has
no vault, phase-diagram, corrected-entry, evaluator, or unqueried-outcome
capability.

Three hulls and five metrics are frozen:

| Hull / metric | Frozen definition |
|---|---|
| policy-visible causal hull (H_t^{\mathrm{causal}}) | frozen MP phases plus WBM phases revealed before round (t) |
| selected-history final hull (H_T^{\mathrm{selected}}) | frozen MP phases plus every phase selected in the completed trace; never unselected entries |
| offline oracle benchmark hull (H^{\mathrm{oracle}}) | frozen MP phases plus every WBM phase in the frozen exact-system evaluation universe; computed only after the trace |
| causal discovery | stable against the pre-reveal (H_t^{\mathrm{causal}}) |
| selected-final confirmation | selected item stable against (H_T^{\mathrm{selected}}) |
| selected-history invalidation | causal discovery but not selected-final confirmation |
| oracle-final true discovery | selected item stable against (H^{\mathrm{oracle}}) |
| benchmark false confirmation | selected-final confirmation but not oracle-final true discovery |

For a 16-candidate pool, the oracle benchmark universe is every frozen cleaned
WBM entry in that exact chemical system, not only the sampled 16. Oracle-final
quantities are offline evaluation labels and are prohibited from policy state,
acquisition, retention, calibration, pool selection, and online hull updates.

The label-blind expansion rule is also frozen before the informativeness audit.
Eligible systems are all binary/ternary exact systems with at least 16 cleaned
candidates. Deterministic strata use candidate count, frozen-prediction
hull-margin spread, and SOAP diversity only; corrected WBM energy and stability
labels are forbidden. Each exact system can enter at most one pool. The current
eight systems remain immutable engineering pools. Their oracle support may
classify the pilot as informative or underpowered but may not replace a pool.
Claim-grade system count will be selected only from pilot-estimated
between-system variance and a predeclared precision target; candidates, budget
cells, and random seeds are not independent statistical units.

P2 begins only after P0, P1, and the informativeness gate pass. Its minimal
matrix is Random; frozen CHGNet ranking; uncertainty plus independent FIFO;
residual acquisition plus persistent FIFO; free reconstruction of the same
FIFO; query-specific compatible top-(K); and full revealed history. The free
same-FIFO pair must match every action. Full history is one (K=\infty) method,
not duplicated across finite (K). CAL-style GP and CAW-Joint remain P3 methods.

Scientific discovery and access economics are separate experiments. The former
uses the frozen 16-candidate pools and reports the five hull-aware metrics. The
latter replays identical reveal traces at scaled archive sizes and reports raw
reads, SOAP comparisons, bytes, certification, index/cache work, wall time and
peak memory under free and measured retrieval. Pareto and break-even summaries
precede any weighted utility aggregation.

### P0 implementation audit (2026-07-16)

The formal runner now uses a fixed policy worker subprocess, an allow-listed
serialized state, opaque query IDs, a ledger-gated single-use oracle vault,
separate per-atom observations and total-energy phase entries, and an fsync'd
action record before every reveal. The causal phase diagram is rebuilt in the
evaluator from frozen MP entries plus revealed same-system WBM entries. Hull
references passed to formation-energy policies are converted to the formation-
energy basis; corrected total energies remain confined to phase entries.

The superseded in-process WBM reviser, preloaded WBM candidate/oracle pairs,
and old exact-emulation runner were deleted. Failure-capable tests cover policy
serialization, counterfactual unrevealed energies, cross-system isolation,
pre-action authorization, duplicate reveal, total/per-atom unit separation,
hypothetical copies, three-hull separation, deterministic ordering, event-log
replay, and zero-mismatch persistent/reconstructed FIFO.

The subprocess is a Python object-capability boundary, not an operating-system
sandbox: it receives no evaluator or vault reference, but runs under the same
OS account. Formal runs therefore permit only the frozen repository worker and
declared policy names; arbitrary third-party policy executables are prohibited.
P0 passing does not imply P1 parity, an informative pool, or a scientific
effect. The auditable implementation record is
[`WBM_P0_EXECUTION_AUDIT.md`](WBM_P0_EXECUTION_AUDIT.md).

## Decision question and cost units

The pilot asks whether keeping a persistent, certified working set has positive
economic value when every revealed DFT result remains in an immutable archive.
It does not tune CAW-Joint or add heuristic scenarios. For policy \(\pi\),

\[
J_{\alpha,\beta,\gamma}(\pi)=D_{\mathrm{oracle-final}}(\pi)
-\alpha C_{\mathrm{oracle}}(\pi)
-\beta C_{\mathrm{activate}}(\pi)
-\gamma C_{\mathrm{online}}(\pi).
\]

\(D_{\mathrm{oracle-final}}\) is the discovery count that survives the full
frozen exact-system oracle benchmark hull. Selected-final confirmation is
reported separately and may not substitute for oracle-final truth in a
scientific GO decision.
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

### Data-feasibility amendment required before formal execution

An oracle-blind count over the frozen 256,963 cleaned WBM IDs was run on
2026-07-16 and retained externally as
`E:\DATA\EviMem-RL\outputs\exploratory\wbm-exact-system-feasibility-v1.json`.
There are 75,480 exact chemical systems; the largest contains 46 cleaned
candidates, with 0 systems containing 64 or 256. Therefore the exact-system,
256-candidate pool specification below is **not executable** on WBM and the
formal policy matrix remains suspended. This is a data-feasibility finding, not
a discovery result: it used neither WBM energy, hull label, nor prediction
error.

Any subsequent formal WBM design must be a versioned amendment made before
evaluation outcomes are accessed. It must either reduce the exact-system pool
size to an attainable fixed value or define a chemically justified multi-system
pool and revise causal-hull, clustering, and estimand definitions accordingly.
It must not silently combine systems or selectively choose systems after seeing
outcomes. The remaining text records the originally frozen, now infeasible
256-candidate proposal; it is retained for traceability rather than executed.

### Exploratory small-pool amendment v1

The first executable WBM pilot uses **eight** oracle-blind exact-system pools,
not the infeasible twelve 256-candidate pools: two hash-selected systems in
each of `2-element × {small, large}` and `3-element × {small, large}` strata.
Each system contributes 16 hash-selected candidates after removal of only
byte-identical serialized CSE structures. The maximum exact-system count for
four-or-more-element systems is seven, so those systems are explicitly out of
scope rather than merged or padded. The pilot uses (B\in\{4,8,12\}) and
(K\in\{0,2,4,8\}); it may support only binary/ternary mechanism evidence,
not a general WBM claim. Exact MP-overlap/prototype canonicalization remains a
separate unresolved gate, so results produced before it is implemented are
exploratory and cannot change the frozen method-level NO-GO decision.

The implementation rebuilds a composition-dependent phase diagram from the
frozen MP phase set plus revealed WBM phases after every oracle call. The
deprecated scalar minimum-energy update is prohibited and has no runtime
fallback.

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

Report query-time causal discoveries, selected-final confirmations,
selected-history invalidations, oracle-final true discoveries, benchmark false
confirmations, conflicts/abstentions, oracle calls, every cost component,
end-to-end latency, peak memory, action disagreement, and archive fraction
retrieved/activated. Keep per-pool results and paired policy differences.

**Access-GO (estimand A)** requires exact behavioral parity and a positive
persistent-minus-reconstruction net-utility difference with a chemical-system-
clustered 95% lower bound above zero. It must span two adjacent utility-weight
points, two adjacent \(K\), and two budgets at a frozen p50 or p90 cold/warm
price, and remain positive under every preregistered leave-one-system-or-family-
out analysis. Discovery difference must be exactly zero. This criterion is not
applied to the eight engineering systems; the claim-grade system count and
minimum cross-system support are frozen from pilot variance and a declared
precision target before P2 outcomes are inspected.

**Evidence-policy GO (estimand B)** requires persistent FIFO to be non-inferior
to query-specific top-\(K\) and full history in oracle-final discoveries (paired 95%
lower bound above -0.5 discoveries per 100 queries) and to have positive net
utility against the best matched acquisition/access-policy baseline. The same
multi-weight, adjacent-\(K\), two-budget, claim-grade cross-system, oracle-final,
and leave-one-cluster robustness rules apply. This establishes only a composite strategy
advantage. A pure working-set claim requires Access-GO; evidence-policy
superiority alone is insufficient.

**NO-GO** holds if benefit appears only above frozen p90 price, in one
cell/pool/family, disappears on the oracle-final hull, or uncertainty, CAL-style GP, or
free/costed on-demand access dominates. An unexplained exact-emulation mismatch
is a hard stop, not a scientific result. Only Access-GO plus a non-dominated
estimand-B result authorizes design of a MADE experiment.
