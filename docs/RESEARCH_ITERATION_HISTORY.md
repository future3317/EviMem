# Materials-memory research iteration history

This file is a preserved technical annex. Its unique chronology and evidence
judgments are consolidated in `docs/EXPERIMENT_LEDGER.md`, the single canonical
audit trail. Do not delete this annex merely to shorten the active project; it
may be reduced only after verifying that every unique experiment identity,
correction, result, validity label and recovery point remains in the ledger.
Git tags preserve exact code and documents; datasets and run outputs remain
outside Git.

**Current status must be read first:** P3C is stopped, AKSC is not authorized as
the WBM paper's next main method, and the paper remains NO-GO. The complete
artifact-level chronology, including invalid and interrupted runs, is in
[`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md). The sections below are history,
not a menu of currently active hypotheses.

## Recovery points

| Research state | Git reference | Meaning |
|---|---|---|
| Corrected CAW-Joint | `caw-method-no-go-2026-07-15` / `0fa1e1f` | Frozen method-level NO-GO, including the old synthetic runners and tests |
| Working-set economics preregistration | `wbm-preregistration-freeze-2026-07-16` / `ca80720` | Frozen access-economics design and exhaustive-solver feasibility tables |
| Secure WBM P0 | `e313499` | Oracle-isolated subprocess execution and removal of the unsafe fallback path |
| Composition-dependent causal hull | `1b37686` | Real phase-diagram transition used by the current WBM runner |
| DACC implementation | `0fb29eb` | Decision-aware calibration coreset and survival-conditioned acquisition |
| First engineering WBM result | `76b8612` | P1/P1.5 audit and small matched-trace pilot |

To inspect or reconstruct an old experiment without restoring it to the active
architecture, use `git show <reference>:<path>` or create a temporary worktree
at the corresponding tag.

## What the retired iterations established

### CAW-Joint corrections

- Information gain must compare the same residual candidate pool before and
  after a hypothetical query; otherwise removing the queried item creates a
  false gain.
- `K` is an active/certified witness budget, never permission to delete DFT
  history. The immutable archive and online working set are distinct.
- Boundary-risk comparisons require query-fixed weights. Interval conflicts
  fail closed and do not imply a calibrated probability distribution.
- The paper's original `|M'| <= K` retention objective was not implemented by
  exact-size-`K` enumeration. A constructed conflict required deleting two old
  witnesses. The corrected exhaustive solver was exponential in `K`.
- Causal discoveries, final-hull confirmations, invalidated discoveries,
  variable oracle cost and discovery regret are separate quantities.

After these corrections, joint acquisition/retention did not beat the strongest
independent acquisition baseline. Its narrow stress-test gains disappeared in
larger budgets and did not justify an ICLR method-superiority claim. Adding more
scenarios or weights was therefore stopped.

### Working-set economics

The next iteration separated two estimands: matched access economics uses the
same active-set selector and requires action-by-action parity, while evidence
policy comparisons may change the selected witnesses and cannot identify a pure
persistence effect. It also separated policy execution, physical timing and
offline utility-price recomputation. The preregistered 12 x 256 exact-system
WBM construction was later found infeasible because cleaned WBM exact chemical
systems are much smaller; this was a design-support failure, not a reason to
replace zero-positive pools after seeing outcomes.

These documents are retained at `wbm-preregistration-freeze-2026-07-16`, but
their CAW-specific execution matrix is no longer a gate for the DACC method.

### Secure real-WBM infrastructure

The live runner has one path: persist action, reveal exactly one oracle record,
update the composition-dependent hull, construct the observable future pool,
admit evidence, then refit before the next decision. A scalar minimum-energy
synthetic hull is not a valid phase diagram and has been removed. The secure
oracle vault, append-only event log, replay audit and exact persistent versus
same-FIFO reconstruction parity tests remain live.

### Retired DACC and survival acquisition

At that iteration, the primary hypothesis compressed residual calibration with the
facility-location objective

```text
F_t(M) = sum_u max_{m in M} G_t(u,m),
G_t(u,m) = w_t(u) [R_t(u) - R_t(u | m)]_+.
```

The per-round gain matrix is fixed, non-negative and tested for monotonicity and
submodularity. Streaming admission is exact only over the current `K` witnesses
plus the new witness; it is not a global full-archive optimum.

The old three-seed synthetic DACC smoke was useful only for debugging: DACC
matched full history in the handcrafted recurrence case, but diversity did too
and was faster; the IID negative control warned that the frozen GP can overfit
full history. The smoke runner was removed because it is neither claim-grade nor
used by the current paper.

Survival-conditioned acquisition was worse in both early engineering cells and
was paused. Its zero-weight/redundant-fantasy failure tests remain useful, but
it is not a current contribution and must not be tuned on evaluation pools.

## Current real-data evidence

The P1/P1.5 engineering audit covers 128 candidates in eight fixed 16-candidate
binary/ternary pools. Historical and modern adapters had zero corrected-energy,
hull, label or phase-membership mismatches; five of eight pools contain an
oracle-final stable candidate. External audit:

```text
E:\DATA\EviMem-RL\manifests\wbm-engineering-p1-p15-audit-v1.json
SHA-256 f3941364f2df317fffea3ab63286f66e624449af88f0c48a2f60585551b68e96
```

In the `B=8, K=2` matched-frozen-action pilot, every retention method followed
the same actions. DACC had the best mean residual RMSE (`0.0656`) and Gaussian
NLL (`-0.4841`); diversity had the best Brier score (`0.0555`). DACC improved
RMSE over full history in five systems and degraded it in three. Result:

```text
E:\DATA\EviMem-RL\outputs\engineering\wbm-calibration-matched-b8-k2-v1\summary.json
SHA-256 7c6ed468f8bb7e31e6dcd8389cbc7fc0df373daad78bd20be869984a63becbf8
```

This is preliminary mechanism evidence, not dominance and not paper-level GO.
Claim-grade WBM still needs canonical/prototype overlap auditing, more chemical
systems, calibration-only parameter freezing, paired uncertainty and measured
compute. MADE remains out of scope until that evidence justifies expansion.

## Live architecture after cleanup

- `calibration_utility.py`, `coreset.py`, `residual_posterior.py`: DACC core.
- `acquisition.py`: frozen, random, GP uncertainty and survival policies only.
- `wbm_secure.py`, `wbm_policy_worker.py`, `hull_engine.py`: sole WBM execution
  boundary and causal-hull protocol.
- `wbm.py`, `wbm_raw.py` and `tools/*wbm*`: official artifact, cleaned-ID,
  parity, pool, SOAP and engineering-run infrastructure.
- `cards.py`, `identity.py`, `protocols.py`, `residual.py`, `risk.py`: reusable
  scientific contracts and fail-closed calibration primitives.

Deleted code consisted of CAW boundary potentials/retention, the old generic
active evaluator and economics ledger, exact binary DP, scalar synthetic hull,
legacy acquisition heuristics, six iteration-only runners and their dedicated
tests. Those paths must be recovered from the frozen tags if a historical audit
is needed; they should not be copied back as compatibility adapters.

## 2026-07-20 closure: P3C and WBM posterior compression

The DACC paragraph above records an earlier iteration and is not the current
method status. DACC was replaced by P3C proper posterior projection, then P3C
was stopped after two structure-correct, disjoint panels totaling 32 exact
systems failed to reproduce the original probability-metric signal. Historical
parity-energy drift and relaxed-structure SOAP leakage are now type-level
provenance failures rather than accepted data variants.

The fixed-GP ceiling has some mean and boundary-probability signal but is badly
underdispersed: mean LOO squared standardized residual is 16.90 and nominal 90%
coverage is 62.4%. Effective dimension averages 12.42, so K=2 is also far below
the kernel spectrum's characteristic scale.

A separate, checkpointed compute-relevance panel used the three longest
eligible exact systems and complete B40 full-history traces. The largest GP
numerical share of round-pipeline time was 0.689%, versus a preregistered 9.09%
Amdahl threshold for a possible 10% ideal speedup. Consequently:

- P3C is terminated as a main method;
- AKSC/all-outcome kernel sketching is not authorized as this WBM paper's main
  direction;
- WBM remains useful for correctness and calibration diagnostics, not evidence
  of a posterior-compression compute frontier;
- the paper remains NO-GO.

The authoritative record is
`docs/WBM_LONG_ARCHIVE_COMPUTE_GATE_2026-07-20.md`.

The full transition must be stated precisely. DACC's singleton
facility-location objective did not match the joint GP used for prediction.
P3C repaired that local reference mismatch by projecting a fixed temporary
union posterior, but it still omitted evicted archive outcomes from the
deployed GP and selected those omissions using observed residuals. This is
outcome-*contribution* deletion, not deletion of the immutable scientific
archive. Proper scoring guaranteed fidelity only to the frozen GP reference;
the direct dispersion audit then showed that reference to be severely
underdispersed. AKSC was proposed to keep all outcome contributions in
low-dimensional posterior sufficient statistics while compressing only an
outcome-independent representation. The B40 Amdahl gate failed before AKSC
implementation, so that proposal is not evidence and is not the WBM paper's
next method. See `docs/EXPERIMENT_LEDGER.md` for every experiment, path, hash,
validity label and recovery point.

## 2026-07-20 problem redefinition: decision-sufficient scientific state

The closure above changes the research question, not the P3C implementation.
The project no longer asks which paid residual outcomes should continue to
count in a bounded GP. It asks for the least costly observable and
protocol-valid state that preserves registered scientific decisions relative
to the complete immutable archive.

This reframing has three conditional results: complete information weakly
dominates any deterministic compression when decisions may ignore information
and state cost is absent; uniform `epsilon` action-value approximation gives at
most `2 epsilon` one-step regret; and a component occupying fraction `f` of
end-to-end time permits at most `1/(1-f)` ideal speedup if eliminated. The B40
gate places current WBM in the no-forgetting null regime.

The next material hypothesis is Certified Hull-Decision State: protocol-valid
evidence influence, directed transport uncertainty, hull-support/facet state,
stable/unstable/abstain decisions, and an `epsilon`-optimal action set. It is
not implemented and must reduce exactly to full history for homogeneous
zero-transport-error evidence. See
`docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md` for the formal definition and
pre-implementation gates.
