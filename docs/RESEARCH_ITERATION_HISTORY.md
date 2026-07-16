# Materials-memory research iteration history

This file records conclusions from retired experiments without keeping their
runners, duplicated evaluators, or frozen preregistration machinery on the live
branch. Git tags preserve exact code and documents; datasets and run outputs
remain outside Git.

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

### DACC and survival acquisition

The current primary hypothesis compresses residual calibration with the
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
is paused. The implementation remains because it is the explicitly defined
secondary hypothesis and its zero-weight/redundant-fantasy failure tests remain
useful; it must not be tuned on evaluation pools.

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
