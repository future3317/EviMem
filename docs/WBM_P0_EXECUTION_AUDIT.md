# WBM P0 execution-correctness audit

**Decision (2026-07-16): P0 engineering gate passes; P1, P1.5, P2 and MADE
remain blocked.** This is an implementation decision, not a materials result.
The frozen method-level NO-GO is unchanged.

## Sole formal execution path

`src/evimem/matmem/wbm_secure.py` is the only real-WBM closed-loop runner.
The policy subprocess receives only a JSON `PolicyState` containing observable
candidate fields, current causal-hull identities, budget and already revealed
active witnesses. It returns one opaque query ID. The evaluator then:

1. fsyncs an action and pre-reveal state checksum to an append-only JSONL log;
2. presents the resulting authorization to the single-use vault;
3. creates a `RevealedObservation` in eV/atom for archive/residual use;
4. creates a `CorrectedPhaseEntry` whose energy is validated as total eV;
5. rebuilds the same-system causal phase diagram;
6. persists post-reveal archive, active-set and hull checksums.

The policy never receives the vault, phase diagram, corrected total-energy
entry, evaluator object or an unqueried outcome. The worker subprocess shares
the evaluator's OS account and filesystem, so this is not a hostile-code
sandbox. Formal execution is restricted to the frozen worker and allow-listed
policy names.

## Three non-interchangeable hulls

- `policy-visible causal`: initial frozen MP entries plus entries revealed
  before the current decision;
- `selected-history final`: initial MP entries plus only entries selected in
  the completed trace;
- `offline oracle benchmark`: initial MP entries plus the full frozen WBM
  universe for that exact chemical system, evaluated only after the trace.

The reported metrics are causal discovery, selected-final confirmation,
selected-history invalidation, oracle-final true discovery and benchmark false
confirmation. A selected-final confirmation is not called a true WBM discovery
unless it also survives the oracle-final hull.

## Removed superseded paths

- deleted the old `WBMPhaseDiagramHullReviser`;
- deleted the old WBM vault that returned a card without a persisted action;
- deleted the old `run_fifo_exact_emulation`/trace implementation;
- deleted the exploratory WBM smoke runner that preloaded oracle cards;
- removed query-side oracle cards from both active evaluators; synthetic tests
  now use an evaluator-owned `CardOracleVault`.

The generic synthetic evaluator remains because it reproduces the frozen
synthetic NO-GO and tests method mechanics. It is not an alternative WBM path.

## Failure-capable evidence

`tests/test_wbm_secure_runner.py` checks:

- allow-listed serialization and deterministic tie-breaking;
- counterfactual non-interference from unrevealed energies;
- action persistence before reveal and duplicate-reveal failure;
- corrected total-energy versus formation-energy-per-atom separation;
- hypothetical-state and cross-system isolation;
- selected-final versus oracle-final disagreement;
- deterministic ledger replay;
- exact persistent FIFO versus free same-FIFO reconstruction.

The focused P0/materials suite passed 48 tests on 2026-07-16. The final
repository validation passed 188 tests and a full Ruff check on the same date.
These counts must not be used as evidence of P1 numerical parity.

## Remaining blockers

1. Produce the 128-candidate P1 parity table with frozen same-environment
   experimental values and predeclared tolerances.
2. Complete engineering-level byte-identical identity checks; canonical
   structure, prototype and MP-overlap audits remain mandatory for claim-grade
   execution.
3. Run the frozen-manifest P1.5 support audit without replacing zero-positive
   pools.
4. Only after those gates, run the minimal P2 matrix containing uncertainty,
   full history and on-demand access. CAL-style GP and CAW-Joint remain P3.

No comparative policy result, GO/NO-GO effect estimate, or MADE run is
authorized by this audit.
