# MatPES confirmatory infrastructure

This note records the implementation boundary for the fresh Delta--Hull
replication. It is infrastructure, not a positive result.

## Frozen method

`delta_hull_active_search` remains the MC1024 nested scrambled-Sobol method
with the existing posterior and one-step final-hull membership objective.
No score blend, adaptive Monte Carlo, lookahead, top-k truncation, or new
weight is introduced in this stage. The default hull backend is `pymatgen`.

`fixed_composition` is an action-equivalent optimization only. It caches the
composition geometry and reuses it for each posterior sample. It is disabled
for claims or production runs until `tools/audit_matpes_fixed_hull_parity.py`
reports zero action-trace and sample-membership mismatches.

## Fresh split contract

`tools/build_matpes_confirmatory_task.py` excludes every development system and
pair, applies only candidate/parent-count gates, and selects systems by a
release/system SHA-256 within binary, ternary and quaternary-or-higher strata.
Target outcomes are copied to a separate `confirmatory_sealed_oracle_vault`
after the system set is selected. Selection never reads target values.

The confirmatory runner accepts `--split confirmatory` and requires a frozen
transport artifact. It does not refit transport on evaluation systems. The
development runner may still fit transport only on its disjoint development
fit systems.

## Transport freeze

`tools/freeze_matpes_transport_model.py` writes a JSON artifact containing the
model parameters, optimizer metadata, fit-system IDs, source task/vault
checksums, and a canonical payload checksum. The confirmatory runner loads that
artifact and fails if a fit system intersects a query system.

## Required execution order

1. Build the all-eligible task outside Git.
2. Build the fresh task/vault and record its exclusion manifest.
3. Freeze transport on the registered disjoint fit systems.
4. Run fixed-hull parity; stop on any mismatch.
5. Run `tools/audit_matpes_sobol_seed_stability.py` with independent scramble
   seeds on development systems.
6. Only then run the frozen fresh replication and baselines.

All raw task, oracle and experiment outputs remain outside Git.
