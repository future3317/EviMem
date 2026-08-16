# CAL acquisition kernel performance decision record

**Date:** 2026-08-16  
**Status:** closed for the current numerical contract  
**Scope:** engineering evaluation of the `cal_style_hull_entropy` acquisition
kernel only; no paper estimand, policy trajectory, posterior, or scientific
result is changed by this record.

## Decision

The current `8990d4a` CAL implementation remains the trusted execution
baseline. The dense-projection, raw-CSR, and strict-order fused-kernel paths
are rejected for the current production contract and must not be reintroduced
as formal campaign implementations.

The `B=6` Co-F-Li-O run and the 230-system formal campaign remain **NO-GO** on
the current execution infrastructure. No additional kernel variant or full
campaign should be launched under this decision.

## Evidence

All measurements below use the real Co-F-Li-O round-1 state with the frozen
`M=400`, `K=20`, fixed-composition setup. The baseline worker sweep used the
instrumented/working-set-bounded path descended from `8990d4a`; its outer
`/usr/bin/time` RSS values are reported in KiB.

| Path | Configuration or gate | Wall time | Peak RSS | Numerical result | Decision |
|---|---|---:|---:|---|---|
| Trusted baseline | workers 1 / 2 / 4 / 8 | 1540.10 / 1127.77 / **606.11** / 660.79 s | 461844 / 462168 / 617780 / 943344 KiB | baseline semantics retained | Keep; workers=4 is the best measured topology |
| Dense projection | one round, workers=4 | 213.16 s | higher than baseline | max score difference `1.71e-8` | Reject: changes the numerical path |
| Raw CSR | workers 1 / 2 / 4 | 567.62 / 371.99 / **265.11** s | higher than baseline | max score difference `2.85e-8` | Reject: changes the numerical path |
| Strict-order fused kernel | real-state micro-gate in `equivcompiler` | not run | not applicable | 274/336 query tiles differ; max hull difference `8.881784197001252e-16`; `np.array_equal` failed | Reject before acquisition |

The strict-order gate used NumPy 2.4.6, SciPy 1.17.1, and Numba 0.66.0.
The real state contained 8,000 conditional rows, 46 active energies, 42
evaluation compositions, and 177,235 decompositions. Since the gate failed at
the hull-vector layer, no complete acquisition round was run for that path.

## Interpretation

The baseline result depends on the concrete NumPy/CPU floating-point execution
path in addition to the mathematical formula and decomposition order. The
alternative paths reduce materialization or arithmetic cost, but their
floating-point differences occur before the covariance and `slogdet` steps of
the CAL score. The observed ULP-scale hull differences are therefore not
treated as harmless by silently widening the existing parity requirement.

This is an implementation decision, not a negative scientific result about
Hull entropy (CAL-style). The rejected paths do not provide evidence for or
against the E54 objective comparison.

## Preserved provenance

The implementation history remains in Git:

- `52e5f39`: CAL acquisition optimization;
- `70167dc`: bounded CAL hull working-set memory;
- `8990d4a`: CAL acquisition phase instrumentation;
- `62f8297`: dense projection experiment, followed by its revert;
- `d8fa055`: raw CSR experiment, followed by its revert;
- `b94bbca`: current clean rollback to the trusted baseline path.

The external benchmark roots remain outside Git and are not overwritten or
deleted:

- `matpes_e55_cal_phase_workers_round1_20260816`;
- `matpes_e55_cal_dense_projection_round1_20260816`;
- `matpes_e55_cal_csr_phase_workers_round1_20260816`;
- `matpes_e55_cal_tiling_co_f_li_o_round1_20260816`.

No E32, E53, E54, or E55 scientific artifact is modified or relabeled.

## Future reopening criteria

This route may be reconsidered only as an infrastructure change that fixes
the NumPy/CPU execution environment while preserving the current numerical
contract, or after an explicitly registered scientific decision changes the
parity contract to a semantic tolerance. Neither option is an implicit
performance optimization, and neither is authorized by this record.
