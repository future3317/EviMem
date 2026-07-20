# WBM long-archive compute-relevance gate

This is the terminal WBM posterior-compression gate in the authoritative
research chain recorded by `docs/EXPERIMENT_LEDGER.md`.

**Decision (2026-07-20): FAIL.** WBM does not supply an end-to-end compute
motivation for bounded posterior compression at the longest executable exact
chemical-system trajectories. P3C remains stopped, and AKSC is not authorized
as this paper's WBM main method.

## Frozen question and gate

The benchmark asks a deliberately stronger question than whether a sparse GP
can make a small dense factorization faster:

> If posterior numerical work were eliminated completely, could the real WBM
> closed-loop wall time improve by at least 10%?

For GP fraction (f_{\rm GP}), Amdahl's law gives the ideal upper bound

\[
S_{\max}=\frac{1}{1-f_{\rm GP}}.
\]

The preregistered target (S_{\max}\ge1.10) requires
(f_{\rm GP}\ge0.0909). A B40 fraction below 9.09% therefore stops WBM
end-to-end compute-Pareto claims, irrespective of whether a sparse numerical
routine is faster in isolation.

The oracle-blind manifest was frozen before physical execution at:

```text
E:\DATA\EviMem-RL\manifests\wbm-long-archive-compute-gate-v1.json
```

It ranks exact systems by descending cleaned candidate count, with a
release-bound SHA256 tie-break. Selection reads no energy, residual, stability
label, hull result, or predictor score. Twelve WBM systems have at least 41
candidates. The frozen three-system panel is Fe--S (46), Fe--Zr (44), and Ni--S
(44), using every cleaned candidate in each system.

## Provenance and execution

All policy-visible features use the official WBM `org` initial structures.
Oracle outcomes use frozen `pymatgen==2023.5.10` MP2020 parity energies. The
engineering P1 provenance gate passes for all 134 candidates. P1.5 discovery
support is false, but it is intentionally irrelevant to this timing-only
estimand; the dedicated runner mode forbids policy comparisons and accepts
only frozen-action, full-history, B40 execution.

The read-only pre-run checkpoint is:

```text
E:\DATA\EviMem-RL\checkpoints\wbm-long-archive-compute-pre-run-v2.json
```

It records code-tree SHA
`ffb627336b2d6035fd70c639b9aeaca40507d54f99d7d82f8ca87de83ef91ee7`,
dirty-diff SHA
`2645a620ef8a7901645e7f1416d33f13527d0a88221bd8f656c3ab362681fb52`,
the exact execution-source hashes, external manifest hashes, and a Conda
explicit lock. Its file SHA256 is
`b24d326e57e9b860215c6f6d16176e1cf89ae4090d09adb1b5aac96ecf313ce7`.

The checkpointed physical summary is:

```text
E:\DATA\EviMem-RL\outputs\diagnostics\wbm-long-archive-full-history-b40-v2\summary.json
```

The independent v1 and checkpointed v2 runs have identical selected actions
and trace checksums for all three systems. Timing varies, as expected, but the
research decision is unchanged.

## Real-trace result

Lazy GP fitting performs factorization during prediction, so GP numerical time
is defined as `posterior_fit_seconds + prediction_seconds`. The denominator is
the sum of the same prefix's complete `round_pipeline_seconds`; this excludes
one-time initialization and is therefore conservative in favor of the GP
fraction.

| System | B8 | B12 | B24 | B40 | B40 ideal maximum speedup |
|---|---:|---:|---:|---:|---:|
| Fe--S | 0.738% | 0.721% | 0.776% | 0.689% | 1.00694x |
| Fe--Zr | 0.629% | 0.675% | 0.682% | 0.592% | 1.00595x |
| Ni--S | 0.717% | 0.705% | 0.714% | 0.674% | 1.00678x |

The maximum observed B40 fraction is `0.6888%`, thirteen times below the
`9.09%` gate. Even perfect elimination of all GP numerical work could improve
the real round pipeline by at most about `0.69%`.

Peak parent RSS is approximately `8.73 GiB`, while a dense B40 float64 kernel
matrix is only 12,800 bytes. This does not attribute all memory precisely, but
it rules out the B40 kernel matrix as a material share of observed process
memory; PPD, structures, SOAP, Python objects, and subprocess infrastructure
dominate.

## Fixed-probe numerical benchmark

To prevent the shrinking remaining pool from hiding dense-GP scaling, every
real archive prefix is also evaluated against a fixed, oracle-blind 32-query
SOAP probe matrix. Each operation performs fresh dense linear algebra, BLAS is
limited to one thread, and 20 warm-ups precede 200 measured repetitions.
Factorization-plus-marginal-prediction at B40 has system medians of roughly
`53--59 microseconds`; P95 is below `220 microseconds`. Kernel construction,
factorization-only, marginal prediction, and full-covariance prediction are
reported separately in the immutable result:

```text
E:\DATA\EviMem-RL\outputs\diagnostics\wbm-long-archive-compute-relevance-v2.json
```

SHA256:
`6c907f93226cf174b84c2b2cef5636b8081c52882b79204d1557845494f2a13f`.
The fixed-probe result demonstrates numerical scaling only; it is not a causal
materials metric and cannot rescue the failed end-to-end gate.

## Fixed-GP dispersion confirmation

The separate 32-system LOO diagnostic directly confirms overconfidence:

| Quantity | System-macro mean | 95% system bootstrap interval |
|---|---:|---:|
| Mean squared standardized residual | 16.90 | [10.60, 24.31] |
| Central 50% interval coverage | 34.2% | [28.8%, 39.8%] |
| Central 80% interval coverage | 52.6% | [46.4%, 58.8%] |
| Central 90% interval coverage | 62.4% | [55.7%, 69.0%] |

The result is external at
`E:\DATA\EviMem-RL\outputs\diagnostics\wbm-fixed-gp-loo-dispersion-32sys-v1.json`
with SHA256
`e63c0cee32071ce389c43ca080078d499ae0794c82a1d6d190d807cf982bcaa5`.
It uses no stable-probability threshold inversion.

## Final research decision

- Stop P3C as a main method; do not tune divergence, weights, or capacity.
- Do not implement AKSC as the main method for this WBM paper. WBM provides
  neither a calibrated full-GP target nor a meaningful end-to-end GP bottleneck.
- Preserve AKSC only as a possible numerical-scalability hypothesis for a
  genuinely long-archive setting where the same Amdahl gate is first passed.
- The exact-threshold ceiling v2 rerun remains useful for cleaning approximate
  prior/oracle Brier and log-loss values, but it cannot reverse the compute gate.
- Paper status remains NO-GO.
