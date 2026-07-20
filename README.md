# matmem

**Auditable decision state for closed-loop materials discovery.**

The live research question is **Decision-Sufficient Scientific State**: what is
the least costly observable, protocol-valid online state that preserves the
scientific decisions available from the complete immutable archive? The next
materials hypothesis is a Certified Hull-Decision State whose active evidence
has a legal, uncertainty-aware influence under the target protocol. Its
all-outcome state, source-environment transport and robust hull-certificate
algorithm are implemented and tested. The fresh v4 evaluation is a
method-level NO-GO rather than a positive result.

The corrected CAW-Joint method remains a frozen method-level NO-GO. The current
real-WBM diagnosis covers two disjoint initial-structure panels: 32 exact
systems and 683 candidates at `B=8,K=2`. It found and fixed both a parity-energy
bug and post-DFT relaxed-structure leakage in SOAP. On the corrected panels,
P3C's pooled Brier difference from GP variance is essentially zero, while CRPS,
log loss, RMSE, Gaussian NLL, and runtime are worse on average. The fresh
next-16 panel does not reproduce the earlier weak probability-metric signal.
P3C is therefore a stopped method-level NO-GO, not a superiority claim. AKSC was
considered as a separate all-outcome kernel-sketch architecture, but a
checkpointed B40 compute-relevance gate now rejects it as this paper's WBM main
method: full-history GP numerical work is at most 0.689% of real round-pipeline
time, so perfect elimination would yield at most a 1.00694x ideal speedup. WBM
is consequently the homogeneous, low-compute-cost null regime in which full
history should remain active. The
[authoritative experiment and evidence ledger](docs/EXPERIMENT_LEDGER.md) is the
single required starting point before proposing another method, editing the
paper's historical account or rerunning an old experiment. It preserves the
complete CAW-Joint -> DACC -> P3C -> outcome-contribution
deletion/reference-mismatch -> AKSC -> B40 -> JARVIS--MP v1/v3/v4 chain, evidence
dispositions, external paths and hashes, invalidated/interrupted/sealed runs,
recovery tags, statistical boundaries and future authorization rules. Detailed
technical annexes remain available through its source map; they do not override
the ledger and must not be deleted until any unique audit content has been
migrated there. The separate
[live decision-state specification](docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md)
defines the current paper-facing hypothesis rather than the historical record.

A real JARVIS--MP multi-protocol task has now been constructed from 1,658
same-material structure-matched pairs. The v1 global-affine evaluation passes
all oracle/null/replay/no-deletion implementation gates but is a scientific
NO-GO: held-out transport violations are 45.32%, and the rank-16 state does not
improve hull decision cost or regret. A fresh v3 composition-aware calibration
also stops before evaluation because its exact-system-clustered radius is
0.177264 eV/atom, above the frozen 0.15 ceiling. Its 12 fresh evaluation systems
remain unopened. See the
[JARVIS--MP preregistration and frozen results](docs/JARVIS_MP_MULTIFIDELITY_PREREGISTRATION.md).

The v4 continuation replaced hand-built transport rules with frozen
CHGNet-0.3.0 source-structure features, exact-system conformal calibration,
all-target-outcome natural parameters and a robust hull LP. It used 210 fresh
calibration systems and opened 72 disjoint evaluation systems only after the
decision-level gate passed. Certified decisions were sound whenever the
simultaneous interval event held, but event coverage was 50/56 supported
systems (89.29%, below the frozen 90% gate), and the method was significantly
worse than naive source-as-target on hull error and regret. See the
[v4 certificate audit](docs/ENVIRONMENT_HULL_CERTIFICATE_2026-07-20.md).

## Package overview

`src/matmem/` provides:

- `cards`, `identity`, `protocols`: native material records, identity, and compatibility contracts;
- `activation`, `sufficient_state`: fail-closed protocol influence and
  fixed-rank linear--Gaussian natural parameters updated by every legal
  outcome; neither module exposes a capacity/similarity deletion path;
- `environment_transport`, `hull_certificate`: frozen source-environment
  directed transport, leverage-scaled clustered intervals, robust
  stable/unstable/abstain hull LPs and certified action sets;
- `residual`, `residual_posterior`: certificate-compatible residual retrieval and a fixed-kernel GP over SOAP embeddings;
- `ceiling_diagnostics`: evaluator-only effective-dimension, residual--kernel
  alignment, LOO dispersion/coverage, and compute-ceiling checks;
- `calibration_utility`, `coreset`: fixed-reference proper divergences,
  asymmetric reference-decision regret, exact online drop-one P3C, and an
  archive-exact diagnostic; the older facility objective remains an explicit
  legacy comparator;
- `acquisition`, `baselines`: oracle-blind acquisition policies and bounded-memory baselines;
- `hull_engine`, `wbm`, `wbm_secure`: composition-dependent causal hull, WBM artifact audit, and the sole secure real-WBM runner.

The repository intentionally contains no downloaded datasets, model checkpoints, or
raw experiment outputs. It contains aggregate audit/pilot reports, but does not claim
formal paper results beyond the preliminary engineering pilot.

## Environment and installation

Python 3.11 or newer is required. The maintained development environment is the Conda
environment named `llm`:

```powershell
conda create -n llm python=3.11 -y
conda run --no-capture-output -n llm python -m pip install -e ".[dev]"
```

## Running the WBM calibration engineering pilot

```powershell
conda run --no-capture-output -n llm python tools/run_wbm_calibration_engineering_pilot.py --help
```

Frozen-grid execution requires an external calibration-freeze manifest and the
matching registered repository config; both hashes are embedded in every trace
and summary. Real inputs and outputs remain external:

```powershell
conda run --no-capture-output -n llm python tools/build_wbm_frozen_grid_manifest.py --help
conda run --no-capture-output -n llm python tools/run_wbm_frozen_grid.py --help
```

The old DACC grid was manually paused and is not resumed under P3C. The original
P3C P1 diagnostic compared legacy facility DACC, joint self-risk, four proper
projection variants, a theory-fixed zero-decision-regret variant, and GP
variance under the same frozen GP and acquisition trace. Its immutable summary
SHA-256 is `4facffd371820bf25678e16e8311bb4c1b7c798f363661c53a1e55102a6109fa`.
The authoritative follow-up is external to Git at
`E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-reference-path-selection-b8-k2-v5`:
the summary SHA-256 is
`0d25f251a1d1ede6dc2b63c5e2ed7c8782fde716f984b45d9c93060ea4b2f9b3`
and the deterministic decomposition SHA-256 is
`149ed9562d5a6c6d550c550f99711542df2733c58236cc3f7daf802a9461ef1d`.
Failed `v2`/`v3` attempts have no valid summary and are excluded; `v4` is
scientifically trace-equivalent but has an incomplete lazy-GP timing definition,
so only `v5` may be cited. No broader P3C grid or MADE run is authorized.

## Development

```powershell
conda run --no-capture-output -n llm python -m pytest -q
conda run --no-capture-output -n llm ruff check .
```
