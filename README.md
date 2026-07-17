# matmem

**Decision-aware calibration coresets and protocol-aware residual memory for materials discovery.**

The current paper-facing hypothesis is a **Decision-Aware Calibration Coreset (DACC)**
for a frozen materials predictor. `K` bounds the residual witnesses used for online
calibration; every revealed DFT result remains in an immutable audit archive. Real-WBM
execution uses one allow-listed subprocess policy path, an oracle-isolated reveal
boundary, and a composition-dependent causal hull.

The corrected CAW-Joint method remains a frozen method-level NO-GO. The small WBM
matched-trace pilot gives preliminary DACC mechanism evidence, not a superiority
claim; survival-conditioned acquisition is currently negative and paused. See the
[current DACC specification](docs/WBM_CALIBRATION_CORESET_AMENDMENT.md),
[engineering WBM result](docs/WBM_ENGINEERING_P1_P15_AND_PILOT.md), and
[research iteration history](docs/RESEARCH_ITERATION_HISTORY.md).

## Package overview

`src/matmem/` provides:

- `cards`, `identity`, `protocols`: native material records, identity, and compatibility contracts;
- `residual`, `residual_posterior`: certificate-compatible residual retrieval and a fixed-kernel GP over SOAP embeddings;
- `calibration_utility`, `coreset`: decision-aware calibration coresets under
  `F_t(M)=sum_u max_m G_t(u,m)` with exact streaming one-swap updates;
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

The frozen exact-system grid is constructed and executed with external
manifests and outputs only:

```powershell
conda run --no-capture-output -n llm python tools/build_wbm_frozen_grid_manifest.py --help
conda run --no-capture-output -n llm python tools/run_wbm_frozen_grid.py --help
```

The grid uses `B={4,8,12}`, valid `K={1,2,4}`, all candidates from each selected
exact chemical system, GP variance as the primary calibration baseline, and
joint-posterior risk only in the two frozen sentinel cells. Survival is absent.

## Development

```powershell
conda run --no-capture-output -n llm python -m pytest -q
conda run --no-capture-output -n llm ruff check .
```
