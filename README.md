# matmem

`matmem` is an oracle-isolated, protocol-aware materials-discovery research
package. It studies sequential target-protocol queries when a frozen
low-fidelity/source representation is available and every paid target outcome
must remain in an append-only scientific archive.

## Current research status

There is no paper-level positive superiority result. The active development
line is the corrected MatPES PBE--r2SCAN posterior with exact composition-
dependent hull updates, frozen Source-Rollout/IC-SARR components, and the new
system-level **Campaign-Gated IC-SARR** comparison. The campaign gate compares
complete source-margin and IC-SARR policies once at the initial state; it does
not impose a per-round selected-history gate.

The WBM single-protocol task remains a negative/null control for state
compression. Earlier coreset, certificate, and local-gate variants are stopped
or historical diagnostics. Their evidence disposition is retained
in [`docs/EXPERIMENT_LEDGER.md`](docs/EXPERIMENT_LEDGER.md), but their retired
exploratory runners are not part of the live method path.

## Live architecture

- `src/matmem/protocol_closed_loop.py`: typed candidates, observable policy
  state, append-only action/reveal records, oracle vault, and causal hull.
- `src/matmem/protocol_knowledge_gradient.py`: frozen protocol transport,
  hierarchical discrepancy posterior, scrambled-Sobol hull rollout,
  source-margin continuation, and IC-SARR.
- `src/matmem/campaign_gate.py`: one-time campaign-level gate between complete
  source-margin and IC-SARR policies under paired posterior worlds.
- `src/matmem/protocol_policy_worker.py`: oracle-free subprocess policies.
- `src/matmem/wbm*.py`: WBM data contracts, secure runner, raw-release audit,
  and fixed-pool infrastructure.
- `src/matmem/hull_certificate.py` and `src/matmem/environment_transport.py`:
  protocol certificates and robust hull decisions for the separate
  multi-fidelity diagnostics.

## Development commands

Use the `llm` Conda environment locally:

```powershell
conda run --no-capture-output -n llm python -m pip install -e ".[dev]"
conda run --no-capture-output -n llm pytest -q
conda run --no-capture-output -n llm ruff check src tests tools
```

Real datasets, checkpoints, oracle vaults, event logs, and experiment outputs
stay outside Git. The campaign-gated API is exercised by
`tests/test_campaign_gate.py`; a real-data smoke must use a new development
identity and must not reuse opened attribution systems for tuning.

## Scientific constraints

New paper-facing work must preserve all outcomes, use exact chemical-system
units, preserve stoichiometry, isolate the oracle, and report causal-time and
terminal/full-pool metrics separately. Any state compression must reduce to
full-history use in the homogeneous zero-transport-cost null. Read the ledger
and [`docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md`](docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md)
before changing a method or launching an experiment.
