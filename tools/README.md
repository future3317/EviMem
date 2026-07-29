# `tools/` — matmem development scripts

This directory contains standalone development and diagnostic scripts. They are
not part of the installed `matmem` package and are intended to be run from the
repository root.

## Categories

### Data preparation

- `build_matpes_protocol_task.py` — build a MatPES protocol-aware development task.
- `build_matpes_confirmatory_task.py` — build a confirmatory-system task with frozen transport.
- `build_jarvis_mp_multifidelity_task.py` — build a JARVIS--MP multi-fidelity task.
- `freeze_matpes_transport_model.py` — fit and freeze a cross-protocol transport model.
- `augment_matpes_structure_embeddings.py` — add structure embeddings to a MatPES task.

### Audits and diagnostics

- `audit_matpes_protocol_pairs.py` — validate protocol pair integrity.
- `audit_matpes_fixed_hull_parity.py` — check fixed-composition hull parity against pymatgen.
- `audit_matpes_sobol_seed_stability.py` — verify scrambled-Sobol seed stability.
- `audit_matpes_source_rollout_opportunity.py` — inspect source-rollout action distributions.
- `diagnose_matpes_horizon_mismatch.py` — diagnose dual-horizon mismatch cases.
- `audit_wbm_official_artifacts.py` — audit official WBM release artifacts.
- `build_wbm_*_manifest.py` — build various WBM candidate/audit manifests.

### Closed-loop runners

- `run_matpes_protocol_closed_loop_exploratory.py` — run an oracle-isolated MatPES exploratory campaign.
- `run_campaign_gated_ic_sarr.py` — run a campaign-gated IC-SARR development pilot.

### Attribution and analysis

- `run_dual_horizon_attribution.py` — offline attribution of Dual-Horizon SARR failure modes.
- `summarize_matpes_source_rollout_opportunity.py` — aggregate source-rollout opportunity summaries.
- `summarize_matpes_confirmatory.py` — summarize confirmatory campaign results.
- `plan_matpes_source_rollout_numerical_audit.py` — plan numerical audits for source-rollout.
- `build_matpes_source_rollout_crossfit.py` — build cross-fitting manifests for source-rollout.

## Convenience aliases

A `Makefile` provides aliases such as `make test`, `make lint`, and example
invocations for the main runners. Run `make help` for details.

All runner scripts require `--task`, `--output`, and protocol-specific inputs.
Example:

```bash
make run-matpes-exploratory TASK=task.json VAULT=vault.json OUTPUT=output.json
```
