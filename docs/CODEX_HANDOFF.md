# Codex handoff: matmem

Status date: 2026-08-03. This is the operational starting point for a new
coding agent. Read this document, `AGENTS.md`,
`docs/EXPERIMENT_LEDGER.md`, and
`docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md` before changing a method,
restarting an experiment, or editing the manuscript.

## Project in one paragraph

`matmem` studies closed-loop materials discovery when a cheap/source protocol
is available and expensive target-protocol energies are revealed sequentially.
The live MatPES task is PBE -> r2SCAN on exact chemical systems. A policy sees
only source-side observables, the causal phase diagram, and prior reveals; the
oracle vault owns target energies and final-hull labels. Every legal reveal is
append-only and conditions the next posterior. This is not a training project:
CHGNet is frozen and the transport model is a small frozen ridge/kernel fit.

## Scientific status

There is **no paper-level positive state-compression result and no unopened
MatPES holdout**.

| Topic | Current evidence | Required interpretation |
|---|---|---|
| IC-SARR vs source margin | P0-v3 complete B=1..6 core curve on five cross-fit folds, 230 exact systems: B=6 terminal full-pool confirmations `+0.170/system`, 95% CI `[+0.087,+0.257]`, sign-flip `p=0.00012`, 51/160/19 win/tie/loss | A real but development-only terminal-confirmation signal; not external confirmation or universal discovery superiority |
| Causal metrics | B=6 final causal confirmations `+0.009/system`; causal announcements `+0.330/system`; unqueried-pool invalidation changes by `-0.161/system` | Do not call the terminal effect a causal-time discovery or final-causal confirmation gain |
| Cost | P0-v3 core B=6 IC-SARR costs about `+11.28 s/system` on the shared server; targeted amendment reports `+11.02 s/system` | A phase-diagram propagation bottleneck, not a training/GPU bottleneck; shared-server time is not a stable speed claim |
| Mechanism audit | Delta-Hull `+0.165`, source rollout `+0.143`, diagonal covariance `+0.109`, ungated rollout `+0.183`, IC-SARR `+0.170` paired `T` differences at B=6 | Joint worlds matter; the numerical gate is an integration safeguard, not a superiority or calibration guarantee |
| Local Dual-Horizon | Correct implementation, but local double gate has poor oracle alignment and is stopped | Do not tune its thresholds, increase MC, add chemistry rules, or resurrect it on opened systems |
| Campaign-Gated IC-SARR | Implemented, fixture-tested two-policy campaign-level API | Its only real-data smoke was interrupted before an atomic result; it is not evidence and must not be resumed under the old output identity |
| WBM / DACC / P3C / AKSC / CHIC / JARVIS certificates | Stopped or negative lines | Preserve their audit history; do not restore retired runners or use old results as a new claim |

The exact development result, metric definitions, and scope caveats are in
`docs/IC_SARR_FIVE_FOLD_RESULTS.md`. In particular,
`D` = causal-time announcements, `F` = selected-history retained
confirmations, and `T` = complete oracle-pool adjudicated confirmations; every
trace must satisfy `T <= F <= D`.

The currently provisioned MatPES artifact contains 324 eligible exact systems.
The five IC-SARR folds, prior fold-0 development, and the 48-system historical
repartition collectively exhaust them. A genuine MatPES primary evaluation
therefore requires a new upstream release/pool and a newly frozen split; do not
rename any existing systems as a holdout. The JARVIS--MP v4-natural pool is a
different multi-protocol task, not an IC-SARR MatPES holdout.

The current alternative route is the external MAD-1.5 v1 PBE-to-r2SCAN
protocol-shift task. It is documented in
`docs/MAD_1_5_PROTOCOL_SHIFT_TASK.md`; the raw data and generated task/vault
are outside Git at `E:\DATA\MAD-1.5-v1`. This route evaluates a fixed-budget
acquisition curve as a task-level mechanism result, not as an independent
MatPES holdout. MAD provides atomization energy rather than formation energy,
so the first task is explicitly an isolated-atom-reference atomization-hull
proxy. Do not call it a standard solid-state formation hull or silently use
total energy as formation energy.

## Locations and environments

| Resource | Location / command |
|---|---|
| Local code repository | `E:\CODE\EviMem-RL` (`https://github.com/future3317/EviMem.git`) |
| Local paper repository | `E:\PAPER` (`https://github.com/future3317/PAPER.git`) |
| Main manuscript | `E:\PAPER\LEARNING WHAT TO REMEMBER\LEARNING WHAT TO REMEMBER.tex` |
| Local data root | `E:\DATA\EviMem-RL` (never commit its contents) |
| Remote host | `ssh lrh@100.110.148.20` |
| Remote code repository | `~/EviMem-RL/` |
| Remote data / outputs | `/home/workspace/lrh/DATA/EviMem-RL/` |
| Local test environment | `conda run --no-capture-output -n llm ...` |
| Remote experiment environment | `conda run --no-capture-output -n equivcompiler ...` |

For non-interactive SSH sessions, `conda` is not on `PATH` until the remote
Miniconda profile has been loaded. Use:

```bash
source /home/workspace/lrh/miniconda3/etc/profile.d/conda.sh
cd ~/EviMem-RL
conda run --no-capture-output -n equivcompiler pytest -q
```

The canonical remote MatPES task, v5 vault, and fold-0 transport are recorded
with paths and SHA-256 values in the remote
`manifests/matpes-canonical-development-v1.json`; the local repository does not
ship these artifacts. Superseded data are recoverable under
`DATA/EviMem-RL/archive/superseded-20260722/` and are not active inputs.

Last verified baseline at commit `99e1311`: local `llm` and remote
`equivcompiler` each had 207 passing tests. Both passed Ruff lint and format
checks. The remote pre-sync work remains recoverable in
`stash@{0}: codex-remote-presync-20260729`; inspect it with `git stash show`
and do not pop it into the live tree.

## Live code architecture

New code must use focused modules, not the legacy compatibility shim.

- `protocol_closed_loop.py`: typed observable state, immutable action/reveal
  log, oracle vault, causal hull and secure execution.
- `transport.py`: source-to-target ridge/kernel transport fitting and routing.
- `posterior.py`: frozen hierarchical Gaussian target-energy posterior.
- `hull_geometry.py`: shared fixed-composition geometry with exact binary and
  oriented-facet ternary lower-hull specializations. Any speed change requires
  action, sampled-membership, hull, and reveal parity against the existing
  implementation.
- `protocol_acquisition.py`: source margin, Delta-Hull, source rollout,
  IC-SARR, and development diagnostics.
- `campaign_gate.py`: restricted campaign-level source-vs-IC-SARR gate. It is
  an API, not yet a live policy-worker option.
- `tools/build_mad15_protocol_task.py`: builds the external MAD-1.5
  PBE-to-r2SCAN task and keeps target atomization outcomes in the external
  vault.
- `protocol_policy_worker.py` and `worker_subprocess.py`: policy-facing,
  oracle-free subprocess boundary.
- `configs/`, `constants.py`, `utils.py`, and `policy_registry.py`: shared
  configuration and conventions.

`protocol_knowledge_gradient.py` is a thin re-export compatibility shim. Do
not add functionality to it. Import from the focused modules above.

## Non-negotiable scientific and safety rules

1. The policy must never read target energies, final-hull labels, evaluator
   data, or the oracle vault before it commits an action.
2. Every selected target result is appended to the immutable archive and
   contributes to the deployed posterior. Do not introduce outcome-selected
   memory/coreset deletion.
3. Use exact chemical systems as split and statistical units; preserve cell
   stoichiometry and distinguish total from per-atom energy.
4. Rebuild composition-dependent causal hulls from legal phase entries. Do not
   substitute a scalar hull margin or mix future competitor phases into a live
   decision.
5. Keep `D`, `F`, and `T` separate. Candidate rows and sequential rounds are
   not independent replicates; uncertainty is system-clustered.
6. Do not tune a posterior, score, threshold, seed, continuation, or chemistry
   heuristic on opened development systems and then present it as validation.
7. New state-compression work must reduce exactly to full history in the
   homogeneous zero-transport-cost null and must first measure a binding
   end-to-end constraint.

## Normal development workflow

1. Inspect `git status`, `AGENTS.md`, the ledger, and the active task manifest.
2. State in the experiment note which failed assumption a proposed method
   changes. If it changes none, it is not an authorized new experiment.
3. Make a small coherent change in the focused module and add/update a focused
   regression test. Prefer Pydantic, NumPy, SciPy, scikit-learn, and pymatgen
   rather than local substitutes.
4. Locally run:

   ```powershell
   conda run --no-capture-output -n llm pytest -q
   conda run --no-capture-output -n llm ruff check src/matmem tests tools
   conda run --no-capture-output -n llm ruff format --check src/matmem tests tools
   ```

5. Before a real remote run, verify the frozen task/vault/transport checksum,
   oracle isolation, and resource limits. Write outputs outside Git with a new
   output identity; never overwrite an incomplete or historical artifact.
6. Use the remote `equivcompiler` environment for real data experiments and
   rerun the relevant parity tests there before interpreting a timing or policy
   result.
7. Update the ledger when a result changes an evidence disposition; update the
   manuscript only with the correct scope statement and after the relevant
   result is complete.

## Style, Git, and handoff discipline

- Python: Python 3.11, Ruff line length 100, typed/Pydantic data contracts,
  deterministic hashes/seeds/tie-breaks, fail-closed validation, and no hidden
  fallback that changes a policy.
- Keep code DRY by using the shared layers; do not copy hull/posterior/worker
  implementations into a new runner.
- Do not commit datasets, artifacts, checkpoints, vaults, event logs, outputs,
  caches, credentials, `.env`, or user review notes.
- Preserve unrelated dirty-worktree files. The root Chinese Markdown notes are
  user documents and are deliberately untracked.
- Commit and push after every coherent, validated unit of work: typically one
  method/bug fix plus its tests and documentation. Do not batch several
  unrelated refactors or include generated data. Use a concise imperative
  commit message and report the commit hash in the handoff.
- Keep `main` buildable. Before merging or pushing a substantial change, run
  the full local test and lint commands above. If remote code is synchronized,
  record its commit and whether its test inventory differs from local.

## First actions for the next agent

1. Confirm local and remote `main` point to the intended commit and read the
   canonical remote manifest without copying data into Git.
2. Decide whether the next work is an engineering-parity optimization or a
   genuinely new campaign-level development task. Do not silently turn the
   interrupted campaign-gate smoke into either.
3. For an engineering change, use the frozen IC-SARR traces as action/hull/
   reveal parity regressions before timing it.
4. For a scientific change, create a new development identity and explain why
   it addresses the documented joint-advantage/gate failure rather than merely
   tuning an opened MatPES task.

The current engineering freeze is complete at the ternary specialization:
local and remote `equivcompiler` both pass 207 tests and Ruff, and the remote
four-system fixed-backend audit passed action and sampled-membership parity.
The provisioned DATA roots still contain no second MatPES release or new
same-configuration protocol pool. The public `v2025.3.10` MatPES tag is a
software release, not evidence of a second paired data release, and the
expected `MatPES_2025_2` PBE/r2SCAN objects were unavailable. The next
scientific action is therefore to obtain or independently construct and audit a
genuinely new same-configuration pool, freeze its development/holdout split
before fitting or opening outcomes, and evaluate frozen IC-SARR once; the
existing 324 systems and JARVIS v4-natural remain closed to that purpose.
