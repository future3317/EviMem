# E55 Numerical Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a frozen MatPES numerical-convergence panel for Delta-Hull and CAL-style acquisition, and extract the already completed E32 two-step rollout budget curve without regenerating E32 trajectories.

**Architecture:** A deterministic roster builder creates five three-system CAL sub-folds from the existing E52 cross-fit manifest. A stage-selectable launcher emits write-once Delta and CAL units through the existing secure closed-loop runner. One summarizer audits E55 outputs and one read-only summarizer extracts the E32 rollout curve; neither changes acquisition code.

**Tech Stack:** Python 3.11, Pydantic/NumPy/SciPy, pytest, Ruff, existing `matmem` secure closed-loop runner, external JSON artifacts, PowerShell locally, SSH to `dbcloud` for execution.

## Global Constraints

- Use `conda run --no-capture-output -n llm ...` for Python, pytest, and Ruff.
- Preserve the oracle/reveal boundary and append-only archive semantics.
- Do not modify, copy over, or regenerate E32/E53/E54 trajectories.
- Use task/vault/cross-fit identities from the E52 100% development pool.
- Keep raw manifests, traces, logs, failures, and summaries outside Git.
- Preserve unrelated dirty files in the local and remote worktrees.
- E55 is development-only MatPES numerical evidence, not an external validation.

---

### Task 1: Deterministic CAL convergence roster

**Files:**
- Create: `tools/build_e55_convergence_manifest.py`
- Create: `tests/test_e55_convergence_manifest.py`

**Interfaces:**
- Consumes: E52 task JSON and five-fold cross-fit JSON.
- Produces: `build(task_path: Path, development_crossfit_path: Path, output: Path) -> dict[str, Any]` and a five-fold manifest with three query systems per fold, original fit systems, candidate-count strata, hashes, and an outcome-independence declaration.

- [ ] Write tests using a synthetic task/cross-fit fixture that assert deterministic low/middle/high selection, original fit-roster preservation, fit-element support, task mismatch rejection, and overwrite refusal.
- [ ] Run `conda run --no-capture-output -n llm pytest -q tests/test_e55_convergence_manifest.py` and verify the tests fail because the builder is absent.
- [ ] Implement the minimal builder and CLI. Candidate counts come from task rows grouped by exact chemical system; no vault is accepted by the interface.
- [ ] Run the focused test and `conda run --no-capture-output -n llm ruff check tools/build_e55_convergence_manifest.py tests/test_e55_convergence_manifest.py`.

### Task 2: Write-once E55 campaign launcher

**Files:**
- Create: `tools/run_matpes_e55_convergence.py`
- Create: `tests/test_matpes_e55_convergence.py`

**Interfaces:**
- Consumes: E52 full-pool root, E55 CAL manifest, output root, unified runner, and explicit stages.
- Produces: 25 Delta units (`5 folds x M={64,128,256,512,1024}`) and 45 CAL units (`5 folds x M={100,200,400} x K={5,10,20}`), each with an identity manifest, output, log, and terminal failure marker.

- [ ] Write tests that assert exact unit counts, grids, policies, budgets, seed, backend, workers, timeout, source hashes, output names, resume validation, and refusal to overwrite failure roots.
- [ ] Run `conda run --no-capture-output -n llm pytest -q tests/test_matpes_e55_convergence.py` and verify the tests fail because the launcher is absent.
- [ ] Implement the launcher by reusing the command shape in `tools/run_matpes_cal_style_campaign.py`; Delta units use one policy and the original five-fold manifest, CAL units use one policy and the derived five-fold manifest.
- [ ] Run focused pytest and Ruff for the new launcher/test.

### Task 3: E55 and E32 read-only summaries

**Files:**
- Create: `tools/summarize_matpes_e55_convergence.py`
- Create: `tools/summarize_e32_rollout_curve.py`
- Create: `tests/test_matpes_e55_summaries.py`

**Interfaces:**
- `summarize_e55(input_root: Path, output: Path) -> dict[str, Any]` audits all 70 E55 units and compares each configuration with its declared high-precision reference.
- `summarize_e32(input_root: Path, output: Path) -> dict[str, Any]` reads E32 `B=1,...,6` artifacts, verifies 230 unique systems and frozen identities, and emits only `B=2,...,6` Delta-Hull versus anchored-rollout metrics plus trapezoidal area.

- [ ] Write fixture tests for missing/duplicate units, hash/config/policy mismatch, action agreement on common states, score/rank availability, terminal `T`, runtime quantiles, E32 paired effects, and output overwrite refusal.
- [ ] Run the focused test and verify it fails because the summarizers are absent.
- [ ] Implement compact shared private helpers only where both summarizers use identical logic; otherwise keep the scripts independent.
- [ ] Run focused pytest and Ruff for both summarizers/tests.

### Task 4: Contract and repository consolidation

**Files:**
- Modify: `docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md`
- Modify: `docs/PROTOCOL_INDEX.md`
- Modify: `README.md` only if its stable entry-point list is stale after consolidation.
- Modify/remove/archive only files proven superseded by code references, Git history, and the consolidation audit.

**Interfaces:**
- Current truth remains the short paper contract plus authoritative experiment ledger.
- Historical scientific dispositions remain in `docs/EXPERIMENT_LEDGER.md`; no new cleanup report is created.

- [ ] Record E55's frozen design, development-only scope, and the fact that the rollout curve is a read-only E32 reanalysis.
- [ ] Compare `codex/e52-primary-calibration` with `main`; integrate unique valid work or document why it is superseded before removing its worktree/branch.
- [ ] Classify root review drafts and redundant protocol documents using current references and Git history; merge unique current facts, archive provenance-bearing retired material, and delete only content with no unique information.
- [ ] Search for stale links/names and run focused documentation/code-entry checks affected by each action.

### Task 5: Verify, deploy, and launch

**Files:**
- External only: `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e55_numerical_convergence_20260812`

**Interfaces:**
- The committed `main` revision is synchronized to `/home/workspace/lrh/EviMem-RL` without disturbing its unrelated dirty E54 runtime files.
- The launcher PID and top-level log are recorded in the E55 external root.

- [ ] Run focused pytest, then the relevant secure-runner/CAL tests, then Ruff on all E55 files.
- [ ] Inspect local diff/status and commit only E55/consolidation files.
- [ ] Transfer the committed files or fast-forward the remote repo while preserving remote dirty files; verify exact code revision and input paths.
- [ ] Build the external CAL manifest, run a one-system smoke in a separate smoke root, audit it, then launch the full E55 Delta and CAL stages with bounded parallelism.
- [ ] Create or update a three-hour heartbeat monitor that reports completed units, failure markers, launcher/worker status, and transitions to summary/paper work only after all audits pass.

### Task 6: Results and manuscript integration

**Files:**
- Modify later, only after audit: `E:\PAPER\LEARNING WHAT TO REMEMBER\appendix_delayed_search.tex`
- Modify later if needed: `E:\PAPER\LEARNING WHAT TO REMEMBER\main_story.tex`
- Modify later if needed: paper figure source/assets and final PDF.

**Interfaces:**
- Consumes only audited E55 and E32 read-only summaries.
- Produces a compact appendix convergence result and complete rollout curve; existing headline effects remain unchanged.

- [ ] Generate both external summaries and verify hashes, rosters, complete grids, no failures, and reference configurations.
- [ ] Add only evidence-supported convergence and rollout statements/tables; do not promote runtime or numerical diagnostics into headline claims.
- [ ] Rebuild the paper with halt-on-error, verify main text at most nine pages, references/labels/overfull/font embedding, and visually inspect affected pages.
- [ ] Commit only task-related paper source/assets/PDF to the paper repository's current `main`.
