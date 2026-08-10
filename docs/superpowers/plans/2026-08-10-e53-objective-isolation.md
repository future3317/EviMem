# E53 Objective-Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a posterior-sampling-matched local-hull probability comparator, run frozen development and secondary MatPES reruns, produce unified inference and theory-linked diagnostics, and restructure the paper around the isolated adjudication effect.

**Architecture:** A new canonical policy computes current-history hull membership on the same Sobol posterior worlds used by Delta-Hull. A new E53 campaign reuses the frozen E52 task/vault/split manifests without overwriting E52 outputs. Standalone summarizers produce immutable external JSON summaries for objective contrasts and theory diagnostics; the manuscript consumes only audited summary values and rendered vector figures.

**Tech Stack:** Python 3.12, NumPy, SciPy, Pydantic, pytest, Ruff, Matplotlib, LaTeX/latexmk, Poppler, Git.

## Global Constraints

- Work directly on current `main`; do not create a worktree, branch, repository, or remote.
- Preserve all pre-existing dirty files in `E:\CODE\EviMem-RL` and stage only E53 files/edits.
- Keep every raw task, vault, trace, and experiment output outside Git.
- Use `conda run --no-capture-output -n llm ...` for Python, pytest, and Ruff.
- Policy code never receives oracle energies or complete-pool labels before legal reveal/evaluation.
- The 94-system result is a secondary held-out MatPES rerun, never untouched or external validation.
- Modify paper files only inside `E:\PAPER\LEARNING WHAT TO REMEMBER`; do not touch other `E:\PAPER` projects.
- Use `apply_patch` for source edits.

---

### Task 1: Matched local-hull probability acquisition

**Files:**
- Modify: `src/matmem/protocol_acquisition.py`
- Modify: `src/matmem/policy_registry.py`
- Modify: `src/matmem/protocol_policy_worker.py`
- Modify: `src/matmem/__init__.py`
- Test: `tests/test_protocol_knowledge_gradient.py`
- Test: `tests/test_protocol_closed_loop.py`
- Test: `tests/test_matpes_protocol_task.py`

**Interfaces:**
- Produces: `matched_local_hull_probability(...) -> DeltaHullActiveSearchResult`
- Produces policy identity: `matched_local_hull_probability`
- Consumes the same `ProtocolTargetEnergyPosterior`, sample count, seed, costs, and immutable IDs as Delta-Hull.

- [ ] **Step 1: Add failing acquisition tests**

Add tests showing that the policy (a) uses sampled probabilities rather than posterior means, (b) evaluates each candidate against supplied current competing-hull energies, (c) rejects unequal costs, and (d) is deterministic for a fixed seed.

- [ ] **Step 2: Run focused tests and confirm failure**

Run:

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_protocol_knowledge_gradient.py -k matched_local_hull_probability
```

Expected: failure because the function is not defined/exported.

- [ ] **Step 3: Implement the acquisition function**

Use `_sample_gaussian(mean, covariance, sample_count, seed)`, compare each sampled self-energy with the fixed current competing-hull energy using the registered tolerance semantics, return sample-mean probabilities in `DeltaHullActiveSearchResult`, and reject invalid/non-equal costs exactly as Delta-Hull does.

- [ ] **Step 4: Add the canonical worker policy**

Register `MATCHED_LOCAL_HULL_PROBABILITY`, mark it transport-required and hull-aware, dispatch it in `protocol_policy_worker.py`, log candidate IDs/probabilities/selection, and preserve immutable-ID tie-breaking.

- [ ] **Step 5: Add worker/registry tests**

Extend policy roster tests to prove transport is required, diagnostics are emitted, and the secure closed-loop runner can execute the new policy without oracle access.

- [ ] **Step 6: Run focused policy tests**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_protocol_knowledge_gradient.py tests/test_protocol_closed_loop.py tests/test_matpes_protocol_task.py
conda run --no-capture-output -n llm ruff check src/matmem/protocol_acquisition.py src/matmem/policy_registry.py src/matmem/protocol_policy_worker.py src/matmem/__init__.py tests/test_protocol_knowledge_gradient.py tests/test_protocol_closed_loop.py tests/test_matpes_protocol_task.py
```

Expected: all tests pass and Ruff reports no errors.

### Task 2: Frozen E53 campaign launcher

**Files:**
- Create: `tools/run_matpes_e53_objective_isolation.py`
- Test: `tests/test_matpes_e53_campaign.py`
- Modify: `docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md`
- Modify: `docs/EXPERIMENT_LEDGER.md`

**Interfaces:**
- Produces five development units at `development/fold{1..5}-b6.json`.
- Produces one secondary unit at `secondary/heldout-b6.json` only when explicitly invoked after development freeze.
- Policy roster is exactly target margin, matched Local-prob, and Delta-Hull.

- [ ] **Step 1: Write failing launcher tests**

Test exact unit counts, policy roster/order, `B=6`, seed, posterior sample count, external-output guard, development fold indices, and secondary manifest use.

- [ ] **Step 2: Run the launcher tests and confirm failure**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_matpes_e53_campaign.py
```

Expected: import failure because the launcher does not exist.

- [ ] **Step 3: Implement the launcher**

Reuse the E52 full-pool task/vault and split manifests. Separate `development` and `secondary` stages so the latter cannot start accidentally. Refuse overwrites, persist failure markers, verify hashes and policy rosters on resume, and keep outputs outside Git.

- [ ] **Step 4: Register E53 before execution**

Append a concise E53 contract/ledger entry containing policy identity, 230/94 boundaries, numerical settings, output roots, estimands, inference, and the prohibition on calling the 94 systems untouched/external.

- [ ] **Step 5: Verify launcher and documentation**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_matpes_e53_campaign.py tests/test_matpes_e52_cleanroom_manifests.py
conda run --no-capture-output -n llm ruff check tools/run_matpes_e53_objective_isolation.py tests/test_matpes_e53_campaign.py
git diff --check -- docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md docs/EXPERIMENT_LEDGER.md
```

Expected: all checks pass.

### Task 3: Unified paired inference and objective summary

**Files:**
- Create: `src/matmem/paired_randomization.py`
- Create: `tools/summarize_e53_objective_isolation.py`
- Test: `tests/test_paired_randomization.py`
- Test: `tests/test_e53_objective_isolation_summary.py`

**Interfaces:**
- Produces `paired_sign_randomization(values, confidence, draws, seed)` with mean effect, inverted interval, p-value, and W/T/L.
- Produces absolute `T` and paired contrasts for `B=1..6` plus integrated budget effect for development and secondary panels.

- [ ] **Step 1: Write failing inference tests**

Use small vectors with exact-enumeration reference results. Test symmetry, zero-effect behavior, deterministic Monte Carlo, interval/test consistency, and all-tie handling.

- [ ] **Step 2: Confirm inference tests fail**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_paired_randomization.py
```

- [ ] **Step 3: Implement sign-randomization and interval inversion**

Use exact sign enumeration when the nonzero pair count is small; otherwise use deterministic Rademacher draws. Invert the same two-sided test over an additive shift by monotone root search. Store draw count, seed, analysis unit, and resolution.

- [ ] **Step 4: Write failing summary tests**

Fixtures contain `B=6` selected IDs and evaluator final labels. Test prefix-derived absolute means, Delta-minus-Local and Delta-minus-target contrasts, roster/hash/count rejection, and development/secondary separation.

- [ ] **Step 5: Implement the E53 summarizer**

Refuse output inside Git or overwrite; verify full policy roster and system uniqueness; derive prefixes; compute frozen inference; emit auditable input hashes and result metadata.

- [ ] **Step 6: Verify inference and summary tools**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_paired_randomization.py tests/test_e53_objective_isolation_summary.py
conda run --no-capture-output -n llm ruff check src/matmem/paired_randomization.py tools/summarize_e53_objective_isolation.py tests/test_paired_randomization.py tests/test_e53_objective_isolation_summary.py
```

Expected: all checks pass.

### Task 4: Theory-linked observed-path diagnostics

**Files:**
- Create: `tools/summarize_e53_rank_diagnostics.py`
- Test: `tests/test_e53_rank_diagnostics.py`

**Interfaces:**
- Consumes completed Delta-Hull decision diagnostics.
- Produces equal-system membership drift, top-rank preservation, full-rank preservation, and linked two-step headroom summaries.

- [ ] **Step 1: Inspect one existing E52 trace schema read-only**

Confirm candidate IDs and pre-reveal probabilities exist in each decision diagnostic. Do not modify/open new oracle data.

- [ ] **Step 2: Write failing diagnostic tests**

Use two synthetic systems with unequal candidate/state counts to prove equal-system weighting. Test candidate intersection, immutable-ID rank order, top-rank preservation, full-rank preservation, and malformed trace rejection.

- [ ] **Step 3: Implement the diagnostic summarizer**

Join consecutive policy states only within the same system and trajectory. Compare candidates legal in both states. Emit pooled counts only as secondary metadata; all reported point estimates are equal-system means.

- [ ] **Step 4: Verify diagnostics**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_e53_rank_diagnostics.py
conda run --no-capture-output -n llm ruff check tools/summarize_e53_rank_diagnostics.py tests/test_e53_rank_diagnostics.py
```

Expected: all checks pass.

### Task 5: Code freeze and remote E53-A development execution

**Files:**
- External outputs only under `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e53_objective_isolation_20260810`

**Interfaces:**
- Consumes committed code and frozen E52 manifests.
- Produces five complete development JSON units and no failures.

- [ ] **Step 1: Run the relevant local regression suite**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_protocol_knowledge_gradient.py tests/test_protocol_closed_loop.py tests/test_matpes_protocol_task.py tests/test_matpes_e53_campaign.py tests/test_paired_randomization.py tests/test_e53_objective_isolation_summary.py tests/test_e53_rank_diagnostics.py tests/test_matpes_e52_cleanroom_manifests.py tests/test_e52_objective_prefixes.py tests/test_matpes_membership_calibration.py
conda run --no-capture-output -n llm ruff check src/matmem/protocol_acquisition.py src/matmem/policy_registry.py src/matmem/protocol_policy_worker.py src/matmem/paired_randomization.py tools/run_matpes_e53_objective_isolation.py tools/summarize_e53_objective_isolation.py tools/summarize_e53_rank_diagnostics.py
```

- [ ] **Step 2: Commit only E53 code/docs/tests**

Inspect `git status` and `git diff --cached`; exclude every pre-existing dirty file. Commit the frozen implementation to current `main`.

- [ ] **Step 3: Synchronize committed code to `dbcloud`**

Use Git transport without stashing, resetting, or touching unrelated server processes. Verify remote HEAD and relevant file hashes.

- [ ] **Step 4: Launch only E53-A**

Run the five development units with the frozen 100% pool manifests and 1024 posterior worlds. Do not invoke the secondary stage.

- [ ] **Step 5: Monitor and audit E53-A**

Check only the E53 output root. Require five complete JSONs, zero failure markers, exact task/vault/crossfit hashes, 46 query/184 fit systems per fold, 230 unique query systems, three-policy roster, `B=6`, and prefix consistency.

- [ ] **Step 6: Run development summaries**

Generate external objective and rank-diagnostic JSON summaries. Record hashes. If any audit fails, stop before E53-B and manuscript changes.

### Task 6: Frozen E53-B secondary execution

**Files:**
- External outputs only under the E53 root.

- [ ] **Step 1: Verify the secondary gate**

Recheck committed HEAD, development summary hashes, frozen analysis code, 94 query systems, 230 fit systems, disjoint fit/query rosters, and `secondary_confirmation_is_untouched=false`.

- [ ] **Step 2: Run the secondary stage once**

Execute the three-policy `B=6` unit with posterior MC1024 and no tuning or intermediate outcome-based changes.

- [ ] **Step 3: Audit and summarize E53-B**

Require one complete JSON, zero failures, exact hashes, 94 query/230 fit systems, three-policy roster, `B=6`, and consistent prefixes. Run the already frozen summarizer once and record hashes.

- [ ] **Step 4: Freeze scientific interpretation**

Write the completed counts, hashes, contrasts, and supported claim boundary to the E53 ledger entry. Do not relabel the panel as untouched/external regardless of result.

### Task 7: Publication figure and table artifacts

**Files:**
- Create: `tools/render_e53_objective_isolation_figure.py`
- Test: `tests/test_render_e53_objective_isolation_figure.py`
- Create/replace inside paper repo: `figures/e53_objective_isolation.pdf`

**Interfaces:**
- Consumes audited external E53 summary JSON only.
- Produces a vector two-panel figure: absolute policy performance and paired complete-minus-local effect over budget/development-secondary panels.

- [ ] **Step 1: Read the scientific-figure-making skill completely**

- [ ] **Step 2: Write a failing rendering test**

Verify expected panels, policy labels, vector PDF creation, and rejection of an incomplete summary.

- [ ] **Step 3: Implement and render the figure**

Use marker plus line-style encoding, colorblind-safe colors, consistent paper font sizes, no paragraph-length text in panels, and three-decimal manuscript precision.

- [ ] **Step 4: Verify the figure**

Run focused pytest/Ruff, inspect PDF metadata/font embedding, render to PNG, and visually check at final column width.

### Task 8: Manuscript restructuring and notation cleanup

**Files:**
- Modify: `E:\PAPER\LEARNING WHAT TO REMEMBER\LEARNING WHAT TO REMEMBER.tex`
- Modify: `E:\PAPER\LEARNING WHAT TO REMEMBER\main_story.tex`
- Modify: `E:\PAPER\LEARNING WHAT TO REMEMBER\main_method.tex`
- Modify: `E:\PAPER\LEARNING WHAT TO REMEMBER\appendix_delayed_search.tex`
- Modify: `E:\PAPER\LEARNING WHAT TO REMEMBER\iclr2026_conference.bib` only if a cited source is missing
- Modify: paper figure PDFs only when referenced by the final source

- [ ] **Step 1: Run `git status` in the paper repository**

Require current `main`; preserve any user changes discovered after the design freeze.

- [ ] **Step 2: Rewrite the abstract and introduction**

Remove protocol bookkeeping. State the globally adjudicated problem, Delta-Hull as the one-step Bayes action, the theoretical greedy boundary, and the strongest E53 matched result with accurate MatPES scope.

- [ ] **Step 3: Reorganize experiments**

Use subsections for protocol, adjudication isolation, lookahead after alignment, and mechanism/efficiency. Move controlled POMDP, finite-world DP, MAD proxy, numerical convergence, calibration, and evaluator sensitivity to appendix scope.

- [ ] **Step 4: Update tables and figures**

Change the main policy table to `Policy | Adjudicator | Score | Lookahead`. Add a compact result table with absolute `T`, paired effect, and unified interval. Make E53 the core Figure 2 and retain a simplified graphical Figure 3.

- [ ] **Step 5: Clean notation and claims**

Define updated history once, use posterior-world `\tilde e`, move finite-MC caveats to the appendix, replace overstrong theory-explains phrasing with the strongest wording supported by the observed-path diagnostics, and unify hyphenation/precision.

- [ ] **Step 6: Update appendix evidence**

Add full E53 curves, inference definition, absolute outcomes, rank diagnostics, membership calibration, and accurate 94-system provenance. Remove redundant defensive prose while retaining one concise protocol/provenance note.

### Task 9: Full verification and commits

**Files:**
- Modify generated paper PDF: `E:\PAPER\LEARNING WHAT TO REMEMBER\LEARNING WHAT TO REMEMBER.pdf`

- [ ] **Step 1: Run full relevant code checks**

Run all E53 tests plus the existing E52 and protocol regression tests. Run Ruff on every task-modified Python file. Confirm zero failures/errors from fresh output.

- [ ] **Step 2: Compile paper halt-on-error**

Run the repository's LaTeX build until references stabilize. Require exit code zero, no undefined references/citations, and no overfull boxes.

- [ ] **Step 3: Audit the PDF**

Check main text is at most 9 pages, references start after main text, all fonts are embedded, and page count is plausible. Render the title page, all main figure/table pages, conclusion, and affected appendix pages; visually inspect readability and clipping.

- [ ] **Step 4: Verify repository scope**

Run `git status` and inspect diffs in both repositories. Confirm no raw outputs, unrelated dirty files, other paper projects, worktrees, branches, or remotes were added.

- [ ] **Step 5: Commit code and paper separately on current main**

Stage only E53-related code/docs/tests in `EviMem-RL` and only requested manuscript/figure/PDF files in the independent paper repository. Review staged diffs, then create concise commits.

- [ ] **Step 6: Report evidence-backed completion**

Report experiment counts, key matched results, secondary-panel scope, test/Ruff/build outcomes, page/font/layout checks, commit hashes, and clickable final PDF/source links.
