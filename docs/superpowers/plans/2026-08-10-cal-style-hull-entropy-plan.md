# CAL-style Hull-Entropy Acquisition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement and formally run a matched-posterior CAL-style joint hull-entropy acquisition baseline in the existing MatPES unified experiment.

**Architecture:** Add one focused acquisition function and immutable result model to `protocol_acquisition.py`, reuse the existing Gaussian sampler and complete-pool hull evaluator, register one policy name in the existing worker, and extend the existing campaign/summary schema instead of creating a second experiment stack. The formal runner will generate CAL traces on the frozen 230-system development and authorized secondary 94-system rosters; authoritative existing comparator traces remain read-only.

**Tech Stack:** Python 3.11, NumPy, SciPy, Pydantic, pymatgen, pytest, `conda run --no-capture-output -n llm`.

## Global Constraints

- Use only the current joint PBE--r2SCAN posterior; do not create an independent GP.
- Use the existing completed-pool hull backend, duplicate convention, reference phases, tolerance, reveal boundary, cross-fitting, and immutable-ID tie-breaking.
- The entropy vector contains unique reduced compositions from the visible candidate pool; reference-only compositions remain hull inputs but are not entropy coordinates.
- Formal defaults are `m=200`, or `max(200, c_max + 32)` when required by the maximum grid size, `K=10`, and relative covariance ridge `1e-10`.
- Never use oracle target outcomes or final labels in policy-facing acquisition code.
- Raw datasets, vaults, checkpoints, and experiment outputs stay outside Git.
- Preserve all existing dirty files and do not stage unrelated changes.
- Use TDD: each production behavior begins with a failing test.

---

### Task 1: Add the pure hull-grid and entropy primitives

**Files:**
- Modify: `src/matmem/protocol_acquisition.py`
- Test: `tests/test_protocol_knowledge_gradient.py`

**Interfaces:**
- Produce `_unique_query_composition_grid(query_compositions)` returning a deterministic tuple of normalized reduced-composition dictionaries.
- Produce `_gaussian_hull_entropy(hull_values, relative_ridge=1e-10)` returning a finite scalar and raising `ValueError` on invalid or non-positive log-determinants.
- Produce `_condition_gaussian_on_scalar(mean, covariance, index, outcome)` returning `(conditional_mean, conditional_covariance)` with the queried coordinate fixed and covariance symmetrized.

- [ ] **Step 1: Write failing tests** for duplicate-composition removal, exclusion of reference-only coordinates, finite entropy on a full-rank covariance, deterministic ridge behavior on a rank-deficient covariance, and exact scalar Gaussian conditioning.
- [ ] **Step 2: Run the focused tests** with `conda run --no-capture-output -n llm pytest tests/test_protocol_knowledge_gradient.py -k 'hull_grid or hull_entropy or condition_gaussian' -q`; verify failure is due to missing helpers.
- [ ] **Step 3: Implement the three helpers** using `_normalized_composition_key`, NumPy symmetrization/eigendecomposition, `np.linalg.slogdet`, and the standard Gaussian conditioning equations. Do not add a marginal Bernoulli entropy path.
- [ ] **Step 4: Re-run the focused tests** and verify they pass.
- [ ] **Step 5: Commit** only the helper tests and production changes with `feat: add hull entropy primitives`.

### Task 2: Implement `protocol_hull_entropy`

**Files:**
- Modify: `src/matmem/protocol_acquisition.py`
- Modify: `src/matmem/__init__.py`
- Test: `tests/test_protocol_knowledge_gradient.py`

**Interfaces:**
- Add immutable `ProtocolHullEntropyResult` with fields `scores`, `expected_entropy_reductions`, `current_entropy`, `expected_conditional_entropies`, `evaluation_composition_count`, `posterior_sample_count`, `fantasy_count`, and `relative_ridge`.
- Add `protocol_hull_entropy(posterior, *, query_compositions, reference_compositions, reference_energies, costs, posterior_sample_count=200, fantasy_count=10, relative_ridge=1e-10, seed=0, fixed_template=None, fixed_runtime_plan=None)`.
- Export the model and function from `matmem` and the compatibility `protocol_knowledge_gradient` module.

- [ ] **Step 1: Write failing tests** for deterministic replay, zero-variance candidate score zero, permutation-invariant score/action ranking, and rejection of unequal costs, invalid ridge, insufficient samples, and non-finite entropy.
- [ ] **Step 2: Run the focused tests** and verify the new API is missing or rejects the intended behavior.
- [ ] **Step 3: Implement the acquisition**: build the frozen candidate grid, draw shared current posterior worlds, evaluate `_final_hull_values` on that grid, compute current joint entropy, then for each candidate draw exactly `K` scalar fantasies from its marginal Gaussian, condition the joint posterior without refitting, reuse common normal draws for conditional worlds, evaluate conditional hull vectors, and score expected entropy reduction per equal cost.
- [ ] **Step 4: Ensure metadata records the actual sample count** (`max(200, c_max + 32)` when required), grid size, ridge, and fantasy count.
- [ ] **Step 5: Run focused and existing acquisition tests** with `conda run --no-capture-output -n llm pytest tests/test_protocol_knowledge_gradient.py -q`.
- [ ] **Step 6: Commit** with `feat: implement matched CAL hull entropy acquisition`.

### Task 3: Register the policy in the worker and protocol registry

**Files:**
- Modify: `src/matmem/policy_registry.py`
- Modify: `src/matmem/protocol_policy_worker.py`
- Modify: `src/matmem/__init__.py` if exports require it
- Tests: `tests/test_protocol_closed_loop.py`, `tests/test_matpes_protocol_task.py`

**Interfaces:**
- Register policy string `cal_style_hull_entropy`.
- Worker dispatch must serialize `kind`, `scores`, selected index/ID, entropy metadata, and numerical settings using the existing decision-round schema.
- Worker must keep the same legal-candidate mask, reveal behavior, fallback behavior, and D/F/T evaluator path as existing policies.

- [ ] **Step 1: Write failing registry/worker tests** asserting the policy is accepted, uses no oracle labels before reveal, and emits a complete strategy trace.
- [ ] **Step 2: Run focused tests** and verify the policy is rejected or unhandled.
- [ ] **Step 3: Add the registry value and dispatch branch** using the existing posterior and hull inputs; use stable lexicographic candidate-ID tie-breaking.
- [ ] **Step 4: Run focused closed-loop tests** and the full protocol worker test module.
- [ ] **Step 5: Commit** with `feat: register CAL hull entropy policy`.

### Task 4: Add campaign task/summary support

**Files:**
- Modify: the existing unified MatPES runner identified by `rg -n 'active_policies|delta_hull_active_search' tools/run_matpes_*`
- Create or modify: the corresponding CAL summary tool beside the existing `tools/summarize_matpes_*` scripts
- Tests: a new `tests/test_matpes_cal_style_campaign.py` and the existing campaign-manifest tests

**Interfaces:**
- Campaign policy roster includes `cal_style_hull_entropy` alongside the frozen existing policies without changing their traces.
- Summary emits per-budget mean `T`, Delta-Hull-minus-CAL paired contrasts using the current system-level interval/test procedure, trapezoidal area, runtime/state cost, and grid/sample-setting metadata.
- Summary validates task/vault/cross-fit hashes, exact 230/94 roster counts, fold disjointness, and policy roster before reporting results.

- [ ] **Step 1: Write failing fixture tests** for policy roster validation, B=6 prefix derivation, paired contrasts, and rejection of missing CAL units or hash mismatches.
- [ ] **Step 2: Run the fixture tests** and verify the new summary path fails because CAL is absent.
- [ ] **Step 3: Extend the existing runner/configuration** without duplicating posterior fitting, evaluator code, or comparator traces.
- [ ] **Step 4: Implement summary calculations** using exact chemical system as the resampling unit and the existing paired uncertainty helper.
- [ ] **Step 5: Run campaign/unit tests** and a small local fixture campaign.
- [ ] **Step 6: Commit** with `feat: integrate CAL into MatPES campaign summaries`.

### Task 5: Run the formal 230-system and secondary 94-system CAL campaign

**Files:**
- Modify only generated task/config files outside Git; do not add raw outputs.
- Use the existing remote launcher and task/vault manifests authorized by the current experiment contract.

- [ ] **Step 1: Freeze and record** policy roster, posterior/hull settings, grid rule, `m`, `K`, ridge, seeds, task/vault/cross-fit hashes, and system counts.
- [ ] **Step 2: Run the 230-system five-fold development campaign** at B=6 and derive B=1..6 prefixes from each trajectory.
- [ ] **Step 3: Audit all units** for complete JSON, failures, hash consistency, zero fit/query overlap, legal reveals, and absence of oracle access in policy traces.
- [ ] **Step 4: Run the authorized 94-system secondary rerun once** only after development code and summary procedure are frozen; label it secondary held-out MatPES rerun, never external or untouched.
- [ ] **Step 5: Run the CAL summary** and save only external summary artifacts required for audit/reporting.
- [ ] **Step 6: Commit only code, tests, and documentation**; keep raw outputs and generated reports outside Git.

### Task 6: Final verification and manuscript decision gate

**Files:**
- Modify: none until all campaign audits pass.
- Optional later paper patch: `E:\PAPER\LEARNING WHAT TO REMEMBER` only after results are authoritative.

- [ ] **Step 1: Run the full relevant test suite** with `conda run --no-capture-output -n llm pytest -q` and `conda run --no-capture-output -n llm ruff check src tests tools`.
- [ ] **Step 2: Verify no raw outputs are staged** and the code repository contains only related source/tests/docs changes.
- [ ] **Step 3: If and only if summaries pass all audits, report CAL vs Delta-Hull effects and decide whether a minimal paper Table 2/definition patch is scientifically warranted.**
