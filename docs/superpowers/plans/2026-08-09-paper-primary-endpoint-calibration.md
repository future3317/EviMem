# Paper Primary Endpoint and Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make budget two the paper-facing primary scalar endpoint, add equal-system and decision-top-3 membership calibration from existing traces, and revise the manuscript into a concise positive objective-first presentation.

**Architecture:** Extend the existing evaluator-side calibration summarizer without changing acquisition or oracle access. Preserve pooled summary fields, add equal-system summaries and a top-3 candidate population, regenerate one external JSON artifact, then drive Figure 2 and the manuscript from the new fields. Objective and planning evidence remain separate complementary executions.

**Tech Stack:** Python 3.11, NumPy, scikit-learn, pytest, Ruff, Matplotlib, LaTeX/latexmk, Poppler, Git, SSH.

## Global Constraints

- Run Python, Ruff, and pytest with `conda run --no-capture-output -n llm` locally and `/home/workspace/lrh/miniconda3/bin/conda run --no-capture-output -n llm` remotely.
- Never add raw experiment outputs, datasets, oracle vaults, or checkpoints to Git.
- Do not open or run the 94-system secondary MatPES panel.
- Do not rerun acquisition, refit a posterior, introduce a new policy, or present Hull-KG as a distinct baseline.
- Preserve all unrelated dirty files in `E:\CODE\EviMem-RL` and all other projects in `E:\PAPER`.
- Before execution, invoke `superpowers:using-git-worktrees`; use code branch `codex/e52-primary-calibration` in `E:\WORKTREES\EviMem-RL-e52-calibration` and paper branch `codex/learning-what-to-remember-calibration` in `E:\WORKTREES\PAPER-learning-what-to-remember-calibration`.
- Use the `scientific-figure-making` skill for the Figure 2 regeneration and the `pdf` skill for final manuscript QA.
- Call B=2 the paper-facing primary scalar endpoint, not a preregistered endpoint.
- Keep main text at or below nine pages.

---

### Task 1: Add equal-system and decision-top-3 calibration summaries

**Files:**
- Modify: `tests/test_matpes_membership_calibration.py`
- Modify: `tools/summarize_matpes_membership_calibration.py`

**Interfaces:**
- Consumes: current E52 result JSON schema containing `policy_decision_rounds`, `final_stability_probabilities`, selected IDs, and evaluator-side final labels.
- Produces: `_equal_system_weights(rows) -> np.ndarray`, `_decision_top_k(rows, k=3) -> list[dict]`, weighted `_metrics` and `_reliability`, and population summaries containing `equal_system_metrics`, `equal_system_cluster_bootstrap_95`, and `equal_system_reliability_bins`.

- [ ] **Step 1: Write the failing equal-system and top-3 tests**

Add a strategy helper that accepts arbitrary labels and probabilities, then add this public-output test:

```python
def test_membership_summary_reports_equal_system_and_top3_metrics(tmp_path: Path) -> None:
    payload = {
        "task_sha256": "task",
        "systems": {
            "A-B": {
                "strategies": {
                    "delta_hull_active_search": _scored_strategy(
                        "A-B",
                        labels={"a": 1, "b": 1, "c": 0, "d": 0},
                        probabilities={"a": 0.9, "b": 0.9, "c": 0.1, "d": 0.1},
                        selected="a",
                    )
                }
            },
            "C-D": {
                "strategies": {
                    "delta_hull_active_search": _scored_strategy(
                        "C-D",
                        labels={"z": 1},
                        probabilities={"z": 0.1},
                        selected="z",
                    )
                }
            },
        },
    }
    input_path = tmp_path / "input.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    result = summarize(
        input_paths=(input_path,),
        output=tmp_path / "summary.json",
        bin_count=5,
        bootstrap_count=20,
    )
    group = result["groups"]["task:delta_hull_active_search"]
    assert result["schema_version"] == 2
    assert group["all_candidates"]["metrics"]["brier_score"] == pytest.approx(0.17)
    assert group["all_candidates"]["equal_system_metrics"]["brier_score"] == pytest.approx(0.41)
    assert group["top3_candidates"]["metrics"]["record_count"] == 4
    assert "equal_system_cluster_bootstrap_95" in group["top3_candidates"]
```

Add a second direct helper test with equal-probability candidates inserted in reverse lexical order:

```python
def test_decision_top_k_keeps_selected_then_breaks_score_ties_by_id() -> None:
    rows = [
        {"system": "A-B", "state_checksum": "state", "pair_id": "z", "probability": 0.9, "label": 1, "selected": True},
        {"system": "A-B", "state_checksum": "state", "pair_id": "c", "probability": 0.5, "label": 0, "selected": False},
        {"system": "A-B", "state_checksum": "state", "pair_id": "b", "probability": 0.5, "label": 1, "selected": False},
        {"system": "A-B", "state_checksum": "state", "pair_id": "a", "probability": 0.5, "label": 0, "selected": False},
    ]
    assert [row["pair_id"] for row in _decision_top_k(rows, k=3)] == ["z", "a", "b"]
```

- [ ] **Step 2: Run the tests and verify the new behavior is absent**

Run:

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_matpes_membership_calibration.py
```

Expected: failure because `schema_version` is still 1 and the equal-system/top-3 fields do not exist.

- [ ] **Step 3: Implement weighted metrics and reliability**

Add `_equal_system_weights(rows)` using row weight `1 / (S * n_s)`. Extend `_safe_ranking_metric`, `_metrics`, and `_reliability` to accept an optional NumPy weight vector. Normalize supplied weights before Brier, NLL, ECE, ROC--AUC, and average precision calculations. Preserve raw `record_count` and `positive_count`.

For equal-system reliability, each bin reports weighted mean probability and weighted empirical frequency using the global equal-system row weights restricted to that bin.

- [ ] **Step 4: Implement decision-top-3 extraction and population summaries**

Add `_decision_top_k(rows, k)` that groups rows by `(system, state_checksum)`, requires exactly one selected row per state, emits that row first, and fills remaining slots by sorting unselected rows with key `(-probability, pair_id)`. Add `_population_summary(...)` so all-candidate, selected-action, and top-3 populations share the same pooled/equal-system metrics, bootstrap intervals, and reliability code.

During equal-system cluster bootstrap, sample exact systems with replacement and give each sampled occurrence total weight `1 / S`; do not collapse duplicate draws by the original system name. Increment `schema_version` to 2 and retain existing pooled keys unchanged.

- [ ] **Step 5: Run focused tests and Ruff**

Run:

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_matpes_membership_calibration.py
conda run --no-capture-output -n llm ruff check tools/summarize_matpes_membership_calibration.py tests/test_matpes_membership_calibration.py
```

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 6: Commit the summarizer implementation**

```powershell
git add tools/summarize_matpes_membership_calibration.py tests/test_matpes_membership_calibration.py
git commit -m "Add equal-system decision calibration summaries"
git push -u origin codex/e52-primary-calibration
```

### Task 2: Regenerate and audit the external calibration artifact

**Files:**
- Modify: `docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md`
- Modify: `docs/EXPERIMENT_LEDGER.md`
- External create: `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/summary/membership-calibration-equal-system-top3.json`

**Interfaces:**
- Consumes: the five existing `objective/100/fold1-b6.json` through `fold5-b6.json` files.
- Produces: immutable schema-v2 summary metrics and a SHA-256 recorded in the ledger; no policy output changes.

- [ ] **Step 1: Update the remote code checkout without touching experiment outputs**

```powershell
ssh dbcloud "cd /home/workspace/lrh/EviMem-RL && git fetch origin codex/e52-primary-calibration && git switch --detach origin/codex/e52-primary-calibration"
```

- [ ] **Step 2: Generate the new non-overwriting summary**

```powershell
ssh dbcloud "cd /home/workspace/lrh/EviMem-RL && /home/workspace/lrh/miniconda3/bin/conda run --no-capture-output -n llm python tools/summarize_matpes_membership_calibration.py --input /home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/objective/100/fold1-b6.json /home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/objective/100/fold2-b6.json /home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/objective/100/fold3-b6.json /home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/objective/100/fold4-b6.json /home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/objective/100/fold5-b6.json --output /home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/summary/membership-calibration-equal-system-top3.json"
```

Expected: status `e52_final_hull_membership_calibration_complete`, schema version 2, and no overwrite of `membership-calibration.json`.

- [ ] **Step 3: Audit invariants against the prior summary**

Run an external read-only Python check that asserts identical `input_sha256`, task SHA, policy name, 217 systems, 1,302 states, 37,665 all-candidate records, and 1,302 selected-action records. Assert top-3 record count is between 1,302 and 3,906 and every population contains finite equal-system Brier/NLL/ECE and valid ranking metrics in `[0,1]`. Compute and record the new summary SHA-256.

- [ ] **Step 4: Copy only the derived summary to a local temporary path**

```powershell
New-Item -ItemType Directory -Force -Path C:\Users\LRH\.codex\tmp\e52-calibration | Out-Null
scp dbcloud:/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_e52_reduced_campaign_20260809/summary/membership-calibration-equal-system-top3.json C:\Users\LRH\.codex\tmp\e52-calibration\membership-calibration-equal-system-top3.json
```

- [ ] **Step 5: Record the paper-facing hierarchy and exact summary results**

Update the contract to identify B=2 on the 100% pool as the paper-facing primary scalar endpoint, with the B=1--6 curve secondary, AUC descriptive, and 70%/85% reruns robustness analyses. Record equal-system all-candidate, selected-action, and top-3 point metrics, counts, summary hash, and the unchanged 94-system boundary in the ledger.

- [ ] **Step 6: Commit the protocol and provenance update**

```powershell
git add docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md docs/EXPERIMENT_LEDGER.md
git commit -m "Freeze paper endpoint and decision calibration reporting"
git push
```

### Task 3: Drive Figure 2 from equal-system calibration

**Files:**
- Modify: `tests/test_render_e52_validation_figure.py`
- Modify: `tools/render_e52_validation_figure.py`
- Regenerate: `E:\WORKTREES\PAPER-learning-what-to-remember-calibration\LEARNING WHAT TO REMEMBER\figures\e52_objective_calibration.pdf`

**Interfaces:**
- Consumes: unchanged objective-prefix summary plus schema-v2 calibration summary.
- Produces: Figure 2 panel b from `equal_system_reliability_bins` and `equal_system_metrics`.

- [ ] **Step 1: Write a failing renderer-data test**

Add `_calibration_panel_data(group)` to the desired interface in the test before it exists:

```python
def test_calibration_panel_uses_equal_system_fields() -> None:
    group = {
        "all_candidates": {
            "metrics": {"brier_score": 0.01},
            "reliability_bins": [{"mean_predicted_probability": 0.1}],
            "equal_system_metrics": {"brier_score": 0.41},
            "equal_system_reliability_bins": [
                {"record_count": 2, "mean_predicted_probability": 0.4, "empirical_frequency": 0.5}
            ],
        }
    }
    bins, metrics = _calibration_panel_data(group)
    assert metrics["brier_score"] == 0.41
    assert bins[0]["mean_predicted_probability"] == 0.4
```

- [ ] **Step 2: Run the renderer test and verify import failure**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_render_e52_validation_figure.py
```

Expected: failure because `_calibration_panel_data` does not exist.

- [ ] **Step 3: Implement the equal-system panel selection**

Add `_calibration_panel_data(group)` and use it in `render`. Keep the all-candidate population in panel b, but source reliability points and Brier/NLL/ROC--AUC callouts exclusively from equal-system fields. Preserve vector PDF output, font type 42, restrained blue/teal/neutral palette, and existing figure dimensions.

- [ ] **Step 4: Run renderer tests, Ruff, and regenerate the figure**

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_render_e52_validation_figure.py
conda run --no-capture-output -n llm ruff check tools/render_e52_validation_figure.py tests/test_render_e52_validation_figure.py
conda run --no-capture-output -n llm python tools/render_e52_validation_figure.py --objective C:\Users\LRH\.codex\tmp\e52-calibration\objective-prefixes.json --calibration C:\Users\LRH\.codex\tmp\e52-calibration\membership-calibration-equal-system-top3.json --output "E:\WORKTREES\PAPER-learning-what-to-remember-calibration\LEARNING WHAT TO REMEMBER\figures\e52_objective_calibration.pdf"
```

Copy the existing external `objective-prefixes.json` to the stated temporary path before rendering. Delete the generated PNG sibling after visual inspection because the manuscript references only the PDF.

- [ ] **Step 5: Render and inspect the standalone figure**

Use Poppler at 180 dpi and inspect the PNG. Confirm readable labels at manuscript size, no clipped marks, reliability points inside `[0,1]^2`, and callout values matching schema-v2 equal-system metrics.

- [ ] **Step 6: Commit the renderer changes**

```powershell
git add tools/render_e52_validation_figure.py tests/test_render_e52_validation_figure.py
git commit -m "Render equal-system membership calibration"
git push
```

### Task 4: Revise the manuscript hierarchy and presentation

**Files:**
- Modify: `E:\WORKTREES\PAPER-learning-what-to-remember-calibration\LEARNING WHAT TO REMEMBER\LEARNING WHAT TO REMEMBER.tex`
- Modify: `E:\WORKTREES\PAPER-learning-what-to-remember-calibration\LEARNING WHAT TO REMEMBER\main_story.tex`
- Modify: `E:\WORKTREES\PAPER-learning-what-to-remember-calibration\LEARNING WHAT TO REMEMBER\main_method.tex`
- Modify: `E:\WORKTREES\PAPER-learning-what-to-remember-calibration\LEARNING WHAT TO REMEMBER\appendix_delayed_search.tex`
- Regenerate: `E:\WORKTREES\PAPER-learning-what-to-remember-calibration\LEARNING WHAT TO REMEMBER\LEARNING WHAT TO REMEMBER.pdf`

**Interfaces:**
- Consumes: audited schema-v2 equal-system and top-3 metrics plus the existing objective and solver summaries.
- Produces: one primary scalar endpoint, secondary curve/AUC/pool-shift labels, compact positive solver positioning, and an appendix calibration table with explicit weighting.

- [ ] **Step 1: Define the primary hierarchy in the main experiment section**

Replace the first execution description with "a fresh rerun under a frozen, outcome-independent protocol." State before the effect estimate that B=2 on the 100% pool is the primary scalar endpoint because it is the first budget at which an observation can change a subsequent acquisition. Report `+0.104`, its 95% CI, and sign-flip p-value there. Describe B=1--6 as the secondary budget curve, `+0.404` AUC as descriptive, and 70%/85% pools as robustness reruns. Do not call B=2 preregistered.

- [ ] **Step 2: Replace pooled calibration claims with equal-system and top-3 evidence**

Use the exact schema-v2 fields for equal-system all-candidate Brier/NLL/ROC--AUC in the main paragraph. Add one compact sentence reporting selected-action and top-3 discrimination/calibration from the appendix table. Explain weighting once: each exact chemical system contributes equal total weight.

- [ ] **Step 3: Make objective and solver analyses complementary rather than directly compared**

Open the solver paragraph with "A complementary matched-posterior development execution isolates planning after objective alignment." Keep the `+0.0391` effect and primary sign-flip p-value, but describe it as limited incremental terminal value despite frequent action changes. Move detailed high-precision convergence numbers to the appendix and retain only one main-text sentence that the numerical audit supports a shared two-step Bellman target.

Replace the appendix phrase "is not pooled with" with a positive statement that the execution isolates the planning question under a shared posterior and evaluator. Do not write a direct statistical or rhetorical comparison between `+0.104` and `+0.0391`.

- [ ] **Step 4: Tighten novelty and notation**

After Eq. `eq:main-delta-hull`, define `lexargmax` as maximization with deterministic immutable-ID tie-breaking. Add to Related Work: "ENS and knowledge-gradient methods optimize future acquisition value under conventional search labels; our distinction is the separation between immediate query observations and globally adjudicated selected-item utility." Normalize `ROC--AUC`, `final-hull`, and related hyphenation throughout touched text.

- [ ] **Step 5: Update Figure 2 caption, appendix table, and page balance**

Use the Figure 2 title "Objective alignment persists under query-pool shift, with strong discrimination and good aggregate calibration." State that panel b uses equal-system weighting. Add an appendix table with rows `All legal candidates`, `Selected action (top-1)`, and `Decision top-3`, and columns for count, equal-system Brier, NLL, ECE, ROC--AUC, and average precision.

Reduce Figure 3 from `0.96\linewidth` to `0.86\linewidth`. Keep Figure 2 full width. Do not force a page break unless the compiled layout remains overcrowded.

- [ ] **Step 6: Consolidate limitations**

Keep one compact limitations paragraph: evidence is within MatPES plus an atomization-energy protocol-shift stress test; target-query cost is unmeasured; external formation-energy validation remains future work. Keep the 94-system non-use statement in the appendix protocol boundary rather than repeating it in the main narrative.

### Task 5: Verify, compile, commit, and push

**Files:**
- Verify all Task 1--4 files.
- Commit only task-specific code/docs in the code worktree and only `LEARNING WHAT TO REMEMBER` source, referenced figures, and compiled PDF in the paper worktree.

**Interfaces:**
- Consumes: completed code and manuscript changes.
- Produces: pushed `codex/` branches with reproducible tests and a submission-ready PDF.

- [ ] **Step 1: Run the complete relevant code verification**

```powershell
conda run --no-capture-output -n llm ruff check tools/summarize_matpes_membership_calibration.py tools/render_e52_validation_figure.py tests/test_matpes_membership_calibration.py tests/test_render_e52_validation_figure.py
conda run --no-capture-output -n llm pytest -q tests/test_matpes_membership_calibration.py tests/test_render_e52_validation_figure.py tests/test_matpes_e52_cleanroom_manifests.py tests/test_e52_objective_prefixes.py tests/test_e52_two_step_equivalence.py tests/test_e52_two_step_convergence.py
```

Expected: Ruff clean and all selected tests pass with zero failures.

- [ ] **Step 2: Compile with halt-on-error**

```powershell
& 'D:\texlive\2025\bin\windows\latexmk.exe' -pdf -interaction=nonstopmode -halt-on-error -file-line-error 'LEARNING WHAT TO REMEMBER.tex'
```

Run from the paper directory. Re-run until references stabilize.

- [ ] **Step 3: Check structural and PDF invariants**

Search the log for overfull boxes, undefined references, and missing citations. Use `pdfinfo` to record total pages and verify references begin after main text, with main text at most nine pages. Use `pdffonts` to confirm every font is embedded.

- [ ] **Step 4: Render and visually inspect affected pages**

Render the abstract page, Figures 2--3 page(s), Related Work/limitations, and the appendix calibration pages at 150--180 dpi. Confirm readable figure text, no text-only panel, no clipped table, consistent hyphenation, and balanced whitespace.

- [ ] **Step 5: Inspect scoped Git diffs and commit**

```powershell
git diff --check
git status --short
```

In the code worktree, stage only the summarizer, renderer, tests, contract, ledger, design, and plan. In the paper worktree, stage only the four manuscript sources, regenerated Figure 2, and final PDF. Preserve all unrelated dirty files outside the worktrees.

- [ ] **Step 6: Push both task branches**

```powershell
git push -u origin codex/e52-primary-calibration
git push -u origin codex/learning-what-to-remember-calibration
```

Report commit IDs, branch names, exact test count, main-text page count, external summary path/hash, and the final PDF path.
