# CAL Hull Runtime Plan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse fixed CAL hull geometry within one acquisition decision without changing the numerical objective.

**Architecture:** A private immutable plan holds the validated fixed template, duplicate-composition index groups, and causal envelope for one selection state. The entropy policy constructs it once and supplies it to every current and conditional hull evaluation; calls without a plan retain the existing construction path.

**Tech Stack:** Python 3.11, NumPy, SciPy, pytest, Ruff.

## Global Constraints

- Use `conda run --no-capture-output -n llm` for Python, pytest, and Ruff.
- Preserve fixed-composition values, seeds, ordering, posterior MC, and fantasy semantics.
- Keep raw outputs outside Git and never modify the active recovery.

---

### Task 1: Prove the reusable geometry contract

**Files:**

- Modify: `tests/test_protocol_knowledge_gradient.py`
- Modify: `src/matmem/protocol_acquisition.py`

- [ ] Write a failing test that constructs `_CalHullRuntimePlan` and asserts bitwise equality between planned and original `_cal_hull_values` on a fixed backend.
- [ ] Run `conda run --no-capture-output -n llm pytest tests/test_protocol_knowledge_gradient.py -q -k runtime_plan` and observe the missing API failure.
- [ ] Add the private immutable plan with `fixed_template`, ordered grouped query indices, and `_CausalHullEnvelope`; make `_cal_hull_values` consume an optional plan.
- [ ] Re-run the focused test and confirm exact equality.

### Task 2: Wire the plan through CAL entropy

**Files:**

- Modify: `src/matmem/protocol_acquisition.py`
- Test: `tests/test_protocol_knowledge_gradient.py`

- [ ] Create the plan once in `protocol_hull_entropy` when the fixed backend is active.
- [ ] Pass the same read-only plan to current and conditional hull evaluations.
- [ ] Re-run the serial-versus-parallel exact score/action test.

### Task 3: Verify and commit

- [ ] Run `conda run --no-capture-output -n llm pytest tests/test_protocol_knowledge_gradient.py tests/test_protocol_closed_loop.py tests/test_matpes_cal_style_campaign.py -q`.
- [ ] Run `conda run --no-capture-output -n llm ruff check src/matmem/protocol_acquisition.py tests/test_protocol_knowledge_gradient.py`.
- [ ] Commit only the runtime-plan code and regression tests with `perf: reuse CAL hull geometry`.
