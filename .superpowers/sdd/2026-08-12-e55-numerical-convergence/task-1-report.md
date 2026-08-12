# E55 Task 1 Report

## Result

Implemented the deterministic, outcome-independent CAL convergence roster builder and its focused tests.

Files changed for the task:

- `tools/build_e55_convergence_manifest.py`
- `tests/test_e55_convergence_manifest.py`

The builder exposes `build(task_path: Path, development_crossfit_path: Path, output: Path) -> dict[str, Any]`, uses task-public candidate counts, preserves each fold's original fit complement, applies integer terciles, enforces fit-element support, records input hashes and provenance, rejects Git-local outputs, and refuses overwrite.

## TDD verification

RED command:

```text
conda run --no-capture-output -n llm pytest -q tests/test_e55_convergence_manifest.py
```

RED output:

```text
ERROR collecting tests/test_e55_convergence_manifest.py
ModuleNotFoundError: No module named 'tools.build_e55_convergence_manifest'
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
```

GREEN command:

```text
conda run --no-capture-output -n llm pytest -q tests/test_e55_convergence_manifest.py
```

GREEN output:

```text
.....                                                                    [100%]
5 passed in 0.11s
```

Lint command:

```text
conda run --no-capture-output -n llm ruff check tools/build_e55_convergence_manifest.py tests/test_e55_convergence_manifest.py
```

Lint output:

```text
All checks passed!
```

## Self-review

- Exactly five folds are required and exactly three query systems are selected per fold.
- Candidate counts are derived only from task `development_pairs` grouped by exact `chemical_system`.
- Selection uses the frozen protocol hash string and low/middle/high integer rank terciles.
- Query systems are filtered by representation in the original fold fit roster.
- The original fit roster is preserved in each output fold.
- Task/cross-fit identity, release identity, output location, and write-once behavior are validated.
- No vault input, target outcome, policy result, or trajectory is read.
- Focused pytest and Ruff passed after the final implementation change.

## Commit

`b4edad5` (`feat: add E55 CAL convergence roster manifest`), containing exactly the two task files.

## Concerns

None for Task 1. The report remains outside the task commit by instruction; unrelated dirty files were not staged or modified.

## Fix Round 1

Addressed all Critical/Important findings and the inconsistent fallback.

### Fixes

- Output paths are rejected when the path or any ancestor is inside any Git repository, including a Git worktree `.git` file and a second temporary Git root.
- Cross-fit input now requires exactly five distinct fold indices `{0,1,2,3,4}`.
- Original query systems must be unique within folds, pairwise disjoint across folds, and cover the explicit `eligible_systems` list exactly.
- The builder requires explicit unique `eligible_systems`; it no longer derives eligibility from task rows.
- Every fold must explicitly provide `fit_systems`; the supplied order is preserved and its set must equal `eligible_systems` minus that fold's original query systems.
- The builder verifies that exactly 15 selected systems are unique.
- Tests independently verify the exact integer tercile boundaries and recompute each expected lowest SHA-256 winner from the frozen protocol string.

### TDD verification

Fix-round RED command:

```text
conda run --no-capture-output -n llm pytest -q tests/test_e55_convergence_manifest.py
```

Fix-round RED output:

```text
6 failed, 4 passed
```

The failures covered fit-roster order preservation, explicit `eligible_systems`, duplicate fold IDs, overlapping query systems, missing fit rosters, and output under a second Git repository.

Fix-round GREEN command:

```text
conda run --no-capture-output -n llm pytest -q tests/test_e55_convergence_manifest.py
```

Fix-round GREEN output:

```text
..........                                                               [100%]
10 passed in 0.20s
```

Ruff command:

```text
conda run --no-capture-output -n llm ruff check tools/build_e55_convergence_manifest.py tests/test_e55_convergence_manifest.py
```

Ruff output:

```text
All checks passed!
```

### Self-review and concerns

- No fallback remains for missing `eligible_systems` or missing `fit_systems`.
- The Git-boundary check walks the resolved output path to the filesystem root and detects both `.git` directories and `.git` files.
- The output preserves the original fit-roster order while sorting only the query systems for deterministic candidate-count ranking.
- No vault, target outcome, trajectory, or policy result is read.
- Unrelated dirty files remain untouched and unstaged.

Fix-round commit: `b4edad5` is the preceding Task 1 implementation commit; this fix round is committed separately after the verified changes above.

## Fix Round 2

Added a focused non-divisible tercile regression test using a 46-system fold in a readable five-fold 230-system fixture. The test independently asserts the exact frozen integer boundaries `15/15/16` and recomputes the lowest SHA-256 winner for each bin from `release_id || e55-cal-convergence-v1 || fold_index || bin || system`.

### TDD verification

The deliberately boundary-sensitive RED assertion first expected `16/15/15`; the unchanged production implementation failed on the bin boundary comparison, confirming the test would detect the incorrect partition. Production code was not modified.

Final focused test command:

```text
conda run --no-capture-output -n llm pytest -q tests/test_e55_convergence_manifest.py
```

Final test output:

```text
...........                                                              [100%]
11 passed in 0.22s
```

Ruff command:

```text
conda run --no-capture-output -n llm ruff check tools/build_e55_convergence_manifest.py tests/test_e55_convergence_manifest.py
```

Ruff output:

```text
All checks passed!
```

### Self-review and concerns

- Production code is unchanged in this round.
- The new fixture has 46 query systems in the primary fold and 46 systems in each of five folds, matching the real 15/15/16 non-divisible case while retaining fit-element support.
- The test checks exact ordered bins and independently recomputes each expected SHA-256 winner.
- Unrelated dirty files remain untouched and unstaged.
- A concurrent Conda invocation briefly hit a temporary lock; sequential pytest and Ruff verification passed afterward.

Fix-round-2 commit: pending final commit.
