# E32-A parallel recovery

This is an execution-only recovery of the failed E32-A run. The task, vault,
cross-fit manifest, seven policies, budgets, posterior worlds, continuation
count, seed, hull backend, transport family and evaluation metrics remain
unchanged. The original failed root is preserved and is not overwritten.

Changes are limited to:

- independent `(fold, budget)` units run concurrently;
- BLAS/OpenMP threads pinned to one per unit;
- rollout-worker response timeout increased from 300 to 900 seconds;
- a separate external recovery root.

The scheduler is `tools/run_delayed_label_followup_parallel.py`. It records
its execution identity outside Git. Completed B=1 outputs may be referenced
read-only from the failed root; missing units run with the same scientific
configuration. The recovery is not paper-facing until all 30 outputs pass
the existing policy-roster, protocol-hash and summary checks.
