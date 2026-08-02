# Controlled delayed-label benchmark

Status: completed synthetic mechanism check. This is not a materials dataset,
an empirical validation, or a tuning surface for IC-SARR.

## Purpose

The benchmark isolates the claim that a numeric query outcome can be immediate
while its discovery value is delayed by a complete-pool convex-hull decision.
It is deliberately independent of the MatPES and MAD runners, their manifests,
and every oracle vault.

## Frozen construction

`src/matmem/controlled_delayed_label_benchmark.py` defines a five-candidate,
four-equally-likely-world finite pool. A query reveals one discrete target
energy. Terminal reward is the number of queried candidates that are vertices
of the complete one-dimensional lower hull in the realized world.

The registered grid is the complete Cartesian product:

- budget: `1, 2, 3`;
- source signal: `0.0, 0.5, 1.0`; and
- delayed-label coupling: `0.0, 0.5, 1.0`.

The four fixed comparators are source margin, greedy current-final membership,
gated source-policy rollout, and exact belief-state dynamic programming. The
rollout uses the source policy as its continuation and only replaces the source
action on a strictly positive terminal-value advantage. The exact DP is an
upper reference under the same finite pool and reveal model; it is not a
deployable policy claim.

## Checks and permitted interpretation

The exhaustive grid has 27 cells. The exact checks verify that gated rollout is
never worse than source margin, exact DP is never worse than gated rollout,
budget-one rollout agrees with greedy current-final action selection, and at
least one grid cell has strict DP headroom. This supports the information and
objective mechanism only. It cannot support a material-discovery, external
generalization, final-causal, cost-aware, bounded-memory, or deployment claim.

Run:

```powershell
conda run --no-capture-output -n llm pytest -q tests/test_controlled_delayed_label_benchmark.py
```
