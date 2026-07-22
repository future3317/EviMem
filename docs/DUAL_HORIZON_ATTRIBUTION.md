# Dual-Horizon SARR failure attribution

Status: development diagnostic only. This note does not authorize a holdout
run, a backbone change, or a paper-level superiority claim.

## Question

Dual-Horizon SARR requires a non-negative selected-history advantage and a
positive terminal full-pool advantage at every decision. The attribution
experiment separates four possible failure modes without changing the frozen
posterior or policy:

1. no action is jointly feasible under the exact oracle;
2. a feasible action exists but the posterior does not identify it;
3. the posterior point estimate is feasible but its numerical lower-bound gate
   rejects it;
4. the first-action ranking is sound but the complete continuation differs.

For every state and every legal first action, the offline evaluator computes
the exact oracle advantages `(Delta_T*, Delta_F*)`, posterior point
advantages, posterior bounds, feasibility masks, and regret of source, point,
dual-gate, and recorded trace actions. The oracle vault is read only by this
offline evaluator; no policy subprocess receives oracle outcomes.

## Frozen input and convergence runs

The canonical development task is v6 (324 exact chemical systems), with the
fold-0 source-margin trace and fold-0 cross-fitted transport model. The first
diagnostic tranche is the same eight systems used by the earlier MC128 result:

`Ag-S`, `Al-O-P`, `B-Fe-Li-Mn-O`, `B-Li`, `Ba-Mg-Mn-O`, `Ba-Mn-O`,
`Bi-Li-O-P`, and `C-Ca`.

The evaluator is run at MC128 (historical baseline), MC512, and MC1024 with
identical states, seeds, and action candidates. Outputs remain outside Git in
the canonical data tree. MC512 and MC1024 are convergence diagnostics, not
independent replicates.

Recorded output SHA-256 values are:

```text
MC128:  372a138a479b3864fdbabae07dfc8c8c4c6e72ed9b6d9e8ee8fc6568668d4504
MC512:  953daf1b093a1314b81dcb68115c12e3e609310717a4d776b5217c62b5cdb325
MC1024: 0eb967bc73eca1bce2724f63f2c2fe995be9290bc27b0b4e743c20bce2fd7db5
```

## Interpretation rules

- Oracle-feasible existence near zero would support structural T/F conflict.
- Feasible oracle actions with low posterior recall indicate advantage or
  covariance misspecification, not automatically a bad encoder.
- High point-feasible-to-gate rejection indicates an over-conservative
  numerical gate or inadequate uncertainty integration.
- Low first-action regret but poor complete-trace regret indicates a
  continuation/policy-horizon mismatch.

The evaluator also reports global confusion counts and deterministic strata by
exact-system element count (`binary`, `ternary`, `quaternary_or_higher`) and
remaining candidate count (`<=16`, `17--32`, `>32`). These are descriptive
diagnostics; systems, not rounds or candidates, remain the scientific unit.

## Code and tests

`tools/run_dual_horizon_attribution.py` is intentionally an offline evaluator,
not a policy runner. Its strata and aggregation helpers are covered by
`tests/test_dual_horizon_attribution.py`. The evaluator records interval
coverage but does not reinterpret a nominal numerical interval as a posterior
calibration guarantee.

Before convergence was completed, the working hypothesis was that the local
two-gate constraint was too conservative and that posterior
advantage/covariance estimation contributed materially. The completed
convergence result is recorded below; no backbone replacement is justified by
this tranche alone.

## MC convergence result (8 systems, 48 states)

The three runs are numerical-integration checks on the same states and seeds,
not independent experiments:

| posterior samples | oracle-feasible state rate | posterior recall of oracle-feasible actions | point-feasible actions rejected by gate | terminal sign accuracy | selected-history sign accuracy | terminal interval coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.250 | 0.333 | 0.938 | 0.432 | 0.309 | 0.593 |
| 512 | 0.250 | 0.417 | 0.726 | 0.419 | 0.301 | 0.325 |
| 1024 | 0.250 | 0.417 | 0.628 | 0.418 | 0.300 | 0.277 |

There are 12 oracle-feasible actions across 48 states. Six are in ternary
states and six in quaternary-or-higher states; no binary state in this tranche
has a jointly feasible action. The posterior recovers only five of the twelve
at MC512/1024, while the gate retains very few point-feasible actions. The
terminal and selected-history sign accuracies are stable near 0.42 and 0.30,
and nominal interval coverage falls as sample count increases. This is
diagnostic of a misaligned/over-conservative numerical gate and/or misspecified
joint rollout advantage, not evidence that more Monte Carlo samples will rescue
the policy.

The evidence supports local dual-gate conservatism and posterior joint-
advantage estimation as leading causes; T/F structural conflict is present but
not universal. A backbone replacement is not justified. Any future dual-horizon
method must be a separately defined campaign-level constrained rollout, not a
tuned version of this local gate.

At MC1024, mean oracle terminal regret was 0.250 eV/atom for the source action,
0.250 for the recorded trace action, and 0.271 for the dual-gate action. The
point selector's regret was 0.188 eV/atom. Among all action candidates, the
posterior point mask recovered 5 oracle-feasible actions and the numerical gate
recovered none; it nevertheless marked 15 oracle-infeasible actions as gate
feasible. This is a direct action-level diagnostic of both missed feasible
actions and false gate positives, not a claim about realized campaign reward.
