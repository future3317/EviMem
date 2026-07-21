# MatPES confirmatory infrastructure

This note records the implementation boundary and completed 48-system
Delta--Hull repartition. The result is a method-level NO-GO, not a positive
claim.

## Frozen method

`delta_hull_active_search` remains the MC1024 nested scrambled-Sobol method
with the existing posterior and one-step final-hull membership objective.
No score blend, adaptive Monte Carlo, lookahead, top-k truncation, or new
weight is introduced in this stage. The default hull backend is `pymatgen`.

`fixed_composition` is an action-equivalent optimization only. It caches the
composition geometry and reuses it for each posterior sample. It is not the
package default; a future claim-grade panel should first run
`tools/audit_matpes_fixed_hull_parity.py` on its registered support.

The completed run uses the cached backend after stable-mask property tests and
full six-action agreement on the three real MC1024 systems completed by the
original reference backend. The slow reference run is preserved as incomplete
diagnostic evidence, not combined with the completed output.

## Fresh split contract

`tools/build_matpes_confirmatory_task.py` excludes every development system and
pair, applies only candidate/parent-count gates, and selects systems by a
release/system SHA-256 within binary, ternary and quaternary-or-higher strata.
Target outcomes are copied to a separate `confirmatory_sealed_oracle_vault`
after the system set is selected. Selection never reads target values.

The confirmatory runner accepts `--split confirmatory` and requires a frozen
transport artifact. It does not refit transport on evaluation systems. The
development runner may still fit transport only on its disjoint development
fit systems.

## Transport freeze

`tools/freeze_matpes_transport_model.py` writes a JSON artifact containing the
model parameters, optimizer metadata, fit-system IDs, source task/vault
checksums, and a canonical payload checksum. The confirmatory runner loads that
artifact and fails if a fit system intersects a query system.

## Required execution order

1. Build the all-eligible task outside Git.
2. Build the fresh task/vault and record its exclusion manifest.
3. Freeze transport on the registered disjoint fit systems.
4. Run fixed-hull parity; stop on any mismatch.
5. Run `tools/audit_matpes_sobol_seed_stability.py` with independent scramble
   seeds on development systems.
6. Only then run the frozen fresh replication and baselines.

All raw task, oracle and experiment outputs remain outside Git.

## Completed repartition result

The 324 eligible systems are split into 276 transport-fit and 48 evaluation
systems before refitting. Delta-Hull obtains 3.6250 oracle-final confirmations
per system versus 3.5625 for source margin. The paired difference is +0.0625,
bootstrap 95% CI [-0.1042,+0.2292], exact two-sided sign-flip p=0.6291. The
48 evaluation systems are opened and must not be used for further tuning.

The runner now separates policy execution from optional posterior-hull
diagnostics. Setting `--posterior-diagnostic-sample-count 0` skips only the
post-trace sampled-hull evaluator; it does not change the posterior used by an
acquisition action, oracle reveal, causal hull, energy diagnostic or final
discovery metric.

## Post-replication development boundary

The negative replication closes further tuning of myopic Delta-Hull on these
48 systems. Source-Rollout Delta-Hull is a new finite-horizon policy and is
developed only through six-fold cross-fitting on the former 276 transport-fit
systems. The opened 48-system artifact may be read by the attribution-only
horizon diagnostic, but it is not a development fold and cannot define a
rollout parameter, posterior, threshold or score.

The rollout uses the same source-margin selector as the deployed baseline,
adds each sampled target outcome to an exact cached composition-dependent
causal hull, and evaluates terminal confirmations on the complete sampled
final hull. The fixed-composition envelope has independent pymatgen parity
tests. Its sixteen-block Bonferroni-simultaneous paired-Sobol fallback controls
only numerical integration noise and must not be described as a calibrated
policy-safety guarantee.

The first 46-system cross-fit fold was evaluated at MC512 and MC1024 without
changing the method. System-level effects agree in 45/46 systems and first
actions in 41/46, but complete traces agree in only 31/46 and individual
actions in 220/276. Consequently folds 1--5 are paused. This is a numerical
integration gate, not authorization to change the transport posterior,
terminal reward, source continuation, or selection threshold.

## Conformal One-Deviation Source-Rollout (development implementation)

The live code now contains a separate `conformal_source_rollout_delta_hull`
policy. It does not alter the frozen Source-Rollout objective or the SARR gate.
On disjoint exact-system calibration traces, the helper
`source_rollout_system_score` computes the nonconformity score

\[
S_s = \max_{t,x}\bigl[\widehat A_{s,t}(x)-A_{s,t}(x)\bigr]_+,
\]

and `fit_conformal_source_rollout_calibration` freezes the finite-sample
system-clustered quantile. At deployment, a candidate can replace the source
action only when its estimated source-relative advantage minus its RQMC
standard-error radius is strictly larger than that frozen threshold. The
runner records `conformal_deviation_used` in the policy state; after the first
accepted deviation, every later round executes the source-margin policy. The
source action remains in the legal action set and is the fail-closed fallback.

The threshold is not a posterior standard deviation, and the calibration
radius is not a blanket safety or superiority guarantee. This implementation
is proposal/development infrastructure only: no calibration artifact or
evaluation result has yet been produced, and the current SARR fold-0 rerun
must be completed before this continuation is used for a paper-facing claim.

For the pending SARR fold-0 rerun, the external result record contains the
observable pre-reveal decision diagnostics required for a numerical audit:
candidate ordering, sixteen RQMC block scores, paired advantages, simultaneous
lower bounds, source/selected action, fallback reason, comparison count and
horizon. These records are emitted before the selected-action-only reveal and
do not expose unqueried target outcomes. They are stored as
`policy_decision_rounds`, separately from post-trace evaluator rounds.
