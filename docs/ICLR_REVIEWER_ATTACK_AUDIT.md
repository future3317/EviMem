# ICLR reviewer attack audit

Status: post-E44 exact separation, P0-v3 direct mechanism audit, and the
2026-08-03 exact-world numerical-gate audit.
This is a claim and submission-readiness audit, not an experiment plan or a
license to tune any opened task.

## Single paper position

The paper is a falsification-and-mechanism study of decision-sufficient
scientific state, protocol transport, and delayed full-pool acquisition under
materials-discovery constraints.  Its formal centerpiece is an exact
information-structure separation; its real-data result is a narrow,
cost-limited acquisition mechanism, not empirical bounded-memory deployment
superiority.

## Attacks and evidence-based response

| Likely reviewer attack | Evidence already in the paper | Honest residual limitation | Required response |
| --- | --- | --- | --- |
| “The title promises joint memory, but experiments are acquisition-only.” | Section 2.5 gives an exact `B=3,K=1` state-feedback separation, formal proof, nulls and Figure 1. | The separation is synthetic and does not test retention competition. | State “exact state-feedback separation,” never “empirical memory superiority.” |
| “The comparator is straw-man.” | The comparator is explicitly nonadaptive: later acquisition cannot use a witness reveal or retained state; all 336 deterministic sequences are bounded algebraically and randomization is covered by convexity. | An adaptive policy that reads retained state reaches `2`; no theorem about all decoupled policies is possible here. | Name the class “nonadaptive,” show the timing, and never broaden it to “any decoupled policy.” |
| “The toy witness is not a material memory constraint.” | The construction is labeled synthetic, protocol-compatible and non-hull; its purpose is information necessity. | No real task has yet measured an actually binding second resource. | Present this as the main empirical gate remaining, not as a solved deployment problem. |
| “IC-SARR only improves oracle labels.” | MatPES reports `+0.170/system` oracle-final development gain; MAD reports AUC `+0.2500`, final-causal `+0.0260` with CI crossing zero. | Final-causal superiority is unsupported. | Always pair any oracle claim with the final-causal limitation. |
| “The method is too expensive.” | MAD primary utility is `-0.6074`; wall-time AUC rises `+8.5739`. | No cost-aware superiority or runtime advantage. | Keep cost co-primary; do not use the E45 secondary timings to soften E43. |
| “MAD is a relabeled holdout or a real stability task.” | Manuscript identifies it as a frozen external protocol-shift panel, public paired data and atomization-energy hull proxy. | It is opened and not formation-energy thermodynamics. | Retain the exact scope language and no pristine/sealed/confirmatory wording. |
| “The baseline ladder was selected after the outcome.” | E45 has a pre-run policy and parameter lock; `B=1` is complete with all five canonical policies. | `B=2..6` are not run, so there is no ladder-wide curve or ranking. | Keep E45 out of primary claims and label it a single-budget secondary checkpoint if cited at all. |
| “Numerical integration creates the effect.” | The registered 1,000-instance exact-world audit reproduced all IC actions, observed zero exact false acceptances, and measured mean rollout regret `0.0005494`. | This is a finite-suite audit under the world model; it does not validate the material posterior. | Call the gate a numerical screen/safeguard and keep posterior misspecification separate. |

## Submission-safe claims

1. State feedback from a retained witness can be strictly necessary against a
   declared nonadaptive acquisition class in the finite-pool information structure.
2. The homogeneous full-history null and fail-closed unsupported-protocol
   behavior are required, tested boundaries of the formulation.
3. IC-SARR has a small, reproducible oracle-final delayed-full-pool mechanism
   signal across MatPES development folds and the MAD protocol-shift panel.
4. Neither final-causal, cost-aware, bounded-memory, runtime, deployment nor
   universal-generalization superiority is established.

## Pre-submission stop conditions

- Do not add a new empirical DBBM superiority claim without a real measurable
  state/access constraint and a separately frozen evaluation.
- Do not rerun or retune MatPES/MAD to improve the primary result.
- Do not report the E45 budget-one checkpoint as a curve, a significance test,
  or a new main table.
- The exact-world numerical-gate audit is complete; do not use it to claim
  posterior calibration or material-discovery safety.
