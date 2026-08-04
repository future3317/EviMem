# ICLR reviewer attack audit

Status: current manuscript integration after the E44 exact separation, the
P0-v3 direct mechanism audit, the E32-A objective/lookahead follow-up, the
MAD-1.5 direct-mechanism curve, and the exact-world numerical-gate audit. This
is a claim and submission-readiness audit, not an experiment plan or a license
to tune any opened task.

## Single paper position

The paper is **Globally Adjudicated Active Search for Convex-Hull Discovery.**
Its central contribution is a
general active-search problem in which observations arrive immediately but
labels are structured functions of a latent candidate pool and are adjudicated
only after the campaign. Materials convex-hull discovery is the motivating
instance. The formal core combines a greedy counterexample, failure of
adaptive submodularity, greedy-sufficiency conditions, delayed-information
value, and an exact information-structure separation. The real-data evidence
is a development-level mechanism study; it is not a claim that one solver or
gate is universally superior.

The manuscript packaging checkpoint is paper commit `9f65ac5`: the main text
is nine pages before references, references precede the appendix, and the
rollout schematic, controlled heatmaps, policy ablations, posterior/hull
implementation details, and numerical-gate audit are appendix material. This
was an integration of the registered E32-A objective/lookahead and MAD
direct-mechanism results; it does not authorize further method or task changes.

## Attacks and evidence-based response

| Likely reviewer attack | Evidence already in the paper | Honest residual limitation | Required response |
| --- | --- | --- | --- |
| Is delayed structured labeling genuinely different from ordinary active search? | The formulation separates immediate reveal `O_x` from terminal structured label `Y_x=g_x(Z_C)`; the Bellman boundary integrates the unrevealed complete pool, and the controlled benchmark verifies the information order. | The material label is evaluated retrospectively; it is not online feedback. | Explain full-pool utility as campaign-level discovery durability, not delayed communication latency. |
| Is the theory only a collection of toy examples? | The paper gives a no-uniform-ratio repeated-greedy construction, an adaptive-submodularity counterexample, rank-stability/weak-coupling conditions, strict delayed-information value, and an exact separation. | These are structural results under declared assumptions, not universal claims about materials. | State every theorem with its assumptions and use real data only as mechanism evidence. |
| If greedy is often enough, why study rollout? | The exact suite and MatPES direct contrasts quantify the greedy-sufficiency boundary; SARR equals DP on 84.8% of the registered synthetic instances and does not beat Delta-Hull on the MatPES roster. The E32 trace audit shows Q-gap predicts action changes ($\rho=0.816$), while realized $T$ gain is only weakly related ($\rho=0.108$). | The current data do not establish a broad rollout advantage or a reliable rank/coupling subgroup law. | Present nonmyopic value as a measured action-versus-utility boundary and treat Delta-Hull strength as a positive boundary result. |
| Is the exact state-feedback separation a memory/deployment result? | E44 exhaustively separates a declared nonadaptive class in a finite synthetic information structure. | It does not measure a binding materials memory or access budget. | Call it an exact information-structure result, never empirical bounded-memory superiority. |
| Is the positive result only an oracle metric? | MatPES has a source-relative `+0.170/system` complete-pool `T` signal; MAD has complete-pool-proxy `T` AUC `+0.2500`, selected-history `F` AUC `+0.0260` with CI crossing zero, and cost utility `-0.6074`. | The terminal utility is retrospective and no selected-history or cost-aware gain is established. | Explain why durability is the declared campaign objective and pair every terminal result with `F`, cost, and scope. |
| Is the method too expensive? | MAD wall-time AUC increases by `+8.5739 s*budget/system`, and the fixed cost utility is negative. | No cost-aware or runtime advantage is supported. | Keep cost co-primary and do not use secondary timings to soften this result. |
| Is MAD a relabeled holdout or a formation-energy stability task? | It is documented as a public protocol-shift panel with an atomization-energy hull proxy. | It is opened task-level evidence and is not formation-energy thermodynamics. | Never use pristine, sealed, confirmatory, or formation-energy wording for MAD. |
| Does the source-relative gain prove that the proposed solver wins? | P0-v3 directly compares source margin, Delta-Hull, ungated SARR, diagonal covariance and IC-SARR. IC-SARR minus Delta-Hull is `+0.0043` (CI `[-0.0609,+0.0696]`, `p=1.0000`) and minus ungated SARR is `-0.0130` (CI `[-0.0565,+0.0304]`, `p=0.6930`). | The evidence supports the delayed objective and a greedy-sufficiency boundary, not solver superiority. | Make direct paired contrasts primary and keep IC-SARR as an appendix numerical screen. |
| Does the full-pool objective add anything beyond a better target posterior? | The evaluator-only same-posterior contrast gives Delta-Hull minus target-margin differences of `+0.0826,+0.0870,+0.0826,+0.0696,+0.0217,-0.0043` over `B=1..6`; AUC is `+0.3000` (95% CI `[+0.0522,+0.5587]`, `p=0.0247`). | The advantage is concentrated at early budgets and vanishes by `B=6`; this is an efficiency claim, not a terminal-value or solver-superiority claim. | Keep the objective contrast in the main figure and state the system-level bootstrap unit explicitly. |
| Does numerical integration create the effect? | The registered 1,000-instance exact-world audit reproduced all IC actions, found zero exact false acceptances, and measured mean rollout regret `0.0005494`. | This is a finite-suite audit under the registered world model; it does not validate the material posterior. | Call the gate a randomized-QMC numerical screen/safeguard and separate posterior misspecification from integration error. |

## Submission-safe claims

1. Delayed structured labels define a distinct active-search objective, with
   complete-pool adjudication measuring retrospective discovery durability.
2. Repeated greedy can fail in general, while rank stability and weak coupling
   explain regimes in which greedy is optimal or nearly optimal; delayed
   observations can have positive information value.
3. Under a shared target posterior, full-pool greedy improves early-budget
   efficiency over target-margin greedy, while the two are indistinguishable by
   the final budget. The exact controlled benchmark, cross-fitted MatPES suite
   and MAD protocol-shift task provide the corresponding mechanism evidence.
4. Neither IC-SARR/SARR solver superiority, final-causal, cost-aware,
   bounded-memory, runtime, deployment nor universal-generalization
   superiority is established.

## Pre-submission stop conditions

- Do not turn the exact synthetic state-feedback separation into an empirical
  DBBM or bounded-memory superiority claim.
- Do not rerun or retune MatPES/MAD to improve the primary result.
- Do not report the E45 budget-one checkpoint as a curve, a significance test,
  or a new main table.
- The exact-world numerical-gate audit is complete; do not use it to claim
  posterior calibration or material-discovery safety.
- Do not call MAD a formation-energy hull or an untouched/sealed holdout.
- Keep IC-SARR in the optional numerical-safeguard role unless a separately
  frozen future study changes that decision.
