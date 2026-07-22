# Authoritative experiment and decision ledger

**Status: active source-relative full-budget MatPES rollout development after
the completed v4, CHIC and myopic Delta-Hull NO-GOs (2026-07-21).** There is still no paper-level
positive method. This is the first file
a future maintainer or coding agent should read before changing the method or
launching another experiment. It records why the research moved from CAW-Joint to DACC,
from DACC to P3C, why P3C exposed outcome-contribution deletion and reference
mismatch, why AKSC was proposed, why the WBM compute gate stopped AKSC as the
paper's next main line, and why two real JARVIS--MP transport certificates did
not authorize a positive protocol-aware state claim.

Raw datasets, checkpoints and experiment outputs remain outside Git. Paths and
hashes below are provenance pointers, not instructions to add those artifacts
to the repository. Old negative, invalidated and interrupted results must not
be overwritten or selectively replaced.

## Canonical-record contract

This file is the **single authoritative research audit trail**. It preserves
the complete iteration logic and the evidence judgment attached to every
result, including results that no longer belong in the paper's main text. A
manuscript edit may move old methods to an appendix or summarize them as an
ablation, but it must not erase their scientific disposition from this file.

The remaining files in `docs/` are live scientific or data-contract notes; they
do not override this ledger. Superseded formulas, configs and runners were
removed from the live tree after their unique experiment identity, checksum,
failure attribution and stopping decision were consolidated here. Git tags are
the recovery mechanism for retired code; restoring an old runner to the live
architecture is not a preservation mechanism.

Every evidence item in this ledger has one of the following dispositions:

| Disposition | Meaning | Permitted use |
|---|---|---|
| **Authoritative valid** | Correct data contract, frozen estimand and complete output | May support the claim stated here, but no broader claim |
| **Mechanism diagnostic** | Valid for implementation or causal attribution, not a superiority result | May explain behavior or motivate a test |
| **Historical invalidated** | A later provenance, leakage, semantic or evaluator defect changes its scientific meaning | Preserve for audit; never cite as claim-grade performance |
| **Incomplete/interrupted** | Physical execution or summary is incomplete | Preserve; never merge, extrapolate or use for inference |
| **Sealed/unopened** | Evaluation outcomes were intentionally not accessed because a calibration gate failed | Preserve the seal; never use for development |
| **Proposal only** | Formula, architecture or research plan without an authorized positive experiment | May define a future gate; never describe as a result |
| **Stopped/NO-GO** | The registered hypothesis failed its decision, inference or systems gate | Do not repeat without stating which failed assumption and gate changed |

Evidence judgments are monotone in one direction: a later experiment may
invalidate or narrow an older claim, but it cannot silently promote a
diagnostic, interrupted run or opened evaluation set into confirmatory
evidence. A new implementation must receive a new identifier rather than
relabel an old result.

## Current decision in one paragraph

CAW-Joint failed after correcting self-removal information gain, memory
semantics, fixed boundary weights and exact retention. DACC then replaced the
interval heuristic with a monotone facility objective, but that objective
combined prior-to-singleton gains even though the deployed GP conditions on a
joint witness set. P3C fixed this local objective mismatch by projecting a
fixed `(K+1)`-witness reference posterior onto exact size-`K` drop-one
neighbors under proper divergences. P3C nevertheless continued to omit evicted
archive outcomes from the deployed posterior and selected omissions using the
observed outcomes themselves. Its properness was only relative to a
misspecified fixed GP reference, not to causal truth. Two structure-correct,
disjoint WBM panels totaling 32 exact systems and 683 candidates did not
replicate the early probability-loss signal. AKSC was therefore proposed to
compress an outcome-independent kernel representation while accumulating
*all* revealed outcomes in posterior sufficient statistics. Before
implementation, a checkpointed B40 gate showed that GP numerical work is only
0.689% of the real WBM round pipeline, below the 9.09% Amdahl threshold; even
perfect GP elimination permits only `1.00694x` ideal speedup. The subsequent
JARVIS--MP task confirmed strong same-candidate low-fidelity signal, but a
global affine certificate violated its held-out radius on 45.32% of pairs and
a composition-aware certificate exceeded its frozen calibration ceiling before
fresh evaluation. P3C is stopped, AKSC is not authorized for WBM, and safe
protocol-aware reuse remains NO-GO under the tested certificates. The v4
runner was subsequently found to use a fixed hash reveal trajectory, so its
counterfactual action scores did not test active query selection. CHIC changes
that failed assumption by making the selected action the only oracle reveal
and by selecting training gradients for an explicitly expensive optimizer.
Its first calibration-system experiments are executable but remain negative.
The subsequent MatPES task found and fixed a stoichiometric total-energy defect
that invalidates its earliest closed-loop traces. On the corrected task, a
composition-referenced hierarchical discrepancy posterior substantially
improves energy and hull inference. Delta-Hull Active Search then shows a
`+0.25` oracle-final confirmation signal on 24 development systems at MC1024.
The effect is unchanged from MC512, but an outcome-independent 48-system
repartition with transport refitted on the other 276 systems shrinks the gain
over source margin to `+0.0625`, CI `[-0.1042,+0.2292]`, exact two-sided
`p=0.6291`. Delta-Hull is non-degenerate and improves over two posterior-margin
baselines, but does not beat the strongest simple baseline. It is therefore a
method-level NO-GO on the tested MatPES task. Source-Rollout Delta-Hull changes
the failed one-step horizon assumption: it evaluates each first action under a
complete posterior energy sample and then uses source margin as continuation
for the remaining budget. It is implemented and the first 46-system
cross-fitted fold remains positive from MC512 to MC1024 at the effect level,
but only 31/46 complete traces agree. The other folds are paused for a
numerical-integration-only diagnosis; no paper-level positive result is
claimed.

## Current research decision (not an experiment)

The stopping chain does not authorize another outcome-selected posterior
approximation. CHIC tested a real optimizer-input constraint without deleting
outcomes, but its JARVIS task did not show an advantage. The active development
hypothesis is now a scientifically different object: nonmyopic active search
with delayed convex-hull labels and source margin as a strong continuation
policy. It retains the composition-referenced hierarchical PBE--r2SCAN
posterior on exact same-configuration MatPES pairs. Every paid target
outcome remains in the archive and conditions the posterior. WBM remains the
homogeneous low-compute null, and opened JARVIS evaluation systems remain
closed to development. The decision-sufficient-state definition still governs
archive, protocol and null behavior; no MatPES superiority claim is currently
supported.

## Terminology that must not be blurred

- **Scientific archive:** append-only record of every paid DFT reveal. No live
  method deletes it.
- **Outcome-contribution deletion:** fitting the next residual GP only to the
  selected active set, so evicted archive residuals no longer contribute to
  `P(f | D_M)`. This is the deletion exposed by the P3C diagnosis.
- **DACC reference mismatch:** facility-location maximizes clipped
  prior-to-single-witness risk reductions, whereas prediction uses the joint GP
  posterior of the whole active set.
- **P3C reference/truth mismatch:** a proper divergence guarantees fidelity to
  the chosen reference `Q`, not calibration to unknown causal outcomes. The
  frozen SOAP--Matern GP reference is directly measured to be underdispersed.
- **Selection mismatch:** P3C deploys `P(f | D_M)` after choosing
  `M=S(D_Z)` from observed residuals without conditioning on that selection.
  Conditioning only on `S(D_Z)=M` would still not restore the information in
  all archived residuals.
- **Representation compression:** AKSC's proposed alternative: select a basis
  without outcomes, but update low-dimensional natural parameters with every
  revealed outcome. It was a proposal, not a positive experimental result.

## Evidence-adjudication matrix

This table is the compact judgment layer over experiments E0--E23. Detailed
paths and hashes appear in the numbered entries below.

| Stage | Evidence disposition | What is established | What is not established | Rerun/continuation rule |
|---|---|---|---|---|
| Corrected CAW-Joint | Stopped/NO-GO; synthetic mechanism history | Self-removal, memory semantics, fixed weights and exact retention were corrected; one narrow coupled mechanism exists | Joint superiority or a universal retention principle | Recover only from the frozen tag for audit; do not restore to live code |
| Working-set economics | Mechanism/preregistration diagnostic | Access economics and evidence selection are distinct estimands; matched free retrieval must be exactly equivalent | A persistence advantage on WBM | Reuse the estimand definitions only if a measured access cost exists |
| Secure WBM P0 | Authoritative infrastructure | Oracle isolation, append-only reveal, parity energy, initial structure and causal hull contracts | Any method-performance claim | Must remain a hard prerequisite for WBM work |
| Early DACC pilot | Historical invalidated for paper claims | The implementation produced nonconstant behavior and a weak engineering signal | Claim-grade WBM advantage; old structure/energy semantics were later corrected | Never cite absolute performance; recovery tag only |
| DACC joint-risk diagnostic | Mechanism diagnostic | Singleton facility gain is not identical to joint GP risk; joint risk is correlated and more expensive | That joint-risk one-swap is a superior replacement | No wider grid; retained as decision-mismatch evidence |
| Interrupted frozen grid | Incomplete/interrupted | Resource accounting and some physical traces exist | Any grid-level comparison or confidence interval | Never resume, merge or fill missing cells selectively |
| P3C local projection | Mechanism diagnostic and stopped/NO-GO | Proper projection is implemented and solves its local reference problem | Truth calibration, selection-aware inference or causal superiority | Do not tune divergence, weights, capacity or run a larger effect-estimation panel |
| Parity/structure repair | Authoritative defect finding | Modern-energy drift and relaxed-structure leakage were real; type-level fixes close both paths | That pre-repair metrics remain scientifically valid | Preserve old results only as data-audit history |
| Corrected 32-system P3C replication | Authoritative stopping evidence | Early probability signal does not replicate; P3C is slower and not Pareto superior | A material positive P3C effect | P3C main line is closed |
| Fixed-GP dispersion/ceiling | Authoritative diagnostic with explicitly approximate legacy probability fields | Reference GP is underdispersed and K=2 is below effective-dimension scale | A universal full-history optimum or calibrated oracle posterior | Exact-threshold cleanup cannot reverse the stopping decision |
| B40 compute gate | Authoritative systems NO-GO | GP numerical work is not a WBM end-to-end bottleneck | That representation compression is useless in all workloads | Reconsider only after a new workload independently passes the Amdahl gate |
| JARVIS--MP v1 | Authoritative negative mechanism result | Real paired signal exists; hard implementation gates pass; global transport and rank-16 state fail | Safe cross-system reuse or a positive compressed-state result | Opened systems are development-closed and cannot calibrate v2+ |
| JARVIS--MP v3 | Sealed calibration-stage NO-GO | Element offsets reduce but do not certify protocol discrepancy | Any v3 evaluation performance | Keep all 12 evaluation systems sealed; never relax 0.15 retrospectively |
| Environment-conditional transport + robust hull certificate | Authoritative fresh-system NO-GO | All-outcome correction and robust hull decisions work; inlier certificate soundness holds | Superiority over naive/global source reuse or the frozen 90% interval gate | Evaluation systems are opened and development-closed; do not tune this method on them |
| CHIC fixed-trace subset | Mechanism diagnostic | Hull-gradient matching is non-degenerate, but it does not beat diversity/hard-example selection | A useful training-compute frontier | Do not tune on these eight systems; change the failed task/model assumption |
| CHIC action-driven loop | Exploratory negative | Selected actions are sole reveals; causal-hull lookahead improves pure influence but source margin remains stronger | Closed-loop superiority | Use only calibration/development systems until a new task has a real training bottleneck |
| LeMat PBE/SCAN pair audit | Authoritative data-quality NO-GO | The downloaded Unique configs are intact but have zero ID/fingerprint overlap | Same-structure multi-protocol pairs | Do not pair by formula; use MatPES PBE--r2SCAN instead |
| MatPES pair/task and stoichiometry audit | Authoritative infrastructure plus historical invalidation | Exact PBE--r2SCAN pairs exist at scale; action-driven reveal works after preserving cell stoichiometry | Any result from the old normalized-composition/total-energy path | Never cite pre-repair MatPES closed-loop traces |
| Hierarchical MatPES posterior + Delta-Hull Active Search | Development mechanism followed by repartitioned NO-GO | Composition/reference correction and a local Matern posterior are non-degenerate; the 24-system signal is real on that panel | Superiority over source margin or a paper-level method claim | The 48-system repartition gives `+0.0625`, CI crossing zero and `p=0.6291`; do not tune Delta-Hull on the opened systems |
| Source-Rollout Delta-Hull | Implemented development method with an unresolved numerical gate | Full-budget posterior rollout, exact simulated causal-hull updates and source continuation pass independent pymatgen and reveal-boundary tests; fold-0 system effects agree 45/46 from MC512 to MC1024 | Cross-fitted superiority, action-level numerical convergence or a paper-level positive claim | Pause folds 1--5; diagnose integration only on the 276-system cross-fit partition and never use the 48 opened systems for development |

## Cross-iteration cause judgments

The following attributions summarize which explanations have been actively
tested rather than inferred from a negative headline.

| Possible explanation | Final judgment | Evidence basis |
|---|---|---|
| Selector or runner is numerically degenerate | Low for corrected P3C and JARVIS v1 | Constructed failure tests, nonzero objective ranges, selector disagreement and exact persistent/replay checks |
| Oracle information or target structure leaked into policy state | Confirmed historically, fixed | Fourteen Au--Te energy drifts, three label flips and relaxed-structure SOAP leakage; typed parity/structure contracts now fail closed |
| Input ordering or persistent/replay asymmetry explains the result | Low after hard gates | Deterministic order rules, checksum replay and exact homogeneous/emulation parity |
| Too few WBM candidates caused P3C failure | Low for the registered B8/K2 estimand | Two disjoint panels contain 32 exact systems and 683 candidates; the fresh panel reverses the early signal |
| WBM trajectories are too short for a systems-compression claim | High, then experimentally closed for WBM | B40 on the three longest eligible systems still yields at most 0.689% GP share |
| Frozen GP/reference is misspecified | High | Mean squared standardized LOO residual 16.90 and nominal 90% coverage 62.4% |
| DACC objective mismatches deployed joint inference | High | Singleton facility and joint-risk choices disagree; joint posterior is the deployed predictor |
| P3C preserves reference but not causal truth | High | Proper projection headroom is recovered locally without corrected cross-system decision superiority |
| Outcome-dependent contribution deletion induces selection mismatch | Medium--high | Retained/evicted residual differences and descriptive retention AUC; complete archive remains the correct target |
| JARVIS--MP pair construction is invalid | Low for the frozen task | Composition agreement plus frozen `StructureMatcher` leaves 1,658 real pairs; oracle and target structure remain isolated |
| Global transport generalizes across exact systems | Rejected | V1 held-out violation rate is 45.3202%; 88.459% of source--target shift variance lies between systems |
| Element-only reference correction is sufficient | Rejected at calibration | V3 clustered radius 0.177264 eV/atom exceeds the frozen 0.15 ceiling |
| Rank-16 failure proves representation compression is intrinsically bad | Unsupported | Rank 16 was not justified by a measured bottleneck and projection RMSE is 0.41698 eV/atom; this is an intentionally restrictive representation, not a lower bound |
| Early MatPES hull policies fail because their objective is wrong | Unsupported after audit | Their causal hull was corrupted after reveal by attaching a cell total energy to a normalized one-atom composition |
| More disjoint fit systems rescue the coarse random-intercept posterior | Rejected | Eight query systems with 88 disjoint fit systems still leave source margin stronger and posterior hull acquisition worse |
| Element reference and local discrepancy are irrelevant | Rejected in development | Element fractions reduce between-system variance and a local Matern posterior materially improves energy and hull errors; claim remains development-only |
| Source margin leaves no active-search headroom | Rejected on the 24-system development panel | The finite-pool oracle ceiling contains 19 additional budget-feasible confirmations; Delta-Hull recovers six | That the same headroom or capture ratio generalizes to a fresh split | Report ceiling and captured headroom with every future result |
| MC1024 exactly resolves the derived hull probability | Not established, but effect-level stability improved | MC512/1024 have identical discovery effects; first actions agree in 23/24 systems, complete traces in 21/24 and rounds in 134/144 | Exact trace convergence or fresh-split superiority | Freeze MC1024; carry the residual numerical limitation into fresh-split replication |
| The budget-six failure is entirely a one-step-horizon mismatch | Plausible but not established | On the opened 48-system attribution trace, five of seven persistent final losses begin only in rounds 5 or 6; the round-prefix effect is non-monotone | That full-budget rollout improves new systems | Test Source-Rollout only by out-of-fold development on the disjoint 276-system set |

## Manuscript treatment versus evidence preservation

The current paper-facing story and the complete audit record serve different
purposes. The main paper should not present two mutually exclusive methods.
The live non-deletion control is all-outcome representation compression; the old
definition `M_t subset A_t, |M_t| <= K`, the fixed-card figures, DACC and P3C
retention formulas belong to a historical appendix or negative-results
analysis. Moving them out of the main narrative does **not** authorize deleting
them from this ledger.

After implementing the robust hull certificate without obtaining a positive
decision--cost frontier, the paper has one currently supportable identity and
one explicitly future possibility:

1. the current negative-results paper on Decision--Inference--Systems
   Alignment, with all-outcome state as a control that excludes outcome
   deletion; or
2. a future positive method paper only after a scientifically different
   transport or constraint passes fresh transport, hull-decision and systems
   gates.

The hybrid claim "a positive certified method has been introduced" is not
supported. The exact linear--Gaussian sufficiency result is a necessary
implementation theorem, but it is standard conjugate sufficiency and is not by
itself the paper's theory contribution.

## Statistical and metric boundaries that must remain attached to E13--E14

1. V3 used ten exact systems for 90% system-clustered split conformal
   calibration with the maximum within-system absolute error as its score. The
   finite-sample index is

   \[
   \left\lceil(10+1)(1-0.1)\right\rceil=10,
   \]

   so the reported 100% calibration-cluster coverage is largely constructive:
   the radius is the maximum of ten system scores. The effective generalization
   sample size is the number of exact systems, not 677 pairs.
2. The frozen 0.15 eV/atom ceiling is an immutable v3 stopping rule. It must not
   become the next method's scientific objective. Future gates must ask whether
   the uncertainty interval certifies a candidate's hull decision or action,
   because the same energy radius has different meaning near and far from the
   hull.
3. V1's paired-affine action regret (`0.00096`) and hull decision cost
   (`0.81615`) measure different registered decisions. Regret scores only the
   selected next action; hull cost scores classification over the remaining
   pool. A policy can rank the best next query correctly while misclassifying
   many unqueried candidates. This separation is expected and must not be
   described as a metric inconsistency.
4. The conformal absolute-error radius is not a Gaussian standard deviation.
   Any pilot likelihood that inserts it as a variance scale is a working model,
   not a distribution-free posterior-calibration guarantee.
5. Rank 16 is a historical pilot choice. Without a measured bottleneck, future
   work may use full rank or a calibration-frozen higher rank; prediction loss
   caused by an arbitrary rank restriction is not evidence of a useful
   compression frontier.

## V4 hypothesis and completed disposition

The latest review narrowed the continuation to three layers, all now
implemented:

1. **Environment-conditional directed transport.** Predict target--source
   discrepancy from observable source structure, local coordination,
   element-conditioned environments, composition complexity and protocol-pair
   identity. Unseen elements, unsupported environments and excessive OOD score
   abstain.
2. **Same-candidate transport base plus all-target-outcome correction.** Keep
   the low-fidelity same-candidate prediction outside the global residual
   projection. Every revealed target residual relative to that base updates
   fixed-dimensional sufficient statistics; no accepted target contribution
   is deleted.
3. **Robust hull-decision certificate.** For simultaneous certified intervals
   `E_i in [L_i,U_i]`, compute lower and upper feasible hull energies over legal
   composition mixtures. Certify stable, unstable or abstain directly, and
   bound the epsilon-optimal action set. Do not convert an interval certificate
   into an unproved Gaussian probability.

This method changed the failed assumption from global/composition-only
transport to environment-conditional transport and changes the gate from a
global energy radius to decision-level selective coverage. It still requires
new system-level calibration, fresh evaluation, a simultaneous-coverage or
risk-control argument, strong target-only/naive-pooling/paired-delta/full-replay
baselines, and at least one measured access, certification, communication or
compute constraint. The calibration prerequisites passed, but the one-time
72-system evaluation did not. The method is therefore **stopped/NO-GO**, not
proposal-only. See E15.

## Research iteration chain

| Stage | What changed | Why it changed | Final status |
|---|---|---|---|
| CAW-Joint | Coupled acquisition and interval-witness retention | Original information gain mixed queried-item removal with evidence gain; capacity and hull claims were inconsistent | Corrected and frozen as method-level NO-GO |
| Working-set economics | Separated access cost from evidence selection and required exact persistent/on-demand emulation | Persistence cannot create value when reconstruction is free and selectors are matched | Preregistered design retained; not evidence for a method win |
| Secure WBM | Added oracle isolation, initial MP phase diagrams, composition-dependent causal hulls and exact chemical-system pools | Scalar hulls and policy-visible oracle information were scientifically invalid | Infrastructure gate passed |
| DACC | Replaced interval heuristic with decision-weighted singleton facility gains | Sought a submodular, observable calibration coreset | Retired: facility objective did not match the deployed joint posterior |
| Joint-risk diagnostic | Refit the actual GP for each one-swap candidate set | Tested whether the DACC proxy caused the result | Highly correlated with DACC and much slower; no stable advantage |
| P3C | Projected a fixed union posterior onto exact drop-one subsets under proper divergences | Corrected the DACC objective/reference mismatch locally | Mechanism works; causal-calibration hypothesis fails |
| P0 provenance repair | Bound oracle energies to frozen parity values and SOAP to official WBM `org` structures | Found 14 Au--Te energy drifts, three label changes and relaxed-structure leakage | Code/data defects fixed; affected results invalidated for claims |
| Structure-correct replication | Ran two disjoint 16-system panels with initial structures and parity energies | Distinguished method failure from implementation/data failure | Early probability signal did not replicate; P3C stopped |
| AKSC proposal | Outcome-independent kernel basis plus all-outcome natural parameters and a separate calibration layer | Avoided deleting paid outcome contributions and separated operator fidelity from calibration | Proposal only |
| Fixed-GP and B40 gates | Measured reference dispersion, effective dimension and real end-to-end GP share | A representation-compression method needs a credible target and material compute bottleneck | Both gates fail for current WBM; AKSC not authorized here |
| JARVIS--MP v1 | Built a real structure-matched OptB88vdW--MP/GGA task and tested global affine transport plus an all-outcome rank-16 state | Moved from homogeneous WBM to genuine protocol heterogeneity | Implementation gates pass; transport and state decision gates fail |
| JARVIS--MP v3 | Added element-fraction reference offsets and exact-system-clustered conformal calibration on fresh systems | Tested whether composition-dependent reference shift explains v1 failure | Radius 0.177264 exceeds frozen 0.15 ceiling; fresh evaluation remains unopened |
| JARVIS--MP v4 | Frozen CHGNet source features, leverage-scaled exact-system intervals, all-target-outcome correction and robust hull LP | Tested environment-conditional reuse against direct decision preservation | Certificate mechanism is sound on interval inliers, but 90% coverage and superiority gates fail; method stopped |

## Experiment ledger

### E0. Corrected CAW-Joint synthetic falsification suite

- Recovery point: tag `caw-method-no-go-2026-07-15`, commit `0fa1e1f`.
- Tests included recurrence, IID residuals, no recurrence, protocol reversal,
  retention competition, budget/capacity curves, input permutations and an
  exact beta--Bernoulli dynamic-programming comparator.
- The handcrafted retention-competition case showed a narrow joint mechanism,
  but uncertainty plus FIFO was stronger in key scenarios; gains disappeared
  across budgets and were not globally optimal.
- Claim status: historical mechanism/negative result only. Removed runners must
  be recovered from the tag, not copied into the live package.

### E1. Working-set economics preregistration

- Recovery point: tag `wbm-preregistration-freeze-2026-07-16`, commit
  `ca80720`.
- Froze acquisition-by-access factorization, exact zero-cost emulation,
  physical-cost measurement versus offline price recomputation, cache rules,
  chemical-system clustering and execution feasibility.
- The proposed `12 x 256` construction was later rejected because exact WBM
  chemical systems are much smaller. No outcome-based pool replacement is
  permitted.
- Claim status: design history, not an empirical win.

### E2. WBM raw data, parity and secure reveal gates

- Engineering P1/P1.5 audit:
  `E:\DATA\EviMem-RL\manifests\wbm-engineering-p1-p15-audit-v1.json`.
- SHA256:
  `f3941364f2df317fffea3ab63286f66e624449af88f0c48a2f60585551b68e96`.
- Established raw WBM checksums, cleaned IDs, MP phase membership, parity
  energies, initial hulls, oracle-vault isolation, append-only reveals and
  composition-dependent causal hull transitions.
- Claim status: infrastructure/correctness only.

### E3. Early small-pool DACC engineering pilots

- B4/K4 smoke:
  `E:\DATA\EviMem-RL\outputs\engineering\wbm-calibration-coreset-b4-k4-v1\summary.json`.
- B8/K2 closed loop:
  `E:\DATA\EviMem-RL\outputs\engineering\wbm-calibration-coreset-b8-k2-v1\summary.json`.
- Matched B8/K2 trace:
  `E:\DATA\EviMem-RL\outputs\engineering\wbm-calibration-matched-b8-k2-v1\summary.json`,
  SHA256 `7c6ed468f8bb7e31e6dcd8389cbc7fc0df373daad78bd20be869984a63becbf8`.
- DACC had the best early RMSE/NLL while diversity or GP variance had better
  probability metrics. Eight systems and one cell could not support
  superiority.
- Claim status: historical engineering signal only; later provenance findings
  prevent using it as claim-grade WBM evidence.

### E4. DACC objective-fidelity and joint-risk diagnostic

- Result:
  `E:\DATA\EviMem-RL\outputs\engineering\wbm-objective-fidelity-gpvariance-matched-b8-k2-v1\summary.json`.
- SHA256:
  `1cf8336f8b78c2223246aec0bf142077ea77c526bd39133550d38211571415b6`.
- At saturated neighborhoods, facility and joint-risk ranks had mean Spearman
  about 0.878 and selected the same action about 84.8% of the time. Joint-risk
  was substantially more expensive and did not show stable metric superiority.
- Claim status: diagnostic. It exposed the objective mismatch but did not
  justify replacing DACC with joint self-risk as a main method.

### E5. Frozen GP calibration and interrupted DACC grid

- Calibration summary:
  `E:\DATA\EviMem-RL\outputs\calibration\wbm-gp-margin-calibration-v3-streaming\summary.json`,
  SHA256 `29c8ea370e2900b7e3f3a60816588bc1f4671ca12f3a3b190b9fbc9b67e74c7b`.
- Freeze manifest:
  `E:\DATA\EviMem-RL\manifests\wbm-gp-and-noninferiority-calibration-freeze-v1.json`,
  SHA256 `0f63e146bdc98bca96051cbf7bf07f3896a7bbd3b312eb69926f80583315055e`.
- Complete physical K1 group:
  `E:\DATA\EviMem-RL\outputs\engineering\wbm-frozen-grid-v1-streaming\physical\primary-k1-b12\summary.json`,
  SHA256 `cb5b9abe76c592f656fd815067b1d13477a45e14f766761880dd5c48a1ac586e`.
- K2 was interrupted after 47/64 ledgers. The partial grid is an interruption
  audit only and must never be merged with a later rerun or used for inference.
- Claim status: calibration freeze and execution record, no grid-level result.

### E6. P3C matched-action mechanism panel

- Summary:
  `E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-objective-fidelity-matched-b8-k2-v1\summary.json`,
  SHA256 `4facffd371820bf25678e16e8311bb4c1b7c798f363661c53a1e55102a6109fa`.
- Analysis SHA256:
  `285dbfaa248d57bc5f2f9b664d08660f9f2aa99c088e0de7f44c3f726a173313`.
- Compared GP variance, legacy DACC, joint self-risk and four proper P3C
  divergences under identical actions. Only NLL had a bootstrap interval below
  zero for one P3C variant; CRPS superiority plus Brier/log non-inferiority did
  not hold.
- Claim status: mechanism validation and method-level NO-GO.

### E7. P3C reference/path/selection diagnosis

- Authoritative summary:
  `E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-reference-path-selection-b8-k2-v5\summary.json`,
  SHA256 `0d25f251a1d1ede6dc2b63c5e2ed7c8782fde716f984b45d9c93060ea4b2f9b3`.
- Decomposition SHA256:
  `149ed9562d5a6c6d550c550f99711542df2733c58236cc3f7daf802a9461ef1d`.
- Separated union/archive references from online/archive subset search,
  decomposed NLL mean/variance effects and measured outcome-dependent
  retention. Archive search changed subsets but did not yield stable causal
  benefit.
- Failed v2/v3 directories have no valid summary. v4 has the same scientific
  traces but incomplete lazy-GP timing. Only v5 is citable.
- Claim status: authoritative historical P3C mechanism diagnosis; later
  structure-stage audit prevents treating its absolute WBM metrics as final.

### E8. Frozen-energy and initial-structure provenance repair

- Fourteen Au--Te oracle energies drifted by 0.105--0.241 eV/atom when modern
  pymatgen recomputed values instead of reading frozen 2023.5.10 parity
  energies; three stable/phase labels changed.
- The SOAP cache used post-DFT relaxed CSE structures rather than official WBM
  `org` initial structures. All 334 selected structure hashes changed after the
  correction.
- The live type system now rejects relaxed policy features and non-parity oracle
  energies. Pre-reveal actions are invariant to unrevealed oracle changes.
- Claim status: confirmed code/data defect, fixed. All affected results remain
  immutable audit evidence but are invalid for scientific claims.

### E9. Structure-correct P3C replication

- First disjoint 16-system panel:
  `E:\DATA\EviMem-RL\outputs\exploratory\wbm-p3c-16sys-b8-k2-initial-org-v3\summary.json`,
  SHA256 `b1d5de011a67495338280f584831858a2af283d024898d2e1a32797194dd7250`.
- Fresh next-16 panel:
  `E:\DATA\EviMem-RL\outputs\exploratory\wbm-p3c-next16-b8-k2-initial-org-v1\summary.json`,
  SHA256 `4dd03bd81afebff63c5c9fa534ff722f01d381fbe36697db00805946c359441e`.
- Combined scope: 32 exact binary/ternary systems, 683 candidates, matched
  frozen actions, B8/K2. Pooled P3C-minus-GPV Brier is `-0.000003`; CRPS,
  log loss, RMSE, NLL and runtime are worse on average. Fresh-panel NLL is
  significantly worse (`+0.275486`, 95% CI `[+0.020473,+0.631204]`).
- Claim status: authoritative stopping evidence. Do not run a larger P3C panel
  merely to estimate this near-zero effect more precisely.

### E10. Fixed-GP ceiling and representation diagnosis

- Result:
  `E:\DATA\EviMem-RL\outputs\diagnostics\wbm-fixed-gp-ceiling-32sys-b8-k2-v3.json`,
  SHA256 `bc1e836aefa2a72ac98d46f5788d5479d775ba4781703a9fc3d64ef3adcd31ba`.
- Mean regularized effective dimension is 12.42; every system exceeds four.
  Kernel Moran autocorrelation is near zero and full-history LOO NLL is poor.
- The legacy input saturated 336 probabilities, so prior/oracle Brier and
  log-loss values are approximate. CRPS and residual quantities are exact.
  Current schema v2 requires explicit residual thresholds and fails closed.
- Claim status: evaluator-only ceiling diagnostic, not a policy result.

### E11. Direct fixed-GP dispersion audit

- Result:
  `E:\DATA\EviMem-RL\outputs\diagnostics\wbm-fixed-gp-loo-dispersion-32sys-v1.json`.
- SHA256:
  `e63c0cee32071ce389c43ca080078d499ae0794c82a1d6d190d807cf982bcaa5`.
- Mean squared standardized residual is 16.90; central 50%, 80% and 90%
  coverage is 34.2%, 52.6% and 62.4%.
- Claim status: direct evidence that the frozen GP is underdispersed. It uses no
  stable-probability threshold inversion.

### E12. Long-archive B40 compute-relevance gate

- Frozen manifest:
  `E:\DATA\EviMem-RL\manifests\wbm-long-archive-compute-gate-v1.json`.
- Read-only checkpoint:
  `E:\DATA\EviMem-RL\checkpoints\wbm-long-archive-compute-pre-run-v2.json`,
  SHA256 `b24d326e57e9b860215c6f6d16176e1cf89ae4090d09adb1b5aac96ecf313ce7`.
- Physical summary:
  `E:\DATA\EviMem-RL\outputs\diagnostics\wbm-long-archive-full-history-b40-v2\summary.json`,
  SHA256 `49ff84a63f6f7ed04cdc5aeae41847b6608878bcfb480237a511d44495b897a8`.
- Compute analysis:
  `E:\DATA\EviMem-RL\outputs\diagnostics\wbm-long-archive-compute-relevance-v2.json`,
  SHA256 `6c907f93226cf174b84c2b2cef5636b8081c52882b79204d1557845494f2a13f`.
- Fe--S (46), Fe--Zr (44) and Ni--S (44) use all 134 candidates. Independent
  v1/v2 actions and trace checksums match. Maximum GP fraction at B40 is
  0.6888%; the frozen gate is 9.09%. Peak RSS is about 8.73 GiB while the B40
  dense float64 kernel matrix is 12,800 bytes.
- A fixed 32-probe, one-thread microbenchmark confirms microsecond dense-GP
  scaling, but is not a causal or end-to-end metric.
- Claim status: authoritative FAIL. It closes WBM posterior compression as the
  next main research line.

### E13. Real JARVIS--MP global-affine protocol pilot

- External directory:
  `E:\DATA\EviMem-RL\multifidelity\jarvis-mp-v1`.
- Task manifest SHA256:
  `d98e87545198c47d318dab67802a95dc87049ff2ab159ad296916ef92359b281`.
- Calibration freeze SHA256:
  `3b6c525eb0458f75ce7a17f36d50bd2fb379332d8b2c73a8e68f6cd0cc44ce4a`.
- Pilot result SHA256:
  `02fddb48f285eb0bd1102479c64ac028b8b33dc0a4c89cd72a599e6e6a904da6`.
- Failure attribution SHA256:
  `5461a223a18110ba0ded4f99d51bcfdae53905a87544886ce4d42e856a658d4e`.
- The source audit found 75,993 unique JARVIS IDs; 49,905 carried an explicit
  MP reference, 44,056 referenced the frozen MP snapshot, and 38,329 pure-GGA
  records had composition agreement. Frozen `StructureMatcher` settings
  retained 1,658 real matched pairs.
- Ten calibration exact systems contain 212 pairs; ten disjoint evaluation
  systems contain 203 pairs across four binary, four ternary and two
  quaternary-or-higher systems.
- Target-only/rejection, persistent/replay and homogeneous-null trajectories
  are exact, and every legal outcome contributes to the numerical state.
- The global affine map has slope 0.970174, intercept -0.189481 eV/atom and
  radius 0.099940 eV/atom, but held-out violation is 45.3202%. Paired transport
  has strong signal; the rank-16 state does not improve hull cost or regret.
- 88.459% of formation-energy shift variance is between exact systems, while
  rank-16 projection RMSE to transported residuals is 0.41698 eV/atom.
- Claim status: authoritative mechanism NO-GO. It validates the task and hard
  implementation gates, not safe transport or a positive compressed state.

### E14. Composition-aware clustered-certificate gate

- External directory:
  `E:\DATA\EviMem-RL\multifidelity\jarvis-mp-v3`.
- Task manifest SHA256:
  `4765bc82de2237ae6156b7a5642c27f62a3565e940876c13e250f8d8f0dd6872`.
- Composition calibration freeze SHA256:
  `571fff2aa7cbd31a8d84c9034822bc74ae407f4d8a262349bac122c77f5722d7`.
- Thirty-three calibration systems contain 677 pairs; 23 systems fit a source
  slope plus element-fraction offsets and ten disjoint systems calibrate the
  maximum within-system absolute error. Twelve fresh evaluation systems (261
  pairs) have zero overlap with v1 evaluation systems.
- The fitted source slope is 0.995984. The clustered radius is
  0.177263735 eV/atom, above the frozen 0.15 ceiling, despite 100% empirical
  cluster coverage. The four largest system errors include Cu--Li--O,
  Cd--Mg, C--Li--O and Cu--Li--S.
- `certificate_passed=false` and `evaluation_results_accessed=false`. Three
  evaluation systems with unseen elements were already registered for exact
  target-only fallback; none of the 12 evaluation systems was opened.
- Claim status: calibration-stage NO-GO. Do not relax the ceiling to 0.18 or
  use the unopened systems to develop a richer transport.

### E15. Environment-conditional transport and robust hull decision

- External directory:
  `E:\DATA\EviMem-RL\multifidelity\jarvis-mp-v4-natural`.
- Task manifest SHA256:
  `ba5bfc139364ecb1b97248e8f72fd646e12ffaa9277a6494f4ebbb265eee5cff`.
- Final calibration freeze SHA256:
  `954acb310a156299c25aaa4b5415a3d6cf0e4118286f73c8d6c03e0a2380279c`.
- Fresh evaluation SHA256:
  `644fd0b547284034314c3c105a9b040363c1bd4b1ec3dc54df1ef191e4242f94`.
- All 45 v1/v3 systems were excluded. The task used all remaining eligible
  exact systems: 210 calibration systems/2,056 pairs and 72 evaluation
  systems/717 pairs. CHGNet-0.3.0 source features are bound to checkpoint
  SHA256 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.
- Calibration passed: 48 conformal-inlier systems had zero certified errors;
  macro certified coverage was 17.32% with bootstrap lower bound 12.42%.
- Evaluation opened once. Fifty of 56 supported systems met the simultaneous
  interval event (89.29%, below the frozen 90% gate). Inlier certified errors
  remained zero and the certified-coverage lower bound was 8.08%.
- The method strongly improved on target-only but was significantly worse than
  naive source-as-target: hull-error difference `+0.07410`, 95% interval
  `[+0.02387,+0.12954]`; regret difference `+0.01200`, interval
  `[+0.00422,+0.02031]`. Global paired delta also had lower MAE and regret.
- Online all-outcome correction improves its environment-only base in MAE and
  hull error, proving non-degenerate operation, but not in action regret.
- Claim status: authoritative method-level NO-GO. Preserve the certificate
  theorem/implementation as a valid mechanism and the result as evidence that
  richer representation alone does not beat simple low-fidelity reuse.

### E16. CHIC fixed-reveal training-subset diagnostic

- External K2 result:
  `E:\DATA\EviMem-RL\outputs\exploratory\chic-training-subset-8sys-b6-k2-v2.json`,
  SHA256 `296536123c9bf9e77735981404219e6567c78f18c33ac080e1740d0c7c5016cd`.
- External K1/rho10 result:
  `E:\DATA\EviMem-RL\outputs\exploratory\chic-training-subset-8sys-b6-k1-rho10-v1.json`,
  SHA256 `bbec19d6bb84cfc80b986dc387e7d5cb4cfc6d4fa882cc21a650cd6a383fe759`.
- Both runs use eight calibration-development exact systems, fixed six-query
  reveal traces and an MLP pretrained on disjoint calibration-fit systems.
  Evaluation systems are not accessed.
- At K2, CHIC/GRAD-MATCH select the same subset in all 48 rounds. CHIC mean MAE,
  hull error and action regret are `0.042598`, `0.121402`, `0.016965`, versus
  diversity `0.043014`, `0.116483`, `0.016965` and hard example `0.041720`,
  `0.118797`, `0.016965`. Mean CHIC gradient evaluations are 79.5 versus 11 for
  the simple bounded selectors.
- At K1 with decision-curvature weight 10, CHIC and GRAD-MATCH differ in 6/48
  rounds, proving non-degenerate hull influence. CHIC still has MAE `0.042890`
  and hull error `0.121402`; diversity has `0.042999` and `0.111034`, while hard
  example has `0.042209` and `0.116342`.
- A later causal-target audit found that this version allowed unqueried
  predicted candidates to support the decision-gradient hull, whereas the
  registered online evaluator uses only initial and revealed target phases.
  The code is corrected, but these results remain a mechanism diagnostic.
- Claim status: no training-subset advantage; do not tune capacity or curvature
  on these eight development systems.

### E17. CHIC action-driven closed-loop diagnostic

- First output:
  `E:\DATA\EviMem-RL\outputs\exploratory\chic-closed-loop-8sys-b6-v1.json`,
  SHA256 `a961b85212e41d987d5fd1bfb939ef952129d676b42d91d4aa785fe78fa7b608`.
- Causal-hull corrected output:
  `E:\DATA\EviMem-RL\outputs\exploratory\chic-closed-loop-8sys-b6-v2.json`,
  SHA256 `5aaaac9973cd799bee0cf89651b8a9ad2cc3831ee4f83a900209cf92a1b1aad8`.
- The native runner persists the policy action before reveal, reveals only that
  ID, appends every target outcome to the archive, and rebuilds the
  composition-dependent causal hull. Unrevealed outcome perturbations cannot
  change the first action in failure-capable tests.
- In v1, pure hull influence has system-macro mean action regret `0.182061`
  versus ridge uncertainty `0.184915` and source margin `0.107554`. CHIC and
  uncertainty agree in 44/48 rounds and CHIC loses to source margin in all
  eight systems.
- V2 removes unqueried predicted phases from the causal hull and adds two
  same-predictor controls. Ridge margin obtains regret `0.123046`; a two-step
  causal-hull lookahead with no hand-mixed exploration weight obtains
  `0.145319`; pure influence remains `0.182061`; source margin remains best at
  `0.107554`. Lookahead recovers 1.5 stable discoveries per system, equal to
  source margin, but beats source margin on only 2/8 systems and is slower
  (5.922 versus 4.119 seconds per system).
- Diagnosis: the reveal boundary and hull update are correct. The remaining
  failure is scientific/model-level: a few within-system labels cannot support
  a 64-dimensional local correction, pure information acquisition is
  misaligned with cumulative discovery regret, and the low-fidelity source
  already gives a very strong ranking.
- Claim status: exploratory negative. Move to a paired protocol task with a
  real training-update budget; do not tune the JARVIS systems further.

### E18. LeMat BulkUnique PBE/SCAN pair-key audit

- External output:
  `E:\DATA\EviMem-RL\outputs\exploratory\lemat-pbe-scan-pair-key-audit-v1.json`,
  SHA256 `88a1df038bf5d9780f80c88417fd065be33db66189f49e9c351098d22d31743c`.
- The official snapshot contains 5,005,017 PBE rows in 16 shards and 417,666
  SCAN rows in two shards, with no missing IDs/fingerprints and no duplicate
  SCAN IDs/fingerprints.
- Cross-config intersection is exactly zero for both `immutable_id` and
  `entalpic_fingerprint`. The Unique configs are internally intact but are
  mutually deduplicated sets, not paired calculations.
- Claim status: authoritative data-task NO-GO. Formula-only matching is unsafe.
  Use the official MatPES PBE/r2SCAN 2025.2 files, which expose a common
  `matpes_id` and structure record, and audit exact structure equality before
  any CHIC run.

### E19. MatPES exact-pair task, frozen representation and stoichiometry repair

- Pair audit: 385,890 exact same-configuration PBE--r2SCAN pairs; 84,532 have
  formation energy on both sides. Audit SHA256
  `248684a47ee0b08964efb2c00705cdac291516b47524935d90845169194decab`.
- Frozen development task: 324 exact chemical systems, 10,236 pairs, 16--64
  selected candidates per system and no repeated original Materials Project
  parent within a system. Every selected pair has a frozen CHGNet-0.3.0
  64-dimensional `crystal_fea`. Task SHA256
  `f43c1ab99995e229edd95b47c834f9e9b439d04fc3de0a369cc6d79f7f74d0df`;
  checkpoint SHA256
  `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.
- Development vault SHA256
  `a272d3a2ce6286443ae6fce35726a688751a37284e3df362c5d1f70e2fcb9952`.
  Policy state contains PBE structure/energy and protocol identity but never an
  unrevealed r2SCAN energy or final-hull label.
- Five PBE structures contain 14 atoms isolated at a 6-Angstrom neighbor
  cutoff. They are retained and explicitly audited rather than silently
  filtered after outcomes are inspected.
- Defect found: the earliest MatPES reveal path normalized a cell composition
  to one atom but attached the unnormalized cell total energy. Every trace
  after its first reveal therefore built the wrong causal phase diagram.
  `ProtocolOracleOutcome` now preserves stoichiometry and the causal hull
  converts total and per-atom energies consistently.
- Claim status: authoritative infrastructure plus historical invalidation. No
  pre-repair MatPES performance number is scientifically usable.

### E20. Hierarchical PBE--r2SCAN posterior development diagnostics

- A system-balanced global discrepancy mean, exact-system random intercept and
  local Matérn-5/2 process are fitted on systems disjoint from each query
  system. The mean uses PBE observables and element fractions; the kernel uses
  the frozen CHGNet source representation. Every revealed target outcome
  remains in the posterior.
- Composition-kernel 24-system output SHA256
  `8466ee6abf4afc208321c850d7826630dcda0102b3a67ae461aa1544e7e648c3`:
  posterior ranking improves AP/AUC over source margin, but oracle-final
  confirmations improve only `+0.0833` per system with 95% interval
  `[-0.1667,+0.3333]`.
- CHGNet-kernel pseudo-random-MC128 24-system output SHA256
  `7f8240a6b6e058d885a1bb3f0ace77e0ccf02b8df62c62b4d4989be093b595cb`:
  oracle-final difference `+0.1667` per system, interval `[0,+0.375]`, with
  five wins, eighteen ties and one loss. Full traces were numerically unstable
  under 32/128 sampling, so this is not claim-grade evidence.
- Nested scrambled-Sobol sampling replaces ordinary pseudo-random Monte Carlo.
  For the same seed, every power-of-two design is an exact prefix of the next;
  8-system MC32/128/512 outputs have SHA256 values
  `8c655340806c424f9d683a7a8e211c5c12619a6779c5bde182618b4c00a88752`,
  `82caace7f80f9d4e1c4c8c053ed392ebb3dec322c5fe5cce8f706cc3894c9bab`
  and `a86d5f33361a48acf8f5bf6ede973bf2bbad1c0757dde07e1a4c05c4c0e32193`.
- Claim status: valid mechanism development. It justifies a larger fixed
  development panel, not a superiority claim.

### E21. Delta-Hull Active Search, 24-system development panel

- Scientific change: Delta-Hull does not compress or delete outcomes. It
  propagates the hierarchical protocol posterior through the complete fixed
  target-protocol hull and, for equal costs, selects the candidate with largest
  posterior final-hull-membership probability. The target oracle reveals an
  energy rather than a final label; membership is a delayed structured event
  coupled through all pool energies. This is the exact one-query Bayes action
  for the registered reward, not a finite-horizon optimality claim. Unequal
  costs are rejected rather than handled by an unproved probability/cost
  ratio.
- Frozen panel: 24 hash-selected exact systems, budget six, 300 disjoint fit
  systems/9,405 fit rows, selected-action-only reveal and full-history
  posterior conditioning. Evaluation systems were not accessed.
- Sobol MC128 output SHA256
  `a1cab6edea238fadbf79b050b6063811d87b59a03a1c6ca6bcd2c556ba8bd541`.
  Delta-Hull obtains `3.6250` oracle-final phases per system versus `3.4583`
  for source margin; paired difference `+0.1667`, 95% interval
  `[-0.0833,+0.4167]`.
- Sobol MC512 output SHA256
  `f70c444606e06b452087d6b3bce54cdf73a645c4ae2251f6095168e277306c96`.
  Delta-Hull obtains `3.7083` versus `3.4583`; paired difference `+0.2500`,
  interval `[+0.0417,+0.5000]`, with 6 wins, 17 ties and 1 loss. Source margin
  leaves 19 discoveries below the finite-pool budget ceiling; Delta-Hull
  recovers six (`31.58%`).
- Sobol MC1024 output SHA256
  `0d33010093e1385e74a9c8ef263b43699c399bb168441613a6577f1eba5f6c06`.
  The oracle-final means, paired `+0.2500` difference, bootstrap interval and
  6/17/1 win/tie/loss counts are identical to MC512. MC512/1024 agree on the
  first action for 23/24 systems, the full six-step trace for 21/24 and 134/144
  individual rounds. The three systems with trace changes retain the same
  system-level discovery difference.
- These three artifacts were launched from an isolated pre-rename snapshot and
  record the policy label `protocol_hull_probability`. That implementation is
  the same posterior final-hull-membership rule now named
  `delta_hull_active_search`; the live tree retains no compatibility alias.
  A post-run audit found that the renamed local runner had not added Delta-Hull
  to its transport-model routing predicate. The isolated runs are unaffected
  because their old label matched the old `protocol_hull_` predicate. The live
  routing is corrected and covered by a failure-capable test before any rerun.
- The effect is narrow and objective-specific. Causal discoveries tie at
  `4.3333`; final-causal confirmations are `4.1250` versus `4.0417`; oracle-pool
  invalidation is `0.6250` versus `0.8750`. Prequential energy MAE is worse by
  `+0.00376` eV/atom, while prequential mean-hull MAE improves by `-0.00145`
  eV/atom. Wall time is `22.13` versus `1.99` seconds per system.
- A diagnostic worker profile at commit `090b4cb` localizes this cost rather
  than treating it as GP training overhead. On the 64-candidate Co-F-Li-O
  development system, six rounds at MC1024 invoke 6,144 complete phase
  diagrams, each with 68 total candidate/reference entries. Under `cProfile`,
  final-hull propagation accounts for 175.73 of 175.85 worker cumulative
  seconds, including 167.89 seconds in `PhaseDiagram` construction; posterior
  conditioning and Gaussian sampling account for only 0.052 and 0.057 seconds.
  Profiling magnifies Python-call overhead, so these absolute numbers are
  bottleneck-localization evidence only. The unprofiled MC512-to-MC1024 increase
  from `14.02` to `22.13` seconds/system independently supports sample-wise
  hull propagation as the added online cost. The reported online wall time
  excludes the once-per-panel transport fit and the post-trace evaluator.
- The MC512 effect survives MC1024, so ordinary finite-sample integration noise
  is no longer the leading explanation for the discovery signal. Exact trace
  convergence is still false, and the same development systems supplied both
  levels. This closes only the development effect-stability check.
- Claim status: promising development signal, not confirmatory evidence and not
  paper-level GO. Do not add an acquisition blend or tune posterior parameters;
  freeze MC1024 and test the unchanged method on a fresh exact-system split.

### E22. Delta-Hull 48-system outcome-independent repartition

- Split change: the 324 eligible MatPES exact systems are repartitioned before
  refitting. SHA256 selection reserves 16 binary, 16 ternary and 16
  quaternary-or-higher systems (1,524 candidates, one row per original MP
  parent); the hierarchical transport is refitted on the remaining 276 systems
  and 8,712 pairs. Selection does not use r2SCAN outcomes. Because the overall
  corpus informed earlier development, this is called a repartitioned
  replication rather than a pristine external confirmation.
- Task SHA256
  `2a8e09f2c77fe9a92ede60f02a3ee6a2cf7f1210c4c4fa94cc67890df02d6f36`;
  oracle-vault SHA256
  `ada607c5d041f64c621c5f1139d5c883c3ec983836d429354a9340305a912358`;
  frozen transport checksum
  `sha256:98a2e9f7ca6328e2ecf60294bec9f115a73d2939be107a51bafbfd12eae42f01`.
  The kernel optimizer converges with status zero, gradient norm
  `1.04668e-6`, no active bounds, length scale `1.77719`, signal variance
  `0.00337948` and noise variance `0.000237849`.
- The unchanged policy uses budget six, MC1024 and selected-action-only reveal.
  Result SHA256
  `d838a27a28e151870fc5cc908aaf4d3eed9939370b3e5397b9a30f3b0d74bc21`.
  Delta-Hull obtains `3.6250` oracle-final confirmations/system versus
  `3.5625` for source margin: paired `+0.0625`, bootstrap 95% CI
  `[-0.1042,+0.2292]`, exact two-sided sign-flip `p=0.6291`, with 10 wins,
  31 ties and 7 losses. The 45 transport-supported-system sensitivity is
  `+0.0667`, CI `[-0.1111,+0.2444]`, with the same `p`-value. The three
  unsupported binary systems (N--Pd, N--Re and O--U) correctly fall back to
  source margin.
- Delta-Hull significantly beats ridge margin (`+0.4792`, CI
  `[+0.1875,+0.7708]`, `p=0.00301`) and posterior-mean final margin
  (`+0.6042`, CI `[+0.2917,+0.9167]`, `p=0.000625`). These are weaker than
  source margin and do not establish the paper hypothesis.
- Mechanism diagnosis: source margin reaches the budget ceiling in 24/48
  systems and leaves 35 total budget-feasible confirmations; Delta-Hull
  recovers a net three (`8.57%`). It disagrees with source margin on 28/48
  first actions, 45/48 full traces and 213/288 round actions, so the null is not
  caused by selector collapse. Causal discoveries are slightly lower
  (`-0.0417`/system), final-causal confirmations tie, and oracle invalidations
  fall by `0.1042`/system, but all three paired intervals cross zero.
- Engineering disposition: the first three complete MC1024 real-system traces
  from the original pymatgen backend match the cached fixed-composition backend
  action for action. Binary, ternary and duplicate-composition property tests
  also match stable masks exactly. Removing repeated post-trace sampled-hull
  diagnostics from every policy/round reduces the complete 48-system execution
  to about 9.5 minutes; those diagnostics can be replayed offline. Online
  Delta-Hull time is `5.07` seconds/system versus `1.93` for source margin.
- Claim status: stopped/method-level NO-GO for superiority over source margin
  on this task. All 48 systems are opened and development-closed. Do not tune
  the kernel, Sobol seed, score, horizon or support filter on them.

### E23. Horizon attribution and Source-Rollout Delta-Hull

- Failed assumption changed: repeated myopic final-membership maximization is
  only the exact Bayes action at remaining budget one, while the deployed task
  has budget six. The new method is not a score blend. For every candidate
  first action and complete posterior energy sample, it inserts the sampled
  target energy into a simulated composition-dependent causal hull and runs
  the deployed source-margin policy for every remaining step. Terminal reward
  counts selected phases on that sample's complete final target hull.
- The source action is explicitly included in the rollout action class. Under
  the registered posterior and exact expectation, maximizing the rollout
  value is therefore no worse than continuing source margin. This is a
  model-relative policy-improvement statement, not a real-distribution safety
  theorem. Sixteen paired scrambled-Sobol blocks supply a numerical
  integration safeguard. A Bonferroni-simultaneous one-sided lower bound is
  applied across all non-source candidates; without a positive bound, the
  policy falls back to source.
- Correctness evidence at commit `3078ea1` plus the immediately following
  test-hardening change: cached causal-hull energies match independent
  pymatgen phase diagrams for binary, ternary, quaternary and five-element
  sampled states;
  every candidate's complete rollout action sequence and terminal reward match
  an independent pymatgen implementation; a constructed horizon-two case
  improves reward from one to two; and the subprocess reveal boundary still
  exposes only the persisted selected action. Full suite: 182 tests, Ruff
  clean.
- Attribution-only result:
  `E:\DATA\EviMem-RL\results\matpes-repartition-v1-horizon-diagnostic.json`,
  SHA256 `2d29072617b82247d8840f104ccbd8c0c71ce82ae7a38d705cc969eae41eb501`.
  Prefix Delta-minus-source confirmation means by round are `0`, `-0.0417`,
  `+0.1458`, `+0.1667`, `+0.1042`, `+0.0625`. Five of the seven persistent
  final losses begin only in rounds five or six. On the 24 nonzero-headroom
  systems the descriptive final difference is `+0.2917`. These opened traces
  motivate the horizon test but cannot establish rollout performance.
- Outcome-independent development plan SHA256
  `a76a10a60c021cdf9bcfe922c457ee4809054da99e3e2b7debe5be8d29be5afa`:
  the 276 former transport-fit systems are split into six folds of 46, each
  containing 12 binary, 19 ternary and 15 higher-order systems. All 48 opened
  systems are excluded; each fold posterior uses only the other five folds.
- Development continuation requires all of: mean gain over source above
  `0.15` confirmations/system, losses no more than half the wins, a clearly
  positive nonzero-headroom effect, positive effects in at least two
  complexity strata, stable higher-integration rollout actions/effects, and
  runtime small relative to a target DFT query. Failure stops this method
  without adding a blend, temperature or posterior change.
- First engineering smoke uses fold 0, budget two and MC16; result SHA256
  `9ad76c6238fbf15775ba6a18253f7f177772b2f454a13c80d8214203f905a548`.
  Source-Rollout minus source is `+0.0652` confirmations/system with 7 wins,
  35 ties and 4 losses; myopic Delta-Hull is `+0.0217` with 10/29/7. This
  validates real-data execution and the expected loss-reduction mechanism at
  deliberately low integration fidelity. It is not a GO result and cannot be
  used to tune the method.
- The first budget-six fold-0 pilot uses MC128 without changing the method;
  result SHA256
  `705ea89e00e31ecdb3a5ab7211cff1b666a2874c703c69ad24f3e77e66b70fca`.
  Source-Rollout minus source is `+0.1957` confirmations/system, bootstrap 95%
  interval `[0,+0.4130]`, exact sign-flip `p=0.1245`, with 9 wins, 33 ties and
  4 losses. On the 22 nonzero-headroom systems it is `+0.5455` with 9/12/1;
  ternary and higher-order strata are positive while binary is zero. Runtime
  is 6.60 versus 2.08 seconds/system. The mean, loss/win, headroom and
  two-stratum development signals pass, but one fold and MC128 cannot establish
  numerical convergence or cross-fold replication.
- The unchanged higher-integration fold-0 runs are:
  MC512 SHA256
  `6ea74f54d7e6e90d92c7be83cf3aadbcf6f959161378a2eff58f0b049e789390`
  and MC1024 SHA256
  `58fc2403c649cd6e32484535b074da7d45e6019d2da8397643045f43681710e2`.
  At MC512 the paired effect is `+0.1522`, bootstrap interval
  `[-0.0217,+0.3478]`, exact sign-flip `p=0.1923`, and 10/31/5
  wins/ties/losses. At MC1024 it is `+0.1739`, interval
  `[-0.0217,+0.3696]`, `p=0.1351`, and 11/30/5. The MC1024 effect on the 22
  nonzero-headroom systems is `+0.5455`, interval `[+0.2273,+0.8636]`, with
  11/10/1; binary, ternary and higher-order means are `+0.0833`, `+0.2105`
  and `+0.2000`. Mean online times are 20.02 versus 2.81 seconds/system for
  rollout and source at MC1024.
- MC512/1024 agree on 41/46 first actions, 31/46 complete traces, 220/276
  individual actions and 45/46 system-level effects. Thus the scientific
  effect direction is not collapsing at the higher integration level, but the
  action sequence has not met a defensible convergence gate. Folds 1--5 are
  stopped pending a numerical-integration-only diagnosis. Do not tune the
  posterior, source continuation, terminal objective or acquisition threshold
  in response.
- The above MC512/1024 artifacts were generated before the simultaneous-gate
  correction, with eight marginal-comparison blocks. They remain useful
  attribution diagnostics, but are not results for the current
  Source-Anchored RQMC Racing (SARR) implementation. A fresh fold-0 run with
  sixteen blocks and Bonferroni-simultaneous bounds is required before any
  updated effect claim.
- Claim status: implemented cross-fitted development method plus mechanism
  diagnostic. No superiority or paper-level positive claim. The first
  higher-integration fold is promising at the effect level but fails exact
  action-level convergence; cross-fold execution is paused.

### E24. Conformal One-Deviation Source-Rollout implementation (2026-07-22)

The report-motivated continuation is implemented as a distinct policy,
`conformal_source_rollout_delta_hull`; it does not replace or relabel SARR.
`source_rollout_system_score` computes the exact-system maximum positive
over-estimation of rollout advantage, and
`fit_conformal_source_rollout_calibration` uses the finite-sample order
statistic over disjoint exact systems. Deployment compares each candidate's
RQMC-adjusted advantage with the frozen radius, while retaining source in the
legal action set. The closed-loop runner carries an explicit
`conformal_deviation_used` bit and, after one accepted non-source action,
returns to source margin for every remaining round.

The implementation has unit and subprocess tests, including finite-quantile
failure, one-deviation enforcement, high-threshold source fallback and exact
reveal parity. It has no calibration artifact, no real-data run and no
positive result yet. The method remains proposal/development infrastructure;
the required first action is still the unchanged sixteen-block SARR fold-0
rerun and numerical opportunity-cost audit. Do not tune the conformal radius
or open new evaluation systems before that gate is resolved.

Before that rerun, the closed-loop audit path was extended without changing a
policy decision: every Source-Rollout round now preserves the pre-reveal
candidate ordering, all sixteen block rollout scores, paired mean advantages,
Bonferroni simultaneous lower bounds, source and selected actions, fallback
reason, comparison count and horizon. This instrumentation is required for the
registered numerical opportunity-cost audit; it does not alter the posterior,
reward, source continuation, selection threshold or oracle boundary.
The external result schema stores these as `policy_decision_rounds`, separately
from evaluator-only post-trace `rounds`, so neither record can overwrite the
other.

### E25. Frozen SARR MC8192 opportunity-cost audit (2026-07-22)

**Authoritative numerical diagnostic; not a performance evaluation.** The
precommitted plan selected 196 development-only pre-reveal states from the
union of SARR deviations, positive-but-Bonferroni-unresolved fallbacks, final
win/loss systems and pre-SARR MC512/MC1024 disagreements. Every state was
replayed once at MC8192 with sixteen scrambled-Sobol blocks, using the original
observable state checksum and selected-action-only reveal history. The output
and its read-only summary verify the frozen plan, task and SARR checksums,
exact 196-state coverage, and `evaluation_systems_accessed=false`.

The accepted sixteen-block SARR deviations are numerically well supported: all
87 have a positive independent high-precision advantage (minimum `0.00513`)
and simultaneous lower bound (minimum `0.00171`); their mean high-precision
selected-action opportunity cost is only `0.000925`, with a zero median. No
selected SARR deviation has negative high-precision advantage. Thus the earlier
action instability is not explained by accepted SARR actions being
posterior-score reversals at MC8192.

The conservative fallback is nevertheless materially active. Of the 74 frozen
states that had a positive point advantage but failed the simultaneous gate at
MC1024, 71 have a nonzero MC8192 source opportunity cost; its mean is
`0.01768`, median `0.01074`, and 38 states are at least `0.01`. This is
evidence that the Bonferroni fallback leaves posterior-model-relative rollout
value on the table. It is **not** evidence that any unselected action has a
positive high-confidence lower bound, nor evidence of an oracle-final gain.

Decision: leave `source_rollout_delta_hull` unchanged and keep folds 1--5
closed. The audit authorizes only **Independent-Confirmation SARR (IC-SARR)**,
a separately named, independently seeded two-stage numerical gate: screen a
candidate with the current SARR estimate, then use an independent
high-precision paired comparison for that single preselected candidate. Its
full frozen protocol is in `docs/IC_SARR_NUMERICAL_GATE_PLAN.md`; calibration,
configuration and all evaluation must use unused development folds. Fold 0
cannot choose a threshold or support a superiority claim. The posterior,
reward, source continuation and target oracle boundary remain frozen.

### E26. IC-SARR implementation gate (2026-07-22)

**Development infrastructure; no effect result.** IC-SARR is implemented as
`independent_confirmation_source_rollout`, separately from SARR. It runs the
unchanged MC1024 sixteen-block simultaneous screen. Only when that screen
falls back to source despite a positive point advantage does it preselect the
single maximum-advantage candidate (pair ID tie break) and estimate that one
candidate-source difference at MC8192 with a disjoint sixteen-block stream.
It deviates only when the one-comparison 95% lower bound is strictly positive.
Accepted SARR deviations are returned unchanged; no-positive screens fall back
to source without stage two.

The implementation has deterministic failure tests for accepted-action
identity, no-positive source fallback, independent seed derivation, and
positive/negative stage-two lower-bound decisions. It records all stage-one
and stage-two quantities before the sole selected-action reveal. No real
system, fold, evaluation system, posterior, terminal reward or source
continuation was changed or accessed. The required next step is still one
unused-development-system implementation preflight, followed only by the
precommitted whole-fold execution if that gate passes.

The deterministic first system of unused fold 1, `Ag-F-Li`, has now passed
that preflight without reading an evaluator result: six selected actions were
the six and only oracle reveals, stage two was invoked on three rounds, all
rounds used the fixed MC1024/MC8192 counts, and no stage-two seed equalled its
stage-one seed. The atomic preflight record has SHA256
`9843b8e72f6b11644f884746f6d76a577363048d9b07a1ac537dab3bf66ff243`, matches
the v6 task and cross-fit manifest checksums, and records
`evaluation_systems_accessed=false`. This authorizes the precommitted fold-1
run unchanged; it adds no outcome comparison and does not revise any method.

### E27. IC-SARR five-fold development replication (2026-07-22)

**Authoritative development evidence; not external confirmation.** The
precommitted IC-SARR policy then ran once on each unused cross-fit fold 1--5,
with 46 exact chemical systems per fold, budget six, identical task/vault/
manifest checksums, fixed MC1024/MC8192 integration, and no evaluation-system
access. No posterior, terminal reward, source continuation, gate, seed or
system selection changed after fold 1. The full artifact hashes and
system-level table are recorded in `docs/IC_SARR_FIVE_FOLD_RESULTS.md`.

Against source margin, terminal oracle-pool confirmations improve by `+0.161`
per system over 230 systems, with deterministic system-bootstrap 95% interval
`[+0.083,+0.239]` and 50/162/18 wins/ties/losses. Every individual fold has a
positive point estimate (`+0.109` to `+0.196`). The independent confirmation
gate is materially exercised: 332 positive-but-unresolved states enter stage
two and 224 pass it. This establishes a reproducible, narrow real-data
development signal for nonmyopic source-anchored terminal confirmation.

The result does **not** establish universal discovery superiority. Final causal
confirmation is only `+0.013` per system overall and its interval crosses zero;
myopic action regret increases by `+0.135` eV/atom; and the reference rollout
is slower by `+22.35` seconds/system on a shared server. The correct next
steps are an action-parity-preserving implementation optimization followed by
a newly reserved disjoint evaluation, not tuning the policy against these 230
opened development systems.

### E28. IC-SARR terminal-metric and backend regression audit (2026-07-22)

**Implementation audit; no new method result.** The runner now records the
causal-time announcement count `D`, selected-history retained confirmation
count `F`, and complete oracle-pool adjudicated confirmation count `T` as
different quantities, checks `T <= F <= D` at evaluation time, and records
`D-F` and `F-T` separately. Reanalysis of the immutable five-fold outputs
gives `(D,F,T)=(4.322,4.083,3.622)` for source margin and
`(4.643,4.096,3.783)` for IC-SARR. This explains the narrow effect: the
terminal gain `+0.161` is associated with `+0.309` within-campaign revocations
and `-0.148` unqueried-competitor invalidations, not a final-causal gain.

The fixed-composition lower-hull backend was also checked in a read-only
one-system action/membership parity audit against pymatgen (budget two,
MC32): no action or sample-membership mismatch occurred. Cached geometry took
2.312 seconds versus 5.197 seconds for pymatgen on that fixture. A separate
one-system IC-SARR runner regression verified the updated evaluator and reveal
boundary only. Both outputs remain outside Git under
`/home/workspace/lrh/DATA/EviMem-RL/outputs/audits/ic-sarr-feedback-fixed-backend-parity-v1/`.
Neither audit changes the posterior, reward, continuation, integration gate,
or the five-fold effect estimate.

### E29. Source-rollout suffix-memoization microbenchmark (2026-07-22)

**Engineering NO-GO; removed.** A candidate action-preserving cache reused a
source-continuation transition when two simulated first-action paths reached
the same posterior-sample/selected-set state. It preserved every synthetic
reward exactly, but on the representative fixed-geometry microbenchmark
(`12` candidates, horizon `6`, `512` posterior samples) took 0.114712 seconds
versus 0.107461 seconds for the existing vectorized grouped evaluator
(`0.9368x`). Python dictionary bookkeeping outweighed the small amount of
state convergence at current pools. The cache was removed rather than retained
as inactive complexity. The composition-geometry cache remains because its
separate pymatgen parity audit gives a positive engineering result.

### E30. Exact binary lower-hull specialization (2026-07-22)

**Authorized engineering optimization; no policy change.** Profiling of the
fixed-composition final-hull backend showed that repeated generic Qhull calls,
not posterior conditioning or source continuation, remain the local cost. For
two-element systems only, the backend now evaluates the same lower hull using
the exact monotone chain of composition fraction versus per-atom energy. It
retains the existing duplicate-composition selection, elemental references,
negative-formation filter and the `1e-14` non-vertex collinearity rule used by
pymatgen's Qhull facet check. Ternary and higher systems still use the prior
Qhull path.

The fast path matches pymatgen on 256 random binary energy vectors, duplicate
composition cases, and an exactly collinear non-vertex case; the full rollout
value parity test also remains exact. On the local 32-candidate, 1,024-sample
binary fixture, the fixed backend falls from 0.4677 seconds with generic Qhull
to 0.0660 seconds with the chain (`7.08x`). This is a local computational
benchmark, not a rerun of the five-fold policy comparison. The remote
MC1024/budget-six Source-Rollout regression on the already opened `Ag-S`
fixture has the identical six immutable action IDs and identical
`(D,F,T)=(2,1,0)` against its frozen output. Its single shared-server wall
time is 4.024 seconds versus 4.972 seconds in the older record; this confirms
action/reveal parity but is not used to claim an IC-SARR end-to-end speedup.

## Superseded, invalid and incomplete evidence

| Evidence | Required treatment |
|---|---|
| CAW/DACC handcrafted synthetic wins | Mechanism history only; never general superiority |
| DACC three-seed smoke | Removed runner; recover from tag only |
| Interrupted K2 frozen grid | Never merge, resume or use for inference |
| P3C reference/path v2/v3 | Failed, no valid summary |
| P3C reference/path v4 | Trace-equivalent to v5 but timing incomplete |
| Any relaxed/`opt` SOAP WBM run | Invalid for closed-loop policy claims due post-DFT geometry leakage |
| Any modern-recomputed oracle-energy run | Invalid where it disagrees with frozen parity energy |
| Legacy ceiling Brier/log-loss | Approximate; explicit-threshold rerun is cleanup only |
| AKSC formulas or microbenchmarks | Proposal/diagnostic only; no WBM method claim |
| JARVIS--MP v1 global affine evaluation | Valid negative result; never reuse its opened evaluation systems for transport development |
| JARVIS--MP v3 fresh evaluation partition | Unopened; remain sealed because the calibration certificate failed |
| JARVIS--MP v4 evaluation | Opened exactly once after calibration passed; development-closed and citable only as the registered NO-GO |

## Recovery points and live-code boundary

| State | Git reference |
|---|---|
| Corrected CAW-Joint NO-GO | `caw-method-no-go-2026-07-15` / `0fa1e1f` |
| Working-set economics freeze | `wbm-preregistration-freeze-2026-07-16` / `ca80720` |
| Secure WBM P0 | `e313499` |
| Composition-dependent causal hull | `1b37686` |
| DACC implementation | `0fb29eb` |
| First engineering WBM result | `76b8612` |
| Complete pre-CHIC v4 NO-GO tree | `decision-state-v4-no-go-2026-07-20` / `c905395` |

## Live documents and retired annexes

The live tree keeps this ledger, the decision-sufficient-state definition, the
WBM data/license audit and the reusable robust-hull certificate note. The
duplicated CAW/DACC/P3C/AKSC/WBM-grid and JARVIS v1/v3 preregistration annexes,
their configs and their executable runners were retired on 2026-07-21. Their
last complete state is tag `decision-state-v4-no-go-2026-07-20`; E0--E21 above
retain the claim-relevant counts, hashes and dispositions. Do not copy those
executors back into `src/matmem`.

## Rules before any new method or experiment

1. Read this ledger and the three authoritative diagnoses linked from README.
2. State which failed assumption the new idea changes; a new score on the same
   outcome-selected size-`K` GP is not a new scientific direction.
3. Use frozen parity energies, initial `org` structures, exact chemical systems,
   oracle isolation and composition-dependent causal hulls.
4. Keep every outcome in the immutable archive and say explicitly whether it
   contributes to the deployed posterior.
5. Measure a real bottleneck before proposing compression. A fixed-probe speedup
   cannot substitute for end-to-end Amdahl relevance.
6. Exploration may iterate on calibration/development systems. Before any
   confirmatory evaluation is opened, freeze the method, systems, estimands and
   stopping rule and use a fresh split.
7. Preserve negative and interrupted artifacts; never overwrite, combine or
   silently relabel them.
8. A conformal absolute-error radius is not a Gaussian standard deviation.
   Any Gaussian working likelihood using that number is a modeling heuristic,
   not a distribution-free calibration theorem.

## Confirmatory infrastructure status (2026-07-21)

The current code adds fail-closed wiring for the frozen Delta--Hull transport,
optimizer provenance fields, and an optional fixed-composition lower-hull
backend. The latter is an implementation optimization, not a new method, and
is not authorized until the independent parity audit has zero mismatches.
Fresh split construction and transport freezing are separate tools. The fresh
builder excludes every development system and pair and performs
outcome-independent SHA-256 selection by composition stratum. The confirmatory
runner requires a frozen transport artifact and checks fit/query disjointness;
it never refits on fresh systems. These tools have only passed fixture and unit
tests so far. No fresh-system result, GO decision, or paper-level positive claim
is implied by their existence.

The remote raw MatPES rebuild yields 324 eligible exact systems (88 binary,
130 ternary, 100 quaternary and 6 higher-order). The first attempt correctly
failed because the old transport fit consumed all 324. E22 resolves that
engineering problem by reserving 48 systems first and refitting on 276. The
opened result is negative against source margin, so data-split feasibility is
no longer the blocker; decision headroom and discriminative posterior value
are the measured limitations.

## E30 -- Dual-Horizon SARR implementation (2026-07-22)

`constrained_dual_horizon_source_rollout` is a separate development policy.
It changes the failed single-horizon assumption by requiring a source-relative
terminal advantage and a non-negative selected-history advantage under
the same rollout samples. The source-margin action remains the legal fallback;
no reward mixing, outcome-selected posterior, or IC-SARR parameter change was
introduced. The causal reward is computed from reference phases plus exactly
the simulated selected outcomes, while terminal reward uses the complete
visible candidate pool. Unit tests cover both-gate fallback, deterministic
selection, selected-history isolation, transport wiring and reveal-boundary
parity. The fold-0 pilot is development evidence only because fold 0 was used
for earlier Source-Rollout development; it cannot support external
confirmation.

The first fold-0 pilot used eight exact systems, budget 6, 1024 posterior
samples and the fixed-composition backend. Dual-Horizon selected 12 of 48
actions away from source (SARR: 15/48) and fell back on the dual gate in 36/48
rounds. Its system-macro final-causal confirmations were 3.000 versus 2.875
for source margin; terminal oracle-pool confirmations were 2.750 for both
methods. Mean action regret was 0.1109 eV/atom for Dual-Horizon versus 0.0163
for source margin, and wall time was 40.63 seconds/system versus 1.89. These
are development diagnostics, not a positive result. A four-system rerun after
batching selected-set causal evaluation reproduced every Dual-Horizon action
ID exactly and reduced its wall time to 31.18 seconds/system (SARR 7.37), so
the optimization is numerical only. The signal is insufficient to authorize a
fresh-systems or paper-level run. Any future dual-horizon research must use a
campaign-level constrained rollout over complete policies, as specified in
`docs/CAMPAIGN_LEVEL_CONSTRAINED_ROLLOUT.md`, rather than tune the local
selected-history lower-bound gate on opened systems.

## E31 -- Holdout availability audit (2026-07-22)

The server-side task audit found no untouched MatPES exact-system holdout in the
currently provisioned artifacts. The v6 CHGNet development task and the
all-eligible task contain the same 324 systems and 10,236 pairs. The 230
systems used for IC-SARR folds 1--5, fold 0 used for Source-Rollout
development, and the 48-system historical confirmatory task together cover
those 324 systems; the 48-system task also overlaps the v6 task exactly. Thus
the existing 48-system output cannot be relabeled as a fresh IC-SARR holdout.
No retraining or IC-SARR execution was launched from this audit. A valid
primary test requires a new upstream MatPES pool/release (or an independently
constructed protocol dataset) with a frozen source-vs-IC-SARR split.

There is a separate, technically gated JARVIS--MP v4-natural pool on the
server (2,056 calibration pairs and 717 evaluation pairs across binary,
ternary and higher-order systems). It is not a MatPES replacement: its source
is JARVIS OptB88vdW and its target is MP PBE, with a different task manifest and
protocol semantics. It can support a new multi-fidelity/protocol-aware study
after an adapter and fresh method freeze, but it cannot be used to claim an
IC-SARR MatPES holdout result. LeMat files are not currently a matched paired
protocol task and are likewise not a drop-in holdout.

## E32 -- Canonical cleanup and rebuild smoke (2026-07-22)

Superseded MatPES task snapshots, redundant model/vault files and completed
exploratory/engineering outputs were moved (not deleted) to
`DATA/EviMem-RL/archive/superseded-20260722/`. The active canonical manifest is
`matpes-canonical-development-v1.json`; it contains the v6 frozen-CHGNet task,
v5 oracle vault and fold-0 transport checksum. A two-system budget-four smoke
was rebuilt from these canonical paths with source-margin and IC-SARR. IC-SARR
had oracle-pool confirmations `1.5` versus `1.0` for source, equal final-causal
confirmations (`2.0` each), mean action regret `0.0136` versus `0.0053` eV/atom,
and wall time `6.06` versus `3.38` seconds/system. This is only a rebuild and
implementation smoke; the two systems are development systems and no method
or holdout conclusion is drawn.

## E33 -- Dual-Horizon oracle/posterior attribution (2026-07-22)

The offline evaluator `tools/run_dual_horizon_attribution.py` was run on the
same eight fold-0 development systems and 48 decision states at MC128, MC512
and MC1024. It enumerates every legal first action and compares exact oracle
terminal/selected-history advantages with the frozen posterior point estimates
and simultaneous lower-bound gate. The oracle vault is accessed only by this
offline diagnostic; no policy subprocess or holdout was opened.

Across all runs, 12 of 48 states contain an oracle action with
`Delta_T*>0` and `Delta_F*>=0` (existence rate 0.25), so T/F conflict is not a
universal explanation. At MC512 and MC1024, posterior recall of those actions
is 0.417, terminal sign accuracy is about 0.418 and selected-history sign
accuracy about 0.300. Point-feasible action rejection by the dual numerical
gate falls from 0.938 (MC128) to 0.726 (MC512) and 0.628 (MC1024), but remains
large. Nominal terminal interval coverage is 0.593, 0.325 and 0.277 at
MC128/512/1024; this is a numerical diagnostic only, not a posterior
calibration guarantee. Posterior point-action oracle regret falls from 0.3125
to 0.2500 to 0.1875 terminal confirmation/reward units (not eV/atom).

The feasible oracle actions occur only in ternary and quaternary-or-higher
states of this tranche (six each; none in binary states). The evidence supports
local gate conservatism and joint rollout advantage misspecification as leading
causes, with structural T/F conflict present but not universal. It does not
justify changing the backbone, tuning the gate on these systems, or claiming a
positive method result. A future dual-horizon method must be a new
campaign-level constrained rollout evaluated on a new development split.

## E34 -- Dual-Horizon attribution closure (2026-07-22)

The MC128/512/1024 convergence tranche closes the current local Dual-Horizon
diagnostic. The posterior point ranking still contains signal, so a backbone
replacement is not supported. The failure is attributed to model-relative
joint advantage/covariance mismatch amplified by the per-state dual lower-bound
gate; T/F structural conflict is present but not universal. The project will
not increase MC, tune `LB_F` or other thresholds, add chemistry heuristics, or
replace the gate with the point selector on these opened systems. Dual-Horizon
remains a correct but mechanism-failed development diagnostic. IC-SARR and
exact action-parity hull backends remain frozen. Any continuation of this idea
must be independently named campaign-level constrained rollout work on new
development systems.

## E35 -- Campaign-Gated IC-SARR implementation (2026-07-22)

The first restricted campaign-level construction is implemented in
`matmem.campaign_gated_ic_sarr`. It compares exactly two complete adaptive
policies, source-margin and the frozen IC-SARR, under paired outer posterior
energy worlds. Each simulated policy conditions on its own previously revealed
world outcomes; the same world is used for both policies, while IC-SARR's
inner RQMC stream remains seed-separated from the outer world stream. A single
system-level Bonferroni gate selects IC-SARR only when the campaign terminal
lower bound is positive and the selected-history lower bound is non-negative.

This is a posterior-relative development API, not an oracle evaluator or a
production policy subprocess. It has deterministic fixture coverage, but has
not been run on a new development tranche. The full Pareto/set-valued Bellman
formulation remains proposal-only. No Dual-Horizon threshold, backbone,
chemistry rule, or opened-system result was changed.

## E36 -- Campaign-Gated IC-SARR real-data smoke (2026-07-22)

A two-system MatPES development smoke was started from the canonical v6 task
using outer samples 16/8 and IC-SARR inner samples 32/64 respectively. The
implementation was computationally much more expensive than a normal
single-policy rollout because every outer world simulates two complete adaptive
campaigns and each IC decision nests its own RQMC rollout. The run was stopped
by the user before the first complete JSON output was written. It is therefore
`Incomplete/interrupted`, has no metrics or policy conclusion, and must not be
resumed or partially interpreted as evidence. A smaller rerun may only be
started explicitly as a new development smoke with a new output identity.
