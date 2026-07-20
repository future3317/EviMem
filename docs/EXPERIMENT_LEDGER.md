# Authoritative experiment and decision ledger

**Status: paper-level NO-GO after completed v4 evaluation (2026-07-20).** This is the first file a future
maintainer or coding agent should read before changing the method or launching
another experiment. It records why the research moved from CAW-Joint to DACC,
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

The other files in `docs/` are frozen technical annexes. They may contain
longer formulas, tables, implementation notes or preregistration details, but
they do not override this ledger. Before deleting or shortening an annex, all
unique experiment identity, checksum, correction, failure attribution,
validity label and stopping decision must first be copied here. Git tags remain
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
protocol-aware reuse remains NO-GO under the tested certificates.

## Current research decision (not an experiment)

The stopping chain does not authorize a fourth posterior approximation. The
live problem is now **Decision-Sufficient Scientific State**, specified in
`docs/DECISION_SUFFICIENT_SCIENTIFIC_STATE.md`: preserve the least costly
observable, protocol-valid state that keeps registered scientific decisions
within a frozen distortion tolerance of the complete archive. WBM is retained
as the homogeneous, low-compute-cost null regime, where the correct behavior is
full history. A positive next experiment requires independently measured
protocol incompatibility, shift, access/compute cost, or another explicit
constraint. The all-outcome predictive layer, fail-closed activation path and
robust hull-decision certificate now exist and pass exact replay, null,
no-deletion, self-removal and interval-soundness tests. The v4 evaluation
nevertheless fails the frozen simultaneous-coverage gate and is dominated by
simpler source reuse. Thus the certificate is a validated selective mechanism,
not a positive state-compression result.

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

This table is the compact judgment layer over experiments E0--E15. Detailed
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

## Preserved technical-annex source map

These annexes remain in Git for exact technical detail. Their status and
allowed use are governed by this ledger.

| Annex | Preserved content | Authority boundary |
|---|---|---|
| `RESEARCH_ITERATION_HISTORY.md` | Narrative CAW--DACC--P3C--AKSC chronology and retired-code recovery guidance | Historical annex; this ledger controls current status |
| `WBM_CALIBRATION_CORESET_AMENDMENT.md` | P3C equations, neighborhood search, proper-divergence variants and P1 follow-up | Stopped-method specification, not an active method |
| `WBM_ENGINEERING_P1_P15_AND_PILOT.md` | Infrastructure gates, DACC/joint-risk/P3C engineering tables and frozen calibration details | Mixed historical/diagnostic evidence; validity labels here prevail |
| `P3C_FAILURE_ATTRIBUTION_2026-07-20.md` | Code/data/method/theory decomposition and 32-system corrected replication | Technical support for E8--E11 |
| `AKSC_CEILING_DIAGNOSTIC_2026-07-20.md` | Effective dimension, reference dispersion and proposed all-outcome architecture | Diagnostic/proposal annex; B40 later stops WBM AKSC |
| `WBM_LONG_ARCHIVE_COMPUTE_GATE_2026-07-20.md` | Checkpointed B40 timing and Amdahl decision | Technical support for E12 |
| `WBM_DATA_LICENSE_AND_INFRASTRUCTURE_AUDIT.md` | WBM/MP/CHGNet license, artifact, parity and oracle-boundary provenance | Infrastructure evidence only |
| `JARVIS_MP_MULTIFIDELITY_PREREGISTRATION.md` | Real pair construction, v1 result, v3 calibration and immutable hashes | Technical support for E13--E14 |
| `ENVIRONMENT_HULL_CERTIFICATE_2026-07-20.md` | Natural v4 method, theorem, fresh split, calibration/evaluation hashes and validation report | Technical support for E15 |
| `DECISION_SUFFICIENT_SCIENTIFIC_STATE.md` | Current formal problem, alignment gates and paper-facing method boundary | Live specification; not a positive result |

If an annex is later removed from the active branch, its last commit/tag and
all unique audit facts must first be recorded in this table and in the relevant
numbered evidence entry. Manuscript page pressure is never a reason to destroy
the research audit trail.

The live package keeps the secure WBM path, P3C/legacy selectors as audited
diagnostics, fixed-GP components and failure-capable tests. Removed historical
runners are intentionally absent. Do not add compatibility adapters or copy old
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
6. Freeze calibration systems, parameters, estimands and stopping rules before
   evaluation outcomes are opened.
7. Preserve negative and interrupted artifacts; never overwrite, combine or
   silently relabel them.
8. A conformal absolute-error radius is not a Gaussian standard deviation.
   Any Gaussian working likelihood using that number is a modeling heuristic,
   not a distribution-free calibration theorem.
