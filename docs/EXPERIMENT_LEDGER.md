# Authoritative experiment and decision ledger

**Status: paper-level NO-GO (2026-07-20).** This is the first file a future
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
constraint. The all-outcome fixed-rank predictive layer and fail-closed
activation path now exist and pass exact replay/null/no-deletion tests.
Certified Hull-Decision State remains incomplete: there is no hull-certificate
algorithm, no decision-preservation theorem, and no positive result in this
ledger.

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

## Recovery points and live-code boundary

| State | Git reference |
|---|---|
| Corrected CAW-Joint NO-GO | `caw-method-no-go-2026-07-15` / `0fa1e1f` |
| Working-set economics freeze | `wbm-preregistration-freeze-2026-07-16` / `ca80720` |
| Secure WBM P0 | `e313499` |
| Composition-dependent causal hull | `1b37686` |
| DACC implementation | `0fb29eb` |
| First engineering WBM result | `76b8612` |

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
