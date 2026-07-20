# JARVIS--MP protocol-heterogeneity pilot

**Frozen result status (2026-07-20): v1 evaluation NO-GO; v3 composition
certificate NO-GO with fresh evaluation unopened.** This is Regime C of the
Decision--Inference--Systems program. WBM remains the homogeneous null control;
it is not the positive benchmark. The immutable designs are
`configs/jarvis_mp_protocol_activation_v1.json` and
`configs/jarvis_mp_certified_state_v2.json`.

## Scientific question

Can an online state reuse low-fidelity outcomes under an independently
calibrated directed protocol certificate while retaining every accepted
outcome contribution, rejecting unsupported transport, and preserving target
convex-hull decisions better than target-only history and naive pooling?

This pilot does **not** ask which outcome cards should be forgotten. The
scientific archive is append-only. For a frozen outcome-independent basis
`phi`, every direct or certified transported residual updates

\[
\Lambda_t=\Lambda_{t-1}+\phi_t\phi_t^\top/\sigma_t^2,
\qquad
\eta_t=\eta_{t-1}+\phi_t r_t/\sigma_t^2.
\]

The numerical state is fixed by basis rank rather than archive length.
Protocol activation is only a legal-influence gate: compatible outcomes all
enter the state; incompatible outcomes remain archived but cannot affect the
target state.

## Frozen real task

- Source: JARVIS-DFT 3D, `OptB88vdW`, official 2022-12-12 archive.
- Target: the pure-`GGA` subset of the frozen 2023-02-07 MP CSE with MP2020
  corrections. `GGA+U` targets are excluded rather than collapsed into GGA.
- Candidate join: JARVIS `reference=mp-id` is necessary but insufficient. Each
  retained pair also passes `StructureMatcher(ltol=0.2, stol=0.3,
  angle_tol=5, primitive_cell=True, scale=True)`.
- Split unit: exact chemical system. Calibration and evaluation systems have
  zero overlap and are chosen by release-bound SHA-256 before outcomes.
- Policy features: only the JARVIS low-fidelity relaxed structure. The MP
  target-relaxed structure is used only to certify the database pair and never
  becomes a prediction feature.
- Target outcomes are held in an external oracle vault. Evaluation candidates
  are removed from their initial MP phase diagrams and enter only after their
  deterministic reveal.

The external frozen build contains 1,658 structure-matched pairs. Ten
calibration systems contribute 212 selected pairs; ten disjoint evaluation
systems contribute 203. Evaluation covers four binary, four ternary, and two
quaternary-or-higher systems. The task manifest SHA-256 is
`d98e87545198c47d318dab67802a95dc87049ff2ab159ad296916ef92359b281`.

## Calibration isolation

The ten calibration systems are further divided before evaluation:

1. four systems fit a frozen ridge base predictor;
2. three disjoint systems fit the directed affine residual transport;
3. three further systems calibrate its 90% absolute-error radius.

The observable feature standardizer and rank-16 PCA may use all task
structures because they use no outcomes. The transport certificate is accepted
only when its frozen radius is finite and at most 0.15 eV/atom. Failure rejects
all source-protocol influence; it does not trigger threshold tuning.

## Comparisons and estimands

All strategies use the same deterministic target reveal sequence:

1. target-only history;
2. explicit protocol rejection (an exact parity control for target-only);
3. naive full-history pooling that falsely relabels JARVIS residuals as MP;
4. same-item affine multi-fidelity transport;
5. persistent certified all-outcome sufficient state;
6. same-order certified full-history replay.

The persistent/replay pair estimates representation economics and must be
behaviorally exact. The comparison with naive pooling estimates protocol
certification, not a generic memory advantage. A diagnostic that removes the
same-item low-fidelity observation will determine whether any gain extends
beyond ordinary paired delta learning.

Primary outcomes are system-macro hull-decision cost, one-step discovery
regret, epsilon-optimal action coverage, certificate violations and exact-null
parity. CRPS, Brier, log loss, RMSE and NLL are diagnostics only. Runtime and
state size are reported, but this small pilot cannot establish a systems
bottleneck or speedup claim.

## Interpretation gate

A pilot GO only authorizes a larger protocol-heterogeneity study. It requires
all hard parity/isolation/no-deletion gates, lower system-macro hull decision
cost and one-step regret than both target-only and naive pooling, evaluation
certificate violation at most 15%, and signal beyond one complexity stratum.
If paired affine transport explains the full gain, the result validates a
standard multi-fidelity mechanism but not a novel decision-sufficient state.
If the certificate fails on calibration, or certified reuse worsens decisions,
this protocol pair is a NO-GO and evaluation must not be used to refit it.

## Frozen v1 result: implementation gates pass, scientific gate fails

The v1 external task and result live under
`E:\DATA\EviMem-RL\multifidelity\jarvis-mp-v1`; raw data and outputs are not
tracked by Git. The task manifest SHA-256 is
`d98e87545198c47d318dab67802a95dc87049ff2ab159ad296916ef92359b281`,
the calibration freeze SHA-256 is
`3b6c525eb0458f75ce7a17f36d50bd2fb379332d8b2c73a8e68f6cd0cc44ce4a`,
and the pilot result SHA-256 is
`02fddb48f285eb0bd1102479c64ac028b8b33dc0a4c89cd72a599e6e6a904da6`.

The real join produced 1,658 structure-matched JARVIS--MP pairs. V1 used ten
calibration exact systems (212 pairs) and ten disjoint evaluation systems (203
pairs): four binary, four ternary and two quaternary-or-higher. The global
affine transport had slope `0.970173699`, intercept `-0.189481293 eV/atom`,
radius `0.099940433 eV/atom`, and calibration coverage `92.1875%`. All four
implementation gates passed: target-only/rejection parity,
persistent/full-history replay parity, homogeneous-null parity, and exact
all-outcome counts.

The certificate did not transfer to held-out exact systems: the evaluation
violation rate was `45.3202%`. System-macro results were:

| Method | Hull decision cost | One-step regret | CRPS | RMSE |
|---|---:|---:|---:|---:|
| target only | 0.97385 | 0.04617 | 0.50072 | 0.79519 |
| naive pooling | 0.32366 | 0.04179 | 0.23176 | 0.30564 |
| paired affine transport | 0.81615 | 0.00096 | 0.09573 | 0.13811 |
| certified rank-16 state | 1.03329 | 0.04246 | 0.24751 | 0.34866 |

The strong paired row establishes useful same-candidate low-fidelity signal,
but it does not validate the compressed state. Failure attribution found that
`88.459%` of source--target formation-energy-shift variance was between exact
chemical systems, structure-match RMS correlated only `-0.224` with absolute
transport error, and rank-16 projection RMSE to transported residuals was
`0.41698 eV/atom`. Thus v1 is a scientific NO-GO caused by cross-system
transport failure plus representation loss, not by oracle leakage, null
degeneracy, outcome deletion, or a failed structure join. The immutable
attribution SHA-256 is
`5461a223a18110ba0ded4f99d51bcfdae53905a87544886ce4d42e856a658d4e`.

## Frozen v3 calibration result: composition offsets are insufficient

V3 is external at `E:\DATA\EviMem-RL\multifidelity\jarvis-mp-v3`. It contains
33 calibration systems (677 pairs) and 12 fresh evaluation systems (261 pairs),
with no overlap with v1 evaluation systems. Twenty-three systems fit

\[
E_{\mathrm{target}}=aE_{\mathrm{source}}+\sum_e x_e b_e,
\]

and ten disjoint systems calibrated an exact-system-clustered conformal radius
using the maximum absolute error within each system. The task SHA-256 is
`4765bc82de2237ae6156b7a5642c27f62a3565e940876c13e250f8d8f0dd6872`.
The fitted slope was `0.995984284`, but the frozen radius was
`0.177263735 eV/atom`, above the preregistered `0.15 eV/atom` ceiling. The
freeze therefore records `certificate_passed=false` and
`evaluation_results_accessed=false`; its SHA-256 is
`571fff2aa7cbd31a8d84c9034822bc74ae407f4d8a262349bac122c77f5722d7`.

This is a calibration-stage NO-GO, not an evaluation result. The threshold is
not relaxed to 0.18 after observing the failure. Elemental reference offsets
remove part of the discrepancy but do not certify the remaining
structure/environment-dependent protocol error. The 12 fresh evaluation
systems remain unopened and cannot be used to choose a richer transport.

## Consequence for the next method

The implemented all-outcome linear--Gaussian state proves the engineering
principle that every legal outcome can update fixed-dimensional natural
parameters, and the persistent/replay gate verifies the exact implementation.
It does **not** yet implement a Certified Hull-Decision State theorem or a
distribution-free use of the conformal radius. In particular, treating a
transport radius as a Gaussian working-likelihood scale is an experimental
modeling choice, not a calibration guarantee. No paper claim is authorized
until a new, independently calibrated protocol model passes before evaluation
and its hull decisions, abstentions, violations, and end-to-end costs are
measured on fresh systems.
