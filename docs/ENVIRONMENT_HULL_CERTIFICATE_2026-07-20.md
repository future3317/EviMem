# Environment-conditional transport and robust hull certificate

**Status: authoritative fresh-system NO-GO (2026-07-20).** The implementation,
calibration gate and one-time evaluation are complete. The result validates a
sound decision-certificate mechanism and all-outcome state, but it does not
support method superiority. No parameter may be changed using the opened
evaluation systems.

## What changed relative to v1/v3

The failed global affine and composition-offset assumptions were replaced by a
single natural pipeline:

1. the policy-visible JARVIS source structure is encoded by the frozen
   `CHGNet-0.3.0` crystal representation;
2. a same-candidate ridge delta map predicts
   \(E_T(x)=E_S(x)+g(h_S(x),E_S(x))\), with ridge strength selected by
   leave-one-exact-system-out calibration;
3. every revealed target outcome updates fixed-dimensional Bayesian-ridge
   natural parameters for the remaining target correction; and
4. exact-system clustered conformal calibration produces simultaneous energy
   intervals that are consumed directly by a robust convex-hull program.

There is no capacity, similarity, eviction, manually selected local descriptor,
global energy ceiling or OOD multiplier. An element absent from transport-fit
systems is unsupported by construction and takes the target-only/abstain path.
For supported inputs, predictive leverage widens the interval continuously;
the hull certificate, rather than an arbitrary OOD threshold, decides whether
the uncertainty permits action.

## Robust decision theorem and implementation

For candidate \(x\), every competing phase \(j\) has a simultaneous interval
\(E_j\in[L_j,U_j]\). The candidate is removed from its own competing set. At
composition \(c_x\), define

\[
H_L^{-x}(c_x)=\min_{\lambda\in\Delta^{-x}(c_x)}
\sum_j\lambda_jL_j,
\qquad
H_U^{-x}(c_x)=\min_{\lambda\in\Delta^{-x}(c_x)}
\sum_j\lambda_jU_j.
\]

The implemented decision is

\[
U_x\le H_L^{-x}(c_x)+\tau\Rightarrow\text{stable},
\qquad
L_x>H_U^{-x}(c_x)+\tau\Rightarrow\text{unstable},
\]

and abstain otherwise. If the simultaneous interval event holds, both decisions
are sound: the first bounds the candidate's worst energy against the
competitors' best feasible hull, and the second bounds its best energy against
their worst feasible hull. `src/matmem/hull_certificate.py` solves both hulls
with `scipy.optimize.linprog`; infeasible composition support abstains. Tests
cover all interval endpoints, self-removal, infeasible mixtures and certified
zero-optimal action sets.

The conformal radius is never used as a Gaussian standard deviation. For the
combined transport-plus-online-correction point predictor, the calibration
score is the maximum across every pre-reveal round and every remaining
supported candidate in one exact system of

\[
s_g=\max_{t,x\in U_t}
\frac{|E_T(x)-\widehat E_{T,t}(x)|}
{\sqrt{\ell_{\rm transport}(x)^2+\ell_{\rm correction,t}(x)^2}}.
\]

The split-conformal order statistic is taken over exact systems, not pairs.

## Fresh task and isolation

The natural v4 task is external at
`E:\DATA\EviMem-RL\multifidelity\jarvis-mp-v4-natural`.

- Task manifest SHA-256:
  `ba5bfc139364ecb1b97248e8f72fd646e12ffaa9277a6494f4ebbb265eee5cff`.
- Calibration-only vault SHA-256:
  `288483a5e035a081bc988b56b2d29bd9fd911e019e26023f7a5bf9747a5e6f83`.
- Sealed evaluation vault SHA-256 before opening:
  `cfabf766990b1acde0b16b98c9bc3c39d73aba40f967be8c4bd6572471f338b6`.
- CHGNet checkpoint SHA-256:
  `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`.

All 45 exact systems previously used by v1 or v3 were excluded. Every remaining
system with at least eight structure-matched pairs was used, with an
outcome-independent hash-fold split. The task contains 210 calibration systems
and 2,056 calibration pairs, plus 72 evaluation systems and 717 evaluation
pairs. Calibration and evaluation systems have zero overlap. Policy-visible
rows contain no `target_*` field, the target structure is absent from the
representation path, and calibration/evaluation outcomes are in physically
separate vault files.

The 210 calibration systems were hash-partitioned into 81 transport-fit, 65
transport-radius and 64 decision-trajectory systems. Ridge selection, transport
intervals and final decision intervals used only these partitions. The
evaluation runner was unavailable to calibration and the calibration command
had no evaluation-vault argument.

## Calibration-only gate

The final immutable calibration freeze is
`environment-hull-calibration-freeze-v3.json`, SHA-256
`954acb310a156299c25aaa4b5415a3d6cf0e4118286f73c8d6c03e0a2380279c`.
It is bound to runner SHA-256
`9c94448aa5d2b650ebf0fa3e24ac4c3689ec856d792db009157cd768b46b7447`.

- 52 of 64 decision-calibration systems had transport element support; 12 took
  the registered target-only fallback.
- The 90% system-clustered final normalized quantile was `0.151757347`.
- The 48 conformal-inlier systems had zero certified decision errors.
- System-macro certified coverage was `17.3216%`; its deterministic bootstrap
  95% lower bound was `12.4201%`.
- Same-order streaming/replay and all-target-outcome counts were exact.

The calibration gate therefore passed without changing v3's historical 0.15
eV/atom stopping rule. That old threshold was simply not reused as the new
estimand.

## One-time fresh evaluation

The immutable result is `environment-hull-evaluation-v1.json`, SHA-256
`644fd0b547284034314c3c105a9b040363c1bd4b1ec3dc54df1ef191e4242f94`.
All 72 registered systems and 717 pairs were evaluated; 344 target outcomes
were revealed under the registered half-pool budget. Candidate counts, reveal
counts and all-outcome state counts reconcile exactly, and replay passed in
every system.

| Method | Hull misclassification | One-step regret (eV/atom) | MAE (eV/atom) |
|---|---:|---:|---:|
| Target-only CHGNet ridge + online correction | 0.49158 | 0.06961 | 0.47716 |
| Naive source-as-target | **0.23114** | 0.01203 | 0.13212 |
| Global paired delta | 0.29360 | **0.01155** | **0.10228** |
| Environment transport only | 0.32266 | 0.02365 | 0.19784 |
| Environment transport + all-outcome correction | 0.30524 | 0.02403 | 0.15527 |

Relative to target-only, the proposed state improves hull error by `-0.18634`
(95% system-bootstrap interval `[-0.24245,-0.13203]`) and regret by
`-0.04558` (`[-0.06563,-0.02885]`). The effect appears in binary, ternary and
quaternary-or-higher strata.

Relative to naive source-as-target, however, it is significantly worse:
hull-error difference `+0.07410` (`[+0.02387,+0.12954]`) and regret difference
`+0.01200` (`[+0.00422,+0.02031]`). It is also worse than global paired delta
in MAE and regret. Online all-outcome correction does improve its own frozen
environment base in MAE (`-0.04258`, interval
`[-0.06765,-0.02209]`) and hull error (`-0.01742`,
`[-0.03128,-0.00336]`), but not in one-step regret.

Fifty of 56 transport-supported evaluation systems satisfied the simultaneous
interval event: `89.2857%`, below the frozen 90% gate by one system. The robust
certificate covered 377 of 2,360 supported candidate-round decisions
(`15.97%`) and made five errors overall; all five were outside the simultaneous
interval event, so inlier certified error remained zero. The system-bootstrap
lower 95% bound for certified coverage was `8.0831%`. The possible zero-optimal
action set contained the oracle-best action in every supported round, but this
conservative set result does not rescue point-policy regret.

## Evidence validation and final judgment

This result is **ready to share as a negative result**, not as a positive method
claim. The exact-system grain, joins, counts, denominators, replay, interval
event and paired comparisons were independently recomputed from the immutable
result. The main caveat is inferential: the 90% conformal statement relies on
exchangeability of new exact systems after the registered element-support
filter; it is not candidate-level IID coverage.

The failure is primarily method/generalization, not code degeneracy or missing
data. The environment representation and all-outcome update are active and
improve the environment-only base, while the robust certificate is sound on
its interval event. But the learned environment correction does not outperform
the much simpler source signal. The registered positive hypothesis is therefore
closed. Do not tune CHGNet layers, ridge grids, conformal level, element support
or hull tolerance on these opened systems. A future continuation would require
a scientifically different source of protocol metadata or paired supervision,
not another score or feature tweak on this task.
