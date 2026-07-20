# WBM engineering P1/P1.5 resolution and calibration pilot

The authoritative cross-iteration artifact inventory and current stopping
decision are in `docs/EXPERIMENT_LEDGER.md`. Results below retain their original
engineering role and must not be promoted after the later provenance and
structure-correct replication audits.

**Status (2026-07-19).** P3C has completed its authorized eight-system,
single-cell matched-action diagnostic. Execution gates pass, but statistical
evidence does not support superiority; paper-level status remains NO-GO and
claim-grade comparison and MADE remain unauthorized.

**Exploratory follow-up (2026-07-20).** A 16-system, 334-candidate matched-action
diagnosis found that the original runner recomputed oracle energies with modern
pymatgen instead of consuming the frozen parity corrected energies. Fourteen
Au--Te candidates changed by 0.105--0.241 eV/atom and three labels flipped. The
runner now reconstructs phase entries from the parity manifest and hard-fails
on missing coverage. The corrected `B=8,K=2` result shows weak P3C-Log Brier and
log-loss improvements in 12/16 systems, but both confidence intervals cross
zero; CRPS and RMSE do not improve, and wall time is about 10.5% higher than GP
variance. This does not reopen the paper-level GO. Full evidence and the
code/data/method/theory attribution are in
`P3C_FAILURE_ATTRIBUTION_2026-07-20.md`.
This paragraph records the energy-fix stage chronologically; its numerical
result is superseded by the initial-structure correction below.

**Initial-structure correction and independent replication (2026-07-20).** The
policy SOAP cache was subsequently found to use DFT-relaxed CSE structures.
The official compiler maps `org` to the initial structure and `opt` to the
relaxed final structure. The manifest, SOAP cache, parity identity, and runner
gate now require `org`; failure tests prohibit fallback to `opt`. Repeating the
334-candidate panel and a disjoint 349-candidate next-16 panel removes the old
weak signal. Across the descriptive 32-system aggregate, P3C-Log minus GP
variance is `+0.001461` CRPS, `-0.000003` Brier, `+0.005611` log loss,
`+0.001310` eV/atom RMSE, and `+0.146537` Gaussian NLL. P3C remains slower.
The implementation gate passes, but the main method hypothesis remains NO-GO.

## Why the former blockers were misleading

The original P1 wording required an independently published candidate-level
MP2020-corrected WBM truth table. The available explicit-ID `wbm-summary.txt`
contains raw/legacy fields, while the official compilation procedure applies
MP2020 corrections. This is a semantic mismatch, not a failed download. The
engineering study is therefore scoped as a **fixed historical-pipeline WBM
replay**, never exact official-energy reproduction.

The original 12 x 256 exact-system pool design is also impossible on WBM: the
largest cleaned exact chemical system has 46 candidates. The immutable
oracle-blind engineering amendment contains eight binary/ternary exact systems
with 16 candidates each. P1.5 must judge those pools without replacing a
zero-positive pool.

Finally, license review had conflated local research with redistribution. The
official registry identifies WBM and the frozen MP snapshot as CC-BY-4.0; all
source artifacts remain outside Git. The local non-commercial research gate is
approved conservatively, while redistribution remains false until final human
author sign-off. This is an execution decision, not legal advice.

## Engineering P1 result

`tools/audit_wbm_p1_p15.py` validates the merged 128-candidate parity table,
SOAP IDs, normalization and checksums. The retained external report is:

```text
E:\DATA\EviMem-RL\manifests\wbm-engineering-p1-p15-audit-v1.json
SHA-256 f3941364f2df317fffea3ab63286f66e624449af88f0c48a2f60585551b68e96
```

Across all 128 candidates, modern and parity environments have zero corrected
formation-energy difference, zero initial-hull difference, zero stable-label
mismatches and zero phase-membership mismatches. All 128 SOAP vectors are
finite and normalized (maximum norm error `2.22e-16`). This passes engineering
P1. Primitive/conventional-cell invariant identity, prototype clusters and
WBM-MP overlap remain claim-grade gates.

## Frozen-pool P1.5 result

P1.5 rebuilds each offline oracle hull from the frozen MP phase set plus every
cleaned WBM entry in that exact chemical system, not only the selected 16.
There are 8 oracle-final stable candidates in 5 of 8 pools and 52 candidates
within 50 meV/atom of the oracle-final hull. Fe-Y remains a zero-positive,
zero-near-hull pool and is not replaced. The pools have sufficient support for
an engineering mechanism pilot, but eight systems and byte identity do not
support a claim-grade cross-system conclusion.

## Small real-WBM pilots

All policies execute behind `SecureWBMRunner`; every action is durably written
before reveal. Full SOAP vectors are replaced only inside the finite 16-item
pool by an exact Gram factorization. It preserves every SOAP inner product to
`1e-9` and therefore does not approximate the kernel geometry.

The first `B=4,K=4` run was retained as a diagnostic negative control. Since
the budget never exceeds capacity, no compression occurs. FIFO, free same-FIFO,
full history, diversity and decision coreset consequently produce the same
three oracle-final discoveries. Survival acquisition produces zero. This cell
cannot test retention.

The compression-stress `B=8,K=2` GP-uncertainty run passes exact persistent /
free-reconstruction action parity. Retention changes later acquisition, so its
calibration rows are a composite closed-loop effect. Decision coreset does not
win this comparison: mean remaining residual RMSE is `0.0714`, versus `0.0666`
for FIFO, `0.0611` for diversity and `0.0596` for full history. Survival obtains
4 oracle-final discoveries versus 8 for FIFO and remains a negative secondary
hypothesis.

The primary matched-trace diagnostic instead uses an evidence-blind frozen
acquisition. All retention methods have identical actions in all eight pools:

| Retention | RMSE (eV/atom) | Gaussian NLL | initial-hull Brier | oracle-final discoveries |
|---|---:|---:|---:|---:|
| FIFO K=2 | 0.0712 | -0.2875 | 0.0895 | 6 |
| full history | 0.0697 | 0.5179 | 0.0685 | 6 |
| diversity K=2 | 0.0768 | 0.3847 | **0.0555** | 6 |
| decision coreset K=2 | **0.0656** | **-0.4841** | 0.0637 | 6 |

Decision coreset improves RMSE over full history in 5/8 systems and degrades
in 3/8. It is best on RMSE and NLL but not Brier. This is a preliminary
mechanism signal, not dominance and not paper-level GO. The external summaries
and checksums are:

```text
B4/K4 diagnostic: E:\DATA\EviMem-RL\outputs\engineering\wbm-calibration-coreset-b4-k4-v1\summary.json
B8/K2 closed loop: E:\DATA\EviMem-RL\outputs\engineering\wbm-calibration-coreset-b8-k2-v1\summary.json
SHA-256 2fa9a4959468788ef6f7aeebc2c9f8f5f9c49fa98c03d5b95110ff313f223d0f
B8/K2 matched trace: E:\DATA\EviMem-RL\outputs\engineering\wbm-calibration-matched-b8-k2-v1\summary.json
SHA-256 7c6ed468f8bb7e31e6dcd8389cbc7fc0df373daad78bd20be869984a63becbf8
```

## Objective-fidelity follow-up

The corrected follow-up keeps the same `B=8,K=2` frozen action trace and adds
`JointPosteriorRiskOneSwap`. It also evaluates stability against the updated
causal hull after the eight reveals; these Brier values therefore supersede the
initial-hull Brier column above rather than silently overwriting it.

| Retention | RMSE | NLL | causal Brier | causal log loss | CRPS | observable joint risk |
|---|---:|---:|---:|---:|---:|---:|
| FIFO | 0.0712 | -0.2875 | 0.0767 | 0.2681 | 0.0444 | 0.6079 |
| full history | 0.0697 | 0.5179 | 0.0482 | 0.1819 | 0.0444 | 0.4638 |
| diversity | 0.0768 | 0.3847 | 0.0396 | 0.1530 | 0.0482 | 0.5746 |
| GP-variance one-swap | 0.0762 | 0.3942 | **0.0358** | **0.1419** | 0.0479 | 0.5164 |
| DACC | **0.0656** | -0.4841 | 0.0469 | 0.1826 | **0.0393** | 0.3695 |
| joint-risk one-swap | 0.0688 | **-0.5404** | 0.0447 | 0.1783 | 0.0407 | **0.3634** |

Across all 64 admissions on the DACC trajectory, facility value and negative
joint risk have mean Spearman `0.812`; selection agreement is `82.8%`, and only
11 rounds have positive facility joint-risk regret. Restricting to the 46
saturated `K+1` neighborhoods raises mean Spearman to `0.878` and agreement to
`84.8%`; seven rounds have positive regret. Final DACC and joint-risk active
sets are identical in five of eight systems. Among the three differences,
neither method dominates: DACC wins two systems on NLL/CRPS, while joint risk
wins two on RMSE/Brier/log loss, with five ties for every metric.

The asymmetric weighted decision loss is identical for all methods in this
small final-time view: seven systems have zero loss and the only nonzero system
does not distinguish selectors. It is therefore a valid but uninformative
primary metric here; broader systems and prequential evaluation are required.

```text
E:\DATA\EviMem-RL\outputs\engineering\wbm-objective-fidelity-gpvariance-matched-b8-k2-v1\summary.json
SHA-256 1cf8336f8b78c2223246aec0bf142077ea77c526bd39133550d38211571415b6
```

This diagnostic supports retaining DACC as the simpler primary hypothesis and
keeping joint risk as a fidelity baseline. It does not support promoting joint
risk, claiming superiority, or entering MADE. The capacity-matched GP-variance
baseline is stronger than geometric diversity on causal Brier/log loss in this
cell, while DACC remains better on RMSE/CRPS. This reinforces the need for a
non-single-point capacity grid rather than a winner claim from the current cell.

The rerun also removed a latent engineering truncation: `full_history` no
longer uses a hard-coded 16-witness reconstructed FIFO view and now exposes the
entire revealed archive. This does not change the present `B=8` numbers, but is
required before any `B>16` comparison.

## Decision

- **Continue** the primary calibration-coreset hypothesis to a feasible frozen
  grid over tight capacities and calibration-only hyperparameters; the
  GP-variance one-swap baseline is now implemented and must remain in that grid.
- **Pause** survival-conditioned acquisition; do not tune fantasy counts or
  weights against these evaluation pools.
- Do not claim superiority until canonical/prototype overlap, more independent
  systems, paired uncertainty and measured compute are complete.
- MADE remains blocked.

## Frozen exact-system grid implementation (2026-07-17)

The next evaluation no longer uses fixed-size 16-candidate pools. An
oracle-blind manifest selects every eligible exact chemical system using
`SHA256(release_id || chemical_system)`, with at most eight systems per binary,
ternary and quaternary-or-higher stratum. Every selected system retains all of
its cleaned candidates in a frozen ID-hash order. The realized manifest has 16
systems and 334 candidates: eight binary systems, eight ternary systems, and no
quaternary-or-higher system because none reaches the preregistered `N_s >= 16`
minimum. Exact chemical systems are never mixed.

The reported grid has 37 labels/system but only 15 physical traces/system.
For an identical strategy and capacity, the `B=12` trace is run once and its
immutable prefixes supply eligible `B=4` and `B=8` labels. Full history is run
once, and joint-posterior risk runs only at `(8,2)` and `(12,4)`. Survival and
the already-completed exhaustive subset diagnostic are disabled.

After each reveal, composition-dependent hull update and retention, an
evaluator-only prequential scorer evaluates every remaining candidate. It
reports boundary-weighted causal CRPS, Brier and log loss, residual RMSE/NLL,
false-stable cost, posterior-fit/retention/prediction/pipeline time and parent
RSS. Oracle outcomes remain inside the evaluator and are never serialized to
the policy subprocess. Statistical comparison uses paired exact-system
differences and a deterministic system-clustered bootstrap.

The infrastructure smoke covered all 16 systems at `B=4,K=1` with FIFO. Every
system produced exactly four prequential rounds. This validates execution only;
it is not a comparative result. Immutable external artifacts are:

```text
E:\DATA\EviMem-RL\manifests\wbm-frozen-exact-system-grid-v1.json
SHA-256 1529f1a73d0d74050fba8a6a05a386398d88ea4b4b4ca236f33cc30feb702a14

E:\DATA\EviMem-RL\outputs\engineering\wbm-frozen-grid-prequential-smoke-b4-k1-fifo-v1\summary.json
SHA-256 50d43ea90554aeb16c393b15abbc3a7c0986640a520e667ec9864a923eb5664e
```

At the time of this smoke run, paper-level GO was blocked because causal
Brier/log-loss non-inferiority margins had not yet been frozen on disjoint
calibration systems. The later calibration-only freeze below resolved that
execution prerequisite without consulting evaluation systems; it did not
establish a positive method result.

## Final pre-grid calibration gates (frozen design, subsequently completed)

The complete comparison grid remains blocked by three hard gates. They are
implemented as external-artifact workflows, but their outcomes have not been
used to inspect an evaluation comparison.

1. The current fixed GP is registered without re-searching parameters:
   Matérn-5/2, length scale `0.35`, signal standard deviation `0.08 eV/atom`,
   noise standard deviation `0.01 eV/atom`, and jitter `1e-10`. It must pass a
   full-history prequential numeric sanity check on eight disjoint calibration
   exact systems (four binary and four ternary, each with `N_s >= 16`) before
   its status changes from engineering-only to calibration-frozen.
2. FIFO and GP-variance one-swap, with no DACC result consulted, define the
   Brier and log-loss non-inferiority margins. For each loss `m`, the immutable
   rule is `min(0.10 * GPV_macro_loss, 0.20 * max(GPV_over_FIFO_95%_LB, 0))`.
   A nonpositive paired lower bound therefore freezes a zero margin rather than
   permitting a post-hoc tolerance.
3. Every canonical `B=12` trace must demonstrate behavioral prefix parity at
   the first four and eight rounds after rebuilding the identical history and
   changing only `remaining_budget`. Selected query IDs, active witnesses,
   causal-hull checksums and evaluator metric-input checksums are recorded;
   any action mismatch rejects prefix reuse for that strategy.

The calibration-system selector excludes both the 16 evaluation systems and
the eight earlier GP-development systems using only cleaned membership,
composition, structure bytes and release-ID hashes. A 173-candidate
calibration manifest, SOAP cache and official prediction/PPD join have been
created outside Git.

## Calibration freeze and paused-grid checkpoint (2026-07-19)

The isolated calibration run completed all 24 planned traces: FIFO,
GP-variance one-swap, and full history over eight disjoint exact systems at
`B=12,K=4`. Full-history prequential sanity passed for every system. The GP
configuration is therefore frozen without a new hyperparameter search. The
calibration-only manifest and summary are immutable external artifacts:

```text
E:\DATA\EviMem-RL\outputs\calibration\wbm-gp-margin-calibration-v3-streaming\summary.json
SHA-256 29c8ea370e2900b7e3f3a60816588bc1f4671ca12f3a3b190b9fbc9b67e74c7b

E:\DATA\EviMem-RL\manifests\wbm-gp-and-noninferiority-calibration-freeze-v1.json
SHA-256 0f63e146bdc98bca96051cbf7bf07f3896a7bbd3b312eb69926f80583315055e
```

GP-variance did not have a positive 95% paired-bootstrap lower bound for its
improvement over FIFO on either probability loss. The preregistered rule
therefore freezes both tolerances at zero:

```text
boundary-weighted causal Brier margin    = 0
boundary-weighted causal log-loss margin = 0
```

The 16-system frozen grid subsequently began with the completed
`primary-k1-b12` physical group. Its immutable summary is:

```text
E:\DATA\EviMem-RL\outputs\engineering\wbm-frozen-grid-v1-streaming\physical\primary-k1-b12\summary.json
SHA-256 cb5b9abe76c592f656fd815067b1d13477a45e14f766761880dd5c48a1ac586e
```

Execution was manually paused while `primary-k2-b12` had written 47 of its 64
append-only ledgers and before that group emitted a summary. The partial
directory is retained for interruption auditing only. It is not a comparative
result and must not be combined with a later run: resumption requires a new
immutable output directory and a fresh execution of the entire physical group.
No multi-cell comparison, paired bootstrap, Pareto frontier, or GO/NO-GO claim
has been computed from the paused grid.

## Objective correction: P3C (2026-07-19)

The paused result exposed a structural weakness in both facility DACC and the
joint-self-risk diagnostic. They minimize a candidate posterior's own Bayes
risk, so a realized residual can be retained because it produces an extreme
probability rather than because it faithfully represents all available
evidence. Positive-part singleton gain additionally ignores counter-evidence
that moves an overconfident posterior back toward the decision boundary.

The replacement hypothesis is P3C (Proper Posterior-Projection Coreset). After a new
card is legally revealed and the causal hull updated, the posterior conditioned
on the current active cards plus that card is frozen as a reference. Every
legal drop-one subset is evaluated against that same reference using a proper
Brier, log, Gaussian-KL or threshold-weighted-CRPS divergence. Asymmetric
stability costs enter through reference decision regret, not candidate
self-risk.

P0 code and unit diagnostics now cover:

- exact online drop-one projection;
- exhaustive archive projection over all subsets up to capacity;
- registered-config and calibration-freeze SHA binding in the formal grid path;
- randomized single-witness/general-GP equivalence;
- a wrong-extreme counterexample for the legacy objective;
- union-reference divergence, decision regret, archive optimization gap,
  archive reactivation and retained-residual bias reporting.

## P3C matched-action P1 result (2026-07-19)

The authorized diagnostic completed 72/72 append-only ledgers: eight frozen
exact chemical systems, 16 candidates per system, `B=8,K=2`, and nine
retention strategies under one frozen evidence-blind acquisition trace. Strict
action parity passed for selected query IDs, hull transitions, and discovery
metrics; no checksum, oracle-isolation, or finite-number check failed.

The immutable external result and deterministic exact-system analysis are:

```text
E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-objective-fidelity-matched-b8-k2-v1\summary.json
SHA-256 4facffd371820bf25678e16e8311bb4c1b7c798f363661c53a1e55102a6109fa

E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-objective-fidelity-matched-b8-k2-v1\p1-analysis.json
SHA-256 285dbfaa248d57bc5f2f9b664d08660f9f2aa99c088e0de7f44c3f726a173313
```

Mean prequential losses are shown below; lower is better. These are engineering
aggregates over eight systems, not candidate-level independent samples.

| Retention | causal CRPS | causal Brier | causal log loss | residual RMSE | Gaussian NLL |
|---|---:|---:|---:|---:|---:|
| GP variance | 0.034981 | 0.075365 | 0.258965 | 0.067492 | -0.505813 |
| Legacy DACC | 0.033168 | 0.075101 | 0.260050 | 0.064283 | -0.735903 |
| Joint self-risk | **0.032317** | **0.072442** | **0.254979** | **0.063945** | **-0.933600** |
| P3C-Brier | 0.034035 | 0.077517 | 0.264602 | 0.065472 | -0.751921 |
| P3C-Log | 0.034015 | 0.076162 | 0.261011 | 0.065553 | -0.754441 |
| P3C-Gaussian-KL | 0.034610 | 0.084903 | 0.288137 | 0.067095 | -0.594334 |
| P3C-twCRPS | 0.034085 | 0.079437 | 0.271797 | 0.065745 | -0.733871 |

For P3C-Log minus GP variance, the exact-system paired mean CRPS difference is
`-0.000965` with deterministic 95% bootstrap interval
`[-0.002348, 0.000280]`; Brier is `+0.000798`
`[-0.006841, 0.008896]`, and log loss is `+0.002047`
`[-0.018637, 0.026197]`. Only NLL has an interval below zero:
`-0.248628 [-0.542344, -0.003605]`. P3C-Gaussian-KL is clearly worse than GP
variance on Brier (`+0.009539 [0.001459, 0.019937]`) and log loss
(`+0.029172 [0.006219, 0.056603]`). Thus no P3C variant establishes CRPS
superiority together with probability-loss non-inferiority.

The mechanism diagnostics are nontrivial. Depending on proper score, archive
reactivation would change the selected subset in 16--22 of 64 rounds, and the
mean online-to-archive objective gap ranges from `0.000131` for P3C-twCRPS to
`0.019521` for P3C-Gaussian-KL. Mean absolute retained-minus-archive residual
bias ranges from `0.00791` to `0.01040` eV/atom. All selected P3C projections
have zero union-reference decision regret. Consequently, the theory-fixed
zero-regret P3C-twCRPS variant is identical to unconstrained P3C-twCRPS in all
64 rounds and never invokes its fallback.

This establishes implementation fidelity and demonstrates real online
irreversibility, but it is not a paper-level positive result. Eight systems and
one `B,K` cell cannot support a non-single-point calibration--compute claim;
the P3C probability metrics also fail the frozen zero-margin gate. Status stays
**NO-GO**, and work pauses here. The interrupted legacy grid remains unusable,
and no new P3C grid, survival acquisition, kernel tuning, fantasy hull, or MADE
run is authorized by this result.

## Reference, path, NLL, and selection follow-up (authoritative `v5`)

The frozen follow-up implements the four diagnostics prescribed after P1
without changing P3C or accessing a broader evaluation grid. Its authoritative
external files are:

```text
E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-reference-path-selection-b8-k2-v5\summary.json
SHA-256 0d25f251a1d1ede6dc2b63c5e2ed7c8782fde716f984b45d9c93060ea4b2f9b3

E:\DATA\EviMem-RL\outputs\engineering\wbm-p3c-reference-path-selection-b8-k2-v5\p1-decomposition-analysis.json
SHA-256 149ed9562d5a6c6d550c550f99711542df2733c58236cc3f7daf802a9461ef1d
```

The result contains 48/48 complete runs and ledgers (eight exact systems by six
strategies), 1,920 independently evaluated reference/search snapshots, and 168
P3C-Log selection-effect records. Frozen-acquisition action parity passes and
the analysis has zero quality issues. `v2` and `v3` are failed incomplete
attempts with no valid summary. `v4` is scientifically trace-equivalent to
`v5`, but it does not fully account for lazy GP fit/factorization timing; it is
excluded from all timing claims and only `v5` is cited.

For a causal loss `L`, define reference headroom `H=L_G-L_Q`, compression loss
`C=L_P-L_Q`, and positive-headroom recovery
`rho=(L_G-L_P)/(L_G-L_Q)`. P3C-Log's union reference has positive mean Brier and
log-loss headroom in 6/8 systems, but only 28/64 and 30/64 rounds. Its
headroom-weighted recovery is 0.715 and 0.672. The archive reference has positive
headroom in 7/8 systems and 36/64 rounds, yet archive-reference/archive-search
recovery is only 0.484 and 0.428 with zero median recovery. Thus Gate A passes,
but positive reference information is intermittent and finite-capacity
projection does not preserve it reliably enough to beat GP variance.

The 2x2 causal-loss decomposition appears below. `UO`, `UA`, `AO`, and `AA`
denote union/archive reference crossed with online/archive subset search.

| Method/metric | UO | UA | AO | AA | GP variance |
|---|---:|---:|---:|---:|---:|
| P3C-Log Brier | 0.076162 | 0.076024 | 0.076482 | 0.075625 | 0.075365 |
| P3C-Log log loss | 0.261011 | 0.260714 | 0.262068 | 0.261074 | 0.258965 |
| P3C-Brier Brier | 0.077517 | 0.077096 | 0.076741 | 0.075776 | 0.075365 |
| P3C-Brier log loss | 0.264602 | 0.263524 | 0.262942 | 0.262268 | 0.258965 |

Archive search under the same union reference is a small, non-robust effect.
For P3C-Log it changes Brier by `-0.000138` (exact sign-flip `p=0.65625`) and
log loss by `-0.000298` (`p=0.90625`). The mean reverses after removing the
largest contributor. Historical irreversibility therefore exists internally,
but Gate C does not establish causal-metric value for reactivation.

The aggregate P3C-Log NLL difference from GP variance is `-0.248628`, with
bootstrap interval `[-0.542344,-0.003605]`. Conservative diagnostics change the
interpretation: the median is `-0.019457`, the exact sign-flip p-value is
`0.21875`, and the system vector is
`[+0.053425,+0.087890,-0.677902,+0.000333,-1.084744,0,-0.329115,-0.038915]`
in the frozen system order. Fe--Y contributes 50.91% of gross improvement and
the top three systems 98.17%; removing Fe--Y leaves mean `-0.129183`.
Symmetric NLL Shapley attribution is `-0.088527` from the posterior mean and
`-0.160101` from posterior variance, adding to the total within `5.6e-17`.
The NLL signal is predominantly variance-driven and influential-system-sensitive.

Finally, the descriptive P3C-Log retention model has in-sample ROC-AUC 0.8073.
Raw absolute-residual and sign coefficients are near zero after standardization,
whereas reference-mean and reference-variance influence coefficients are 0.8418
and 1.0096. This verifies outcome-dependent selection through posterior
influence, but neither proves extreme-residual chasing nor identifies selective
inference as the causal source of the probability-metric gap.

The timing gate is now semantically complete: union-reference fit/factorization,
online candidate projection, archive-reference fit/factorization, archive subset
evaluation, prequential evaluation, and causal-hull update are separate fields.
Archive diagnostics are excluded from online retention cost. This passes Gate D
as instrumentation, not as a Pareto-performance claim.

The final method decision remains **paper-level NO-GO**. Gaussian-KL is frozen
as a negative variant, twCRPS is diagnostic-only, and the decision-safe
constraint is inactive. No full grid, acquisition extension, parameter search,
or MADE run is authorized by this follow-up.
