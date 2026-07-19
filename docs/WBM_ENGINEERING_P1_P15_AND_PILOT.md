# WBM engineering P1/P1.5 resolution and calibration pilot

**Status (2026-07-16).** The new decision-aware calibration-coreset method is
no longer blocked from small engineering WBM experiments by the old CAW /
working-set gates. Claim-grade comparison and MADE remain unauthorized.

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

Paper-level GO remains blocked because causal Brier/log-loss non-inferiority
margins have not yet been frozen on disjoint calibration systems. Evaluation
systems cannot be used to choose those margins.

## Final pre-grid calibration gates (frozen design, not results)

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
created outside Git. The calibration runner was deliberately paused before it
produced an immutable summary, so **no GP configuration or margin is currently
frozen** and the comparative grid remains prohibited.
