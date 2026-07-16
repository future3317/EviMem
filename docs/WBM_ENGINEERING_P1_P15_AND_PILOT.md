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

## Decision

- **Continue** the primary calibration-coreset hypothesis to a small frozen
  grid over tight capacities and calibration-only hyperparameters.
- **Pause** survival-conditioned acquisition; do not tune fantasy counts or
  weights against these evaluation pools.
- Do not claim superiority until canonical/prototype overlap, more independent
  systems, paired uncertainty and measured compute are complete.
- MADE remains blocked.
