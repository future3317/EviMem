# Raw-results manifest

This file records the location and provenance of raw experimental outputs. Raw
outputs remain outside Git by design. The manifest is the versioned pointer that
allows a later audit to identify the exact external result roots without making
the code repository a data archive.

Inventory time: 2026-08-03 (Asia/Shanghai)

## Provenance

- Code commit: `47c0891` (`E:\CODE\EviMem-RL`)
- Paper commit: `f3f1f84` (`E:\PAPER`)
- Local data root: `E:\DATA\EviMem-RL`
- Remote data root: `/home/workspace/lrh/DATA/EviMem-RL`
- Remote host: `lrh@100.110.148.20`
- Raw outputs are not copied, staged, committed, or pushed by this manifest.

## Registered campaigns

| Campaign | External root | State at inventory | Registered outputs | Files | Bytes | Failures |
|---|---|---:|---:|---:|---:|---:|
| MatPES mechanism v3 core | `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_v3_20260802` | complete | 5 folds × B=1..6 = 30/30 | 4,274 | 79,084,883 | 0 |
| Reduced targeted ablation v1 | `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_reduced_v1_20260802` | complete | 5 folds × B=6 = 5/5 | 1,391 | 58,737,975 | 0 |
| Exact-DP random suite v3 | `E:\DATA\EviMem-RL\analysis\matpes_ic_sarr_mechanism_v3_20260802\random_exact_dp_suite_v3.json` | complete | 1,000 instances | 1 file | 2,335,785 | 0 |

The two remote campaigns were checked read-only. No active P0 launcher or
closed-loop worker process was present at inventory time. The remote raw roots
are authoritative for the MatPES runs; the local copy currently contains the
synthetic suite and preflight records only.

## Derived post-processing artifacts

These files are also external to Git and are safe to regenerate only with the
same frozen inputs and registered code:

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_reduced_v1_20260802/derived_summary_reduced_v1.json` | 232,812 | `d9607180cb769896a68c7544d60ac97a50e5b3ad420ebb8e98cb8056fbc9d6ab` | complete; superseded for manuscript mechanism mapping |
| `E:\DATA\EviMem-RL\analysis\matpes_ic_sarr_mechanism_reduced_v1_20260802\derived_summary_reduced_v2_corrected.json` | 241,692 | `4923624a86d00be55d422960f48263b3fdf06123f78cf92ce1398af2050e2c87` | complete; corrected post-processing mapping |
| `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_v3_20260802/derived_ic_sarr_calibration_v3.json` | 1,768,548 | `a21b41b4912f35cf82043ceda7bbf8961f8e7ead3f938a5b4d78d41f60cdec0b` | complete |
| `E:\DATA\EviMem-RL\analysis\matpes_ic_sarr_mechanism_reduced_v1_20260802\derived_direct_mechanism_comparisons_v1.json` | 111,515 | `54a86971d849002537db2e25335b1f7c799dcf956144b5a25657b13f62eec9d2` | complete; direct same-system mechanism contrasts |
| `E:\DATA\EviMem-RL\analysis\matpes_ic_sarr_mechanism_v3_20260802\random_exact_gate_audit_v3.json` | 733,351 | `1CD3A1FC973EECFDB0A81333EA6245B995A200B32B351708019B5CBD9C421456` | complete; exact-world numerical-gate audit |
| `E:\DATA\EviMem-RL\analysis\matpes_ic_sarr_mechanism_v3_20260802\derived_budget_direct_comparisons_v1.json` | 22,560 | `95a73859177f57474b25203773729d4b9c70dc3ee94c4d6943f11793573a5a6d` | complete; B=1..6 direct paired policy contrasts |
| `E:\DATA\EviMem-RL\analysis\matpes_ic_sarr_mechanism_v3_20260802\derived_prequential_calibration_v1.json` | 22,710 | `0e7693d477a415c5ff228790c4511fe1edf1854545e946f6b9026d4a7d362d21` | complete; B=6 prequential energy diagnostics |

The calibration audit initially exposed and then fixed a task-schema adapter
bug: initial reference phases store corrected total energy and composition,
so the audit now converts total energy by atom count, matching the frozen
closed-loop implementation. This does not alter any experiment or estimand.

## Corrected v3 verification (2026-08-03)

The remote roots were rechecked after the post-processing jobs completed. The
core root contains 4,274 files totaling 79,084,883 bytes, including all
30/30 core outputs, and the reduced root contains 1,391 files totaling
58,737,975 bytes, including all 5/5 reduced outputs. Neither root contains a
failure marker, and no launcher, worker, summary, or calibration process is
active.

The completed calibration artifact is 1,768,548 bytes with SHA-256
`a21b41b4912f35cf82043ceda7bbf8961f8e7ead3f938a5b4d78d41f60cdec0b`.
The complete core budget curve and reduced B=6 mechanism audit are contained
in the completed reduced-summary artifact listed above. The legacy full-suite
summarizer was also run read-only against the v3 core root; it failed closed on
the first absent legacy v2 ablation path and wrote no output. This is a
post-processing roster mismatch, not an experiment failure.

For manuscript mechanism decomposition, the corrected post-processing artifact
maps `F-T` to the runner's `unqueried_competitor_invalidations` field. It is
not raw data and remains outside Git.

## Local checksums

SHA-256 values below cover the locally available v3 directory at inventory
time. They are not hashes of the remote MatPES directories.

| File | Bytes | SHA-256 |
|---|---:|---|
| `analysis/matpes_ic_sarr_mechanism_v3_20260802/random_exact_dp_suite_v3.json` | 2,335,785 | `B9DCE8220B5B6783DDF2561DC913909E7480489B7A3E769FB17FEBD2C6ACE1C0` |
| `analysis/matpes_ic_sarr_mechanism_v3_20260802/preflight/matpes-p0v3-preflight-single-b1.json` | 356,141 | `291C99B4FDC65343096CB89FF82A7A34B57CB48F1F898AB8459B42398BBE1588` |
| `analysis/matpes_ic_sarr_mechanism_v3_20260802/preflight/query_manifest.json` | 128 | `83102E840D46FC845175A623AD99E7A19AC2537F46F7DB3E74BE7239F4AD9429` |

## Use and update policy

Only complete, predeclared outputs may be summarized into the manuscript. A
summary must retain the campaign root, code commit, protocol/manifest identity,
output counts, failure count, and any result checksum available at the source.
The periodic monitor may read remote outputs and run registered summaries, but
must not restart jobs, alter policies/seeds/manifests, or copy raw outputs into
Git. Derived tables, figures, and paper text belong in their respective Git
repositories only after the complete result set has passed the registered
audit.
