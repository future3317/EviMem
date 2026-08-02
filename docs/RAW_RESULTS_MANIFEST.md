# Raw-results manifest

This file records the location and provenance of raw experimental outputs. Raw
outputs remain outside Git by design. The manifest is the versioned pointer that
allows a later audit to identify the exact external result roots without making
the code repository a data archive.

Inventory time: 2026-08-02 (Asia/Shanghai)

## Provenance

- Code commit: `65b015b` (`E:\CODE\EviMem-RL`)
- Paper commit: `eb0d7ed` (`E:\PAPER`)
- Local data root: `E:\DATA\EviMem-RL`
- Remote data root: `/home/workspace/lrh/DATA/EviMem-RL`
- Remote host: `lrh@100.110.148.20`
- Raw outputs are not copied, staged, committed, or pushed by this manifest.

## Registered campaigns

| Campaign | External root | State at inventory | Registered outputs | Files | Bytes | Failures |
|---|---|---:|---:|---:|---:|---:|
| MatPES mechanism v3 core | `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_v3_20260802` | complete | 5 folds × B=1..6 = 30/30 | 4,273 | 77,316,335 | 0 |
| Reduced targeted ablation v1 | `/home/workspace/lrh/DATA/EviMem-RL/analysis/matpes_ic_sarr_mechanism_reduced_v1_20260802` | complete | 5 folds × B=6 = 5/5 | 1,390 | 58,505,163 | 0 |
| Exact-DP random suite v3 | `E:\DATA\EviMem-RL\analysis\matpes_ic_sarr_mechanism_v3_20260802\random_exact_dp_suite_v3.json` | complete | 1,000 instances | 1 file | 2,335,785 | 0 |

The two remote campaigns were checked read-only. No active P0 launcher or
closed-loop worker process was present at inventory time. The remote raw roots
are authoritative for the MatPES runs; the local copy currently contains the
synthetic suite and preflight records only.

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
