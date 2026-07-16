# WBM phase-1 data, license, and infrastructure audit

Status date: 2026-07-16. Scope is limited to the five preregistered
infrastructure gates. No comparative policy matrix or MADE run is authorized.

## Source and license findings

The Matbench Discovery registry at commit
`d7ac0770033c01cffe164b5cd4151ca4087c30a0` identifies:

| Artifact | Frozen source candidate | Registry license | Current decision |
|---|---|---|---|
| WBM structures/energies | Materials Cloud `2021.68` CSEs plus the upstream compiler's Google Drive structures/summary | CC-BY-4.0 | downloaded externally; raw-to-cleaned compiler gate produced exactly 256,963 IDs and an external manifest. Human license decision still pending. |
| MP initial phase set | `2023-02-07-mp-computed-structure-entries.json.gz` (Figshare `40344436`, MD5 `76fc748db6b175bb80de4c276d27c235`) | CC-BY-4.0 | downloaded outside Git at `E:\DATA\EviMem-RL\artifacts`; official MD5 and frozen SHA-256 verified. Human license decision remains pending. |
| MP initial parity reference | `2023-02-07-ppd-mp.pkl.gz` (Figshare `48241624`, MD5 `60d19d691fa1d338aa496a40a9641bef`) | CC-BY-4.0 | downloaded outside Git; official MD5 and frozen SHA-256 verified. CSE/PPD phase-membership audit is available in the parity environment. |
| Matbench Discovery code/registry | GitHub commit above | MIT | code license only; it is not substituted for the data licenses |
| Frozen predictor output | CHGNet 0.3.0 discovery predictions (Figshare `66646268`, MD5 `fd7cd3781a24be465aaeadf97663ce58`) | BSD-3-Clause model/checkpoint; prediction artifact pending audit | downloaded outside Git; official MD5 and frozen SHA-256 verified. It is the required primary predictor artifact; the checkpoint remains non-substituting. |
| WBM structure artifact for SOAP | not yet selected | unresolved | **blocked** pending exact structure file, license, release, and checksum |

Evidence URLs:

- <https://github.com/janosh/matbench-discovery/blob/d7ac0770033c01cffe164b5cd4151ca4087c30a0/data/datasets.yml>
- <https://github.com/janosh/matbench-discovery/blob/d7ac0770033c01cffe164b5cd4151ca4087c30a0/data/wbm/readme.md>
- <https://archive.materialscloud.org/record/2021.68>
- <https://docs.materialsproject.org/changes/database-versions#v2022.10.28>

This is a source-level audit, not legal advice. The executable gate requires a
human-reviewed `DataLicenseDecision` and SHA-256 for each external WBM, MP,
prediction, and structure artifact. It fails if a file is missing, the checksum
differs, research use is not explicitly approved, or the file is inside the Git
repository. Redistribution permission is recorded separately from research
use. No dataset, prediction file, SOAP cache, or experiment output is committed.

## Infrastructure gate status

| Gate | Implemented check | Unit status | Real-artifact status |
|---|---|---:|---:|
| 1. Data and license | explicit license authority, external-path enforcement, SHA-256, four required artifact roles | pass | **technical artifact subgate passed:** WBM raw IDs, MP CSE/PPD and official predictor files are external and checksummed. **Formal gate remains blocked** by human license/redistribution approval and the frozen SOAP/structure artifact. |
| 2. MP initial causal hull | composition-specific linear-program hull using only frozen MP phase records; release and phase-set checksum in `HullSnapshot` | pass | **input-parity subgate passed:** CSE and PPD contain the same 154,718 unique `entry_id` values. Candidate-level initial-hull parity remains pending pool freeze. |
| 3. Oracle-isolated WBM reveal | policy-safe observable type has no energy/label fields; vault reveals one selected query once | pass | **raw-source smoke pass:** all five files decode; a real structure decodes with pymatgen and energy remains absent from the observable object. Curated WBM IDs/structures are still blocked. |
| 4. Frozen prediction and SOAP cache | predictor/config/content checksum, fixed SOAP parameters/species, normalized vectors, structure-identity lookup | pass | **prediction subgate passed:** CHGNet 0.3.0 official file has 256,963 unique IDs, exact cleaned-ID join, and no missing/excess IDs. SOAP remains blocked pending canonicalization/pool freeze and cache build. |
| 5. Exact persistent/on-demand emulation | separate persistent FIFO and archive reconstruction; per-round action, ordered witness state, hull state, discovery state, and canonical trace checksum | pass | not run on WBM pools |

“Unit pass” means the mechanism and failing tests exist using tiny synthetic
records. It is not evidence about WBM performance and does not satisfy a real-
artifact gate. The comparative matrix remains blocked until all five entries in
the last column pass with retained manifests and checksums.

## Deterministic technical choices

- Initial hull optimization minimizes MP formation energy subject to the exact
  target composition. WBM energies cannot enter the builder's `MPPhaseRecord`
  interface.
- SOAP computation is not silently reimplemented. Phase 1 accepts an externally
  generated, audited SOAP artifact and validates cutoff 5 Angstrom, `n_max=8`,
  `l_max=6`, periodic mode, species vocabulary, normalization, and checksum.
  The exact maintained SOAP implementation/version remains a real-artifact gate.
- `scipy.optimize.linprog(method="highs")` is the reference hull solver in the
  current `llm` environment. Dependency/version must be added to the frozen WBM
  environment manifest before the real MP gate.
- Exact-emulation cost instrumentation may add timing fields, but any action,
  active witness, hull, or discovery mismatch is a hard failure.

## Local engineering and parity runtimes

The phase-one raw-source smoke checks used `pymatgen 2026.5.4`, `dscribe
2.1.2`, and `chgnet 0.4.2`, installed in the local `llm` environment on
2026-07-16. These versions establish that maintained interfaces can decode the
external WBM structures; they do **not** freeze a predictor checkpoint or
SOAP cache and they must not be used to regenerate the MP2020-corrected WBM
labels. The latter remains pinned to the historical WBM compiler environment
(for example, 2023-era pymatgen behavior) once the v2022.10.28 MP snapshot is
available.

The separate `wbm-parity` environment is installed with `pymatgen==2023.5.10`
and is reserved for MP2020 correction and initial-PPD parity checks. It has no
authority to declare parity until the checksummed MP CSE and PPD artifacts are
available. Its outputs must be reported separately from the modern adapter
environment. The published PPD needs a read-only unpickle compatibility
workaround because its serialized `MaterialsProject2020Compatibility`
constructor predates the pinned runtime. The workaround restores saved object
state only; it never changes the file or recomputes a correction.

## Verified frozen official inputs

The three registry artifacts were moved from the repository-local staging
directory to `E:\DATA\EviMem-RL\artifacts`, where they are excluded from Git.
`OFFICIAL_ARTIFACT_MD5_VERIFICATION.json` records file ID, size, MD5 and
SHA-256. The reproducible command below additionally validates CSE/PPD phase
membership and strict prediction-to-cleaned-ID parity, writing its report
outside the repository:

```powershell
conda run --no-capture-output -n wbm-parity python tools/audit_wbm_official_artifacts.py `
  --artifact-dir E:\DATA\EviMem-RL\artifacts `
  --cleaned-ids E:\DATA\EviMem-RL\manifests\wbm-cleaned\wbm-256963-cleaned-benchmark-ids.txt `
  --output E:\DATA\EviMem-RL\manifests\official-artifact-audit.json
```

The audit reports technical integrity only. It deliberately leaves
`formal_gate_passed=false` until an accountable human reviews research use,
attribution and redistribution for every source, and it never exposes WBM
oracle energy or stability labels to policy code.

## External raw-to-cleaned ID artifact

`tools/build_wbm_cleaned_id_manifest.py` writes only to an external output
directory. The current run produced
`wbm-256963-cleaned-benchmark-ids.txt` with ID checksum
`sha256:ab2ff77ad9d17168a25e2a4724c98438f52bc8ef160a74e01d8eacf9b42e9bc1`.
The accompanying JSON manifest records all source-file checksums, the six
step-3 source-ID anomalies, the two missing step-5 initial structures, the
un-normalized summary/CSE ID differences, the normalized ID equality gate, and
the `257489 -> 257487 -> 256963` filter chain. It is an identity artifact only,
not a scientific label or policy result.

The current external JSON manifest has SHA-256
`5fa86711e4f4b9c3b42ed46652724b93b94a352dd06ba3f5e6d9fb96932b2edb`.
The separately downloaded official CHGNet 0.3.0 checkpoint has SHA-256
`d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`;
it is retained only as a source artifact until its license manifest is manually
approved and the official discovery prediction file has passed its own MD5 and
ID-join gates.

## P0/P1 amendment (2026-07-16)

P0 execution correctness now passes through the sole secure WBM runner. The
retired gate document is summarized in `RESEARCH_ITERATION_HISTORY.md`; the
full version remains at commit `e313499`. This does not by itself authorize a
claim-grade policy comparison.

The frozen 128 engineering candidates now have a versioned external audit at
`E:\DATA\EviMem-RL\manifests\wbm-parity-128-merged-audit-v2.json`. The audit
separates the raw explicit-ID WBM summary, historical-pipeline replay,
modern-adapter replay, and official CHGNet predictions. Historical
`pymatgen==2023.5.10` and modern `pymatgen==2026.5.4` replay values agree for
all 128 corrected formation energies and initial hull distances in this subset.

Exact-official-energy P1 remains unavailable, but it no longer blocks the
method-adaptive engineering pilot. The available explicit-ID
`wbm-summary.txt` contains raw/legacy energy columns, not an independently
published candidate-level MP2020-corrected compiled summary. Those raw columns
must not be relabeled as official corrected energy or official initial-hull
truth. Prototype clustering, MP-overlap, claim-grade canonical identity and
human publication/redistribution review also remain pending. The engineering
study is now explicitly scoped as a fixed historical-pipeline WBM replay; it
does not make an official-energy reproduction claim. Under that scope,
cross-environment P1 has zero numerical or label mismatches and full-system
P1.5 has been executed. See `WBM_ENGINEERING_P1_P15_AND_PILOT.md`.

## Current authorization boundary

The official artifacts, cleaned IDs, historical-pipeline parity, SOAP cache and
small oracle-isolated engineering pools are available for DACC debugging. They
do not establish claim-grade identity independence, redistribution permission,
or paper-level superiority. Comparative claim-grade execution still requires
the canonical/prototype overlap audit, calibration-only parameter freezing and
the system-count/uncertainty plan listed in the current DACC specification.
