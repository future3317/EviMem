# WBM phase-1 data, license, and infrastructure audit

Status date: 2026-07-16. Scope is limited to the five preregistered
infrastructure gates. No comparative policy matrix or MADE run is authorized.

## Source and license findings

The Matbench Discovery registry at commit
`d7ac0770033c01cffe164b5cd4151ca4087c30a0` identifies:

| Artifact | Frozen source candidate | Registry license | Current decision |
|---|---|---|---|
| WBM structures/energies | Materials Cloud `2021.68`; 256,963-record curated WBM release | CC-BY-4.0 | research use appears compatible with attribution; exact files/checksums not yet frozen |
| MP initial phase set | MP v2022.10.28 static artifact (`figshare` file `40344436`) | CC-BY-4.0 | research use appears compatible with attribution; exact file/checksum not yet frozen |
| Matbench Discovery code/registry | GitHub commit above | MIT | code license only; it is not substituted for the data licenses |
| Frozen predictor output | not yet selected | unresolved | **blocked** pending artifact, model provenance, license, release, and checksum |
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
| 1. Data and license | explicit license authority, external-path enforcement, SHA-256, four required artifact roles | pass | **blocked:** exact predictor/structure artifacts and all local checksums absent |
| 2. MP initial causal hull | composition-specific linear-program hull using only frozen MP phase records; release and phase-set checksum in `HullSnapshot` | pass | not run: MP artifact absent |
| 3. Oracle-isolated WBM reveal | policy-safe observable type has no energy/label fields; vault reveals one selected query once | pass | not run: WBM artifact absent |
| 4. Frozen prediction and SOAP cache | predictor/config/content checksum, fixed SOAP parameters/species, normalized vectors, structure-identity lookup | pass | not run: prediction and structure artifacts absent |
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

## Next permitted work

Only obtain or point to external artifacts, complete the executable manifest,
freeze the maintained SOAP/predictor implementations, and run these five gates.
Do not run uncertainty, CAL-style GP, compatible-residual, random, CAW-Joint,
or any other comparative WBM policy until the real-artifact column passes.
