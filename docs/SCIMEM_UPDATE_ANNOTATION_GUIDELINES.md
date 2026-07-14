# SciMem-Update Annotation Guidelines

## Purpose and Non-goals

This protocol produces human judgments for a 300–400 pair pilot. It does not
create automatic update gold, train the update manager, or authorize publication.
The four annotations are `SemanticRelation`, `ScopeRelation`,
`AuthorityRelation`, and `EvidenceSufficiency`, defined in
`SCIMEM_UPDATE_LABELBOOK.md`.

The annotation task must not display a compiled memory operation. Annotators
must not enter `ADD`, `MERGE`, `LINK`, `CONFLICT`, `SUPERSEDE`, or `IGNORE` in
notes. Operations are generated later and only by the deterministic
`UpdateCompiler` after certificate and store-state checks.

## Annotation Unit

One task contains two bounded claim/evidence records and immutable provenance:

- claim text or a structured claim rendering;
- source document identifier;
- exact evidence locator and checksum;
- timestamp when available;
- source-level update metadata when present.

Sampling strata are balancing hints, not labels. SciFact SUPPORT/CONTRADICT
annotations are not publication, admission, conflict, or supersession labels.
SciREX relations are retrieval supervision and are not update gold.

## Procedure

1. Verify that each displayed claim is supported by its stated locator. If the
   locator is absent or unusable, mark evidence insufficient and record why.
2. Resolve entities without adding unstated aliases. If entity identity remains
   uncertain, use the appropriate insufficient/unknown label.
3. Assign semantic relation using only the visible propositions.
4. Assign scope relation separately across material/population, property or
   endpoint, conditions, time, and method.
5. Compare authority only from explicit certificate, curator, correction, and
   source provenance. Newer date alone is not higher authority.
6. Assign evidence sufficiency last, based on whether steps 1–5 were supported.
7. Add a concise evidence note. Quote only the smallest necessary excerpt.
8. Submit no memory operation.

## Source-level Corrections and Retractions

Crossref `update-to` records and Retraction Watch metadata establish a
document-level status change. They do not identify which scientific claim is
corrected, whether a particular claim is withdrawn, or which replacement claim
is valid. For these tasks:

- retain API source, response checksum, update type, timestamp, and DOI relation;
- label authority conservatively from the visible metadata;
- treat claim-level effect as `awaiting_human_evidence_annotation`;
- never infer claim-level supersession from a document retraction;
- request the relevant correction notice/evidence during adjudication if a
  claim-level decision is needed.

## Conflict and Supersession

Annotators identify semantic contradiction and scope, not operations. A genuine
conflict candidate requires a contradiction under the same scope with sufficient
evidence. A supersession would additionally require an eligible new certificate,
clearly stronger and newer authority, and verified claim-level correction,
retraction, or curator-correction evidence. Since those are compiler/store checks,
the annotation UI intentionally contains no supersession control.

## Disagreement Reason Taxonomy

Use one or more reasons when requesting adjudication:

| Code | Use when |
|---|---|
| `AMBIGUOUS_CLAIM_BOUNDARY` | Annotators selected different proposition boundaries. |
| `ENTITY_RESOLUTION` | Entity identity or aliasing caused the difference. |
| `RELATION_INTERPRETATION` | Direction, modality, negation, or relation semantics differ. |
| `SCOPE_GRANULARITY` | Population/material/condition/time/method containment differs. |
| `TEMPORAL_ORDER` | Relevant event or observation ordering is unclear. |
| `AUTHORITY_UNCLEAR` | Provenance tiers or correction authority cannot be compared. |
| `EVIDENCE_MISSING` | A required evidence span or provenance record is absent. |
| `SOURCE_STATUS_ONLY` | Only document-level correction/retraction metadata is present. |
| `ANNOTATION_ERROR` | A clear instruction or transcription mistake occurred. |

Adjudicators preserve both original submissions, record the selected labels and
reason codes, and may return the pair for more evidence. They do not manufacture
missing context.

## Quality Control

- Double-annotate all pilot pairs; target agreement is reported separately for
  each axis rather than collapsed into an operation.
- Blind annotators to one another and to native dataset labels/sampling strata.
- Include hard-negative calibration examples from the labelbook before live work.
- Audit at least 10% of aligned evidence locators and all source-level update
  tasks.
- Reject exports containing a compiled operation field.
- Do not call the adjudicated set gold until coverage, agreement, evidence
  alignment, and license gates pass.

## Label Studio Import and Export

Use the standard Label Studio project configuration at
`annotation/label_studio_config.xml`. Import
`annotation/scimem_update_pilot_unlabeled.jsonl` as JSON tasks. Every line has
the standard shape `{ "id": ..., "data": ..., "meta": ... }`.

Export completed annotations as Label Studio JSON. Preserve task `id`, `data`,
`meta`, annotator identity, timestamps, and all four choice results. Do not add a
post-processing field for a memory operation to the annotation export. A later,
versioned compiler job consumes validated hierarchical labels together with
certificates and memory-store state.

Argilla may be used instead if the same four categorical fields, evidence note,
adjudication metadata, task IDs, and provenance are preserved. No custom
annotation front end is permitted for Phase 1B.
