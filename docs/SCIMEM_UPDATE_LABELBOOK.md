# SciMem-Update Hierarchical Labelbook

Status: Phase 1B pilot labelbook. The candidate pool is unlabeled and is not
SciMem-Update gold. Annotators label the four axes below; they never label or
see a compiled memory operation.

## SemanticRelation

| Label | Definition | Positive example | Hard negative |
|---|---|---|---|
| `EQUIVALENT` | A and B assert the same scientific proposition after harmless wording or unit normalization. | “The transition is at 90 K” vs “Tc = 90 kelvin” for the same sample and method. | Same value measured on a different composition: this is not equivalent because scope differs. |
| `COMPATIBLE_DISTINCT` | Both propositions can be true, but each contributes distinct information. | “Method M reaches 82 F1 on dataset D” vs “Method M uses a transformer encoder.” | Two same-scope values that cannot both be true are contradictory, not compatible. |
| `CONTRADICTORY` | Under the stated scope, both propositions cannot simultaneously be true. | “Treatment X reduces outcome Y” vs “Treatment X increases outcome Y” for the same population, endpoint, and time. | Opposite effects in adults and children may be different-scope, not contradictory. |
| `UNRELATED` | The records have no scientifically useful semantic relation for memory update. | A microscopy resolution result paired with an unrelated protein-risk claim. | Same method on a different benchmark is normally compatible-distinct, not unrelated. |
| `INSUFFICIENT_CONTEXT` | The visible material is inadequate to decide the semantic relation. | A says “performance improved”; B reports a number but the metric and baseline are absent. | If entities, metric, direction, and conditions are explicit, do not use this as a low-confidence escape label. |

## ScopeRelation

Scope covers the population/material, intervention or method, endpoint/property,
conditions, time, and measurement setting. It is independent of whether two
claims agree.

| Label | Definition | Positive example | Hard negative |
|---|---|---|---|
| `SAME_SCOPE` | All scope-defining dimensions needed for comparison match. | Same alloy composition, temperature, property, and measurement method. | Same material family but a different temperature is not automatically same scope. |
| `NARROWER_SCOPE` | B is a strict subset or more qualified version of A. | A covers all adults; B covers adults over 65 under the same endpoint. | A different endpoint is not merely narrower. |
| `BROADER_SCOPE` | B strictly generalizes A. | A concerns one crystal orientation; B covers all orientations of that material. | A claim about another material is different scope, not broader. |
| `DIFFERENT_SCOPE` | The scopes are comparable enough to identify a material difference, but neither contains the other. | Same method evaluated on two different datasets. | Missing population information should be unknown scope, not different scope. |
| `UNKNOWN_SCOPE` | One or more scope-defining dimensions are absent or ambiguous. | Temperature is required to compare two conductivity values, but one record omits it. | Do not infer a missing temperature from domain convention. |

## AuthorityRelation

Authority is based only on visible provenance: verified correction/retraction
evidence, curator decision, deterministic verification, source policy, and
timestamp. Citation count, fluent wording, or a newer publication date alone is
not authority.

| Label | Definition | Positive example | Hard negative |
|---|---|---|---|
| `NEWER_MORE_AUTHORITATIVE` | B is later and has explicit, stronger authority for the proposition. | A verified publisher correction with claim-level evidence replaces an earlier value. | A newer ordinary paper disagreeing with an older paper is not automatically more authoritative. |
| `OLDER_MORE_AUTHORITATIVE` | A is older but has explicit stronger authority than B. | A curated reference measurement vs a later unverified preprint claim. | Older does not mean authoritative by itself. |
| `EQUAL_AUTHORITY` | The visible provenance places both records at the same authority tier. | Two independently verified records under the same policy and verifier tier. | Equal publication venue is insufficient if one certificate is rejected. |
| `UNRESOLVED` | Evidence suggests an authority comparison matters, but the visible provenance cannot resolve it. | Two verified studies disagree and neither is a correction or curator ruling. | Use `NOT_APPLICABLE` when authority has no bearing on an unrelated pair. |
| `NOT_APPLICABLE` | Authority comparison is irrelevant to this pair. | Two unrelated claims or compatible facts with no replacement question. | Same-scope contradictory claims require an authority assessment, even if it ends unresolved. |

## EvidenceSufficiency

| Label | Definition | Positive example | Hard negative |
|---|---|---|---|
| `SUFFICIENT` | The visible evidence and provenance support all four requested labels without inventing fields. | Both claims have aligned evidence, matching entity/scope details, and certificate provenance. | A document-level retraction flag alone is not sufficient for claim-level replacement. |
| `PARTIAL` | Some labels are supportable, but a material comparison dimension or provenance fact is incomplete. | The semantic direction is clear, but the measurement temperature is absent for one record. | If the relation itself cannot be decided, use insufficient rather than partial. |
| `INSUFFICIENT` | The pair cannot be responsibly labeled from the visible evidence. | One side is only a title and the claim needed for comparison is absent. | Sparse wording can still be sufficient when all required entities, relation, scope, and provenance are explicit. |

## Relation Is Not Scope

Semantic relation answers “can these propositions both be true?” Scope relation
answers “do they refer to the same scientific setting?” A pair can be lexically
opposite but `COMPATIBLE_DISTINCT` + `DIFFERENT_SCOPE`; it can also be
`EQUIVALENT` in meaning while `UNKNOWN_SCOPE` prevents safe merging.

## Conflict Is Not Supersede

A conflict is an unresolved same-scope incompatibility that should remain
visible with both lineages. Supersede is a destructive store action that marks
an older record inactive for default retrieval. Annotators do not choose either
operation. The deterministic `UpdateCompiler` may compile a destructive
supersession only when scope identity, certificate eligibility, higher authority,
sufficient evidence, newer evidence, and verified claim-level correction basis
all pass. A Crossref or Retraction Watch document status alone never satisfies
claim-level supersession.

## Missing Information

- Preserve missing material, condition, time, method, certificate, and evidence
  fields as missing; never infer them from nearby records or domain convention.
- Use `UNKNOWN_SCOPE` when missing information blocks the scope comparison.
- Use `INSUFFICIENT_CONTEXT` when it blocks the semantic comparison.
- Use `PARTIAL` when some axes are defensible but at least one required fact is
  missing; use `INSUFFICIENT` when responsible annotation is impossible.
- Record the missing dimension in the evidence note.
