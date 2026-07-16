# DeepSeek V4 Pro SciFact holdout audit

## Scope and decision

This is an internal, visible-text quality audit of blind-primary API candidates.
It is not human review, gold-label creation, a training-data acceptance test, or
a paper result. Every API call used `deepseek-v4-pro` with thinking disabled;
the API saw only an external-safe packet. The cited runtime directories are
ignored and contain no repository dataset artifact.

The decision is **do not accept either holdout output as SciMem-Update labels**.
Both remain `not_gold` review-queue artifacts.

## Fresh holdouts

| Protocol | Fresh SciFact packets | Validated at run time | Unannotated fail-closed | API tokens | Assessment |
|---|---:|---:|---:|---:|---|
| V3 evidence-bound scope | 20 | 18 | 2 | 18,296 | Rejected: directional scope and unsupported entity-link errors remain frequent. |
| V4 claim-link gates | 20 | 18 | 2 | 22,842 | Rejected: improved authority handling, but visible-text entity/scope decisions remain unreliable. |

Each packet set excluded all earlier development, V2, and prior-holdout packet
IDs. Ledger verification passed for both runs. Failed responses were left
unannotated; no rule-derived label was written.

## V3 visible-text findings

The V3 audit found repeated failure modes, including reversed directional scope
(`scifact:0004`, `0010`, `0064`, `0066`), invented entity/alias connections
(`0023`, `0058`, `0103`), unsupported programme identity (`0034`), and a
broad-claim versus subset-result pair incorrectly called same scope (`0141`).
Several same-scope contradictions also used `EQUAL_AUTHORITY` despite no
packet-visible claim-level authority evidence. These are protocol violations,
not disagreements that can be resolved by majority voting.

## V4 visible-text findings

V4 corrected the authority output for its same-scope contradiction candidates,
but it still made high-impact visible-text errors. Representative categories
were:

- unsupported identity or causal connection: `0035`, `0057`, `0071`;
- incorrect strict-subset direction: `0014`, `0024`, `0156`;
- broad claim versus a specific subgroup/result treated as same scope: `0087`,
  `0110`, `0114`, `0115`;
- endpoint conflation: `0155`.

Six candidates appeared labelbook-consistent under the strict visible-text
review (`0038`, `0043`, `0135`, `0137`, `0146`, `0147`). This is only 6 of 18
emitted candidates (33%); it is far below the threshold for accepting a
model-only annotation set. The remaining records are either concrete errors or
too ambiguous to treat as accepted without targeted expert review.

After this audit, the canonical validator was tightened further: an external
safe packet may use `UNRESOLVED` authority only for a same-scope contradiction;
otherwise it must use `NOT_APPLICABLE`. Under that new gate, the recorded V4
outputs for `0105` and `0155` are also rejected. The historical runtime files
were not altered.

## Consequence

The API runner, provenance ledger, safe-packet projection, and fail-closed
validation are working as intended. The model's scientific relation judgement
is not reliable enough to replace expert annotation. Further work should focus
on reducing the review workload through deterministic rejection/routing and a
small targeted expert audit, not on converting agreement or API output into
silver/gold training labels.
