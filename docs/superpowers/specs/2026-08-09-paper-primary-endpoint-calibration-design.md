# Primary Endpoint and Decision-Relevant Calibration Design

## Goal

Strengthen the paper-facing statistical hierarchy without expanding the
experimental story. The revision will identify one theoretically motivated
primary scalar endpoint, report calibration at the chemical-system level and
among decision-relevant candidates, and present objective and planning evidence
as complementary analyses rather than as a direct effect-size comparison.

## Scope

This is a post-processing and manuscript revision only. It does not rerun
acquisition, fit a posterior, open the 94-system secondary MatPES panel, add a
new policy, or turn the knowledge-gradient numerical implementation into a
baseline. Existing oracle and reveal boundaries remain unchanged.

## Statistical hierarchy

The paper-facing primary objective-effect endpoint is

\[
\Delta T(B=2) = T_{\text{Delta-Hull}}-T_{\text{target margin}}
\]

on the 100% query pool. Budget two is the smallest sequential budget at which
the first observation can change a subsequent acquisition, directly matching
the paper's two-step theory. The manuscript will call this the primary scalar
endpoint, not a preregistered endpoint. The complete B=1--6 curve is secondary,
the area under that curve is a descriptive aggregate, and the 70% and 85% pool
reruns are robustness analyses. No co-primary endpoint or cross-budget
superiority claim is introduced.

## Calibration populations and weighting

The existing candidate-state pooled summaries remain available as secondary
descriptive results. The primary calibration point estimates use equal total
weight per exact chemical system. If system `s` contributes `n_s` rows, each of
its rows receives weight `1 / (S n_s)`, where `S` is the number of represented
systems. Weighted Brier score, Bernoulli NLL, ECE, ROC--AUC, average precision,
and reliability bins use these weights. Cluster bootstrap intervals resample
exact systems with replacement and treat each sampled occurrence as one
equal-weight system.

Three candidate populations are reported:

1. all legal candidates at every pre-reveal state;
2. the selected action at every state (the decision top-1 population); and
3. the selected action plus the two highest-probability remaining candidates
   at every state (the decision top-3 population).

Top-3 ties are resolved deterministically by candidate ID after sorting by
descending posterior final-hull probability. States with fewer than three legal
candidates contribute all available candidates. This diagnostic is evaluator-
side post-processing and cannot alter any action.

## Output schema and figure

`tools/summarize_matpes_membership_calibration.py` will preserve the current
pooled keys for compatibility and add, for each population:

- `equal_system_metrics`;
- `equal_system_cluster_bootstrap_95`; and
- `equal_system_reliability_bins`.

It will also add a `top3_candidates` population with the same pooled and
equal-system summaries. The schema version increments because the output gains
new public fields.

Figure 2 panel b will use equal-system reliability points and equal-system
metric callouts. Its caption will describe strong discrimination and good
aggregate calibration rather than claim universal calibration. The panel title
and terminology will consistently use `ROC--AUC`, `final-hull`, and
`membership probability`. Figure 2 remains visually dominant; Figure 3 is
slightly reduced so the page reads as primary objective evidence followed by a
compact planning diagnostic.

## Manuscript narrative

The first occurrence of the execution is written as "a fresh rerun under a
frozen, outcome-independent protocol," with "fresh rerun" used thereafter.
The main experiment section states the B=2 hierarchy before reporting its
estimate and treats the budget curve, AUC, and pool shifts as secondary or
robustness evidence.

The planning paragraph starts by identifying its evidence as a complementary,
separate matched-posterior development execution. It does not numerically test
or rhetorically compare `+0.104` against `+0.039`; the supported synthesis is
that the fresh objective rerun confirms low-budget objective alignment while a
complementary analysis finds limited incremental terminal value from the tested
lookahead. This boundary is stated once, positively, rather than repeated as a
defensive disclaimer.

Related Work explicitly distinguishes the contribution from known nonmyopic
search: ENS and knowledge-gradient methods value future acquisitions under
conventional search labels, whereas this paper separates immediate query
observations from globally adjudicated selected-item utility. `lexargmax` is
defined once as deterministic maximization with immutable-ID tie-breaking.

## Testing and acceptance criteria

Implementation follows test-first development. Unit tests must fail before the
new weighting and top-3 behavior exist, then pass after implementation. Tests
cover equal-system weighting under unequal pool sizes, weighted reliability,
top-3 construction and deterministic ties, cluster-bootstrap schema, and
renderer use of equal-system fields.

The externally stored E52 summary is regenerated from the existing five 100%
pool B=6 trajectories without overwriting the prior summary. Input hashes and
record/state/system counts must match the existing audit. The final manuscript
must compile with halt-on-error, have no undefined references or overfull boxes,
keep main text at or below nine pages, embed all fonts, and pass visual inspection
of the abstract, Figures 2--3, the objective/solver prose, the calibration table,
and limitations. Only task-specific code, documentation, manuscript source,
referenced figures, and the compiled PDF are committed.
