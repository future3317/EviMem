# E53 Objective-Isolation Design

## Goal

Isolate the empirical contribution of complete-pool adjudication from the
choice of probability scoring, then update the manuscript around the resulting
matched comparison without overstating the status of previously opened data.

## Scientific question

The existing headline comparison changes two ingredients at once:
posterior-mean target margin versus posterior complete-pool membership
probability. E53 adds one matched policy that changes only the adjudicator:

- **Local-prob:** posterior probability that a legal candidate is stable on
  the current history-consistent hull, excluding all still-unqueried
  competitors.
- **Delta-Hull:** posterior probability that the same candidate is stable on
  the completed visible-pool hull.

Both policies use the same conditioned energy posterior, common Sobol Gaussian
worlds, posterior sample count, hull tolerance/backend, immutable-ID tie
breaking, and legal reveal history. Their only scientific difference is the
set of competitors used for adjudication.

We do not add a full 2x2 factorial. A "complete-pool mean margin" has no unique
decision-theoretic definition and would introduce an artificial policy rather
than isolate an existing ambiguity.

## Experiment identity and data boundaries

E53 is a new experiment identity. It does not overwrite or pool outputs from
E32 or E52.

### E53-A: development objective isolation

Run one `B=6` trajectory per cross-fitting fold on the frozen 230-system E52
development universe at the 100% visible query pool. The policy roster is:

1. posterior-mean target margin;
2. matched local-hull membership probability; and
3. Delta-Hull complete-pool membership probability.

Each fold fits the transport posterior only on the other 184 development
systems. Prefixes of each completed trajectory define `B=1..6`; no independent
per-budget reruns are used. The predeclared identification contrast is
Delta-Hull minus Local-prob. Delta-Hull minus target margin remains a secondary
continuity comparison with E52.

### E53-B: secondary held-out MatPES rerun

After E53-A code, policies, numerical settings, estimands, and inference are
frozen, fit the transport model once on all 230 development systems and run the
same three-policy `B=6` protocol once on the complementary 94 systems. Derive
`B=1..6` by prefixes.

The 94-system panel is not untouched or external: 46 systems previously
appeared in other-fold posterior fits and 48 were opened in E22. E53-B is
reported only as a **secondary held-out MatPES rerun under the frozen current
protocol**. Its target outcomes may not be used to alter methods, thresholds,
endpoint definitions, figures, or statistical procedures.

## Matched local-probability policy

Add a new canonical policy identity instead of changing the historical
`posterior_current_hull_probability` implementation. At each decision state:

1. condition the shared target-energy posterior on all legal reveals;
2. draw the same deterministic Sobol Gaussian worlds used by Delta-Hull;
3. for every legal candidate and sampled self-energy, evaluate stability
   against the fixed current competing hull built from references and revealed
   candidates only;
4. score by the sample mean of this local stability indicator; and
5. select by immutable pair-ID lexicographic argmax.

The policy logs candidate probabilities before reveal. Oracle energies and
complete-pool labels remain inaccessible to policy code.

## Statistical analysis

The exact chemical system is the analysis unit. Report, for each policy and
budget, absolute mean terminal confirmations `T`. Report paired differences
for Delta-Hull minus Local-prob and Delta-Hull minus target margin.

Use one paired sign-randomization procedure for both testing and interval
estimation. The two-sided p-value tests a zero mean paired effect. The 95%
confidence interval is obtained by inverting the same sign-randomization test
over an additive paired-effect shift. Use deterministic seeds and enough draws
to make Monte Carlo resolution finer than manuscript precision. Report
win/tie/loss counts as descriptive diagnostics.

The development analysis reports the complete `B=1..6` curve and its
predeclared integrated budget effect. E53-B reports the same frozen summaries;
no endpoint is selected after reading its outcomes.

## Theory-linked MatPES diagnostics

Use only policy-side probability traces joined after completion. For each
Delta-Hull trajectory, measure:

- absolute membership-probability drift for candidates legal in consecutive
  states;
- preservation of the top-ranked candidate and of the full lexicographic
  ranking across actual reveal transitions; and
- the existing two-step planning-headroom distribution from matched rollout
  diagnostics.

Weight system-level summaries equally. These are observed-path diagnostics,
not counterfactual verification of the sufficient conditions. The manuscript
may state that they connect limited headroom to stable posterior rankings only
to the extent supported by the measured results.

## Manuscript restructuring

The paper repository remains `E:\PAPER\LEARNING WHAT TO REMEMBER`, current
`main`; no worktree, new repository, or remote is created. Paper edits stay
inside that repository.

The main text will be reorganized around:

1. protocol and evaluation boundary;
2. whether global adjudication changes acquisition;
3. whether lookahead adds value after objective alignment; and
4. concise mechanism and efficiency evidence.

The main policy table will expose `Adjudicator`, `Score`, and `Lookahead`.
Figure 2 will prioritize the matched Local-prob versus Delta-Hull comparison;
membership calibration moves to the appendix. The controlled POMDP, finite-
world audit, high-precision numerical audit, MAD proxy, and evaluator-only
sensitivity remain appendix evidence. The abstract removes protocol
bookkeeping and contains only the problem, formulation, theoretical boundary,
and strongest supported empirical result.

Notation will use one history symbol per context, define updated history once,
and denote posterior rollout worlds by `\tilde e` rather than evaluator energy
`e`. The rollout finite-Monte-Carlo caveat moves out of the main method.

No manuscript claim is updated until all required output hashes, policy
rosters, fit/query disjointness checks, system counts, prefix consistency,
and inference summaries pass.

## Verification and deliverables

Code verification includes focused unit tests, full relevant pytest, Ruff, and
read-only audits of every external output. Raw outputs stay outside Git.

Paper verification includes halt-on-error compilation, main-text page count,
undefined-reference/citation checks, overfull-box checks, font embedding, and
rendered-page visual inspection. Only task-related source, figure, summary
metadata, and final PDF files are committed.

## Repository constraints

- Preserve all pre-existing dirty files in `E:\CODE\EviMem-RL`; stage and
  commit only E53-related changes.
- Use `conda run --no-capture-output -n llm ...` for Python, pytest, and Ruff.
- Do not add datasets, oracle vaults, traces, or raw experiment outputs to Git.
- Work directly on the current `main` branches because the user explicitly
  authorized that workflow.
- Do not modify any other project under `E:\PAPER`.
