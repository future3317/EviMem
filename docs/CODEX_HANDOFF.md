# Codex handoff: EviMem-RL

**Status date: 2026-08-09.** Start with `AGENTS.md`,
`docs/CURRENT_PAPER_EXPERIMENT_CONTRACT.md`, and
`docs/PROTOCOL_INDEX.md`. Consult `docs/EXPERIMENT_LEDGER.md` for provenance,
opened-data boundaries, and the disposition of an older experiment identity.

## Current paper

The paper studies active search with globally adjudicated utility. A materials
query reveals a target-protocol energy immediately, while final utility is the
complete-pool hull membership of queried candidates. Its argument is:

1. objective alignment matters because provisional and final labels differ;
2. repeated greedy can fail, but rank stability and weak coupling explain when
   it is sufficient; and
3. on MatPES, the tested lookahead methods change many actions but add little
   terminal utility beyond objective-aligned Delta-Hull.

Delta-Hull is deliberately simple and should not be sold as a complicated new
algorithm. The main empirical ladder is posterior target margin, Delta-Hull,
and Delta-Hull-anchored lookahead. SARR, IC-SARR, Hull-ENS, and selective gates
are appendix or historical diagnostics.

## Active work: E52

- Add `protocol_hull_knowledge_gradient` as a matched two-step final-label
  baseline under the same posterior and evaluator.
- Record pre-reveal candidate-level final-hull membership probabilities and
  evaluate reliability, Brier score, clipped NLL, and ranking after traces are
  complete.
- Build outcome-independent nested 70/85/100% candidate pools and rerun both
  posterior and acquisition on each pool.
- Do not claim external formation-energy validation. The 2026-08-09 raw
  MatPES rebuild exactly duplicated all 324 systems and 10,236 pair IDs in the
  opened task, so it provides no external roster.

Instrumentation is currently modified locally in:

- `src/matmem/protocol_policy_worker.py`;
- `tools/run_matpes_protocol_closed_loop_exploratory.py`.

The policy diagnostics expose candidate IDs and predicted membership
probabilities before reveal; the evaluator adds complete-pool labels only
after trace completion. Targeted tests previously passed 50 tests and Ruff.
Revalidate after the E52 pool builder and summarizer are added.

## Scientific boundaries

- The oracle vault and evaluator are the only sources of target outcomes and
  final labels. Policy-facing code never receives them.
- Every reveal is append-only and conditions the posterior.
- Existing MatPES systems are development evidence; repartitioning them cannot
  create an external holdout.
- MAD-1.5 is an atomization-energy hull-proxy protocol-shift stress test, not a
  formation-energy benchmark.
- Raw data, vaults, traces, checkpoints, and summaries remain outside Git.

## Locations and environments

| Resource | Location / command |
|---|---|
| Local code | `E:\CODE\EviMem-RL` |
| Local paper | `E:\PAPER\LEARNING WHAT TO REMEMBER` |
| Remote host | `ssh lrh@100.110.148.20` |
| Remote code | `/home/workspace/lrh/EviMem-RL` |
| Remote data | `/home/workspace/lrh/DATA/EviMem-RL` |
| Local Python/tests | `conda run --no-capture-output -n llm ...` |
| Remote experiment Python | `/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python` |

## Immediate sequence

1. implement and test the deterministic pool-subset builder;
2. implement the system-clustered calibration summarizer;
3. run one-system parity smokes for target margin, Delta-Hull, and knowledge
   gradient at each pool fraction;
4. freeze output identity, policy roster, seeds, budgets, and hashes;
5. run E52-A--C remotely and audit counts/failures before manuscript use.

Do not resume an incomplete historical root or fold new outputs into E32,
E49, E50, or E51.
