# Facet-calibrated Safe Hull-ENS: registered protocol v1

**Status:** registered new method-development line, 2026-08-04. Raw outputs
remain outside Git. This line is explicitly authorized by the user after the
E32-A audit; it does not overwrite or extend E32-A, P0-v3, or MAD outputs.

## Scientific question

Does a one-fantasy, full-remaining-budget approximation to the delayed
complete-pool objective improve directly over repeated Delta-Hull, and does a
Delta-Hull-relative independent screen reduce action changes that are not
supported by posterior value?

This changes the finite-horizon solver, not the task definition. The source and
target protocols, candidate pool, reference phases, reveal boundary, posterior
family, immutable archive, seeds of the retained experiments, and
`fixed_composition` hull backend remain separate and auditable. The new line
uses a new method identity and seed.

## Policies

The first P0 roster is deliberately small so that the first result is
interpretable:

1. `source_margin` — historical source-derived baseline;
2. `delta_hull_active_search` — repeated one-step complete-pool greedy
   baseline;
3. `hull_ens` — delayed-label Expected Next-best Search (Hull-ENS);
4. `safe_hull_ens` — Hull-ENS with a Delta-Hull-relative independent screen.

The posterior and all four policies receive the same observable state and
common task/fold/budget inputs. No policy reads an unrevealed target energy or
label.

## Hull-ENS objective

For remaining budget `b`, candidate `x`, and current posterior state `h`,

\[
Q_b^{\mathrm{HENS}}(x\mid h)=p_h(x)+
\mathbb E_{E_x\mid h}\left[\sum_{j=1}^{b-1}
p^{(j)}_{h,x,E_x}\right],
\]

where the `p^(j)` are the largest conditional complete-pool membership
probabilities among the remaining candidates after one fantasy observation.
The implementation uses common posterior worlds, Rao--Blackwellized
membership probabilities for each conditional branch, and one conditional
Gaussian draw per future batch rank. It does not simulate future action paths.

At `b=1`, the future sum is empty and the action is required to have exact
Delta-Hull parity. At `b=2`, the population objective is the exact two-step
Bayes value; finite posterior/fantasy sampling is reported as numerical
approximation. For `b>2`, the top-`b-1` conditional batch is an explicit
approximation, not an exact Bellman solution.

## Delta-relative safe gate

Let `x_Delta` be the Delta-Hull action. The safe policy forms

\[
\widehat\Delta_h(x)=\widehat Q_b^{\mathrm{HENS}}(x)-
\widehat Q_b^{\mathrm{HENS}}(x_\Delta)
\]

on an independent iid posterior-world stream. The certificate set is the
union of the top-`K` posterior membership candidates and top-`K` posterior
information candidates, with `K=8` capped by the legal pool and always
including `x_Delta`. With `M` independent worlds and failure probability
`delta=0.05`, the registered radius is

\[
r_M=b\sqrt{2\log(2|\mathcal C_h|/\delta)/M}.
\]

The policy may select the estimated best candidate only if its paired lower
bound over Delta-Hull is positive. Otherwise it returns `x_Delta`. The gate is
a posterior-value numerical safeguard; it is not a model-calibration,
oracle-correctness, deployment, or causal guarantee.

## Numerical settings

- method seed: `20270804`;
- posterior sample count: `128` for the initial P0 run;
- fantasy sample count: `8` (capped in the worker at eight);
- independent certificate stream: `128` iid Gaussian worlds;
- gate failure probability: `0.05`;
- budgets: `B=1..6`;
- five outcome-independent cross-fit folds;
- minimum legal candidate count: `12`;
- hull backend: `fixed_composition`;
- transport family: `hierarchical_matern52_frozen_structure`;
- no unequal-cost ratio heuristic is permitted.

Each `(fold, budget)` unit has an independent output file. The scheduler sets
BLAS/OpenMP/NumExpr threads to one per unit and parallelizes only independent
units. This is a runtime optimization; it does not change posterior draws,
policy semantics, or the reveal protocol.

## Required checks and reporting

Before any manuscript integration, the complete summary must verify:

- all `5 x 6` output files exist and contain the registered roster;
- no failure marker exists and each output is complete;
- task, vault, cross-fit, runner, worker, acquisition, registry, and protocol
  hashes match the executor identity;
- 230 exact development systems are covered at each budget;
- B=1 Hull-ENS action parity with Delta-Hull;
- direct paired B=1..6 contrasts for `T`, `F`, `D`, wall time, and W/T/L;
- safe-gate selection and fallback rates;
- action disagreement and value gain relative to Delta-Hull;
- certificate radii and candidate-set sizes;
- no raw output is copied into Git.

Only a complete, checked result can update the paper. If Hull-ENS or the safe
gate does not beat Delta-Hull on the opened development roster, the result is
reported as a negative solver test and the paper remains a delayed-objective /
greedy-sufficiency mechanism paper. No claim of formation-energy holdout,
final-causal superiority, cost-aware superiority, deployment benefit, or
universal superiority is authorized by this protocol.

## Deferred facet calibration

Facet-calibrated posterior fitting is a separate P1 identity. It must use
training-fold outcomes only, preserve the frozen posterior's all-outcome
archive semantics, predeclare active-facet construction, and be compared to
the same posterior with only the calibration component changed. It is not
combined with the first Hull-ENS P0 result, so any change can be attributed.
