# Review traceability: DUALMEM to certified active witnesses

This record maps `审查意见.md` to the corrected mathematics, implementation,
tests, experiments, and paper text. It is intentionally outcome-changing.

| Review issue | Mathematical correction | Code | Failure-capable evidence | Paper outcome |
|---|---|---|---|---|
| `K` was presented as deletion/storage capacity | Add immutable archive `A_t`; define `M_t ⊆ A_t`, `|M_t|≤K` as certified active witnesses | `cards.py`, `active.py`, `baselines.py` | archive grows while active set stays ≤K; K=0 works | Reframed throughout; on-demand archive top-K added as a strong counter-baseline |
| Queried-item self-removal in information gain | Compare both potentials on `V_t^x=U_t\{x}` and apply explicit hypothetical hull transition | `acquisition.py` | `test_information_value_excludes_queried_item_removal` requires exact zero | Eq. common-pool replaces old Eq. 11 |
| Memory-dependent boundary weights | Freeze `ω_t(x)` from base query and causal hull, clipped by `ω_0` | `boundary.py` | working sets shift centers but weights remain identical | Potential and proposition rewritten |
| Adaptive selection of the most convenient interval | Intersect prior and all compatible witness intervals | `boundary.py` | intersection narrows; empty intersection conflicts and abstains | No ordinary conformal or adaptive-selection guarantee claimed |
| Weak/vacuous regret proposition | State only a conditional fixed-weight error bound on simultaneous event `E_t` | `boundary.py` and offline evaluator separation | tests verify online code never receives hidden oracle outcomes | Cumulative regret proposition deleted |
| Interval treated as a probability distribution | Rename semantics to two-scenario heuristic; add `stable_score_kind` audit field | `acquisition.py`, `active.py` | step records distinguish model probability, fixed prior, and scenario weight | No calibrated-expectation language remains |
| Variable costs conflated with round count | Define `T_B`; filter unaffordable candidates; report call regret and cost regret separately | `cards.py`, `active.py` | budget cannot overspend and stops when no candidate is affordable | Metrics and formulas separated |
| Causal and final hull discoveries conflated | Record query-time status, final-hull confirmation, and provisional invalidation | `active.py` | constructed run produces 2 causal, 1 final, 1 invalidated | Three metrics named explicitly |
| Complexity understated | Use exact brute-force retention reference; state `O(n²K²)` | `boundary.py`, `acquisition.py` | random small instances match manual exhaustive search | Old `O(n²K)` claim removed; measured runtime reported |
| First-round-only oracle isolation | Require identical next actions whenever revealed histories agree | type-separated evaluator and deterministic IDs | hidden unselected outcomes changed after round 1 without changing actions | Leakage section updated |
| Input-order/artificial stress risk | Deterministic ID tie-breaking, phase diagram, exact binary DP, order audit | `exact_dp.py`; three new tools | 105 permuted rerun comparisons: zero mismatches | Positive claim restricted to a narrow mechanism region |
| Missing CAL/MADE/WBM positioning | Add CAL as direct hull-aware acquisition; WBM fixed-pool and MADE closed-loop gates | no data adapter added before license/identity audit | explicitly not executed | Related work and real-data gate expanded |
| DUALMEM naming conflict | Rename paper method to `CAW-Joint` (certified active witnesses) | docs and paper-facing names | n/a | Title and abstract no longer use DUALMEM |

## Corrected experiment decision

- Mechanism GO: joint beats its matched decoupled control at `(B,K)=(5,1)`,
  `(5,2)`, and `(10,2)` in the designed retention-competition environment.
- Long-horizon warning: exact binary DP at `(4,2)` is 2.3125 expected
  discoveries versus 2.2500 for one-step joint.
- Method-level NO-GO: uncertainty+FIFO obtains 10.0 discoveries at `(10,2)`
  versus joint 9.0 in the stress test; joint also loses to random on IID residuals.
- Materials NO-GO: no WBM, MADE, CAL, or real-DFT evaluation has been run.
