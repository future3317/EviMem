# Exact state-feedback theorem attack audit

Status: completed 2026-07-30. This is a formal-scope audit of E44, not a new
materials experiment and not evidence for an empirical bounded-memory claim.

## Verdict

The original numerical claim, `2 > 5/3`, is correct, but the former label
``action-state-decoupled'' was too easy to read as a claim about a natural or
comprehensive class of separately designed acquisition--retention policies.
That broad reading is false. A policy that sees the retained witness before its
second query can reproduce the state-feedback policy and obtain `2`, whether or
not its retention update was implemented in a separate module.

E44 is therefore narrowed and renamed: it is an **exact state-feedback versus
nonadaptive acquisition separation**. It establishes that, in the registered
delayed-full-pool instance, feedback from one retained revealed outcome can be
strictly valuable. It does not establish that joint optimization of competing
retention choices is necessary.

## Formal timing and observability

For each round, both policy classes share the same public state: the round,
the unqueried action set, the known pool, the world prior, and all registered
tie breaks. They share the same legal query actions, budget `B=3`, and reveal
kernel. The only random variable is an equally likely world `w in {0,1,2}`.

```text
public state -> choose query -> reveal outcome -> immutable archive append
             -> choose retained token -> next policy state -> terminal utility
```

- Querying witness `p` reveals `w` to both classes at the reveal event.
- The state-feedback policy stores `w` in its single policy-memory slot and
  reads it when choosing later queries.
- A nonadaptive acquisition policy cannot use the reveal or retained token in a
  later query. On this symmetric instance it is exactly an ordered sequence of
  three distinct actions. It may globally optimize that sequence.
- Retention after a reveal may be chosen optimally by the state-feedback policy.
  The comparator has no subsequent acquisition at which a retained token can
  matter; allowing it to retain the token does not change its action sequence.

Thus the joint policy receives no additional oracle information. The separation
comes entirely from state feedback, which the comparator explicitly forbids.

## Exhaustiveness and upper bound

There are eight legal actions and three rounds, hence `8*7*6 = 336` ordered,
deterministic nonadaptive sequences. The implementation enumerates all of them
in `enumerate_nonadaptive_sequence_values`; its test asserts that every value
is at most `5/3` and that the maximum is exactly `5/3`.

Algebraically, for any such sequence let `n_s` be the safe query count, `n_p`
the witness count, and `n_w` counts for the three world groups. Its expected
terminal utility is

`n_s + (n_0+n_1+n_2)/3 = n_s + (3-n_s-n_p)/3 <= 5/3`.

The bound is tight for `safe` plus two candidates from one fixed group. A
randomized nonadaptive policy is a probability mixture over these deterministic
sequences. Since expected utility is linear in the mixture weights, it cannot
exceed `5/3`. Tie-breaking picks a representative maximizer only and cannot
affect this bound.

The state-feedback Bellman solver has value `2`: query `p`, retain its revealed
world code, then query the matching pair. Each branch obtains `0+1+1=2`.
The `K=0` Bellman policy also has value `5/3`; without a persistent token it
cannot turn the probe reveal into a later query choice. This is a memoryless
state-feedback null, not merely a named baseline comparison.

## Reward and delayed-label audit

The reward is symmetric across the three worlds. `p` has zero terminal reward;
`safe` has reward one in every world; each group candidate has reward one in
exactly its matching world. No action has an outcome-dependent reward hidden
from the comparator. The selected-history count is intentionally separate:
every non-probe candidate appears stable before the omitted competitor is
adjudicated, so the nonadaptive maximizer has selected-history count `3` but
only terminal utility `5/3`. The construction is therefore a stylized
delayed-full-pool label problem, not a formation-energy convex-hull model.

## Null audit

The registered exact checks pass only under their stated meanings:

| Condition | Exact value | Interpretation |
| --- | ---: | --- |
| `K=0` | `5/3` | No persistent state feedback; matches nonadaptive optimum. |
| `K>=B` | `2` | Capacity no longer limits the state-feedback policy. |
| Zero state-access cost | `2` | Full history is available; capacity is non-binding. |
| Uninformative witness | `5/3` | No informative state can be retained. |
| Unsupported witness | `5/3` | Witness is removed from legal actions; policy abstains. |

These tests do not turn the synthetic construction into deployment evidence,
and they do not test a real multi-witness retention competition.

## Submission-safe theorem statement

> There exists a three-world delayed-full-pool instance with `B=3` and `K=1`
> in which the optimal state-feedback policy has expected terminal utility `2`,
> while every randomized nonadaptive acquisition policy has utility at most
> `5/3`.

Forbidden shorthand: “joint planning is necessary against all decoupled
policies,” “the theorem proves bounded-memory superiority,” or “the theorem
demonstrates retention competition.” A real DBBM superiority claim still needs
a task with a measured binding state constraint and a frozen comparison against
strong adaptive, separately specified retention controls.
