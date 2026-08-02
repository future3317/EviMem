# Exact state-feedback separation protocol

Status: registered synthetic necessity audit.  This document defines an exact,
observable finite-pool construction.  It is not a MAD, MatPES, WBM, or
deployment result, and it does not reopen any empirical panel.

## Question and policy classes

The audit asks a deliberately narrow question:

> Can a bounded policy state make a retained observation necessary for a later
> acquisition action under delayed full-pool adjudication, compared with a
> nonadaptive acquisition rule?

The comparator is called **nonadaptive acquisition**, not a universal
``decoupled'' class. Its exact restriction is that a later acquisition action
cannot read either a witness outcome or the post-retention policy state. On the
registered symmetric instance its observable decision inputs are only the
round and remaining actions, so each deterministic comparator is a fixed
three-query sequence. It may optimize that sequence globally; the restriction
is only the absence of state feedback.

The state-feedback policy observes exactly the same witness outcome at the
same reveal time. It writes that outcome into its one policy-memory slot and
may read the retained token at the next acquisition decision. Thus the two
classes differ in use of an observed result, not oracle access, action set,
budget, world prior, or tie break. A more permissive policy that can read the
retained token before its next query can implement the same mapping and reaches
`2`; it is deliberately outside the comparator. Consequently, this audit is
an exact adaptivity/state-feedback separation, **not** a theorem about every
colloquial ``decoupled'' acquisition--retention policy or about joint retention
competition.

The timing is fixed:

```text
public state (round, remaining actions) → choose query → reveal outcome
→ append immutable archive → choose retained token → next policy state
→ terminal delayed-full-pool utility.
```

The exact comparison contains:

- the optimal state-feedback policy with policy-memory capacity `K=1`;
- the best nonadaptive sequence, enumerated exhaustively;
- a terminal-myopic policy with the same deterministic tie break;
- the full-history policy; and
- the optimal joint policy with `K=0`.

## Finite delayed-hull abstraction

There are three equally likely complete oracle worlds, `w=0,1,2`, and query
budget `B=3`.  The visible pool contains one non-hull protocol-compatible
calibration witness `p`, one always-finally-stable candidate `s`, and two
candidate members for each world, `g[w,0]` and `g[w,1]`.

- Querying `p` reveals the world index but is never a final-hull member.
- `s` is a final-hull member in every world.
- `g[w,j]` is a final-hull member exactly in world `w`.
- A selected `s` or `g[w,j]` is stable on the selected-history hull.  The
  omitted world-specific competing phase can later invalidate the wrong
  `g[w,j]`, so selected-history and complete-pool labels differ by design.

The witness is a toy protocol-compatible energy observation correlated with
the omitted competitor; it is not an online final label.  This gives a
minimal delayed-adjudication POMDP rather than a claim about a particular
chemical system.

With `K=1`, the state-feedback policy queries `p`, retains its world code across
the two remaining rounds, then queries the two corresponding `g` candidates.
Its expected terminal confirmation count is therefore `2`. A nonadaptive
sequence cannot condition either later action on that code. Exhaustive
enumeration is registered to establish its optimum rather than assuming a
particular baseline. The expected strict gap is `2 - 5/3 = 1/3` terminal
confirmations.

## Registered null and safety checks

The implementation must exactly verify all of the following.

1. `K=0` removes the state-feedback advantage and matches the best
   nonadaptive value.
2. If `K >= B`, joint and full-history values agree.
3. If archive/state access has zero cost, state capacity is non-binding and
   the policy reduces to full history.
4. If the witness is uninformative about the future boundary, the joint policy
   reverts to the best nonadaptive value.
5. If the witness protocol is unsupported, it is excluded from the legal
   action set and the policy abstains/reverts to the same baseline value.

Every policy report must include terminal confirmations, final-causal
confirmations, false-stable and false-unstable counts, action paths, and
memory turnovers.  The terminal objective is primary; selected-history
confirmation is deliberately reported separately rather than relabeled as
the final-hull outcome.

## Claim boundary

Passing this audit establishes only an existential state-feedback separation
against the declared nonadaptive class. The construction has one informative
witness and therefore does not test choice among competing retained outcomes.
It does not establish that DBBM is empirically superior, that the synthetic
witness has a real materials counterpart, or that IC-SARR gains final-causal
or cost-aware superiority.
