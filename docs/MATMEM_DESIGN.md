# Certified Active Witnesses for Pool-Based Materials Discovery

## Evidence status

This is a corrected synthetic mechanism prototype, not a materials-discovery result.
The current evidence supports only:

> A certified active-witness constraint can change the preferred query sequence.

It does **not** support superiority of joint acquisition and retention. In the
corrected retention-competition stress test, joint obtains 9.0 discoveries but
uncertainty acquisition with independent FIFO obtains 10.0. WBM and MADE have
not been audited or run.

## Scientific meaning of K

Every selected oracle result is appended to an immutable audit archive `A_t`.
No DFT result is deleted. `K` bounds only the residual witnesses that have been
protocol-checked, hull-version-checked, indexed, and activated for online use:

\[
S_t=(U_t,A_{t-1},M_{t-1},H_t,b_t),\qquad
M_{t-1}\subseteq A_{t-1},\quad |M_{t-1}|\le K.
\]

The persistent working-set policy admits from `M_{t-1} ∪ {m_t}`. Reactivating
an arbitrary archived result is not assumed free. A strong on-demand archive
top-K kNN baseline grants that retrieval to test whether persistent retention
is scientifically necessary.

## Causal and final hulls

Formation energy and protocol are native observations. Energy above hull is a
derived value tied to a versioned phase set. Online ranking uses only a hull
built from already revealed phases. Evaluation reports separately:

- query-time causal discoveries;
- final-hull-confirmed discoveries;
- causal discoveries invalidated by a later phase.

The simplified synthetic hull transition lowers a same-system reference. A
real WBM/MADE adapter must recompute a composition-dependent convex hull.

## Protocol-compatible interval intersection

For a compatible witness `i` and query `x`,

\[
\mu_i(x)=\hat E^0(x)+T_{i\to x}(r_i)-H_t(x),
\]
\[
\rho_i(x)=L d(x,i)+u_{i\to x}+q_c.
\]

The prior and every compatible witness interval are intersected:

\[
I_t(x;M)=I_t^0(x)\cap
\bigcap_{i\in\mathcal C_M(x)}
[\mu_i(x)-\rho_i(x),\mu_i(x)+\rho_i(x)].
\]

An empty intersection is a calibration conflict and fails closed. The code no
longer selects whichever single interval produces the smallest risk.

## Fixed-weight boundary potential

Working sets are compared with a query-fixed weight:

\[
\omega_t(x)=\max\left\{\omega_0,
\exp\left(-\frac{|\hat E^0(x)-H_t(x)-\tau|}{\sigma}\right)\right\}.
\]

The midpoint of a nonempty intersection defines `predicted_stable`. The
asymmetric ambiguity loss is

\[
\ell_t(x;M)=\omega_t(x)
\begin{cases}
c_{FS}\mathbf 1\{\bar h_t(x;M)>\tau\},&\hat s_M(x)=1,\\
c_{FU}\mathbf 1\{\underline h_t(x;M)\le\tau\},&\hat s_M(x)=0.
\end{cases}
\]

\[
\Phi(V,H,M)=\sum_{x\in V}\ell_t(x;M).
\]

The weight may change after a causal hull update, but never merely because the
working set being compared changed.

## Conditional proposition only

Define the simultaneous coverage event over all reachable working sets:

\[
\mathcal E_t=\{h_t(x)\in I_t(x;M),\ \forall x\in U_t,
\ \forall M\in\mathcal M_t^{reachable}\}.
\]

Conditional on `E_t` and a correct causal hull, the fixed-weight asymmetric
decision error is bounded by `Phi`. The project does not claim ordinary split
conformal calibration supplies this simultaneous event. The earlier cumulative
discovery-regret proposition was removed because unrestricted information
overrides make it vacuous.

## Common-pool two-scenario acquisition

For candidate `x`, define the common residual pool `V_t^x = U_t \ {x}`. Both
sides of information gain use exactly this pool:

\[
\Delta_t^{info}(x,r)=
[\Phi(V_t^x,H_t,M_{t-1})-
\Phi(V_t^x,\mathcal T_H(H_t,x,r),M_t^*(x,r))]_+.
\]

This excludes the mechanical benefit of removing the queried item. A dedicated
test constructs an orthogonal, non-hull-changing query and requires zero
information value.

The interval supplies two boundary-straddling scenarios and geometric weights.
Those weights are a **heuristic**, not a residual probability distribution or
calibration consequence:

\[
a_t(x)=\frac{\lambda_{disc}q_x^-+
\lambda_{info}\sum_{j\in\{-,+\}}q_x^j
\Delta_t^{info}(x,r_x^j)}{C(x)}.
\]

## Costs and offline metrics

Variable query costs use

\[
T_B=\max\{T:\sum_{t=1}^{T}C(x_t)\le B\}.
\]

The evaluator filters unaffordable candidates and stops when none remain. It
reports call-count discovery regret and its cost analogue separately; neither
is used online.

## Complexity

The auditable reference retention solver enumerates every legal subset of the
streaming candidate set, including sets smaller than `K`. With at most `K+1`
candidates this requires up to `2^(K+1)-1` potential evaluations, because the
full `K+1` set is infeasible. Each potential scans `O(nK)` query-witness pairs,
so two-scenario acquisition has direct per-round runtime `O(n^2 K 2^K)`.
This exponential reference is intentional until a restricted neighborhood or
an exactly equivalent cached solver is derived. Random small instances compare
solver output with manual exhaustive search, and a constructed conflict test
requires simultaneous removal of two old witnesses.

## Falsification suite

At `B=10`, `K=2`, 30 candidates, and 10 seeds:

- retention competition: joint 9.0, matched decoupled 8.0, uncertainty+FIFO 10.0;
- recurrence: joint 9.0, compatible kNN 9.0, uncertainty+FIFO 10.0;
- IID residual: joint 4.7, random 5.8;
- nonrecurring chemistry: joint 4.0, uncertainty+FIFO 5.2;
- protocol reversal: compatible methods about 9.2, joint 9.0, unsafe reuse 4.0;
- local boundary correlation: joint 9.5, residual priority 9.6.

The five-seed `B × K` phase diagram finds a joint-vs-decoupled increment only
at `(5,1)`, `(5,2)`, and `(10,2)`; it disappears at `B=20`. Five permutations
of every synthetic pool for joint, decoupled, and uncertainty policies produce
zero action-sequence mismatches.

The exact binary diagnostic at `(B,K)=(4,2)` gives expected discoveries 2.3125
for exact joint, 2.2500 for one-step joint, and 2.2361 for decoupled. This is a
mechanism diagnostic, not materials evidence.

## Real-data gate

The next evidence step is not more handcrafted synthetic tuning:

1. audit WBM licenses, canonical identities, frozen predictions, reveal order,
   MP initial hull, and sequential hull construction;
2. evaluate CAL, uncertainty, on-demand archive top-K, decoupled, and joint on
   the same `B × K` grid;
3. run the planner/scorer in MADE against its supplied planners and MLIP ranks;
4. keep causal and final hull metrics separate;
5. require joint to beat the best independent acquisition+retention control.

Until that gate passes, the project status is mechanism GO and paper-level
joint-superiority NO-GO.
