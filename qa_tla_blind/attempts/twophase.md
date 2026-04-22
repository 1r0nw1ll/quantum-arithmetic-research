# TwoPhase — Mode B Reproduction Attempt

**Blindness pact:** Author read ONLY `prompts/twophase.md`, `mode_b/wildberger_object_model.md`, `qa_tla_blind/README.md`, `CLAUDE.md`. Has NOT read any TwoPhase / transaction_commit TLA+ file, no prior attempts, no scorer diffs, no candidates, no fidelity audit.

## 1. Object model primitives actually used

- **Points in `Z²`** (primitive §1) — used to position each RM's local state as a lattice point `(b_i, e_i)`; the integer pair is the load-bearing carrier of the RM's decision variable.
- **Red quadrance `Q_r`** (primitive §3, Lorentzian) — used as the **disagreement kernel**. `Q_r((b_i,e_i),(b_j,e_j)) = (b_i-b_j)² - (e_i-e_j)²` and the "no disagreement" safety invariant is expressed as a null-cone condition on `Q_r` between any two terminally decided RMs.
- **Blue quadrance `Q_b`** (primitive §3, Euclidean) — used for **non-trivial motion detection**: `Q_b > 0` between two RM state points certifies they are distinct; used inside the TPTypeOK type-check as the non-degeneracy witness.
- **Translations** (transform §1) — each monotonic action `RMPrepare`, `RMRcvCommitMsg`, `RMRcvAbortMsg`, `TMCommit`, `TMAbort`, `TMRcvPrepared` is encoded as a **fixed integer translation** on the RM state point `(b,e)` (or on the TM point, or on a message-pool counter). Translations preserve all three quadrances; that closure is the reason `Q_r` disagreement is preserved along legal traces.
- **4-tuple `(b, e, d, a)` with `d = b+e`, `a = b+2e`** (primitive §9) — used to reconcile A1/A2 with the 4-valued RM alphabet: `(b, e)` live in `{1,...,5}²`, `d = b+e` is the **phase projection** (working vs decided), `a = b+2e` is the **decision projection** (commit-like vs abort-like). The 4-tuple's consistency `d = b+e, a = b+2e` is what makes "terminal decision" a derived predicate rather than an independent assignment.
- **Chromogeometric Pythagorean identity `Q_b² = Q_r² + Q_g²`** (§`Three-metric…`) — used to derive the joint invariant: two RMs disagreeing on `Q_r` (null red-cone violation) can still coincide on `Q_b`, and Pythagorean identity gives the conserved tri-metric budget that distinguishes "same decision" from "disagreement."

**Not used.**
- **Cross-ratio** (§6): 2PC's safety is *pairwise* ("no two RMs disagree"), not quadruple-projective. Cross-ratio is the wrong arity.
- **Spread / spread polynomials `S_n`** (§4, §7): 2PC has no rotational substructure. States are monotone-accumulating, not periodic.
- **Hexagonal ring `T_{d+1} + b·e`** (§8): there is no parametric sum+bilinear count to decompose here (N is parametric but the invariant is pairwise, not a ring count).
- **Mutation moves** (§5): no Coxeter-Dynkin graph appears in 2PC; the action set is not a root system (see §3 below).
- **Projective maps** (§4-transforms): TCConsistent is a metric/null-cone statement, not a projective invariant; using projective maps would lose the disagreement kernel.
- **Reflections** (§2-transforms): no coordinate swap in 2PC; the action set is strictly monotone (pool is add-only), reflections would violate monotonicity.
- **TQF collinearity** (§5): considered in §4 below and rejected for TCConsistent — TQF is a *ternary* predicate, and 2PC safety is *binary*.

## 2. QA state encoding (Wildberger-native)

**Per-RM state.** Each RM has a local state ∈ `{working, prepared, committed, aborted}`. I pick **Option Y: lattice point in `Z²`** (specifically in `{1,...,5}²` to honour A1 and keep all four coordinates in `{1,...,N}`).

Coordinate assignment (chosen so that `d = b+e` separates "pre-decision" from "post-decision" and `a = b+2e` separates "commit-side" from "abort-side"):

| State       | `(b, e)` | `d = b+e` | `a = b+2e` | Interpretation                                   |
|-------------|----------|-----------|------------|--------------------------------------------------|
| `working`   | (1, 1)   | 2         | 3          | pre-decision, neutral                            |
| `prepared`  | (2, 1)   | 3         | 4          | pre-decision, commit-leaning (TM-visible)        |
| `committed` | (3, 1)   | 4         | 5          | terminal, commit-branch (a odd)                  |
| `aborted`   | (1, 3)   | 4         | 7          | terminal, abort-branch (a odd but larger)        |

Key derived signature: **`a mod 2`** is not the discriminator (all four `a` values are odd); instead, the terminal decision discriminator is **`sign(b - e)`** after the 4-tuple is evaluated:
- `committed`: `b - e = +2`  ⟹  `Q_r((3,1),(1,3)) = (3-1)² - (1-3)² = 4 - 4 = 0`  ⟹  **red null**.
- This is the load-bearing geometric fact: `committed` and `aborted` live **on the red null-cone from each other** — which is exactly the chromogeometric shape of "these two decisions are maximally opposed in red."

`Q_r(committed, committed) = 0` (same state, trivially null).
`Q_r(aborted, aborted) = 0`.
`Q_r(committed, aborted) = (3-1)² - (1-3)² = 0` — **the disagreement itself lies on the red null-cone**.

This is the geometric content: **the forbidden disagreement pattern is a non-trivial red-null pair of DECIDED states**. "Non-trivial" means: not both at the same lattice point. `Q_b(committed, aborted) = (3-1)² + (1-3)² = 8`, so `Q_b > 0` certifies distinctness; paired with `Q_r = 0`, that is exactly a genuine null-cone disagreement, not a self-pair.

**TM state.** TM has states `{init, committed, aborted}`. Same lattice, using three of the four RM points:
- TM `init` → `(1, 1)` (mirrors `working`)
- TM `committed` → `(3, 1)`
- TM `aborted` → `(1, 3)`

TM has no `prepared` analog, which is why the set has three elements, not four — the RM-only `prepared` point `(2, 1)` is excluded.

**`tmPrepared` (set of RMs seen prepared).** This is a **subset of the RM index set**; encoded as an indicator vector `(χ_1, …, χ_N) ∈ {1, 2}^N` (A1: `1` = not-yet-seen, `2` = seen), one coordinate per RM.

**Message pool `msgs`.** Encoded as three monotone integer counters `(m_P, m_C, m_A) ∈ Z_{≥1}³` with A1 shift (initial = `(1,1,1)`, each send increments by 1). Since the pool is receipt-without-consumption, counters are monotone non-decreasing. `Prepared` carries a sender so `m_P` is actually an N-vector `(m_{P,1}, …, m_{P,N})`.

**Global state.** A point in the product lattice
```
G  ∈  {1..5}² × {1..5}² × {1,2}^N × Z_{≥1}^N × Z_{≥1} × Z_{≥1}
      RM₁..RM_N   TM      tmPrepared   m_P per RM    m_C   m_A
```
(The RM-factor is `({1..5}²)^N` expanded.) Dimension scales linearly with `N`.

**A1 (No-Zero) resolution.** No coordinate can take the value `0`. The lowest lattice point used is `(1,1)`. `tmPrepared` uses `{1, 2}` not `{0, 1}`. Message counters start at `1`, not `0`. `Q_r = 0` is a **scalar invariant value** on `Z`, not a state — A1 applies to states, not to predicate outputs, so a null-cone condition `Q_r = 0` is A1-clean.

## 3. Actions as transforms (Wildberger-native)

Each action is a **translation** on one or more coordinates of `G`. Translations are from primitive-transform §1 and preserve all three quadrances.

| Action              | Translation                                                                                   | Guard (integer-only)                                 |
|---------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------|
| `RMPrepare(i)`      | `(b_i, e_i) += (1, 0)`, `m_{P,i} += 1`                                                         | `(b_i, e_i) = (1, 1)` (i.e. working)                  |
| `RMChooseToAbort(i)`| `(b_i, e_i) += (0, 2)`                                                                          | `(b_i, e_i) = (1, 1)` (i.e. working)                  |
| `TMRcvPrepared(i)`  | `χ_i := 2`                                                                                     | `TM = (1,1)` ∧ `m_{P,i} ≥ 2`                          |
| `TMCommit`          | `TM += (2, 0)`, `m_C += 1`                                                                     | `TM = (1,1)` ∧ ∀i: `χ_i = 2`                          |
| `TMAbort`           | `TM += (0, 2)`, `m_A += 1`                                                                     | `TM = (1,1)`                                          |
| `RMRcvCommitMsg(i)` | `(b_i, e_i) := (3, 1)` (equiv: `+= (3-b_i, 1-e_i)` from prepared, or from working)             | `m_C ≥ 2`                                             |
| `RMRcvAbortMsg(i)`  | `(b_i, e_i) := (1, 3)` (equiv: translation into the abort point)                               | `m_A ≥ 2`                                             |

**Note on the "receipt" actions.** `RMRcvCommitMsg` and `RMRcvAbortMsg` are technically **projections-to-a-fixed-point**, not pure translations — they zero out the "distance from terminal." They are expressible as *conditional* translations (`(b_i, e_i) += (3 - b_i, 1 - e_i)`), which is a translation whose offset depends on the current point. **This is a mild primitive strain:** pure translations under §1 are constant-offset. I'm using the slightly broader "integer-offset-by-guard" form. I'm flagging this (see §6). It's still integer, still no float, still A1-respecting.

**Contribution-signal check: group/monoid/set?**

The action set is **a commutative monoid under composition when restricted to the message-pool and `tmPrepared` coordinates** (translations on those coords commute and are non-invertible — the pool is monotone add-only). On the RM-state coordinates, the set is **NOT a group**: there is no inverse for `RMPrepare` (no RMUnprepare), no inverse for receipt. Monotone-accumulating semantics = monoid, not group.

It is **not** the spread-polynomial composition monoid (no rotational structure). It is **not** the mutation-game root system (no Cartan matrix, no Dynkin graph). It is an unstructured **free commutative monoid on the 7 action generators modulo the guards**. This is weaker than the QA control-theorem generator algebra. I note this honestly: the object model's richer generator structures (spread polynomials, mutations) do not match 2PC's action set.

## 4. Invariants as geometric constraints (Wildberger-native)

### TPTypeOK

**Geometric form.** Four conjunctive bounding-box predicates on `G`:

```
TPTypeOK  ≡  (∀i ∈ RM)  (b_i, e_i) ∈ { (1,1), (2,1), (3,1), (1,3) }
          ∧  (TM_b, TM_e) ∈ { (1,1), (3,1), (1,3) }
          ∧  (∀i ∈ RM)   χ_i ∈ {1, 2}
          ∧  (∀i ∈ RM)   m_{P,i} ≥ 1  ∧  m_C ≥ 1  ∧  m_A ≥ 1
```

Each RM state point must lie in the 4-point set that is **the intersection of `{1..3}²` with the "odd-`a`" subvariety** — i.e., `a = b + 2e` is odd for each admissible point (`(1,1): a=3`, `(2,1): a=4` [even — special for the intermediate `prepared`], `(3,1): a=5`, `(1,3): a=7`). More usefully: each point satisfies **`Q_b ≤ 10`** (the four points have blue-quadrance ≤ 10 from origin: `2, 5, 10, 10`). The TPTypeOK bound is a **blue-quadrance bound**: `Q_b((b_i,e_i), (1,1)) ∈ {0, 1, 4, 4}`.

**Preservation.** All seven translations map `G` to `G` (inside the bounding boxes), provided guards hold. Boundedness is preserved because the translations are guarded — no translation fires from a terminal state back into another terminal state.

### TCConsistent (safety — central)

**Central question:** what does "no two RMs disagree" look like geometrically?

I evaluated the four candidates the prompt offered:

- **TQF collinearity:** REJECTED. TQF is ternary; 2PC safety is binary pairwise. Forcing a triple would require an artificial third point.
- **Cross-ratio constraint:** REJECTED. 4-point projective invariant; 2PC has pairwise binary safety. Wrong arity.
- **Inner-SCC membership:** REJECTED. The reachable graph is a DAG (monotone pool), not a strongly connected graph — there are no non-trivial SCCs beyond singletons in the terminal layer. SCC vocabulary would be ornamental.
- **Null-cone condition:** ACCEPTED. The red quadrance between the two terminal points is exactly zero:
  `Q_r((3,1),(1,3)) = 4 - 4 = 0` — the disagreement pair **IS** the non-trivial red null-cone pair.

**Full geometric predicate.**

```
TCConsistent  ≡
  ¬ (∃ i, j ∈ RM,  i ≠ j :
       (b_i, e_i) ∈ Terminal  ∧  (b_j, e_j) ∈ Terminal
       ∧  (b_i, e_i) ≠ (b_j, e_j)
       ∧  Q_r((b_i, e_i), (b_j, e_j)) = 0
       ∧  Q_b((b_i, e_i), (b_j, e_j)) > 0 )
```

where `Terminal = { (3,1), (1,3) }`.

In words: **no pair of distinct RMs sits on a non-trivial red null-cone within the terminal set**. The `Q_b > 0` clause excludes self-pairs (where `Q_r` is trivially zero). The terminal-membership clause restricts the constraint to decided RMs — RMs still `working` or `prepared` do not contribute to disagreement.

**Why this captures TLA+ safety.** The informal safety is "no RM commits while another aborts, and vice versa." That is exactly "no pair `{(3,1), (1,3)}`." The unique non-self red-null pair inside Terminal IS `{committed, aborted}`. Any two committed RMs have `Q_r = 0 ∧ Q_b = 0` — excluded by `Q_b > 0`. Any two aborted RMs likewise. The only surviving case is committed-vs-aborted, which is exactly what the predicate forbids.

**Preservation.** Under legal actions, no RM can reach `(3,1)` unless a `TMCommit` happened (guard: `m_C ≥ 2`), which requires all `χ_i = 2` (guard of `TMCommit`), which requires no prior `TMAbort` (because `TMAbort` fires only from `TM = (1,1)` and `TMCommit` likewise — the TM is linear in its own trajectory, at most one of the two fires). Therefore `m_C ≥ 2 ∧ m_A ≥ 2` is unreachable, so no RM can receive both broadcasts; the disagreement pair cannot form. This is the geometric restatement of the classical 2PC correctness argument. **The red null-cone cannot be reached because the monotone action set forbids both `TMCommit` and `TMAbort` firing in the same trace.**

**Scaling in `N`.** The predicate is ∀∃ over RM × RM and evaluates pointwise on integer lattice points. It survives arbitrary `N` with no re-encoding. Cost is `O(N²)` per check but the predicate itself is `N`-independent in form.

## 5. Contribution self-assessment

Per the README rubric (0-4, DieHard=0, Bakery=1, QA control theorems=4):

- **Generator-relative structure?** Partial. Actions are named translations with guards; but the generator algebra is a free commutative monoid modulo guards, not a rich Weyl/spread-polynomial structure. Compare the QA control theorems' `{σ, μ, λ₂, ν}` with proper algebraic identities. Here the generators don't *compose* into new interesting states — they just reach their respective corners. **Weak generator structure.**
- **SCC / orbit organization?** No. The reachable graph is a DAG (monotone pool). There are no non-trivial SCCs beyond terminal singletons. I did NOT force an SCC story — that would be ornamental.
- **Closed-form counts?** Weak. I can state "the number of reachable RM-configurations at step `k` is `O(5^N)`" but I do not derive a closed form. The pair-count `\binom{N}{2}` enters the invariant cost but not its algebraic content.
- **Failure-class algebra?** YES — modest. The *disagreement failure class* is exactly `Q_r = 0 ∧ Q_b > 0` on Terminal². This is a **null-cone failure algebra**, load-bearing in the encoding: the geometric predicate IS the safety predicate, not decoration. One failure class, one null-cone equation. This is the Wildberger-leveraged piece.
- **Monotonicity under generator expansion?** Yes, trivially: adding more action-generators (e.g. a read-only `RMObserve`) cannot decrease the set of reachable states; red-null disagreement remains forbidden because the TM's trace linearity is preserved.

**Self-score: 2 (Useful).**

**Rationale (3 sentences).** The red-quadrance null-cone predicate for `TCConsistent` is load-bearing, not ornamental — `Q_r((3,1),(1,3)) = 0` is the specific algebraic fact that makes the encoding work, and it is genuinely a Wildberger-chromogeometric insight rather than a renaming. However, the generator side is weak (free commutative monoid, no root-system/spread-polynomial content), there is no orbit/SCC structure to expose, and no closed-form count beyond the trivial pair count. Above Bakery (Contribution 1 per README), below the QA control theorems' 4 — matches Contribution 2.

## 6. Primitive-gap report (Mode B specific)

**Gap 1 — Conditional translations.** The receipt actions `RMRcvCommitMsg` and `RMRcvAbortMsg` are naturally "set the RM to the terminal point." As a transform this is a translation whose offset depends on the current point, not a constant-offset translation. The object model lists translations as `T_{(a,b)}(x,y) = (x+a, y+b)` with constant offsets. A cleaner primitive would be a **projection-onto-fixed-point** or **idempotent endomorphism**, which the Wildberger bundle does not explicitly surface. Not a crisis (it factors as guarded translations), but a minor strain.

**Gap 2 — Monoid, not group/Weyl.** The action set is a free commutative monoid with guards. The object model's strongest generator structures are the **spread-polynomial composition monoid** (rotation group over Q) and the **mutation-game Weyl group** (root-system reflections). 2PC has neither rotation nor reflection — its dynamics are strictly monotone add-only. The object model's rich generator structures do not engage here. **This is the structural reason 2PC can score Contribution 2 but not 3+ under Mode B: Wildberger's richest primitives are geometric-motion primitives, and 2PC is a monotone-accumulation protocol.**

**Gap 3 — Message pool as integer counter.** The monotone message pool is naturally a **counter** (or a multiset). The object model surfaces `Caps(N, N)` as a bounded counter space but does not give a native "unbounded add-only counter" primitive. I worked around by using `Z_{≥1}` with A1 shift, but a distributed-protocol-friendly counter primitive is missing. In full GA, this might be handled by grade-0 (scalar) accumulators with an explicit add-only constraint; the Wildberger bundle uses bounded `Caps` instead.

**Gap 4 — No bivector needed.** The prompt's objection `#4` (whether bivectors are needed) does NOT fire here. 2PC's safety is a pairwise scalar (null-cone) condition, which `Q_r` handles natively. No higher-grade GA primitive is required. This is honest good news — the Wildberger bundle is sufficient on this spec's safety side.

**Overall verdict on the bundle for this spec.** Sufficient for the **invariant**, weaker for the **dynamics**. The bundle's geometric-motion primitives (rotations, reflections, mutations) are idle; the bundle's metric/null-cone primitives (quadrances, chromogeometry) do real work. Augmentation recommendation: if Mode B wants to push distributed-protocol scores above 2, add an explicit **monotone-counter primitive** and a **guarded-translation monoid** generator class — these are not in the current Wildberger bundle and are the specific structural reason 2PC tops out here.

## 7. Self-check

- [x] **A1 (No-Zero):** all state coordinates in `{1,...}`; `Q_r = 0` is a scalar predicate value, not a state. OK.
- [x] **A2 (Derived Coords):** `d = b+e` and `a = b+2e` are computed, not assigned; the four RM lattice points have their derived coords listed in §2.
- [x] **T1 (Path Time):** time is integer step count over the seven translation generators; no continuous time. Model-checking is BFS over integer lattice.
- [x] **T2 / NT (Observer Projection Firewall):** no continuous inputs; the protocol is discrete-input discrete-output; quadrances are integer-valued (no `sqrt`, no float).
- [x] **S1 (No `**2`):** all squared quantities written as `(x-y)*(x-y)` in the intended implementation; the prose `x²` notation is textual, not code.
- [x] **S2 (No float state):** all state is `int`; A1 shifts are integer; no `np.zeros`, no `np.random`.
- [x] **Wildberger primitives used are from the object model's list:** points, `Q_r`, `Q_b`, translations, 4-tuple, Pythagorean identity. No fabricated primitives.
- [x] **`ornamental-overlay` risk self-diagnosed:** `Q_r((3,1),(1,3)) = 0` is load-bearing — if I removed the red null-cone predicate, the safety invariant would have no Wildberger content and would collapse to "`rmState[i]` and `rmState[j]` aren't `{committed, aborted}` as a pair," which is the TLA+ form. The chromogeometric form **adds** the algebraic fact that committed-and-aborted are exactly the non-trivial red-null pair in the chosen lattice. That is a genuine encoding insight, not decoration. However, the generator side (translations only, no spread/rotation/mutation) IS thinner — the Wildberger action vocabulary is underused. I score myself 2, not 3, for this reason.
- [x] **Blindness pact held:** confirmed — no ground-truth / attempts / candidates / diffs / fidelity_audit read this session.
