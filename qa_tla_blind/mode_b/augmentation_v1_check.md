# Augmentation v1 Consistency Check

**Checker:** fresh subagent, 2026-04-22.
**Scope:** axiom compliance + conflict check + precision check. NOT fidelity to Wildberger.

## Primitive A — Monotone multiset

Located at `wildberger_object_model.md` lines 271-310.

### Axiom compliance

- **A1 (No-Zero):** Line 274 defines `M : X → ℤ≥₁`; line 288 states "`M(x) = 1` represents 'absent' (shifted)". This is a correct A1 shift of the native `ℤ≥₀` counting semantics. The derivation holds: the native counting multiset with 0 = absent is shifted by +1 so the domain is `{1, 2, ...}` with 1 as the sentinel for "absent". CLAUDE.md A1 requires states in `{1, ..., N}`; `ℤ≥₁` is the unbounded form of this. **PASS.**
- **A2 (Derived coords):** Line 289-290 names `|M| = Σ M(x)` as a derived coordinate. This is a true derived integer: it is a function of the multiset state, never assigned independently. However, A2 canonically says `d = b+e`, `a = b+2e` — i.e. derived coords are specific named projections of a `(b, e)` state, not arbitrary aggregates. The augmentation extends the A2 idiom ("derived, never independently assigned") to a new coordinate on a non-`(b,e)` state. This is a reasonable extension, but it is a GENERALIZATION of A2, not literal A2. **PASS with note** — derivation holds for the spirit of A2; flag that the object is a multiset, not a `(b,e)` pair, so strict A2 does not apply verbatim.
- **T1 (Path-time):** Line 291-292 identifies step count with number of `add` applications. The monoid only has monotone-add, so every step strictly increases `|M|` by 1 (point-add) or by `|S|` (set-add). Integer path length. **PASS.**
- **T2/NT (Firewall):** Line 293 states "no continuous layer." The multiset carrier is discrete (`X` is a "discrete base set", line 274). No float/continuous primitive crosses into the monoid action. **PASS.**
- **S1 (No `**2`):** No squaring operation is defined in this primitive. The only operations are `add`, `add_all`, pointwise comparison, and cardinality sum. **PASS** (vacuous).
- **S2 (No float state):** Counts are integers (`ℤ≥₁`). Sum `|M|` is an integer. `X` is discrete. **PASS.**

### Conflict with existing primitives?

Checked each object-model primitive:

- Points (lines 46-54): `(x, y) ∈ Z²` — different carrier (`Z²` vs `ℤ≥₁^X`). No overlap.
- Lines (lines 56-58): projective triples over `Z`. Different type. No overlap.
- Quadrance (lines 61-71): integer-squared-distance functional; takes pairs of points. The multiset has no pair-valued "distance" operation. No overlap.
- Spread (lines 74-82): rational angular invariant over line pairs. No overlap.
- TQF (lines 85-92): three-quadrance collinearity predicate. No overlap.
- Cross-ratio (lines 95-98): four-point projective invariant. No overlap.
- Spread polynomials `S_n` (lines 101-105): **composition monoid** `S_n ∘ S_m = S_{n·m}`. The Monotone-multiset monoid is *free commutative strictly monotone* — a different algebraic object. The multiset monoid is idempotent-free (two `add(M, x)` calls yield `M(x) + 2`, not `M(x) + 1`), while the spread-polynomial monoid is multiplicative on indices. They coexist without clash: one acts on counts, the other on rotation indices. **No conflict.**
- Hexagonal ring `ring(a, b) = T_{d+1} + b·e` (lines 108-119): weight-diagram count. The multiset `|M|` is also a count, but there is no identification; they are counts of different objects. **No conflict.**
- 4D diagonal rule (lines 122-134): lattice-point-in-2-plane claim. Different carrier. No overlap.
- Transforms 1-5 (lines 140-167): act on `Z²` or `P²(Z)`, not on multisets. No overlap.
- Chromogeometric quadrances (lines 171-186): Gram-matrix signatures on `Z²`. The lattice order `≤` on multisets is pointwise and has no signature — it's order-theoretic, not metric. **No conflict** — the ordering lives in a genuinely different category than quadrance.
- 4-tuple `(b, e, d, a)` embedding (lines 189-203): concerns QA canonical tuple. No overlap.

**Verdict: no conflicts.**

### Precision check

- **Operation signature:** clear. `add : Multiset × X → Multiset`, `add_all : Multiset × P(X) → Multiset`. Input/output types stated. Line 278-284.
- **Reachability structure:** stated. Line 297-301: "the reachable subset is the *upward closure* of `M₀`, i.e. *every* multiset is reachable." The lattice order is named and its definition given pointwise.
- **Invariants expressible:** named concretely on lines 303-310 (monotonicity, inclusion, threshold, message-pool). Each is given as a formula, not just a gesture.
- **Reproducer can use without inventing:** **yes**, with one minor caveat — `add_all(M, S)` over a *set* `S ⊆ X` is the natural extension, but if a reproducer wants to add a multiset `S'` (counting multiplicities on the "added batch"), the extension is not stated. For Mode B TwoPhase/Paxos message-pool semantics the set form suffices.

**Precision PASS.**

## Primitive C — Lattice-lub / argmax

Located at `wildberger_object_model.md` lines 312-353.

### Axiom compliance

- **A1 (No-Zero):** Line 326 states `T, U` are integer sets after A1 shift. The claim "`lub` stays in `T`" is a correct closure property — `max` of a finite non-empty subset of `T` is a member of `T`. Derivation holds. **PASS** (modulo the background convention that `T` is `{1, ..., N}`-shaped).
- **A2 (Derived coords):** Line 327-328: `lub(S)` is named a "height coordinate" of `S`. This is a derived integer that is a function of `S`, never assigned independently — it is *defined by* maximization. Like the `|M|` case for Primitive A, this is a generalization of A2 beyond the canonical `(b, e, d, a)` tuple. Derivation holds in spirit. **PASS with note.**
- **T1 (Path-time):** Line 329-331 states `lub`/`argmax` are single-step. Discrete operation; does not introduce continuous time. **PASS.**
- **T2/NT (Firewall):** Line 332 "pure integer; no continuous layer." `lub` and `argmax` use only total-order comparison, which is discrete. **PASS.**
- **S1 (No `**2`):** No squaring. **PASS** (vacuous).
- **S2 (No float state):** Carriers `T, U` are integer. Comparison returns ordering only. **PASS.**

### Conflict with existing primitives?

Cert-backed primitives are all rational-arithmetic / geometric / polynomial. `lub` is order-theoretic, a different algebraic layer.

- Quadrance/Spread/TQF/cross-ratio: none use ordering beyond integer comparison for arithmetic. No collision.
- Spread-polynomial composition monoid (lines 101-105): multiplicative-on-indices, not order-theoretic. No collision.
- Rotations/reflections/translations: metric-preserving. They do not *preserve* `lub` in general (a reflection can swap max and min if the carrier is signed) — but this is benign because `lub` is not claimed to be preserved by Wildberger transforms, and the augmentation flags this on lines 349-353: "clearly labeled an order-theoretic augmentation" that is "not claimed as a Wildberger bundle member."
- Mutation moves (lines 164-167): act on root systems. No shared object with `lub`.
- 4-tuple derived coord `d = b + e`: this is a SUM, not a MAX. The augmentation does not conflate these. No collision.

**Integer-exact philosophy:** `lub` IS integer-exact — no rounding, no tie-breaking floats — as long as the stated lowest-index tie-break is used. The augmentation explicitly names the tie-break rule (line 320-321). **Consistent with integer-exact philosophy.**

**No cross-ratio or spread collision:** neither cross-ratio nor spread is an ordering; `lub` produces an *element* not a *ratio*. **No collision.**

**Verdict: no conflicts.**

### Precision check

- **Operation signature:** clear. `lub : P_fin(T) \ {∅} → T`, `argmax : P_fin(S) × (S → U) → S`. Tie-break rule named ("lowest-index") — line 320-321.
- **Reachability structure:** not applicable in the same sense as A (lub is a pure function, not a monoid action); but the lub MONOID `(T, max)` is stated to be "idempotent commutative" (line 338), a sup-semilattice. Line 336-338 gives the monoid laws explicitly.
- **Invariants expressible:** lines 340-346 give three named idiomatic invariants ("last value at highest ballot", "monotone height", "guard by current max"). Paxos `showsSafe` is explicitly exemplified line 342.
- **Reproducer can use without inventing:** **yes**. One minor gap: the `argmax` tie-break says "e.g., lowest-index" — the "e.g." hedges. A reproducer will likely pick a specific rule per use-site. For benchmark reproducibility, tighter would be "always lowest-index unless the spec dictates otherwise."

**Precision PASS with one minor textual hedge** (`e.g.` on tie-break).

## Overall verdict

`augmentation-clean` — both primitives are axiom-compliant, free of conflicts with any existing object-model primitive, and precisely described. The augmentation is safe to commit and use.

One optional textual tightening is recommended but not required:
- **Primitive C, line 320-321:** replace "with a tie-breaking rule (e.g. lowest-index tie-break)" with "with a tie-breaking rule (default: lowest-index; specify at use site if different)". This removes the "e.g." hedge without changing semantics.

## Precision gaps (if any)

- **Primitive A, `add_all` on a multiset (not just a set):** if a reproducer needs to merge two multisets `M ← M ⊎ M'`, that operation is not stated. The set-valued `add_all` is only defined pointwise on an added set `S ⊆ X`. Not a blocker for TwoPhase/Paxos message-pool use (sets suffice there).
- **Primitive C, tie-break hedge:** `e.g. lowest-index` leaves the scorer to adjudicate if a Mode B attempt uses a different tie-break. Minor.

Neither gap blocks commit.

## Note on the deferred primitives (B', D, E)

The deferral reasoning (lines 355-371) is defensible:

- **B' (guarded-translation / idempotent-projection):** the claim is that Primitive A's `add(M, x)` subsumes the common case (guards as multiset-membership tests). This is correct for the dominant TwoPhase/Paxos idiom ("if msg ∈ msgs then ..."). Strict-guard cases not expressible as membership will need B, and that is acknowledged.
- **D (hyperplane-intersection in P^{k-1}, k > 3):** this is genuinely a Wildberger-corpus extension question, not a gap the Mode B object model should invent. Deferring it until k=3 (the P²(Z) line-meets case already in the bundle) is shown insufficient is the right sequencing.
- **E (structured-message payload):** the claim that it factors as a product of Primitive A multisets over a tuple base set is plausible but unverified; deferring until empirical need is appropriate.

**Is there a primitive that clearly should be in v1 but isn't?** No. The two gaps TwoPhase/Paxos actually hit were monotone pools (A) and argmax-receive (C); those are exactly what v1 adds. B', D, E are speculative-or-subsumed. The minimal scope choice is justified.

**Confidence:** high.
