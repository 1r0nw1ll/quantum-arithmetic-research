# QA-as-Wildberger Object Model (Mode B reference)

**Status:** draft, 2026-04-22. Synthesis artifact, not a theorem.
**Scope:** pre-specified QA-native object model for Mode B blind-reproduction
benchmarks. Supplies the **primitives** a reproducer can reach for when
encoding a TLA+ spec, so that Contribution scores measure encoding fit
rather than discoverability.

**Source material (all in-repo, cited):**

- Wildberger, *Chromogeometry* (Math. Intelligencer 2008) — cert [234]
  `qa_chromogeometry_pythagorean_identity_cert_v1`.
- Wildberger, *Divine Proportions* (2005) — full read, underlies quadrance /
  spread / TQF.
- Wildberger, UHG I–IV — certs [232], [233]; theory doc
  `docs/theory/QA_UHG_PROJECTIVE_COORDINATES.md`.
- Wildberger, *Quarks, diamonds, and representations of sl_3* (2003) —
  cert [240] diamond sl(3) irrep dimension.
- Wildberger, *Hyper-Catalan + Geode* (AMM 2025) — cert [231] hyper-Catalan
  diagonal.
- Wildberger, *Spread polynomials, rotations, butterfly effect* (2009) —
  cert [236] spread-polynomial composition monoid.
- Wildberger, *Rational Trigonometry in Higher Dimensions* (2017) — cert
  [237] 4D diagonal rule for 2-planes in R^4.
- Wildberger + Notowidigdo, *Generalised vector products of a tetrahedron*
  (2019) — cert [241] quadruple coplanarity.
- `docs/theory/QA_WILDBERGER_STRUCTURAL_FOLLOWUPS.md` — SL(3) hexagonal
  ring identity `ring(a,b) = T_{d+1} + b·e`, chromogeometric TQF symmetry.

## Why Wildberger

The bundle Wildberger develops — rational trigonometry, chromogeometry, UHG,
spread polynomials, mutation games, hyper-Catalan — is the specific
**rational-arithmetic/geometric-algebra** substrate the QA project has been
integrating since 2026-03. It replaces the continuous-metric `sqrt` /
transcendental-angle layer with integer-valued quadrances, spreads, and
cross-ratios. This is what "QA is essentially geometric algebra" means
operationally: the geometric primitives live in Z or Q, actions are rational
transforms, invariants are polynomial identities.

For Mode B this substrate gives a **pre-specified alphabet of primitives**
the reproducer doesn't have to rediscover from the blinded prompt.

## Primitive objects

### 1. Points

A point is an integer pair `(x, y) ∈ Z²` or a projective class
`[x : y : z] ∈ P²(Z)` (UHG).

Under QA coordinates: a point is `(b, e)` with derived `d = b+e`, `a = b+2e`.
The 4-tuple `(b, e, d, a)` is **one point viewed in four coordinate
projections**; three are derived.

### 2. Lines

A line is a projective triple `⟨l : m : n⟩` satisfying `lx + my + nz = 0`
for incident points. Integer-coefficient.

### 3. Quadrance (integer squared-distance)

Between points `P₁ = (x₁, y₁)` and `P₂ = (x₂, y₂)`:

- **Blue (Euclidean):** `Q_b(P₁, P₂) = (x₁ - x₂)² + (y₁ - y₂)²`
- **Red (Lorentzian):** `Q_r(P₁, P₂) = (x₁ - x₂)² - (y₁ - y₂)²`
- **Green (mixed):** `Q_g(P₁, P₂) = 2 · (x₁ - x₂) · (y₁ - y₂)`

All three are integer-valued on `Z²`. No sqrt, no float.

Under QA coords `(b, e)`: `Q_b = b² + e²`, `Q_r = b² - e² = (b-e)(b+e) = (b-e)·d`,
`Q_g = 2 b e`.

### 4. Spread (integer squared-angle)

Between lines with slopes `m₁, m₂`: `s(L₁, L₂) = (m₁ - m₂)² / ((1 + m₁²)(1 + m₂²))`.

For lines through origin with direction vectors `v_i`:
`s = 1 - (v₁ · v₂)² / (|v₁|² |v₂|²) = (v₁ × v₂)² / (|v₁|² |v₂|²)`.

A spread is a rational number in `[0, 1]`. Over the integers, the cross
products and dot products are exact; spreads are rationals in lowest terms.

### 5. Triple Quad Formula (TQF)

For three points yielding three pairwise quadrances `Q₁, Q₂, Q₃`:

```
TQF(Q₁, Q₂, Q₃) = (Q₁ + Q₂ + Q₃)² - 2(Q₁² + Q₂² + Q₃²)
```

**TQF = 0 ⟺ the three points are collinear.** Under chromogeometry:
`TQF_r = TQF_g = -TQF_b` (cert [246]); `TQF_b = 16 · (signed area)²`.

### 6. Cross-ratio (projective invariant)

For four collinear points: `(A, B; C, D) = (AC · BD) / (AD · BC)` where each
is a signed ratio. Integer numerator and denominator; the ratio is a
rational preserved under any projective transform.

### 7. Spread polynomials `S_n`

Integer polynomials `S_n(s)` with `S_0 = 0, S_1 = s, S_{n+1} = 2(1 - 2s)S_n
- S_{n-1} + 2s`. They form a **composition monoid**: `S_n ∘ S_m = S_{n·m}`
(cert [236]). `S_n` encodes the `n`-fold rotation over rationals without
transcendentals. This is QA's discrete rotation group.

### 8. Hexagonal ring decomposition (SL(3))

For the SL(3) irrep `π[a, b]`: the weight diagram decomposes into concentric
hexagonal rings with outer-ring size

```
ring(a, b) = T_{d+1} + b · e    where d = a + b, T_n = n(n+1)/2
```

Natural **two-component split**: *D-component* `T_{d+1}` (depends only on
the sum) + *B-component* `b·e` (bilinear cross-term). This is a template
for any invariant that has both a sum-only piece and a cross-piece — which
is common in distributed-protocol safety invariants.

### 9. 4D diagonal rule

Cert [237] shows the QA 4-tuple `(b, e, d, a)` is exactly the lattice point
`b·v₁ + e·v₂` in the 2-plane of R⁴ spanned by `v₁ = (1, 0, 1, 1)` and
`v₂ = (0, 1, 1, 2)`. The Gram matrix of this basis is `[[3, 3], [3, 6]]`
with `det = 9`, matching QA's canonical modulus. Perpendicular QA-tuple
pairs in this 2-plane satisfy Wildberger's diagonal rule `Q₁ + Q₂ = Q₃`
(where `Q_i` are quadrances on the 4-tuple).

The related general-R⁴ Plücker-style Gram-determinant expression
`Q(Π) = (u·v)² − (u·u)(v·v)` for orthogonal `{u, v}` coincides with
`-(u·u)(v·v)` and is a 2-plane invariant, but cert [237]'s verified claim
is the diagonal-rule form + the `det = 9` correspondence above, not the
Plücker form directly.

## Primitive transforms

All transforms are **rational-coefficient** maps on `Z²` or `P²(Z)`.

### 1. Translations

`T_{(a, b)}(x, y) = (x + a, y + b)`. Preserve all quadrances (blue / red /
green).

### 2. Reflections

`R_x(x, y) = (-x, y)`, `R_y(x, y) = (x, -y)`. Preserve `Q_b`; flip
`Q_r → Q_r, Q_g → -Q_g` (or symmetric). Generate the discrete dihedral
group on the plane.

### 3. Rotations (rational, via spread polynomials)

A rotation by spread `s` applied to `(x, y)`: use the rotation matrix
parameterised by spread rather than angle. Composition via spread polynomial
monoid. All coords stay rational.

### 4. Projective maps (UHG)

`(x, y, z) ↦ M · (x, y, z)^T` for an integer `3×3` matrix `M` with nonzero
determinant. Preserve cross-ratios; do not generally preserve quadrances
(those are metric, cross-ratios are projective).

### 5. Mutation moves (generalised root systems)

For Coxeter-Dynkin graphs, *mutation* sends a root `α_i → -α_i + Σ |⟨α_i,
α_j⟩| α_j`. Integer-coefficient; generates the Weyl group. Cert [244]
ties this to E_8 construction.

## Three-metric chromogeometric structure

Every plane figure carries three simultaneous geometries:

- **Blue** (positive-definite, Euclidean):           `Q_b = dx² + dy²`
- **Red** (signature (1, -1), Lorentzian):           `Q_r = dx² - dy²`
- **Green** (signature (1, -1), mixed indefinite; Gram `[[0,1],[1,0]]` has eigenvalues ±1; null-cone = coordinate axes):    `Q_g = 2 · dx · dy`

**Pythagorean identity (cert [234]):**  `Q_b² = Q_r² + Q_g²`.

**TQF symmetry (cert [246]):** `TQF_r = TQF_g = -TQF_b`.

**Usage for protocol encoding.** Many binary state invariants have dual
aspects. Chromogeometry gives three parallel "views" of the same geometry:
two states being *equal* vs *opposite* vs *complementary* map naturally
to `Q_r = 0` (null red), `Q_g = 0` (null green), `Q_b = max` (blue extreme).
A safety invariant like "no disagreement" can be phrased as a null cone
condition in one of the three metrics.

## 4-tuple `(b, e, d, a)` embedding

QA's canonical 4-tuple embeds into Wildberger's 4D diagonal rule. For a
protocol state viewed as a point in `Caps(N, N)²` or equivalent:

- `(b, e)` is the primary 2D point
- `d = b + e` is the *sum*-projection (line sum)
- `a = b + 2e` is the *linear*-projection (weighted sum)

The 4-tuple constraint `d = b+e ∧ a = b+2e ∧ a - d = e ∧ d - a = -e` is
**one degree of freedom's worth** of over-specification — it fixes the
internal consistency of the tuple.

**Usage.** Any 2D state that has derived "sum" and "linear" observations
embeds as a 4-tuple; any predicate on the 4-tuple lifts to a constraint on
the 2D state.

## How to encode a TLA+ spec under this object model

### Step 1: position states in a Wildberger space

Instead of mapping each TLA+ variable independently to `(b, e)`, ask:
**what is the natural geometric object the system is acting in?**

- Bounded counter → point in `Caps(N, N)` with 4-tuple `(b, e, d, a)`.
- Process-state lattice (e.g. `{idle, wait, cs, exit}`) → point in `P²(Z)`
  with integer coords; use cross-ratio invariants for transitions.
- Ticketed mutex (Bakery) → point in `Caps(T, P)` where `T` = ticket axis
  (integer), `P` = process axis (integer). Quadrance between two
  `(ticket, pid)` pairs is a chromogeometric distance.
- Commit/abort decisions (2PC) → point on `{-1, 0, +1}^N` lattice (N RMs);
  disagreement = red-quadrance nonzero between two RMs' choices.
- Log-entry sequences (Raft) → point in ordered-tuple space; cross-ratio
  of three log indices is a projective invariant.

### Step 2: map actions to transforms

Not every action is a Wildberger transform. Catalog which are:

- *State-preserving rename* → identity.
- *Coordinate swap* (μ-like) → reflection.
- *Scale doubling* (λ₂-like) → dilation with spread polynomial.
- *Counter increment* (σ-like) → translation by unit vector.
- *Non-geometric guard* (e.g. "if pc = L1") → NOT a Wildberger transform;
  these are the "ornamental" candidates — they need a different primitive.

### Step 3: restate invariants as geometric constraints

- **Mutual exclusion / disagreement** → "not on the null red-cone" or
  equivalent `Q_r ≠ 0` between conflicting state points.
- **Type bounds** → "inside `Caps(N, N)` rectangle" = boundedness.
- **Inductive invariant** → some TQF or cross-ratio equation held by all
  reachable states.
- **Liveness / no deadlock** → every orbit/SCC has an exit edge to a
  target state (cert [236]'s monoid structure may apply here).

### Step 4: score contribution against the QA control theorem signature

The four contribution markers remain the same:
- generator-relative structure (here: chromogeometric / UHG / spread-
  polynomial generators instead of σ/μ/λ₂/ν)
- SCC / orbit organization
- closed-form counts (triangular `T_n`, ring decomposition `T_{d+1} + b·e`)
- failure-class algebra (null-cone, OOB, parity)

## Augmentation v1 (post-TwoPhase + Paxos Mode B, 2026-04-22)

**Purpose.** Close two confirmed bundle gaps identified by consecutive Mode B
distributed-protocol runs (TwoPhase at Contribution 2, Paxos at Contribution
2). Both runs produced a load-bearing geometric primitive on the *safety*
axis (Q_r null-cone; P² line-meets) but hit gaps on the *dynamics* axis
(monotone message pools, conditional updates, argmax-receive). These two
primitives target the dynamics side directly.

**Provenance caveat.** These primitives are **not lifted from Wildberger's
published bundle**. They are benchmark-derived: each fills a specific gap
observed empirically. They are included in this object model because they
are QA-axiom-compliant and integer-exact, not because Wildberger states them.
Scorers must treat them as `augmentation-primitives` and flag any claim that
they are Wildberger-native as a misattribution. A future audit against a
broader Wildberger corpus review may relocate one or both to the cert-backed
tier; for now they sit in their own tier.

### Primitive A — Monotone integer multiset

**Object.** A finite multiset `M : X → ℤ≥₁` over a discrete base set `X`
(after A1 shift; the native `ℤ≥₀` form is A1-shifted to `ℤ≥₁` with `1 =
"zero count"` sentinel). `M(x)` is the count of `x` in `M`.

**Legal transforms (the monotone-add monoid).**

- `add(M, x)`: `M(x) → M(x) + 1`; all other counts unchanged. Unconditional.
- `add_all(M, S)`: apply `add(-, x)` for each `x ∈ S`; the set action extends
  the point action pointwise.
- No `remove`, no `subtract`. The monoid is *free commutative strictly
  monotone* — think of it as the natural numbers on each axis with addition
  only.

**QA axiom compliance.**

- A1 (No-Zero): `M(x) ∈ ℤ≥₁`; `M(x) = 1` represents "absent" (shifted).
- A2 (Derived coords): the total count `|M| = Σ M(x)` is a natural derived
  coordinate. `|M|` is strictly non-decreasing under `add`.
- T1 (Path-time): step count is the number of `add` applications; a single
  path through the monoid.
- T2/NT (Firewall): no continuous layer.
- S1 / S2: integers only, no `**2`, no float state.

**Reachability lattice.** The reachable set of multisets from `M₀ = {1}^X`
(the empty multiset, A1-shifted) under `add` is the product lattice
`∏_{x ∈ X} ℤ≥₁` with the pointwise order — and under `add` dynamics, the
reachable subset is the *upward closure* of `M₀`, i.e. *every* multiset is
reachable. The order itself is the `≤` relation on each coordinate. This
is a free commutative monoid equipped with a dominance order.

**Invariants expressible in the Monotone-multiset algebra.**

- Monotonicity: `∀ x. M'(x) ≥ M(x)` — trivially an invariant of the monoid.
- Inclusion: `M₁ ⊆ M₂ ⟺ ∀ x. M₁(x) ≤ M₂(x)` — the lattice order.
- Quantity thresholds: `|M| ≥ k` or `M(x) ≥ k` — closed-form in counts.
- **Message-pool invariants for distributed protocols** (TwoPhase `msgs`,
  Paxos `msgs`) are all monotonicity + threshold statements over this
  algebra.

### Primitive C — Lattice-lub / argmax over a finite subset

**Object.** A finite subset `S ⊆ T` where `(T, ≤)` is a totally ordered set.
Two operations:

- `lub(S) := max_{t ∈ S} t` — the least upper bound / maximum. Defined when
  `S ≠ ∅`.
- `argmax(S, f) := t* ∈ S` where `f(t*) = lub({ f(t) : t ∈ S })`, with a
  tie-breaking rule (e.g. lowest-index tie-break). `f : S → U` maps into
  another totally ordered `U`; `argmax` returns the element `t*` of `S`
  whose `f`-image is the max.

**QA axiom compliance.**

- A1 (No-Zero): `T, U` are integer sets after A1 shift; `lub` stays in `T`.
- A2 (Derived coords): `lub(S)` is a derived integer, a "height coordinate"
  of `S`.
- T1 (Path-time): `lub` / `argmax` are single-step operations; they do not
  consume path-time beyond their invocation.
- T2/NT: pure integer; no continuous layer.
- S1 / S2: integer comparison only, no `**2` or float state.

**Legal transforms (lub monoid).**

- `lub(S ∪ {t}) = max(lub(S), t)` — insertion is monotone.
- `lub(S₁ ∪ S₂) = max(lub(S₁), lub(S₂))` — associative, commutative.
- `(T, max)` is an idempotent commutative monoid (a **sup-semilattice**).

**Invariants expressible with lub / argmax.**

- "Last value received at highest ballot": `argmax({(b, v) ∈ Promises}, (b,v) ↦ b)` — Paxos `showsSafe` condition.
- "Monotone height": `lub(S_t)` is non-decreasing as `S_t` grows under
  Primitive A.
- "Guard by current max": an action conditioned on `t > maxBal` uses a
  `lub`-comparison with A-side multiset tracking.

**Note on chromogeometric flavor.** `lub` is not a geometric primitive in
Wildberger's sense — it is order-theoretic. This primitive is therefore
**clearly labeled an order-theoretic augmentation** and is not claimed as a
Wildberger bundle member. It is included because TwoPhase and Paxos both
required it and no Wildberger-native primitive can cover it. Benchmark
honesty demands we flag it rather than force a bogus geometric rendition.

### Primitive D (Augmentation v2, 2026-04-22) — Projective subspace lattice in P^{k-1}(Z)

**Purpose.** Paxos Mode B capped at Contribution 2 because the v1 object model
only covered P²(Z) (lines and points). Paxos's quorum intersection is a
Wildberger-adjacent projective-dimension question, but for N≥4 acceptors it
lives in higher projective spaces. This primitive generalizes the
intersection calculus from P² to arbitrary P^{k-1}(Z).

**Provenance caveat.** This IS in the spirit of Wildberger's higher-dim UHG
work (his series on projective metrical geometry extends beyond the plane),
but is not verified by any specific cert in the ecosystem as of 2026-04-22.
Treat as **Wildberger-framework-consistent extension**, not cert-backed.

**Object.** The projective space `P^{k-1}(Z) = (Z^k \\ {0}) / ~`, where `~`
is the integer-coefficient scaling equivalence. A *projective subspace of
dimension d* is the projectivization of a (d+1)-dim linear subspace of `Z^k`
(equivalently the projectivization of the `Z`-span of d+1 linearly
independent points).

Key dimensions:

- `P^0(Z)` = a single point.
- `P^1(Z)` = a projective line (1-dim).
- `P^{k-1}(Z)` = full ambient (k-1-dim).
- A *hyperplane* is a (k-2)-dim projective subspace = codim-1.

**Legal operations.**

- **Span** `span(S)` for a set of points `S ⊆ P^{k-1}(Z)`: the smallest
  projective subspace containing every point in S. Dimension = (rank of
  the matrix of homogeneous coords) − 1. Integer-exact: rank over `Z`.
- **Meet / intersection** `V ∩ W` for two projective subspaces: pointwise
  intersection; again a projective subspace. Dimension computable via
  linear-algebra rank over `Z`.
- **Join / sum** `V + W`: smallest subspace containing both; equals
  `span(V ∪ W)`. Dimension = `dim V + dim W − dim(V ∩ W)` (the standard
  dimension formula).

**Dimension formula (the algebraic kernel).**

```
dim(V + W) + dim(V ∩ W) = dim(V) + dim(W)
```

Rearranged: `dim(V ∩ W) = dim(V) + dim(W) − dim(V + W) ≥ dim(V) + dim(W) − (k − 1)`.

This bound is the primitive's chief algebraic content. If two subspaces
satisfy `dim(V) + dim(W) ≥ k − 1`, they must intersect (i.e., `V ∩ W` has
dimension `≥ 0`, i.e., is at least one projective point). This is
**Grassmann's formula** applied to projective subspaces.

**Grassmann's formula over Z.** Grassmann's formula is typically stated
over a field, but it transfers exactly to the integer-span submodules of
`Z^k`: those submodules are free (every submodule of a free Z-module is
free), and the rank of a matrix of integer column vectors equals its rank
over `Q` (Smith normal form makes this exact). So
`dim_Z(V ∩ W) = dim_Q(V_Q ∩ W_Q)` where `V_Q, W_Q` are the rational
extensions, and the dimension formula holds over `Z` by passing to `Q`
and back. QA's integer-exact philosophy is preserved — no `Q`-valued
quantity appears in the primitive's operations.

**Usage for quorum intersection.**

Encoding: N acceptors as N points in general position in `P^{k-1}(Z)` for a
chosen k. Each quorum `Q ⊆ {acceptors}` with `|Q| = q` spans a projective
subspace `span(Q)` of dimension `q − 1` (assuming general position).

For two quorums `Q₁`, `Q₂`:

```
dim(span(Q₁) ∩ span(Q₂))  ≥  (q₁ − 1) + (q₂ − 1) − (k − 1)
                          =  q₁ + q₂ − k − 1
```

Paxos's pigeonhole bound: `|Q₁ ∩ Q₂| ≥ q₁ + q₂ − N`.

**Comparison.** The projective-intersection bound gives a non-empty
`span(Q₁) ∩ span(Q₂)` whenever `q₁ + q₂ ≥ k`. The pigeonhole bound gives
non-empty `Q₁ ∩ Q₂` (as sets) whenever `q₁ + q₂ > N`. These are different
quantities: projective dimension of an abstract subspace vs cardinality of a
set of named acceptors.

**Correspondence that works cleanly** — when `k = N` (ambient `P^{N-1}`,
one dimension per acceptor) and acceptors are the standard basis points
`e_i = [0 : ... : 1 : ... : 0]`. Then `span(Q)` is *exactly* the coordinate
subspace indexed by `Q`, and `span(Q₁) ∩ span(Q₂) = span(Q₁ ∩ Q₂)`. The
projective-dimension bound reduces *exactly* to the pigeonhole bound:
`dim(span(Q₁) ∩ span(Q₂)) = |Q₁ ∩ Q₂| − 1 ≥ q₁ + q₂ − N − 1`, matching
the pigeonhole statement up to the `-1` offset from projectivization.

**This means the projective-subspace encoding captures Paxos quorum
intersection EXACTLY for arbitrary N, provided acceptors are placed at the
standard basis of P^{N-1}(Z).**

For N=3, standard basis in P²(Z): `e_1 = [1:0:0], e_2 = [0:1:0], e_3 = [0:0:1]`.
Quorums = 2-subsets. Spans = lines through two basis points. Two lines in
P² intersect in exactly one point: the third basis point IS missing, so the
intersection point is... actually the line through `e_1, e_2` is
`{[a:b:0] : (a,b) ≠ 0}`; the line through `e_1, e_3` is `{[a:0:c]}`;
intersection is `{[a:0:0]} = {e_1}`. Matches pigeonhole: `|{e_1,e_2} ∩ {e_1,e_3}| = 1`.

**QA axiom compliance.**

- A1 (No-Zero): homogeneous coords exclude the zero vector. Projective
  points are equivalence classes; the representative point can be shifted
  via A1 scaling as needed.
- A2 (Derived coords): dimension is a derived integer.
- T1 (Path-time): operations are single-step.
- T2/NT: no continuous layer.
- S1 / S2: integer linear algebra; no `**2`, no floats.

**Note on the v1 `P²(Z)` case.** The v1 object model's §Points "or projective
class `[x : y : z] ∈ P²(Z)`" is the k=3 special case of this primitive. The
generalization to arbitrary k subsumes it; v1's P² treatment stays valid.

### Primitive B', E — explicitly deferred

Three other gaps from TwoPhase/Paxos (guarded-translation / idempotent-
projection, hyperplane-intersection in P^{k-1} for k > 3, structured-
message payload) are **NOT** added in Augmentation v1. The scope choice is
deliberate:

- **B (guarded-translation):** subsumed by `add(M, x)` over Primitive A
  when the condition is expressible as a multiset membership test — most
  of TwoPhase's and Paxos's guards are of this form. The remaining strict-
  guard cases may still require B, but we try A+C first.
- **D (hyperplane-intersection in P^{k-1}):** genuine Wildberger-framework
  extension that would require review of Wildberger's higher-dimensional
  projective work. Out of scope for a minimal augmentation. If Paxos stays
  at 2 with A+C, this is the next augmentation to try.
- **E (structured-message payload):** likely expressible as a product of
  Primitive A multisets over a tuple base set. Defer to empirical need.

## What this model DOES NOT include

- **Continuous / transcendental primitives.** No angles in radians, no
  `sqrt`, no transcendental functions. If the protocol has such, they are
  observer-projection inputs only (Theorem NT firewall).
- **Non-integer coefficients** in the transforms. Everything is in Z[1/2]
  at worst (for halving in ν).
- **Axiom-of-choice moves.** No "pick an arbitrary element of an infinite
  set." All quantifiers are bounded.
- **Higher-grade Clifford elements** (bivectors, trivectors as full GA
  multivectors). Wildberger's bundle uses rational trigonometry plus
  projective invariants; it is not the full Hestenes/Clifford GA. If a
  spec really needs bivectors, this object model falls short — flag
  `no-mapping-exists` for that spec under Mode B.

## How Mode B scoring differs from Mode A

Mode A asked: *did the reproducer find any QA structure?* Mode B asks:
*given the Wildberger primitives, does the chosen pre-specified object
model capture the TLA+ invariants?* The axes stay the same (Recovery,
Contribution 0-4, failure taxonomy), but:

- **Recovery** is unchanged.
- **Contribution ≥ 2** requires that the reproducer's encoding *uses* the
  object-model primitives non-trivially (e.g., a quadrance-based MutEx;
  an SL(3)-ring-like decomposition of an invariant; a chromogeometric
  dual view of a binary choice). Nominal renaming still scores 0-1.
- **`ornamental-overlay`** fires when the object-model primitives were
  name-dropped without load-bearing use.
- **`no-mapping-exists`** is strengthened: it now means "even with the
  Wildberger primitives pre-specified, no defensible encoding exists."
  A firm finding, not a guess.

## Authoring note

This artifact is a **synthesis from existing in-repo material**, not new
math. Primitives split into two provenance tiers:

- **Cert-backed** (quadrances blue/red/green, Pythagorean identity, TQF,
  TQF symmetry, spread-polynomial recurrence + composition monoid, SL(3)
  hexagonal ring identity, 4D diagonal rule + Gram correspondence,
  quadruple coplanarity, mutation-game E_8 construction, hyper-Catalan
  and super-Catalan diagonal references): established in the repo's live
  cert families ([231]-[237], [240], [241], [242], [244], [246]).
- **Theory-doc / Divine-Proportions background** (spread scalar form,
  cross-ratio, translations, reflections, rotations-via-spread as a
  transform on points, generic projective maps): standard Wildberger
  primitives cited in `docs/theory/QA_UHG_PROJECTIVE_COORDINATES.md` or
  Divine Proportions / Chromogeometry papers; correct and widely used
  but not individually verified by one of the nine cited certs.

The Mode B claim this artifact makes is: *these primitives, taken
together, form a usable encoding alphabet for the blind-reproduction
benchmark*. That claim gets tested when we run Mode B on a spec — the
first test should be a TwoPhase or similar distributed protocol where
Mode A has returned `ornamental-overlay`.

## Provenance

- Author: main session claude-main-1123, 2026-04-22.
- Source docs: cited inline. No novel primitives introduced; all lifted
  from existing certs or from the Wildberger theory docs cited as
  background (Divine Proportions, Chromogeometry paper, UHG I-IV, spread-
  polynomial paper, Mutation Game paper). Transforms (translations,
  reflections, rotations-via-spread, projective maps) are Wildberger-
  standard background, not cert-verified individually.
- Fidelity audit: `fidelity_audit.md` (fresh-subagent review, aggregate
  `minor-drift` pre-fix, corrected to clean-on-load-bearing-primitives
  before this commit; two-tier provenance split per the audit's
  `inferred-not-cited` finding).
- Next step: apply this model in a Mode B blind reproduction of the
  TwoPhase spec (`qa_tla_blind/ground_truth/specifications/TwoPhase/`).
- Independence note: as a synthesis artifact authored by the main session,
  this inherits degraded independence from the Bakery Mode A interpretation
  slip. A fresh subagent should re-read this artifact against the cited
  source certs for fidelity before Mode B scores are accepted as
  calibrated.
