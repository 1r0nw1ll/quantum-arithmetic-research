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
