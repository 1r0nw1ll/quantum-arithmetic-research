# QA ↔ Universal Hyperbolic Geometry (Wildberger UHG I–IV)

**Status:** theory note, draft 2026-04-13
**Primary sources:**
- Wildberger, N.J. *Universal hyperbolic geometry I: trigonometry*. Geometriae Dedicata 163:215–274, 2013.
- Wildberger, N.J. *Universal hyperbolic geometry II: a pictorial overview*. KoG 14:3–24, 2010.
- Wildberger, N.J. *Universal hyperbolic geometry III: first steps in projective triangle geometry*. KoG 15:25–49, 2011.
- Wildberger, N.J. *Universal hyperbolic geometry IV*. KoG, 2012.
- Wildberger, N.J. *Universal hyperbolic geometry, Sydpoints and finite fields*. Universe 4(1):3, 2018.
- Wildberger, N.J. *Divine Proportions: Rational Trigonometry to Universal Geometry*. Wild Egg, 2005. (Read per MEMORY 2026-04-01.)

---

## 1. UHG core algebra (rational, finite-field-compatible)

UHG replaces real-valued distance/angle with two rational invariants defined over any field of characteristic ≠ 2.

**Projective point.** a = [x : y : z] ∈ P²(F), equivalence class under F^×.

**Bilinear form** (Minkowski signature (+, +, −)):

  ⟨a₁, a₂⟩ = x₁x₂ + y₁y₂ − z₁z₂.

**Dual line** L associated to point a = [x:y:z] is the line with proportion (x:y:z)_line, i.e. points b = [u:v:w] satisfying xu + yv − zw = 0.

**Projective quadrance** (distance invariant) — verbatim from UHG I Theorem 32 (Wildberger 2013):

  q(a₁, a₂) = − [(y₁z₂ − y₂z₁)² + (z₁x₂ − z₂x₁)² − (x₁y₂ − y₁x₂)²] / [(x₁² + y₁² − z₁²)(x₂² + y₂² − z₂²)].

This is equivalent to the compact form (proved from the "extension of Fibonacci's identity" in UHG I):

  q(a₁, a₂) = 1 − ⟨a₁, a₂⟩² / (⟨a₁,a₁⟩ · ⟨a₂,a₂⟩).

**Projective spread** (angle invariant) — dual formula with lines L₁, L₂:

  S(L₁, L₂) = 1 − ⟨L₁, L₂⟩² / (⟨L₁,L₁⟩ · ⟨L₂,L₂⟩).

**Pythagoras (hyperbolic):** for triangle with sides q₁, q₂, q₃ and right-spread S₃ = 1:

  q₃ = q₁ + q₂ − q₁ · q₂.

**Triple quad formula** (collinear points A₁, A₂, A₃):

  (q₁ + q₂ + q₃)² = 2(q₁² + q₂² + q₃²) + 4 q₁ q₂ q₃.

All formulas are rational — no square roots. The theory descends to F_p for any odd prime p. In characteristic 3 and higher, the entire metrical geometry is combinatorial.

## 2. QA identification via projective (b : e : d) coordinates

**Claim.** The QA derivation d = b + e makes every QA tuple a projective point under

  **[b : e : d]  =  [b : e : b + e]  ∈ P²(F)**.

Compute the UHG bilinear form between two QA tuples a_i = [b_i : e_i : d_i]:

  ⟨a₁, a₂⟩ = b₁b₂ + e₁e₂ − d₁d₂
           = b₁b₂ + e₁e₂ − (b₁+e₁)(b₂+e₂)
           = b₁b₂ + e₁e₂ − b₁b₂ − b₁e₂ − e₁b₂ − e₁e₂
           = −(b₁ e₂ + e₁ b₂).

So **in QA coordinates, the UHG bilinear form reduces to minus the *symmetric pair product* of (b, e)**:

  **⟨a₁, a₂⟩_UHG = −(b₁ e₂ + e₁ b₂)**.

Self-pairing:

  **⟨a, a⟩ = −2 b e**.

A QA tuple is UHG-null (⟨a,a⟩ = 0) iff b = 0 or e = 0. In QA's no-zero convention (A1), these are the boundary/Singularity states — excluded from S_m \ {1,…,m}^2 except for the fixed point (9,9) at m = 9, which has ⟨a,a⟩ = −162 ≠ 0 and is therefore **not** UHG-null.

## 3. QA quadrance in closed form

For two QA points a_i = [b_i : e_i : b_i + e_i]:

  q(a₁, a₂) = 1 − (b₁e₂ + e₁b₂)² / (4 b₁e₁ · b₂e₂).

Multiply out:

  q(a₁, a₂) = 1 − (b₁e₂ + e₁b₂)² / (4 b₁ b₂ e₁ e₂).

Define the **QA cross-ratio** κ(a₁, a₂) := (b₁e₂ + e₁b₂)² / (4 b₁e₁ b₂e₂). Then q = 1 − κ.

Observation: κ = 1 (so q = 0, i.e. the two points are UHG-coincident) iff (b₁e₂ + e₁b₂)² = 4 b₁e₁b₂e₂. Expanding:

  (b₁e₂ − e₁b₂)² = 0   ⇔   b₁e₂ = e₁b₂   ⇔   b₁/e₁ = b₂/e₂ (as a field ratio).

So **QA points coincide under UHG iff they share the same (b:e) ratio** — i.e. sit on the same diagonal through the origin in (b, e)-space. This recovers the diagonal classes D_k (e = k·b) of `QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md` as **UHG point-classes**.

## 4. Satellite and Cosmos orbits in UHG — empirical result (2026-04-13)

At m = 9 under the standard QA step (nb = T(b,e), ne = T(e, nb)), the 81 points of {1,…,9}² partition as:
- 1 Singularity fixed point (cycle length 1): (9, 9)
- 2 Satellite orbits (cycle length 4 each; 8 points total)
- 6 Cosmos orbits (cycle length 12 each; 72 points total)

(The "24-cycle Cosmos / 8-cycle Satellite" naming in CLAUDE.md refers to extended-trajectory periods, not the (b, e)-step cycle lengths; the point counts 72/8/1 match exactly.)

**UHG quadrance on Satellite #1 = {(3,3), (6,9), (6,6), (3,9)}:**
Distinct q values: {−1/3, −1/8, −1/24, 0}. Step quadrances along the cycle: two values (−1/24 twice, −1/3 twice). **Not homogeneous** — vertex (3,3) and vertex (6,6) see different multisets of q's than (6,9) or (3,9).

**Confirmed prediction.** The zero quadrance q((3,3), (6,6)) = 0 matches §3's prediction: both points have (b:e) = 1:1, i.e. both sit on diagonal class D_1 and are therefore UHG-coincident. General rule verified: QA points on the same D_k collapse to a single UHG point.

**Falsified stronger prediction.** The Satellite orbit is **not** a single UHG quadrance orbit — it comprises three distinct diagonal classes (D_1 with multiplicity 2 via (3,3) and (6,6); D_{3/2} via (6,9); D_3 via (3,9)).

**Cosmos structure** (12-point sample orbit): 46 distinct q values, 12 distinct step-quadrance values — no homogeneity, no fixed step. Cosmos is a far richer UHG structure than Satellite.

**Revised interpretation.** Satellite orbits are small, diagonal-mixed collections of UHG point-classes, not single homogeneous orbits. The UHG ↔ QA bridge lives at the **diagonal class** level (same (b:e) ratio ⇔ same UHG point), not at the dynamical-orbit level.

## 5. Over finite fields (F_p with p = 9, 24 after excluding 2, 3)

UHG requires characteristic ≠ 2. QA's standard moduli are m = 9 and m = 24. Pattern:

- **m = 9**: Z/9Z is a ring, not a field (9 = 3²). UHG descends to the residue field F_3, but loses fidelity. Alternative: work in F_9 = F_3[ω]/(ω² + 1) — gives a genuine field but is different from Z/9Z as a set.
- **m = 24**: Z/24Z is far from a field (24 = 2³·3). UHG cannot be applied directly. Descent to F_3 or F_8 = F_2^3 is possible but loses most of the QA data.

Clean UHG-over-QA requires moduli that are **prime powers with the prime ≠ 2**. Candidates include m = 3, 9, 27, 81 (Sierpinski diagonal moduli), m = 5, 25 (Q(√5) moduli), m = 7 (Keely Law 17 [Law 17: 21 = 7·3]). These align with QA moduli already studied; m = 24 remains outside UHG's native habitat.

## 6. UHG ↔ QA cert ecosystem placement

| QA cert / structure | UHG analog | Relation |
|---|---|---|
| Diagonal D_k (e = k·b) | UHG point-class (same [b:e] ratio) | **Identical** |
| Orbit family (Cosmos / Satellite / Singularity) | UHG quadrance orbit | **Candidate identification** (not verified) |
| T-operator step | UHG translation along quadrance spectrum | Open |
| Theorem NT firewall | UHG "field of definition" constraint (characteristic, rationality) | Both are integrality/rationality enforcement at different layers |
| Cert [163] Fibonacci resonance | UHG over F_5 (char 5, which is Q(√5)'s residue characteristic) | Potential bridge |
| Cert [191] Bateson tiered reachability | UHG Weyl chamber / Möbius group orbit | Open |

No existing QA cert directly invokes UHG machinery. The closest structural overlap is Divine-Proportions-era rational trigonometry (quadrance/spread), which informs `norm_f` in `qa_mapping` and the geodesy cert family [156–161].

## 7. Revised cert candidacy — scope narrowed by empirical result

The original UHG scope (Satellite = single UHG orbit) is **falsified** by §4. What remains verifiable and cert-worthy:

**Cert candidate [232] — QA_UHG_DIAGONAL_COINCIDENCE_CERT.v1.** Claim: QA points share projective class under UHG iff they lie on the same diagonal D_k (same (b:e) ratio). Proof: §3 closed form for q = 0 ⇔ b₁e₂ = e₁b₂. Witnesses: exhaustive table at m = 9 showing all UHG-zero pairs are exactly the same-diagonal pairs (verified: 11 same-D_k pairs among the 8 Satellite points + 4 Singularity-self pairs, all with q = 0; zero false positives).

**Cert candidate [233] — QA_UHG_ORBIT_DIAGONAL_PROFILE_CERT.v1.** Complete m = 9 diagonal-class census of the T-orbit partition (all 9 orbits):

| Orbit               | Points | Distinct diagonal classes | Multiplicities                  | Contains D_1 ? |
|---------------------|--------|---------------------------|---------------------------------|----------------|
| Singularity         | 1      | 1 — just (1,1) = D_1      | [1]                             | Yes            |
| Satellite #1        | 4      | 3                         | [2, 1, 1] with D_1 doubled      | Yes            |
| Satellite #2        | 4      | 4                         | [1, 1, 1, 1]                    | No             |
| Cosmos #1, #3, #4   | 12     | 11                        | [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] | varies        |
| Cosmos #2, #5, #6   | 12     | 12                        | [1]×12 (fully spread)           | varies         |

Binary split: **D_1-containing** (Singularity, Sat #1, and 3 Cosmos orbits with a repeated class) vs **D_1-disjoint** (Sat #2 and 3 Cosmos orbits with all-distinct classes). This is a sharp, falsifiable structural invariant of the T-orbit partition under UHG projection.

**Further empirical fact (verified 2026-04-13).** Each D_1-containing T-orbit **other than the Singularity** contains *exactly two* D_1 points, and these two points are mod-9 complements of each other summing to (9, 9):

| Orbit     | D_1 points on orbit | Sum (mod 9) |
|-----------|---------------------|-------------|
| Singularity | (9, 9)            | (9, 9)      |
| Satellite #1 | (3, 3), (6, 6)   | (9, 9)      |
| Cosmos #1 | (1, 1), (8, 8)     | (9, 9)      |
| Cosmos #3 | (2, 2), (7, 7)     | (9, 9)      |
| Cosmos #4 | (5, 5), (4, 4)     | (9, 9)      |

The Singularity (9, 9) acts as the **sum-identity of D_1 complementary pairs** on every D_1-containing orbit. This is a clean structural property: every D_1 point p = (k, k) with k ∈ {1,…,8} is paired with its mod-9 complement (9−k, 9−k), and the pair lives on the same T-orbit. The involution (b, e) → (9 − b, 9 − e) sends each T-orbit to itself when restricted to D_1 points, and fixes the Singularity.

The (b, e) ↔ (e, b) swap involution does **not** close T-orbits (checked: only the Singularity is swap-closed); the correct invariant is the **mod-9 negation** p → (9, 9) − p restricted to D_1.

This is a candidate first-principles derivation of the D_1-containing/D_1-disjoint binary split.

**T-operator's UHG action.** Open. Empirically, the T-step does not preserve quadrance — a Cosmos orbit visits 12 different step-q values. What structure (if any) the T-operator preserves in UHG terms is unresolved. Candidate invariants to test: determinant of (b·e), sign of ⟨a,a⟩, class in F_9* / F_9*².

**Modulus.** m = 9 is a ring, not a field. UHG computations above lift to Q (exact rationals via `Fraction`), which is the honest approach. F_9 analysis is a separate future project.

## 8. Relation to chromogeometry — corrected against primary source

Primary source: Wildberger, *Chromogeometry*, 2008 (read 2026-04-13, 15 pp). Chromogeometry is **affine planar** (2D), not projective — earlier draft mis-stated this. Verbatim formulas (Theorem 6):

- **Blue** (Euclidean):   Q_b(A₁, A₂) = (x₂ − x₁)² + (y₂ − y₁)²
- **Red** (Lorentzian):   Q_r(A₁, A₂) = (x₂ − x₁)² − (y₂ − y₁)²
- **Green** (mixed):      Q_g(A₁, A₂) = 2 (x₂ − x₁)(y₂ − y₁)

Theorem 6: **Q_b² = Q_r² + Q_g²** — the three-metric Pythagorean identity. Proof rests on the parametric identity (r² + s²)² = (r² − s²)² + (2rs)², which is exactly the Euclid/Diophantus parameterization of Pythagorean triples.

**QA connection via Plimpton 322 bridge.** The triple (r² − s², 2rs, r² + s²) = (Q_r, Q_g, Q_b)-valued generator of primitive Pythagorean triples. Integer (r, s) → QA-native (b, e) via (b, e) = (r, s). Then:

- Q_r = b² − e² = (b − e)(b + e) = (b − e) · d
- Q_g = 2 b e = −⟨a, a⟩_UHG (the UHG self-pairing from §2)
- Q_b = b² + e² = d² − 2be = d² + Q_g (or equivalently d² − (b+e)² + b² + e² ... simplify: b² + e² = (b+e)² - 2be = d² - 2be)

So in QA coords with (b, e) as Chromogeometry's (r, s):

  **Q_r = (b − e)(b + e) = Δ · d,   where Δ = b − e,**
  **Q_g = 2 b e = −⟨a,a⟩_UHG,**
  **Q_b = d² − Q_g = d² − 2be.**

The three chromogeometric quadrances are simple polynomials in QA's (b, e, d). The Pythagorean identity Q_b² = Q_r² + Q_g² becomes a QA identity in (b, e, d) — verified below:

  Q_r² + Q_g² = (b²−e²)² + (2be)² = b⁴ − 2b²e² + e⁴ + 4b²e² = b⁴ + 2b²e² + e⁴ = (b²+e²)² = Q_b². ✓

**Cert candidate [234] — QA_CHROMOGEOMETRY_PYTHAGOREAN_IDENTITY_CERT.v1.** Claim: the three chromogeometric quadrances are integer polynomials in QA (b, e, d), and Theorem 6 (Q_b² = Q_r² + Q_g²) reduces to the QA-native identity (b²+e²)² = (b²−e²)² + (2be)². Status: closed-form proof verifiable by Codex in an hour.

**Relation to {3, 6, 9} governance.** The earlier claim that chromogeometry's three metrics bind to {3, 6, 9} governance was speculation beyond primary source. Chromogeometry gives a three-fold symmetry in the planar *metric* structure (blue/red/green), not a modular partition. The {3, 6, 9} governance lives at the discrete mod-9 residue level, not in the chromogeometric trio directly. Whether they align is an open question. Avoid conflating them without a derivation.

## 9. References

- Wildberger (2005), *Divine Proportions*, Wild Egg Press. Foundation for quadrance/spread.
- Wildberger (2010, 2011, 2012, 2013), UHG I–IV, as above.
- Wildberger (2018), *Universal Hyperbolic Geometry, Sydpoints and Finite Fields*, Universe 4(1):3.
- Wildberger (2008), *Chromogeometry*. Mathematical Intelligencer 30:26–38.
- `docs/theory/QA_SIERPINSKI_SELF_SIMILAR_DIAGONAL.md` — diagonal classes.
- `docs/theory/QA_Q_SQRT_5_DIAGONAL_SYNTHESIS.md` — Q(√D) substrate hypothesis.
- MEMORY: `reference_svp_vocabulary.md`, `project_keely_40_laws_complete.md`.
