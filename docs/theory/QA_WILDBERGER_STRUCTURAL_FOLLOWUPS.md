# Wildberger Structural Follow-Ups — [240] Ring + [241] TQF Identities

**Status:** theory note, 2026-04-14
**Parent docs:** `QA_WILDBERGER_TIER2_FOLLOWUPS.md`, `QA_WILDBERGER_CORPUS_TRIAGE.md`
**Triggers:** cert [240] QA_DIAMOND_SL3_IRREP_DIMENSION, cert [241] QA_QUADRUPLE_COPLANARITY.

---

## 1. [240] follow-up — sl(3) hexagonal ring size identity

**Base cert [240].** dim π[a, b] = (a+1)(b+1)(a+b+2)/2 matches 22/22 standard Fulton–Harris entries.

**Refinement.** Consecutive-shell differences dim π[a, b] − dim π[a−1, b−1] measure the "outermost hexagonal ring" of the sl(3) weight diagram when (a, b) are both ≥ 1. Expanding:

  ring(a, b) := dim π[a, b] − dim π[a−1, b−1]
            = (a + 1)(b + 1)(a + b + 2)/2 − a·b·(a + b)/2
            = (a² + b² + 4ab + 3a + 3b + 2) / 2
            = (a + b + 1)(a + b + 2)/2 + a·b
            = **T_{d + 1} + a·b**    where d = a + b, T_n = n(n + 1)/2.

In QA coordinates (b_QA, e_QA) = (a, b):

  **ring = T_{d+1} + b · e.**

**Empirical verification.** Exhaustive on (a, b) ∈ [1, 14]², 196 entries, zero failures.

**Samples:**

| (a, b) | ring | T_{d+1} + ab |
|--------|------|--------------|
| (1, 1) | 7    | T₃ + 1 = 6 + 1 = 7 |
| (1, 2) | 12   | T₄ + 2 = 10 + 2 = 12 |
| (2, 2) | 19   | T₅ + 4 = 15 + 4 = 19 |
| (3, 3) | 37   | T₇ + 9 = 28 + 9 = 37 |
| (5, 5) | 91   | T_{11} + 25 = 66 + 25 = 91 |

**Cert candidate [245] — QA_SL3_HEXAGONAL_RING_IDENTITY_CERT.v1.**
- Witness 1: Algebraic identity (a+1)(b+1)(a+b+2)/2 − a·b·(a+b)/2 = (a+b+1)(a+b+2)/2 + a·b, expandable symbolically.
- Witness 2: Exhaustive check on (a, b) ∈ [1, 14]², 196 zero failures.
- Witness 3: QA coordinate form ring = T_{d+1} + b·e ∈ Z_{>0}, integer polynomial in (b, e, d).
- Witness 4: Ring sizes for π[1,1] = 7 (adjoint minus trivial), π[2,2] = 19, π[3,3] = 37, known multiplicities.

**Significance.** The weight diagram of π[a, b] decomposes into concentric hexagonal rings. The outer ring is a triangular-number portion (d+1 lattice points per side, T_{d+1} total) plus a "cross-product" bilinear term b·e. This is a structural decomposition into a **D-component** (triangular number depending only on the sum d) and a **B-component** (bilinear term b·e vanishing on the boundary b = 0 or e = 0).

## 2. [241] follow-up — chromogeometric Triple Quad symmetry

**Base cert [241].** 4-point Cayley–Menger determinant vanishes under all three chromogeometric quadrances (Q_b, Q_r, Q_g) — i.e. every four QA points are coplanar in R³.

**Refinement.** Define the Triple Quad Formula (TQF) residue for three points:

  TQF(Q₁, Q₂, Q₃) := (Q₁ + Q₂ + Q₃)² − 2(Q₁² + Q₂² + Q₃²)

TQF = 0 iff the three points are collinear (rational-trig Triple Quad Formula). On a generic triangle, TQF is nonzero and measures non-collinearity.

**New identity.** For any three points (x_i, y_i) ∈ Z² (in particular any three QA points):

  **TQF_r(P₁, P₂, P₃) = TQF_g(P₁, P₂, P₃) = − TQF_b(P₁, P₂, P₃).**

**Symbolic proof.** Computing TQF over the three chromogeometric quadrances in sympy on generic (x_i, y_i):

  TQF_b = 4 · (x₁y₂ − x₁y₃ − x₂y₁ + x₂y₃ + x₃y₁ − x₃y₂)² = **16 · (signed triangle area)²**

  TQF_r + TQF_b ≡ 0  (simplified to identical zero)
  TQF_g + TQF_b ≡ 0  (simplified to identical zero)

Empirical cross-check: 3000 random triangles from [1..9]², zero violations.

**Sample.** For pts = [(2,7), (5,5), (8,2)]:
- Q_b pairs: 13, 13, 52. TQF_b = (78)² − 2·(169 + 169 + 2704) = 6084 − 6084 = 0. Wait — these are collinear!
  (2,7)→(5,5)→(8,2): deltas (3,−2),(3,−2). Yes collinear. So TQF should be 0. Rechecking:
  (2+5+8)/3 = 5, (7+5+2)/3 = 14/3 — not on average line. Actually slopes from (2,7)-(5,5) = -2/3, (5,5)-(8,2) = -1 — not collinear.
  (x1,y1)=(2,7), (x2,y2)=(5,5), (x3,y3)=(8,2). Signed area = |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|/2 = |2(3) + 5(-5) + 8(2)|/2 = |6 - 25 + 16|/2 = 3/2 → nonzero. Not collinear.
  Empirical: TQF_b = 36, TQF_r = TQF_g = −36. Matches sign identity.

**Cert candidate [246] — QA_CHROMOGEOMETRIC_TQF_SYMMETRY_CERT.v1.**
- Witness 1: Symbolic expansion (sympy) showing TQF_r + TQF_b ≡ 0 and TQF_g + TQF_b ≡ 0 as polynomial identities in (x₁, y₁, x₂, y₂, x₃, y₃).
- Witness 2: Factored form TQF_b = 16·A² where A = (x₁y₂ − x₁y₃ − x₂y₁ + x₂y₃ + x₃y₁ − x₃y₂)/2 = signed triangle area. Always non-negative.
- Witness 3: Exhaustive random sample (e.g. 3000 triangles from [1..9]²) with zero violations.
- Witness 4: Collinearity corollary — TQF_b = 0 iff triangle degenerates; by the identity, TQF_r = TQF_g = 0 too. So collinearity is chromogeometry-invariant.

**Significance.** The three chromogeometric metrics (blue Euclidean, red Lorentzian, green mixed) assign *equal-magnitude, opposite-sign* triple-quad-residues to any triangle. The blue quadrance gives a positive-definite form; the red and green each give forms whose TQF is the negative of the blue. This is a **three-fold algebraic symmetry** of planar triangles under chromogeometry, extending the segment-level identity Q_b² = Q_r² + Q_g² (cert [234]) to triangle-level.

Collinearity is invariant across blue/red/green — any chromogeometric metric agrees on which triples are collinear. But triangular "quadrances" carry a sign that flips under Lorentzian/green relative to Euclidean.

## 3. Cumulative Wildberger → QA cert state (2026-04-14)

| # | Family | Status |
|---|--------|--------|
| [231] | Hyper-Catalan diagonal correspondence | ✓ GREEN |
| [232] | UHG diagonal coincidence | ✓ GREEN |
| [233] | UHG orbit diagonal profile + (9,9)-complement law | ✓ GREEN |
| [234] | Chromogeometry Pythagorean identity | ✓ GREEN |
| [235] | Super Catalan diagonal (A000984) | ✓ GREEN |
| [236] | Spread polynomial composition monoid | ✓ GREEN |
| [237] | 4D diagonal rule (2-plane in R⁴) | ✓ GREEN |
| [239] | Twelve dihedral orderings | ✓ GREEN |
| [240] | Diamond sl(3) irrep dimension (22/22) | ✓ GREEN |
| [241] | Quadruple coplanarity (Cayley-Menger) | ✓ GREEN |
| [242] | Neuberg cubic over F_23 | ✓ GREEN |
| [244] | Mutation Game root lattice (E_8 via mutations) | staged (2020 paper) |
| [245] | sl(3) hexagonal ring = T_{d+1} + b·e | **ready — verified** |
| [246] | Chromogeometric TQF symmetry TQF_r = TQF_g = −TQF_b | **ready — proved** |

**11 green + 3 staged = 14 cert numbers tracked.**

## 4. References

- Wildberger, N.J. *Quarks, diamonds, and representations of sl_3*, UNSW preprint, 2003. (Base for [240].)
- Wildberger, N.J. *Chromogeometry*, Math. Intelligencer 30:26–38, 2008. (Base for [234].)
- Wildberger, N.J. *Divine Proportions: Rational Trigonometry to Universal Geometry*, Wild Egg, 2005. (Triple Quad Formula.)
- Notowidigdo, G.A.; Wildberger, N.J. *Generalised vector products and metrical trigonometry of a tetrahedron*, arXiv:1909.08814, 2021. (Base for [241].)
