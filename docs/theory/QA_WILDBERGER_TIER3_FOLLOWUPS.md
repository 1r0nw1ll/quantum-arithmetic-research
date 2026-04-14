# Wildberger Tier-3 Follow-Ups — Finite-Field Elliptic Curves + Sydpoints

**Status:** theory note, 2026-04-13 (≈ 23:55)
**Scope:** Two additional primary-source mappings beyond Tier-1 + Tier-2 follow-ups.
**Parent:** `QA_WILDBERGER_CORPUS_TRIAGE.md`.

---

## 1. Neuberg Cubics Over Finite Fields (Wildberger 2008)

**Source.** Wildberger, N.J. *Neuberg cubics over finite fields*, arXiv:0806.2495 (2008). Local `/tmp/wild/neuberg.pdf` (16pp).

**Abstract (verbatim).** "The framework of universal geometry allows us to consider metrical properties of affine views of elliptic curves, even over finite fields. We show how the Neuberg cubic of triangle geometry extends to the finite field situation and provides interesting potential invariants for elliptic curves, focussing on an explicit example over **F_23**. We also prove that tangent conics for a Weierstrass cubic are identical or disjoint."

**Key theorems (verbatim).**
- **Theorem 1 (Orthic triangle):** For a triangle A_1A_2A_3 with orthocenter configuration (I_0, I_1, I_2, I_3): the lines I_0I_1 and I_2I_3 are bisectors of the vertex at A_1 in the rational-trig sense s(I_0I_1, A_1A_2) = s(I_0I_1, A_1A_3).
- The explicit spread formula for such bisectors in (x, y) coordinates is integer-polynomial in the coordinates of the three vertices.
- Over F_23, the Neuberg cubic gives explicit triangle-geometric invariants attached to each point on an elliptic curve.

**Why this is high QA relevance.**

QA is intrinsically finite-field (mod 9, mod 24). The Neuberg-cubic construction over F_23 is the first Wildberger paper that *directly* studies finite-field integer invariants — exactly the regime where QA lives. Three concrete bridges:

1. **Elliptic curves over F_p are QA-compatible.** The group law on an elliptic curve E(F_p) is integer arithmetic mod p. The paper shows each point on E(F_p) carries a canonical triangle-geometric structure (its Neuberg cubic invariant). QA can read these invariants as `(b, e)`-lattice data.

2. **F_23 in particular has 23 as a prime ≠ 2, 3.** This is compatible with UHG (which requires char ≠ 2). m = 23 may be a useful modulus for QA + Neuberg bridging — 23 is also the largest prime dividing the Monster group's order (Tesla 3-6-9 orthogonal), and is the "Keely Law 17" adjacency (since 23 = 17 + 6).

3. **Tangent conics theorem:** "For a Weierstrass cubic, tangent conics are identical or disjoint." This is a dichotomy — a finite-field rigidity result. In QA terms: the tangent-conic orbits of E(F_p) partition cleanly, no partial overlap. Echoes cert [222b] (D_1-containing vs D_1-disjoint T-orbits) structure.

**Cert candidate [242] — QA_NEUBERG_CUBIC_F23_CERT.v1.**
- Witness 1: Concrete example of elliptic curve y² = x³ + ax + b over F_23 with explicit Neuberg cubic invariants (take from paper's explicit table).
- Witness 2: Orthic-triangle spread formula as integer polynomial in (x_i, y_i, R_i).
- Witness 3: Tangent-conic dichotomy proof (identical or disjoint, no partial overlap).
- Open: connect F_23 group structure to QA T-orbit partition at m = 23 (untested).

**Priority: HIGH.** This is the cleanest primary-source finite-field-integer Wildberger paper not yet mapped. Strong QA-cert territory.

## 2. Universal Hyperbolic Geometry IV: Sydpoints and Twin Circumcircles (Wildberger + Alkhaldi 2012)

**Source.** Wildberger, N.J.; Alkhaldi, A. *Universal Hyperbolic Geometry IV: Sydpoints and Twin Circumcircles*, KoG 16:43–62 (2012). Local `/tmp/wild/uhg4.pdf` (20pp).

**Abstract (verbatim).** "We introduce the new notion of **sydpoints** into projective triangle geometry with respect to a general bilinear form. These are analogs of midpoints, and allow us to extend hyperbolic triangle geometry to non-classical triangles with points inside and outside of the null conic. Surprising analogs of circumcircles may be defined, involving the appearance of pairs of twin circles, yielding in general **eight circles** with interesting intersection properties."

**Key theorem (verbatim, Theorem 1 Parallel vectors).** If vectors v and u are parallel then Q_v · Q_u = (vu)². Conversely if Q_v · Q_u = (vu)², then either v and u are parallel OR the bilinear form restricted to span(u, v) is degenerate.

**Affine spread:** s(v, u) := 1 − (v·u)² / (Q_v · Q_u). This is the **affine** version of UHG I's projective spread, applied to any bilinear form.

**Sydpoints:** midpoint analogs that handle triangles whose vertices straddle the null conic. The classical hyperbolic midpoint is undefined for such non-classical triangles; sydpoints give a replacement. Each triangle has multiple sydpoints (the "eight circles" arise from sydpoint pairs).

**QA interpretation.**

For QA tuples a = [b : e : d = b + e] with b, e ∈ {1, …, m}:

  ⟨a, a⟩ = b² + e² − d² = b² + e² − (b + e)² = **−2 · b · e**.

Since A1 forbids zero, b · e ≥ 1 always. So **⟨a, a⟩ ≤ −2 < 0 for every QA point** — QA lives strictly inside the null cone of the UHG (+,+,−) bilinear form.

Consequence: classical midpoints (defined inside the null cone) always exist for QA triangles. **Sydpoints are not needed to stay inside the QA region.** The UHG IV extension to non-classical triangles is *not* a QA direct application.

But: **twin circumcircles at m = 9.** The paper shows 8 circles per triangle (under suitable generality). At m = 9, each triangle of QA points gives 8 circles — and the 8 Satellite points are precisely the 8-point "circle" analogs in the T-orbit partition. Potential bridge: the 8-circle family of a QA triangle may equal a Satellite orbit.

**Cert candidate [243] — QA_UHG_IV_EIGHT_CIRCLE_CERT.v1** (speculative, low priority):
- Claim: Each QA triangle (three non-collinear points in {1, …, 9}²) admits 8 UHG IV circles; do any 3 of those triangles yield 8 circles that match the 8-point Satellite orbit structure?
- Witness: exhaustive computation of 8-circle structures for sample triangles at m = 9.
- Priority: **LOW**. Likely no clean correspondence; UHG IV handles configurations QA doesn't reach. Include for completeness.

## 3. Cumulative Wildberger → QA mapping state (2026-04-13 end)

| Cert | Family | Status |
|------|--------|--------|
| [231] | Hyper-Catalan diagonal correspondence | ✓ GREEN |
| [232] | UHG diagonal coincidence | ✓ GREEN |
| [233] | UHG orbit diagonal profile | ✓ GREEN |
| [234] | Chromogeometry Pythagorean identity | ✓ GREEN |
| [235] | Super Catalan diagonal (A000984) | ✓ GREEN |
| [236] | Spread polynomial composition monoid | ✓ GREEN |
| [237] | 4D diagonal rule (2-plane in R^4) | ✓ GREEN |
| [238] | Projective line metrical (optional refinement) | queued, subsumed by [232]+[236] |
| [239] | Twelve conics / twisted icosahedron | 12 dihedral orderings verified; staged for Codex |
| [240] | Diamond model of sl_3 + quark triple | staged (structural bridge) |
| [241] | Tetrahedron quadrances | exploratory, 4-vertex tetrahedron + quadrances verified |
| [242] | Neuberg cubic over F_23 | staged (HIGH priority; finite-field integer QA-native) |
| [243] | UHG IV eight circles | staged (LOW priority; QA stays inside null cone) |

**13 cert numbers tracked.** 7 shipped, 6 staged. Coverage across:
- Hyper-Catalan / Catalan / Fuss-Catalan / Super-Catalan (4 certs)
- UHG projective / diagonal / orbit structure (3 certs)
- Chromogeometry (1 cert)
- Spread polynomial monoid (1 cert)
- 4D geometry (1 cert)
- Icosahedral / dihedral orderings (1 staged)
- sl_3 quark / diamond (1 staged)
- Elliptic curves / Neuberg (1 staged)

## 4. Acquisition state

**Primary sources in `/tmp/wild/`:**
- AMM hyper-Catalan (via Playwright, open access)
- UHG I, UHG IV (arxiv + master.grad.hr)
- Chromogeometry, Simply-Laced AAM, G_2 construction, Minuscule posets, Affine/Projective, Multisets, Diamonds (sl_3) (UNSW direct)
- Spread polynomials + butterfly (arxiv)
- Super Catalan + Fourier over finite fields (arxiv)
- 4D higher-dimensional rational trigonometry (hrcak)
- Tetrahedron vector products (arxiv)
- One-dim metrical geometry (arxiv)
- Pentagrammum Mysticum (heldermann-verlag)
- Neuberg cubics over finite fields (arxiv)

**Not acquired** (paywalled, low-priority, or superseded):
- Mutation Game 2020 — worldscientific paywall; 2003 predecessor (simply-laced AAM) provides the core numbers-game framework.
- Quadrangle centroids in UHG (2016) — low novelty.
- Parabola in UHG I/II (2014/2016) — conic-adjacent, covered partially by UHG I and [235]/[236].
- Various 2018-2020 follow-ups on rotor coordinates, positive definiteness, Feuerbach theorem, Apollonian circles, Pascal hexagrammum illumination — mid-cite, extending existing directions rather than opening new ones.

## 5. References

- Wildberger, N.J. *Neuberg cubics over finite fields*. arXiv:0806.2495, 2008.
- Wildberger, N.J.; Alkhaldi, A. *Universal Hyperbolic Geometry IV: Sydpoints and Twin Circumcircles*. KoG 16:43–62, 2012.
