# Family [125] — QA_CHROMOGEOMETRY_CERT.v1

**Chromogeometric quadrances as QA invariants; Wildberger Theorem 6**

---

## What this family certifies

For any QA generator `(b, e)` with derived coordinates `d = b+e`, `a = b+2e`, the three core
QA invariants C, F, G are exactly the three chromogeometric quadrances of the direction
vector `(d, e)` in Wildberger's chromogeometry:

| QA symbol | Chromogeometric identity | Name |
|-----------|--------------------------|------|
| C | Q\_green(d, e) = 2·d·e | Green quadrance |
| F | Q\_red(d, e) = d²−e² = b·a | Red quadrance (= semi-latus product) |
| G | Q\_blue(d, e) = d²+e² | Blue quadrance |

The identity **C² + F² = G²** is Wildberger's Chromogeometric Theorem 6:

> Q\_blue(v)² = Q\_red(v)² + Q\_green(v)²  for any vector v = (d, e)

This is simultaneously a Pythagorean triple identity — the generator `(b, e)` produces the
Pythagorean triple `(F, C, G) = (d²−e², 2de, d²+e²)`.

---

## Conic discriminant

The discriminant **I = |C − F|** classifies the conic associated with the generator direction:

| Condition | I | Conic type |
|-----------|---|------------|
| C > F | C − F | Hyperbola |
| C = F | 0 | Parabola (requires b = e√2; no integer solution except b=e=0) |
| C < F | F − C | Ellipse |

The signed discriminant is `C − F = 2e² − b²`, but I is always reported as the positive
difference.

---

## Validation checks (CG1–CG7)

| ID | Check | Fail type |
|----|-------|-----------|
| CG1 | `schema_version == 'QA_CHROMOGEOMETRY_CERT.v1'` | SCHEMA\_VERSION\_WRONG |
| CG2 | `C == 2·d·e` | GREEN\_QUADRANCE\_MISMATCH |
| CG3 | `F == d²−e²` | RED\_QUADRANCE\_MISMATCH |
| CG4 | `G == d²+e²` | BLUE\_QUADRANCE\_MISMATCH |
| CG5 | `C²+F² == G²` | PYTHAGORAS\_VIOLATED |
| CG6 | `F == b·a` | SEMI\_LATUS\_MISMATCH |
| CG7 | conic\_type and I match C vs F | CONIC\_TYPE\_MISMATCH |

---

## Fixtures

| File | Generator | (C, F, G) | Result | Notes |
|------|-----------|-----------|--------|-------|
| `cg_pass_hyperbola_3_4_5.json` | b=1, e=1, d=2, a=3 | (4, 3, 5) | PASS | Anchor fixture; 3-4-5 triple; C>F → hyperbola |
| `cg_pass_ellipse_20_21_29.json` | b=3, e=2, d=5, a=7 | (20, 21, 29) | PASS | 20-21-29 triple; C<F → ellipse |
| `cg_fail_green_mismatch.json` | b=1, e=1, d=2, a=3 | C=2 (wrong) | FAIL | Canonical confusion: C=2·b·e instead of 2·d·e; triggers CG2+CG5 |

---

## Mathematical context

**Source**: Wildberger arXiv:0806.2495, arXiv:0806.2789, arXiv:math/0701338.

The three chromatic geometries (blue = standard Euclidean, red = split-complex / Minkowski,
green = multiplicative / null) arise from three isometry groups on the projective line over a
field. For the QA direction vector `(d, e)`:

- **Blue quadrance** G = d²+e² — Euclidean norm squared
- **Red quadrance** F = d²−e² — Minkowski (split-complex) norm squared
- **Green quadrance** C = 2de — multiplicative / null geometry measure

The identity C²+F²=G² is the algebraic statement that the blue quadrance is the sum of squares
of red and green quadrances — a chromogeometric Pythagoras.

**Connection to QA**: QA triples (F, C, G) = (d²−e², 2de, d²+e²) are null points [F:C:G] in
Universal Hyperbolic Geometry (UHG), satisfying the null condition F²+C²−G²=0 (see family
[127] QA\_UHG\_NULL\_CERT). The QA T-operator is a RED isometry (split-complex rotation by φ
in Z[√5]/mZ[√5]) — see family [126] QA\_RED\_GROUP\_CERT.

---

## ok semantics

`ok=True` means the certificate is internally consistent:
- detected failure types == declared `fail_ledger`
- `result` field is consistent (`"PASS"` ↔ empty ledger, `"FAIL"` ↔ non-empty ledger)

A FAIL fixture with correctly declared failures returns `ok=True` — consistency, not correctness,
is what the meta-validator checks.

---

## Cert gap this family closes

Previously noted in MEMORY.md as `QA_CHROMOGEOMETRY_CERT.v1` — no prior family directly
certifies the mapping between QA invariants C, F, G and Wildberger's chromogeometric
quadrances, or the conic discriminant I = |C−F|.
