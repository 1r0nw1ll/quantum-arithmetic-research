# Family [127] — QA_UHG_NULL_CERT.v1

**QA triples as null points in Universal Hyperbolic Geometry; Gaussian integer interpretation**

---

## What this family certifies

Every QA triple `(F, C, G) = (d²−e², 2de, d²+e²)` is a **null point** `[F:C:G]` in
Universal Hyperbolic Geometry (UHG), satisfying the UHG null condition:

```
F² + C² − G² = 0
```

This is the same as C² + F² = G² (family [125] Chromogeometric Pythagoras). The new
content is the UHG interpretation: `[F:C:G]` is a projective null point, meaning the
direction vector `(F, C, G)` is **light-like** in Minkowski 3-space with signature (+,+,−).

---

## Gaussian integer interpretation

The triple `(F, C, G)` arises from the **Gaussian integer** `Z = d + e·i`:

| Component | Formula | Meaning |
|-----------|---------|---------|
| F | Re(Z²) = d²−e² | Real part of Z squared |
| C | Im(Z²) = 2de | Imaginary part of Z squared |
| G | \|Z\|² = d²+e² | Norm squared of Z |

The null condition F²+C²−G²=0 is then:

```
|Re(Z²)|² + |Im(Z²)|² = |Z²|²    (modulus identity for complex numbers)
```

which holds automatically since `|Z²| = |Z|²`.

The **join of null points** in UHG corresponds to Gaussian integer multiplication:
if `Z₁ = d₁+e₁·i` and `Z₂ = d₂+e₂·i`, then the null point from `Z₁·Z₂` encodes
the composition of the two QA generators.

---

## Validation checks (UN1–UN7)

| ID | Check | Fail type |
|----|-------|-----------|
| UN1 | `schema_version == 'QA_UHG_NULL_CERT.v1'` | SCHEMA\_VERSION\_WRONG |
| UN2 | `C == 2·d·e` (Green quadrance = Im(Z²)) | GREEN\_QUADRANCE\_MISMATCH |
| UN3 | `F == d²−e²` (Red quadrance = Re(Z²)) | RED\_QUADRANCE\_MISMATCH |
| UN4 | `G == d²+e²` (Blue quadrance = \|Z\|²) | BLUE\_QUADRANCE\_MISMATCH |
| UN5 | `F²+C²−G² == 0` (UHG null condition) | NULL\_CONDITION\_VIOLATED |
| UN6 | `F=Re(Z²)`, `C=Im(Z²)`, `G=|Z|²` for `Z=d+e·i` | GAUSSIAN\_DECOMP\_MISMATCH |
| UN7 | `null_quadrance field == 0` (if present) | NULL\_QUADRANCE\_WRONG |

---

## Fixtures

| File | (d, e) | (F, C, G) | Result | Notes |
|------|--------|-----------|--------|-------|
| `un_pass_3_4_5.json` | d=2, e=1 | (3, 4, 5) | PASS | 3-4-5 triple; Z=2+i, Z²=3+4i |
| `un_pass_5_12_13.json` | d=3, e=2 | (5, 12, 13) | PASS | 5-12-13 triple; Z=3+2i, Z²=5+12i |
| `un_fail_null_violated.json` | d=2, e=1 | G=6 (wrong) | FAIL | BLUE\_QUADRANCE\_MISMATCH + NULL\_CONDITION\_VIOLATED + GAUSSIAN\_DECOMP\_MISMATCH |

---

## Mathematical context

**Sources**: Wildberger arXiv:0909.1377 (UHG I), arXiv:0806.2789 (Chromogeometry Conics),
arXiv:math/0701338 (1D Metrical Geometry).

**UHG null points**: In UHG with bilinear form `λ(X,Y) = X₁Y₁+X₂Y₂−X₃Y₃` (Minkowski
signature), a point `[X:Y:Z]` is **null** if `λ([X:Y:Z],[X:Y:Z]) = X²+Y²−Z² = 0`.
The set of null points forms the **absolute conic**, which plays the role of the
boundary at infinity in hyperbolic geometry.

**QA triples are null**: Setting `X=F`, `Y=C`, `Z=G`:
- F²+C²−G² = (d²−e²)²+(2de)²−(d²+e²)² = 0 (algebraic identity, always 0)

So ALL QA triples are automatically null. This means QA arithmetic lives on the absolute
conic of UHG — QA is the arithmetic of points at infinity in the universal hyperbolic plane.

**Connection to families [125] and [126]**:
- [125] proves C²+F²=G² (chromogeometric Pythagoras) — equivalent to the null condition
- [126] proves the T-operator is a red isometry — the dynamics preserve the Minkowski structure
- [127] (this family) names the geometric object: QA triples = UHG null points = absolute conic
