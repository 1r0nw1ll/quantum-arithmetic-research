# Family [207] QA_CIRCLE_IMPOSSIBILITY_CERT.v1

## One-line summary

No QA state has C < 4: the circle is structurally impossible. C = 2(b+e)e >= 4 always (A1+A2). d > e always, F > 0 always — no degenerate conics exist. The circle is an observer projection of an ellipsoid where C lies along the viewing axis (Will Dale, 2026-04-08).

## Mathematical content

### Circle impossibility theorem

For any QA state (b,e) with b,e in {1,...,m}:

```
d = b + e          (A2, raw — NOT mod-reduced)
a = b + 2e         (A2, raw)
C = 2de = 2(b+e)e
```

By axiom A1 (No-Zero): b >= 1, e >= 1, so d = b+e >= 2.

```
C = 2de >= 2 * 2 * 1 = 4
```

**No QA state has C < 4.** The circle (C=0) is impossible by a margin of 4.

### No degenerate conics

Since d = b+e and b >= 1: **d > e always**. Therefore:

```
F = d*d - e*e = (d-e)(d+e) = b*(b+2e) = b*a > 0
```

F is always strictly positive. Every QA state defines a proper (non-degenerate) ellipse or hyperbola. There are no collapsed points, no lines, no circles — only proper conics.

### CRITICAL: raw vs mod-reduced coordinates

QA elements (C, F, G, etc.) are computed from **raw** derived coordinates: d = b+e, a = b+2e. The T-operator step function uses mod-reduced d = (b+e-1)%m+1 for state evolution. **These are different operations.** Elements are properties of the direction; mod-reduction is for dynamics.

### Observer projection interpretation (Will Dale)

A circle is the **side view of an ellipsoid** where C lies along the viewing axis. C is nonzero (>= 4) but hidden from the observer. The observed radius is F.

This is **Theorem NT in pure geometry**: the observer sees a circle and concludes eccentricity=0, but the discrete QA state always has C >= 4.

### Hierarchy of conic impossibilities

| Shape | Condition | Why impossible | Min gap | Cert |
|-------|-----------|----------------|---------|------|
| **Circle** | C = 0 | C = 2(b+e)e >= 4, A1+A2 | 4 | [207] |
| **Parabola** | I = C-F = 0 | d/e = 1+sqrt(2) irrational | 1 | [140] |

## Checks

| ID | Description |
|----|-------------|
| CI_1 | schema_version matches |
| CI_C_MIN | Minimum C >= 4 (claimed and computationally verified) |
| CI_EXHAUSTIVE | all_C_ge_4 == true for all S_m states |
| CI_PROJECTION | Observer projection interpretation present (Will Dale) |
| CI_HIERARCHY | Both circle and parabola impossibilities articulated |
| CI_CHROMO | Chromogeometry connection present |
| CI_SRC | Source attribution to Will Dale |
| CI_WITNESS | >= 3 witnesses with correct raw d = b+e derivation |
| CI_F | fail_ledger well-formed |

## Examples

**Minimum C** (1,1): d=2, a=3. C=4, F=3, I=1. The 3-4-5 fundamental. Hyperbolic. |I|=1 = closest to parabola.

**Elliptic** (2,1): d=3, a=4. C=6, F=8, I=-2. Note: DIRECTION (d=2,e=1) is 3-4-5, but STATE (b=2,e=1) has d=3.

**Maximum C** (9,9): d=18, a=27. C=324, F=243, I=81. Hyperbolic. Most non-circular state in S_9.

## Connection to other families

- **[140] QA_CONIC_DISCRIMINANT_CERT.v1**: Parabola impossibility (I=0). This cert adds circle impossibility (C=0) with wider margin (4 vs 1).
- **[189] QA_DALE_CIRCLE_CERT.v1**: Dale's pi=1 works because the circle is always a projection.
- **[125] QA_CHROMOGEOMETRY_CERT.v1**: I=Qg-Qr. Both always positive, never equal to 0.
- **[208] QA_QUADRANCE_PRODUCT_CERT.v1**: Companion cert — product irreducibility.

## Source

Will Dale, 2026-04-08/09. Corrected 2026-04-09: C >= 4 (not 2) — elements use raw d = b+e.
