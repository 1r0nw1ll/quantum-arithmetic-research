# [310] QA Rational Surveying

**Family**: `qa_rational_surveying_cert_v1`  
**Depends on**: [305] Reactive Power Versor Coupling (Wildberger spread in QA)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Pythagorean identity: C²+F²=G² for all (b,e) in {1,...,24}² (576 pairs, unbounded d=b+e, exact integer arithmetic) | PASS |
| C2 | Area = 6L: right-triangle area C·F/2 = 6L where L = d·e·(d²−e²)/6; equivalently C·F = 12L exactly; triangular parcel area = 6 × cubic QA identity L | PASS |
| C3 | Primitivity criterion: gcd(C,F,G)=1 iff gcd(b,e)=1 AND b is odd; equivalently iff gcd(d,e)=1 AND d,e have different parities (Euclid's theorem in QA variables); verified over all 576 pairs, zero exceptions | PASS |
| C4 | Rational spreads: s_P = F²/G² and s_Q = C²/G² are exact Fractions in (0,1) with s_P+s_Q=1; s_R = 1 at the right angle; no π or arcsin enters the QA layer | PASS |
| C5 | Theorem NT: bearing angle arctan(F/C) is transcendental (observer projection only); Wildberger spread F²/G² is exact Fraction (QA measurement layer); bearing never re-enters discrete QA logic | PASS |

## Key results

### The squaring map

Every QA base pair (b,e) generates an integer-sided right triangle via:

```
d = b + e
C = 2·d·e          (short leg — always even)
F = d² − e² = b·a  (long leg)
G = d² + e²        (hypotenuse = Gaussian norm N(d + ie))
L = d·e·(d²−e²)/6  (cubic QA identity)
```

This is Euclid's Pythagorean parametrization expressed in QA variables. The QA system is the generator algebra for **all** integer-sided right triangles.

### Area = 6L (C2)

```
Area = C·F/2 = 2de·(d²−e²)/2 = de·(d²−e²) = 6L
```

The area of the surveying triangle equals 6 times the cubic QA identity. Witness cases:

| (b,e) | Triangle | Area | L | 6L |
|-------|----------|------|---|-----|
| (1,1) | 3-4-5 | 6 | 1 | 6 ✓ |
| (1,2) | 5-12-13 | 30 | 5 | 30 ✓ |
| (2,3) | 16-30-34 | 240 | 40 | 240 ✓ |

### Primitivity criterion (C3)

Standard Euclid: (2mn, m²−n², m²+n²) with gcd(m,n)=1 and m≢n(mod 2) generates all primitive Pythagorean triples. In QA variables (m=d, n=e):
- gcd(d,e) = gcd(b+e, e) = **gcd(b,e)**
- d≢e(mod 2) iff d−e=b is odd, i.e. **b is odd**

Therefore: **primitive ⟺ gcd(b,e)=1 AND b odd**.

| (b,e) | b odd? | gcd=1? | Predicted | Triple | Primitive? |
|-------|--------|--------|-----------|--------|------------|
| (1,1) | ✓ | ✓ | primitive | (4,3,5) | ✓ |
| (2,1) | ✗ | ✓ | scaled | (6,8,10)=2×(3,4,5) | ✗ |
| (1,2) | ✓ | ✓ | primitive | (5,12,13) | ✓ |
| (2,3) | ✗ | ✓ | scaled | (16,30,34)=2×(8,15,17) | ✗ |
| (3,5) | ✓ | ✓ | primitive | (39,80,89) | ✓ |

### Rational spreads and Theorem NT (C4, C5)

For the five canonical Heino sets:

| (b,e) | Triangle | Spread F²/G² | Bearing (×π, observer) |
|-------|----------|-------------|----------------------|
| (1,1) | (4,3,5) | 9/25 | 0.2048π |
| (1,2) | (12,5,13) | 25/169 | 0.1257π |
| (1,3) | (24,7,25) | 49/625 | 0.0903π |
| (2,3) | (30,16,34) | 64/289 | 0.1560π |
| (3,5) | (80,39,89) | 1521/7921 | 0.1444π |

The spread is always an exact rational (Fraction). The bearing is always a transcendental float. The QA layer exposes the spread; the observer layer reads the bearing.

## Connection to GIS

In Geographic Information Systems, all standard calculations — bearings, projected distances, area calculations — use continuous trigonometry (arctan, sin, cos). Under Theorem NT these are **observer projections only**. The QA substrate is the integer (b,e) lattice generating integer-sided parcels. Rational surveying (Wildberger) replaces:

- Distance → **Quadrance** Q = G² (exact integer)
- Angle → **Spread** s = F²/G² (exact Fraction)
- Bearing computation → algebraic spread law (no transcendentals)

This cert is the foundational primitive for a QA GIS arc: discrete integer geometry as the QA layer, continuous coordinate systems as observer projections.
