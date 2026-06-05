# [336] QA Pythagorean 16 Identities: Sum-Difference Squares

**Family**: `qa_pythagorean_16_identities_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* pp.12-13 Ch.1 "Preview"

> "a base C which equals 2de; an altitude F which equals ab; and a hypotenuse G  
>  which equals d²+e². Alternatively, the sum of the hypotenuse and base is a²;  
>  the difference of the hypotenuse and base is b²; the sum of the hypotenuse  
>  and altitude is 2d²; the difference of the hypotenuse and altitude is 2e²;  
>  and the sum and difference of the altitude and base are functionally prime  
>  numbers, H and I."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | C=2de, F=ab, G=d²+e²; C²+F²=G² verified for 8 pairs | PASS |
| C2 | G+C=a² and G-C=b² verified for 13 bead pairs | PASS |
| C3 | G+F=2d² and G-F=2e² verified for 13 bead pairs | PASS |
| C4 | All six identities + L=abde/6=CF/12 hold simultaneously | PASS |
| C5 | Algebraic proof: G±C and G±F identities follow purely from d=b+e, a=d+e | PASS |

## Core Structural Result

### Primary Definitions (C1)

From bead numbers (b, e, d, a) with **d=b+e, a=b+2e** (A2 raw):

| Identity | Formula | Role |
|----------|---------|------|
| C | 2de | Base (always 4-par) |
| F | ab | Altitude |
| G | d²+e² | Hypotenuse (always 5-par) |

$$C^2 + F^2 = G^2$$

### Sum-Difference Identities (C2, C3)

Six identities linking the triangle's sides to squares of bead numbers:

$$G + C = a^2 \quad G - C = b^2$$
$$G + F = 2d^2 \quad G - F = 2e^2$$

### Algebraic Proof (C5)

These follow purely from the definitions d=b+e and a=b+2e=d+e:

| Identity | Proof |
|----------|-------|
| G+C = a² | (d²+e²)+2de = (d+e)² = a² ✓ |
| G-C = b² | (d²+e²)-2de = (d-e)² = b² ✓ |
| G+F = 2d² | (d²+e²)+ab = d²+(e²+b²+2be) = d²+d² = 2d² ✓ |
| G-F = 2e² | (d²+e²)-ab = d²+e²-b²-2be = 2e² ✓ |

### Worked Example: (b,e)=(3,4) → triangle (20-21-29)

| Quantity | Value |
|----------|-------|
| d=b+e | 7 |
| a=b+2e | 11 |
| C=2de | 56 |
| F=ab | 33 |
| G=d²+e² | 65 |
| G+C | 121 = 11² = a² ✓ |
| G-C | 9 = 3² = b² ✓ |
| G+F | 98 = 2×49 = 2×7² = 2d² ✓ |
| G-F | 32 = 2×16 = 2×4² = 2e² ✓ |

### Area Identity (C4)

$$L = \frac{abde}{6} = \frac{CF}{12}$$

Both formulas give the same integer L for every primitive Pythagorean triple.

## Observer Projection Note (Theorem NT)

"Triangle", "base", "altitude", "hypotenuse", "ellipse" are observer-layer labels. The causal structure: four bead numbers (b,e,d,a) and six algebraic identities among them. C²+F²=G² is not a geometric theorem here — it is a polynomial identity in integers. No continuous geometry enters.

**Depends on**: [135] Pythagorean Tree; [151] QA Par Numbers; [334] Koenig Circle Nesting
