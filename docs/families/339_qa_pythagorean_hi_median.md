# [339] QA Pythagorean H,I Median Identity: H²+I²=2G²

**Family**: `qa_pythagorean_hi_median_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* pp.12, 32-33

> "the values of H and I are such that H²+I²=2G². This may seem trivial  
>  in that this equality can be easily derived from C²+F²=G²  
>  or (C+F)²+(C-F)²=2G²."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | H=C+F and I=|C-F| verified for 11 pairs | PASS |
| C2 | H²+I²=2G² (G is median: G²=(H²+I²)/2) verified for 12 pairs | PASS |
| C3 | {H+I, H-I}={2C, 2F}: H+I=2·max(C,F) and H-I=2·min(C,F) | PASS |
| C4 | Algebraic: (C+F)²+(C-F)²=2(C²+F²)=2G² for all coprime pairs | PASS |
| C5 | gcd(H,G)=1 and gcd(I,G)=1 (H,I coprime to hypotenuse G) for 12 pairs | PASS |

## Core Structural Result

### Definitions (C1)

$$H = C + F \quad I = |C - F|$$

H and I are the "functionally prime" numbers arising from the sum and difference of the base and altitude.

### Median Identity (C2)

$$H^2 + I^2 = 2G^2$$

This says **G² is the average of H² and I²** — the hypotenuse-square is the arithmetic mean of the two extreme squares. Iverson calls G the "median" value between I and H.

### Algebraic Proof (C4)

$$H^2 + I^2 = (C+F)^2 + (C-F)^2 = 2C^2 + 2F^2 = 2(C^2+F^2) = 2G^2 \checkmark$$

### Sum-Difference Recovery (C3)

$$\{H+I,\ H-I\} = \{2C,\ 2F\}$$

Specifically:
- $H + I = 2\cdot\max(C,F)$
- $H - I = 2\cdot\min(C,F)$

Both C and F are recoverable from H and I alone.

### Worked Examples

| Triangle | C | F | G | H=C+F | I=|C-F| | H²+I² | 2G² |
|----------|---|---|---|-------|---------|-------|-----|
| 4-3-5 | 4 | 3 | 5 | 7 | 1 | 50 | 50 ✓ |
| 12-5-13 | 12 | 5 | 13 | 17 | 7 | 338 | 338 ✓ |
| 20-21-29 | 20 | 21 | 29 | 41 | 1 | 1682 | 1682 ✓ |
| 8-15-17 | 15 | 8 | 17 | 23 | 7 | 578 | 578 ✓ |

Note: when F>C (e.g., 20-21-29), I=|C-F|=1 and H+I=42=2×21=2F (not 2C).

## Observer Projection Note (Theorem NT)

"Triangle", "hypotenuse", "median" are observer-layer labels. The causal structure: integer bead identities H=C+F, I=|C-F|, and the algebraic identity (C+F)²+(C-F)²=2(C²+F²)=2G². No continuous geometry enters.

**Depends on**: [336] Pythagorean 16 Identities; [334] Koenig Circle Nesting; [338] Pythagorean Gnomon Square
