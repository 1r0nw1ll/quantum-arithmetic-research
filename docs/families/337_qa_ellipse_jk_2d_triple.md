# [337] QA Ellipse J,K Parameters and 2D Triple Decomposition

**Family**: `qa_ellipse_jk_2d_triple_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* pp.37-38, 43-44

> "J=D-C/2 and K=D+C/2. As related to the parametric numbers they are: J=bd and K=ad."  
> "F+G = J+K = 2J+C = 2D (each equals 2d²)"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | J=bd=D-C/2 and K=ad=D+C/2; verified for all test pairs | PASS |
| C2 | J+K=2D=2d² via a+b=2d; 10 pairs | PASS |
| C3 | 2J+C=2D=2d²; 2bd+2de=2d(b+e)=2d² | PASS |
| C4 | F+G=J+K=2J+C=2D simultaneously for 8 pairs | PASS |
| C5 | Algebraic: d-e=b→J=d(d-e)=bd; d+e=a→K=d(d+e)=ad | PASS |

## Core Structural Result

### J and K Definitions (C1)

From D=d²=d·d and C=2de:

$$J = D - \frac{C}{2} = d^2 - de = d(d-e) = db \quad (\text{since } d-e = b)$$
$$K = D + \frac{C}{2} = d^2 + de = d(d+e) = da \quad (\text{since } d+e = a)$$

J and K are the **semi-intervals** at the two foci of the ellipse (observer-layer label).

### Three Decompositions of 2D = 2d² (C2, C3, C4)

$$F + G = J + K = 2J + C = 2D = 2d^2$$

| Decomposition | Formula | Structural reason |
|--------------|---------|-------------------|
| F + G | ab + (d²+e²) | cert [336] G+F=2d² |
| J + K | bd + ad = d(a+b) | a+b = (b+2e)+b = 2d |
| 2J + C | 2bd + 2de = 2d(b+e) | b+e = d |

All three equal 2D because they are the same quantity expressed through different Pythagorean pairings. The rectangle 2D (= 2d²) can be partitioned in exactly three geometrically distinct ways.

### Algebraic Proof (C5)

$$d - e = (b+e) - e = b \implies J = d(d-e) = bd \checkmark$$
$$d + e = a = b+2e \implies K = d(d+e) = da \checkmark$$
$$J + K = d(a+b) = d \cdot 2d = 2d^2 \checkmark \quad (\text{since } a+b = 2b+2e = 2d)$$

### Worked Example: (b,e)=(3,2) → triangle (20-21-29)

| Parameter | Value |
|-----------|-------|
| d=b+e | 5 |
| a=b+2e | 7 |
| C=2de | 20 |
| F=ab | 21 |
| G=d²+e² | 29 |
| D=d² | 25 |
| J=bd | 15 |
| K=ad | 35 |
| F+G | 50 = 2×25 = 2D ✓ |
| J+K | 50 = 2×25 = 2D ✓ |
| 2J+C | 50 = 2×25 = 2D ✓ |

## Observer Projection Note (Theorem NT)

"Ellipse", "focus", "semimajor diameter" are observer-layer labels. The causal structure: J=bd and K=ad are integer bead number products; the triple equality F+G=J+K=2J+C=2d² is a pure algebraic identity following from d=b+e and a=b+2e. No continuous geometry enters.

**Depends on**: [336] QA Pythagorean 16 Identities; [334] Koenig Circle Nesting
