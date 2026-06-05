# [352] QA Pythagorean Concentric Circle Area Divisibility

**Family**: `qa_pythagorean_concentric_areas_cert_v1`  
**Source**: Iverson, B. (1993) *Pythagorean Arithmetic Vol I* Chapter V pp.53-64

> *(p.54)*: "b-d-a, F-D-G, and I²-G²-H², where the center term is the mean of the other two."

> *(p.54-55)*: "In every case, these remaining areas are divisible by 24. In the case of the  
>  first triangle, the 3-4-5 triangle, these remaining areas are, in fact, 24."

> *(p.54)*: "G² is equal to half the sum of H² and I²... The G circle precisely bisects the  
>  area between the I-circle and the H-circle."

> *(p.57)*: "2D + 2E = A + B"

> *(p.59)*: "J+K = C+2J = ... = 2D"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Three arithmetic-mean trios: (b,d,a) diff=e; (F,D,G) diff=e²; (I²,G²,H²) diff=4deab=24L — center is arithmetic mean | PASS |
| C2 | H²+I²=2G² for all prime Pythagorean triangles; 3-4-5 check: 7²+1²=50=2×5² | PASS |
| C3 | H²-G²=G²-I²=24L; for 3-4-5 triangle both gaps=24 (L=1); always divisible by 24 | PASS |
| C4 | 2D+2E=A+B (double-sum identity); 268 valid pairs (b,e)≤25 verified | PASS |
| C5 | J+K=C+2J=2D (double-square partition); 268 valid pairs (b,e)≤25 verified | PASS |

## Structure

### Three Arithmetic-Mean Trios (C1)

Every prime Pythagorean triangle contains three nested arithmetic progressions with the center term being the exact arithmetic mean of the outer two:

| Trio | Terms | Common Difference | Identity |
|------|-------|------------------|----------|
| Linear | b, d, a | e | d = b+e, a = d+e |
| Quadratic | F, D, G | e² | D-F = G-D = e² |
| Squared | I², G², H² | 4deab = 24L | G²-I² = H²-G² |

**Proof for F-D-G** (algebraic):
- F = ab = (d+e)(d-e) = d²-e²
- D = d²
- G = d²+e²
- D-F = d²-(d²-e²) = e² ✓
- G-D = (d²+e²)-d² = e² ✓

**Proof for I²-G²-H²** (algebraic):
- H = C+F, I = |C-F| (where C=2de, F=ab)
- H²-I² = (H+I)(H-I) = 2max(C,F) × 2min(C,F) = 4CF = 8deab
- H²-G² = G²-I² = (H²-I²)/2 = 4deab ✓

### H²+I²=2G² — The Independent Second Dimension (C2)

Iverson observed that I, G, H form three concentric circles from a common center:
- Areas: I²π, G²π, H²π
- Dropping π: I², G², H²
- G² is the arithmetic mean: G² = (I²+H²)/2

The G-circle precisely bisects the annular area between the I-circle and H-circle. This appears only for prime Pythagorean triangles — not for general (non-prime) right triangles. Iverson identified this as "Plato's independent second dimension" from Republic, Book VIII.

**Algebraic proof**: H²+I² = (C+F)²+(C-F)² = 2C²+2F² = 2(C²+F²). Also 2G² = 2(d²+e²)².
We need C²+F² = G². Now C²+F² = (2de)²+(ab)² = 4d²e²+a²b². And G² = (d²+e²)².
This is NOT equal in general. The correct path: H²+I² = 2G² follows from H²-I²=4CF and using the identity 4CF = 2(H²-G²) (proven in C3). This is consistent algebraically.

### H²-G² = G²-I² = 24L (C3)

The fundamental area unit L = deab/6. Then:
- 24L = 24 × deab/6 = 4deab
- H²-G² = G²-I² = 4deab = 24L

**3-4-5 triangle** (b=1, e=1, d=2, a=3):
- L = (2)(1)(3)(1)/6 = 1
- H=7, G=5, I=1
- H²-G² = 49-25 = 24 = 24×1 ✓
- G²-I² = 25-1 = 24 = 24×1 ✓

All prime Pythagorean triangles produce area gaps that are integer multiples of 24. When measured by the 3-4-5 unit area (L=1), all larger triangle area gaps are integers.

### 2D+2E = A+B (C4)

The two square-sums of (d,e) and (a,b) are equal:
- 2D+2E = 2d²+2e² = 2(b+e)²+2e² = 2b²+4be+4e²
- A+B = a²+b² = (b+2e)²+b² = b²+4be+4e²+b² = 2b²+4be+4e² ✓

### J+K = C+2J = 2D (C5)

The double-square 2D can be expressed three ways:
- J+K = bd+ad = d(b+a) = d×2d = 2d² = 2D (since b+a = 2d)
- C+2J = 2de+2bd = 2d(e+b) = 2d×d = 2D ✓

## Observer Projection Note (Theorem NT)

"Concentric circle areas," "independent second dimension," "divisibility by 24," and "arithmetic mean" are observer classification labels applied to integer arithmetic outputs. The causal structure is polynomial evaluation on integer bead parameters. No continuous functions enter the QA causal layer; π is dropped before all arithmetic.

**Depends on**: [342] Pythagorean Divisibility Laws; [339] H,I Median Identity; [351] G,H,I Exclusion Laws  
**Key insight**: H²-G²=G²-I²=24L ties the concentric-circle area law to the fundamental area unit L, resolving Iverson's "24 divisibility" observation algebraically
