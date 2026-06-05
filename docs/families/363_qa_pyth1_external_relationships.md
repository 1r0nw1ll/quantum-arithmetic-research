# [363] QA Pyth-1 External Relationships

**Family**: `qa_pyth1_external_relationships_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol I* Chapter VI pp.67-71

> *(p.62)*: "In the first column, the value of F increases in steps of 2 units. In the remainder of the columns, F increases by 2b units at each step."

> *(p.62)*: "the value C increases by 4e units at each step for the base of the triangles."

> *(p.62)*: "The difference between F and G remains constant in the blocks across the page, and the difference between C and G remains constant, progressing down the columns."

> *(p.62)*: "the value of A becomes the value of B moving to the right e-blocks."

> *(p.62)*: "A given value of D becomes the same value of E at a distance of b-blocks down each column."

> *(p.63)*: "complete absence of all 2-par numbers except as values of e, d, J or K."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Column step F(b,e+1)−F(b,e)=2b; row step C(b+2,e)−C(b,e)=4e | PASS |
| C2 | G−F=2E=2e² (constant along rows); G−C=B=b² (constant along columns) | PASS |
| C3 | A(b,e)=B(b+2e,e): A-value of (b,e) equals B-value of (b+2e,e) | PASS |
| C4 | D(b,e)=E(b,e+b): D-value of (b,e) equals E-value of (b,e+b) | PASS |
| C5 | 2-par integers (≡2 mod 4) appear only in {e,d,J,K}; never in {A,B,C,D,E,F,G,H,I} | PASS |

## Mathematical Details

### C1: Progression Steps in Table 3

Table 3 arranges prime Pythagorean triangles in a grid: rows fix e and vary b (b increases by 2 per block, since b must remain odd); columns fix b and vary e (e increases by 1 per block).

**Column step (fixing b, increasing e by 1)**:

F(b,e) = ab = b(b+2e)  
F(b,e+1) = b(b+2(e+1)) = b(b+2e+2) = F(b,e) + 2b

**Proof**: F(b,e+1) − F(b,e) = b·2 = 2b ✓

Special case b=1 (first column): step = 2·1 = 2. "Increases in steps of 2" ✓

**Row step (fixing e, increasing b by 2)**:

C(b,e) = 2de = 2(b+e)e  
C(b+2,e) = 2(b+2+e)e = 2(b+e)e + 4e = C(b,e) + 4e

**Proof**: C(b+2,e) − C(b,e) = 4e ✓

### C2: G Differences are Structural Constants

**G−F along rows (fixed e)**:

G − F = (d²+e²) − (d²−e²) = 2e² = 2E

This is independent of b. For fixed e, as b varies (columns change), G−F = 2E remains constant. ✓

**G−C along columns (fixed b)**:

G − C = (d²+e²) − 2de = (d−e)² = b² = B

This is independent of e. For fixed b, as e varies (rows change), G−C = B remains constant. ✓

These two facts confirm the table's "orderly array": C and F increase along their respective axes while G compensates to maintain fixed differences.

### C3: A-B Migration Rule

The "a"-bead of triangle (b,e) is a=b+2e. The "b"-bead of triangle (b+2e, e) is b'=b+2e.

Therefore: A(b,e) = a² = (b+2e)² = b'² = B(b+2e, e)

**Proof**: direct substitution ✓

Iverson's statement "moving e-blocks to the right" means: in Table 3, moving from column b to column b+2e (which is e column-pairs away), the B-value equals the A-value of the starting position. For the first row (e=1): moving 1 block right; A(b,1)=(b+2)²=B(b+2,1). ✓

### C4: D-E Migration Rule

The "d"-bead of triangle (b,e) is d=b+e. The "e"-bead of triangle (b,e+b) is e'=e+b.

Therefore: D(b,e) = d² = (b+e)² = (e+b)² = e'² = E(b,e+b)

**Proof**: direct substitution ✓

Iverson's statement "b-blocks down each column" means: in Table 3, moving from row e to row e+b (which is b rows down), the E-value equals the D-value of the starting position.

### C5: 2-par Exclusion from A-I

2-par integers satisfy n≡2 (mod 4). The complete exclusion from {A,B,C,D,E,F,G,H,I}:

| Identity | Form | mod 4 values possible |
|----------|------|----------------------|
| A = a² | square of odd number | ≡1 (mod 4) — 5-par only |
| B = b² | square of odd number | ≡1 (mod 4) — 5-par only |
| C = 2de | one of d,e even → 4\|C | ≡0 (mod 4) — 4-par only |
| D = d² | square | ≡0 or 1 (mod 4) — 4-par or 5-par |
| E = e² | square | ≡0 or 1 (mod 4) — 4-par or 5-par |
| F = ab | both odd | ≡1 or 3 (mod 4) — 5-par or 3-par only |
| G = d²+e² | 5-par (cert [360]) | ≡1 (mod 4) — 5-par only |
| H = C+F | odd (cert [361] C4) | ≡1 or 3 (mod 4) — never 2-par |
| I = \|C−F\| | odd (cert [361] C4) | ≡1 or 3 (mod 4) — never 2-par |

**2-par CAN appear in**: d (when d≡2 mod 4, e.g., b=1,e=1→d=2); J=bd (when d≡2 mod 4); K=ad (when d≡2 mod 4); e (when e≡2 mod 4, e.g., e=2).

**Why**: The 2-par class (≡2 mod 4) contains exactly those integers divisible by 2 but not 4. Squares of such integers are 4-par (not 2-par). Products of two odd integers are odd (not 2-par). Only products with exactly one factor of 2 land in 2-par — which happens for e, d, J=bd, K=ad when d has exactly one factor of 2.

## Theorem NT Note

"Ellipse," "apogee," "perigee," "circumscribe," "orbit of electron" in Iverson's text are observer projections of the underlying discrete arithmetic. The algebraic identities C1-C5 are purely about bead arithmetic — integer differences, squares, and congruences — not geometric constructions.

**Depends on**: [360] Prime Triangle Structure; [361] Primeness Parity Shape (C≡0 mod 4; H,I odd); [353] External Table Laws (G-F and G-C step patterns); [359] Nightside Energy (2-par/3-par/4-par/5-par classification)
