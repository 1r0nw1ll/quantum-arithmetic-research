# [353] QA Pythagorean External Table Laws

**Family**: `qa_pythagorean_external_table_cert_v1`  
**Source**: Iverson, B. (1993) *Pythagorean Arithmetic Vol I* Chapter VI pp.67-71

> *(p.62-63)*: "The difference between F and G remains constant in the blocks across  
>  the page, and the difference between C and G remains constant, progressing down  
>  the columns."

> *(p.62)*: "In the first line, the value of A becomes the value of B in the next  
>  block in succession across the page."

> *(p.62)*: "the value of D becomes the value of E in the next block down the column"

> *(p.62)*: "F increases by 2b units at each step [down]; C increases by 4e units  
>  at each step [across]"

> *(p.62)*: "there are no empty, nonprime blocks in the column where b=1 or in the  
>  lines where e=1 and e=2"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | G-F=2e² constant across rows; G-C=b² constant down columns; 268 pairs (b,e)≤25 | PASS |
| C2 | A(b,e)=B(b+2e,e) — A value transfers to B e-blocks right; gcd preserved; 169 pairs | PASS |
| C3 | D(b,e)=E(b,e+b) — D transfers to E b-blocks down; b=1: D(1,e)=E(1,e+1); 86 pairs | PASS |
| C4 | F increases by 2b per unit e-step down each column; C increases by 4e per 2-step in b | PASS |
| C5 | b=1 column no empty blocks (gcd(1,e)=1 always); e=1,2 rows no empty blocks (gcd(odd,1)=gcd(odd,2)=1) | PASS |

## Structure

### Table 3 Layout (Chapter VI)

Table 3 is a 2D grid of prime Pythagorean triangle "blocks":
- **Rows**: constant e, b increases left-to-right through odd numbers
- **Columns**: constant b, e increases top-to-bottom
- **Empty blocks**: occur when gcd(b,e)>1 — all identities would have the common factor

### C1: Constant-Difference Laws

Two constant-difference identities emerge from the table geometry:

**Across rows (constant e):**
- G - F = (d²+e²) - (d²-e²) = **2e²** — fixed for all blocks in the same row
- Since F = ab = (d+e)(d-e) = d²-e² (RAW, not mod-reduced)

**Down columns (constant b):**
- G - C = (d²+e²) - 2de = (d-e)² = **b²** — fixed for all blocks in the same column
- Since b = d-e always

Both identities are direct polynomial consequences of the bead definitions.

### C2: A-to-B Transfer

For any valid block at (b,e), the A value of that block equals the B value exactly e-steps to the right:

**A(b,e) = B(b+2e, e)**

- A(b,e) = a² = (b+2e)²
- B(b+2e, e) = (b+2e)² ✓
- gcd(b+2e, e) = gcd(b,e) = 1 — the destination block is always prime

For the first row (e=1): A(b,1) = (b+2)² = B(b+2,1) — A transfers to the immediately adjacent block.

### C3: D-to-E Transfer

For any valid block at (b,e), the D value equals the E value exactly b-steps down the column:

**D(b,e) = E(b, e+b)**

- D(b,e) = d² = (b+e)²
- E(b, e+b) = (e+b)² ✓
- gcd(b, e+b) = gcd(b,e) = 1 — the destination block is always prime

**First column (b=1)**: D(1,e) = (1+e)² = E(1, e+1) — D transfers to the immediately adjacent block below.

**General column (b=3)**: D(3,e) = E(3, e+3) — 3 blocks down.

### C4: Step Sizes

**F column steps** (constant b, increasing e):
- F(b,e) = ab = (b+2e)·b → F(b,e+1) - F(b,e) = 2b per unit step in e

**C row steps** (constant e, b increases by 2 per step since b is odd):
- C(b,e) = 2de = 2(b+e)e → C(b+2,e) - C(b,e) = 2·2·e = 4e per 2-step in b

Special case (b=1, first column): F increases by 2·1=2 per step. ✓

### C5: No-Empty-Block Rows and Columns

**Column b=1**: gcd(1,e) = 1 for all e ≥ 1 — every row yields a valid prime triangle.

**Row e=1**: gcd(b,1) = 1 for all b — every column yields a valid prime triangle.

**Row e=2**: gcd(b,2) = 1 for all odd b — odd and even integers are always coprime.

**Rows e=4,8,16,...** (powers of 2): gcd(odd b, 2^k) = 1 always — odd numbers share no factor with powers of 2.

## Compatible Pairs

The five compatible pairs in the external table: **(A,B), (D,E), (F,G), (H,I), (J,K)**.

Within each pair, one member of a block appears as the other member in a neighboring block — the specific transfer rule differs by pair. C2 gives the A-B rule; C3 gives the D-E rule; H-I has a more complex zigzag pattern described separately in the Koenig Series (Ch.VII).

## Observer Projection Note (Theorem NT)

"Table 3 blocks," "empty blocks," "e-steps right," "compatible pairs" are observer classification labels. The causal structure is polynomial evaluation on integer bead parameters (b,e,d=b+e,a=d+e). No mod reduction, no continuous functions.

**Depends on**: [342] Pythagorean Divisibility Laws; [352] Concentric Area Laws (F=d²-e², G=d²+e²)  
**Key insight**: The 2D table structure encodes two constant-difference laws (G-F=2e², G-C=b²) and two value-transfer laws (A→B, D→E) that together let the entire table be extended without computing all identities
