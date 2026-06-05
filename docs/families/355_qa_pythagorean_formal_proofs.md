# [355] QA Pythagorean Formal Proof Statements

**Family**: `qa_pythagorean_formal_proofs_cert_v1`  
**Source**: Iverson, B. (1993) *Pythagorean Arithmetic Vol I* Chapter IX pp.94-100

> *(p.95, Statement 1)*: "Either d or e must be an even number."

> *(p.96, Statement 4)*: "The factor 3 will be represented in every set of bead numbers."

> *(p.98, Statement 8)*: "Within every triangle, the base, C, will contain the factor 4."

> *(p.98, Statement 9)*: "The area of every prime Pythagorean triangle is divisible by 6."

> *(p.99, Statement 13)*: "The difference, B, between the hypotenuse, G, and the base, C,  
>  of a prime Pythagorean triangle is the square of an odd number [B = G-C = b²]."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Statement 1: Exactly one of (d,e) is even; b odd forces parity(d)=parity(e)^1; 268 pairs ≤25 | PASS |
| C2 | Statement 4: Factor 3 in every bead set {b,e,d,a}; all 9 tri-residue combinations yield 3∣bead | PASS |
| C3 | Statement 8: C=2de always divisible by 4; proof: one of (d,e) is even, so C=2de has two 2-factors | PASS |
| C4 | Statement 9: CF/2 always divisible by 6; proof 4∣C and 3∣(C or F) → 12∣CF → 6∣(CF/2) | PASS |
| C5 | Statement 13: G-C=b² (odd perfect square); proof G-C=d²+e²-2de=(d-e)²=b²; 268 pairs ≤25 | PASS |

## Proof Chain

Iverson's Chapter IX contains 13 numbered proof statements building a logical chain. This cert captures the 5 non-trivial foundational proofs; others follow directly.

### C1: Parity of d and e (Statement 1)

**Proof**: b is always odd (Iverson's primeness requirement). If e is even, then d=b+e=odd+even=odd. If e is odd, then d=b+e=odd+odd=even. In all cases, exactly one of (d,e) is even, and the other is odd.

### C2: Factor 3 in Bead Numbers (Statement 4)

**Proof via tri-classification** (classification mod 3):

| b mod 3 | e mod 3 | Which bead is divisible by 3? |
|---------|---------|-------------------------------|
| 0 | * | b |
| * | 0 | e |
| 1 | 1 | a = b+2e ≡ 1+2 = 3 ≡ 0 |
| 1 | 2 | d = b+e ≡ 1+2 = 3 ≡ 0 |
| 2 | 1 | d = b+e ≡ 2+1 = 3 ≡ 0 |
| 2 | 2 | a = b+2e ≡ 2+4 = 6 ≡ 0 |

All 9 combinations of (b mod 3, e mod 3) yield at least one bead divisible by 3. ✓

### C3: C Divisible by 4 (Statement 8)

**Proof**: C=2de. By C1, exactly one of (d,e) is even. Say e=2m; then C=2d×2m=4dm. Or say d=2k; then C=2×2k×e=4ke. Either way, C is divisible by 4. ✓

**Consequence**: C ≡ 0 (mod 4) — C is always a "4-par" number in Iverson's terminology.

### C4: Area Divisible by 6 (Statement 9)

**Proof**: 
1. C is divisible by 4 (C3)
2. Factor 3 appears in {b,e,d,a} (C2)
   - If 3|a or 3|b → F=ab contains factor 3
   - If 3|d or 3|e → C=2de contains factor 3
3. Therefore 3|CF (either directly from C or from F)
4. Combined: 4|C and 3|CF, so 12|CF → Area = CF/2 is divisible by 6. ✓

Iverson notes: "This is the basis for the identity L" where L=CF/12=abde/6.

### C5: G-C = b² (Statement 13)

**Proof** (algebraic):
- G = d²+e²
- C = 2de
- G-C = d²+e²-2de = (d-e)² = b² (since b=d-e by definition)

Since b is always odd, b² is always an odd perfect square. The difference G-C is always the square of the "yin" bead b.

**3-4-5 triangle check**: b=1, d=2, e=1. G=5, C=4. G-C=1=1²=b². ✓

## Complete List of Iverson's 13 Proof Statements (Chapter IX)

| # | Statement | Certified? |
|---|-----------|-----------|
| 1 | Either d or e must be even | C1 ✓ |
| 2 | a is always odd | (follows from C1: a=d+e=odd+even=odd) |
| 3 | Factor 2 in every bead set | (follows from C1) |
| 4 | Factor 3 in every bead set | C2 ✓ |
| 5 | One of {b,e,d,a} or G divisible by 5 | ([351] C2 partial) |
| 6 | C, F, or G contains factor 5 | (follows from 5) |
| 7 | C or F contains factor 3 | (follows from C2) |
| 8 | C divisible by 4 | C3 ✓ |
| 9 | Area=CF/2 divisible by 6 | C4 ✓ |
| 10 | Area divisible by area of 3-4-5=6 | (follows from C4, L=Area/6) |
| 11 | G is always 5-par (≡1 mod 4) | ([351] C1 ✓) |
| 12 | d coprime to b,e; a coprime to b,d,e | ([349] C5 ✓, Euclid VII.28) |
| 13 | G-C=b² (odd perfect square) | C5 ✓ |

## Observer Projection Note (Theorem NT)

"Par," "tri," "pent" number classifications are observer projection labels (residue classes mod 4, 3, 5 respectively). Divisibility criteria are integer modular arithmetic. No continuous functions enter the QA causal layer.

**Depends on**: [342] Pythagorean Divisibility Laws; [351] G,H,I Exclusion Laws; [349] Twin Prime Mod-6 Structure  
**Key insight**: Iverson's 13-proof chain formalizes why the bead identities C,F,G are divisible by 4,3×anything,1 respectively — establishing that the prime triangle area is always a multiple of 6, i.e., L is always a positive integer
