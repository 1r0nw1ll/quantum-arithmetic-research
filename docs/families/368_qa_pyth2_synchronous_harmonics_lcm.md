# [368] QA Pyth-2 Synchronous Harmonics: LCM and Coincidence Periods

**Family**: `qa_pyth2_synchronous_harmonics_lcm_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XIII pp.38-70+

> *(p.48)*: "The cycle of 3 and 5 will coincide every 15 units."

> *(p.48)*: "The cycle of 2 and 3 will coincide every 6 units. 2 and 5 will coincide every 10 units. 3 and 5 will coincide every 15 units. And the cycles of 2, 3, and 5 will coincide every 30 units."

> *(p.49)*: "The complete cycle for 3, 5, and 7 would be 105 units before all will coincide."

> *(p.48)*: "The addition of a 6-cycle to the 2-cycle and 3-cycle does not change the overall cycle because 6 is not prime to 2 or 3."

> *(p.50)*: "the sine is represented as F/G"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | LCM of pairwise-coprime integers = their product; lcm(2,3,5)=30; lcm(3,5,7)=105; lcm(2,3,5,7)=210 | PASS |
| C2 | Non-coprime extension: if c\|lcm(a,b) then lcm(a,b,c)=lcm(a,b); lcm(2,3,6)=lcm(2,3)=6 | PASS |
| C3 | LCM is associative: lcm(a,b,c)=lcm(lcm(a,b),c); pairwise-coprime triple: lcm=product | PASS |
| C4 | QA sine = F/G for all prime pairs; F<G always; F²+C²=G² (Pythagorean identity) | PASS |
| C5 | Half-cycle: all-odd LCM is odd (midpoint=non-integer); even-factor LCM is even (integer midpoint) | PASS |

## Mathematical Details

### C1: LCM of Coprime Cycles = Product (Iverson's Coincidence Claim)

When cycles have lengths that are pairwise coprime, they all coincide exactly at intervals equal to their product:

| Cycles | LCM = period | = product? |
|--------|-------------|------------|
| 2, 3 | 6 | = 2×3 ✓ |
| 2, 5 | 10 | = 2×5 ✓ |
| 3, 5 | 15 | = 3×5 ✓ |
| 2, 3, 5 | 30 | = 2×3×5 ✓ |
| 3, 5, 7 | 105 | = 3×5×7 ✓ |
| 2, 3, 5, 7 | 210 | = 2×3×5×7 ✓ |

**Proof**: For any two integers a, b: lcm(a,b) = ab/gcd(a,b). When gcd(a,b)=1: lcm(a,b) = ab. By induction, for pairwise-coprime sets: lcm(a₁,...,aₙ) = a₁·...·aₙ. ✓

Verified for all 705 coprime pairs (a,b)<50.

### C2: Non-Coprime Extension Lemma

Iverson's observation: adding the 6-cycle to the 2-cycle and 3-cycle doesn't extend the combined period, because 6 shares factors with 2 and 3.

**Specific case**: lcm(2,3,6) = lcm(6,6) = 6 = lcm(2,3).

**General rule**: If c divides lcm(a,b), then lcm(a,b,c) = lcm(a,b).

**Proof**: Let m = lcm(a,b). If c|m then every multiple of c is also a multiple of m's largest factor structure, so lcm(m,c) = m. ✓

This explains why adding a "composite" or "non-coprime" cycle to a system doesn't extend the master cycle — the new cycle's period is already accounted for within the existing LCM.

Verified for 4330 triples (a,b,c)∈[2,29] where c|lcm(a,b).

### C3: LCM Associativity

lcm(a,b,c) = lcm(lcm(a,b),c) for all positive integers a,b,c.

For pairwise-coprime triples: lcm(a,b,c) = abc (product of all three).

Verified: 5832 triples in [2,19] for associativity; 1662 pairwise-coprime triples for product equality.

### C4: QA Sine = F/G

In Quantum Arithmetic, the trigonometric sine of the triangle angle is represented exactly as the rational fraction:

**sine = F/G = altitude/hypotenuse**

Since C, F, G satisfy the Pythagorean theorem (C² + F² = G²), and C > 0 for all prime pairs, we have F < G always, so sine = F/G ∈ (0,1) exactly — no decimal approximation needed.

First few values:

| (b,e) | F | G | sine = F/G |
|-------|---|---|-----------|
| (1,1) | 3 | 5 | 3/5 |
| (1,2) | 5 | 13 | 5/13 |
| (1,3) | 7 | 25 | 7/25 |
| (3,1) | 15 | 17 | 15/17 |
| (3,2) | 21 | 29 | 21/29 |

This is the QA convention for trigonometry: every ratio is exact (Fraction arithmetic), not a decimal approximation. Verified for 369 prime pairs (b,e)≤30.

### C5: Half-Cycle and Integer Midpoints

For a combined cycle of length L = lcm(cycle lengths), the half-cycle midpoint is L/2.

**All-odd cycle lengths**: lcm of any collection of odd integers is odd (since the product of odd numbers is odd, and lcm divides the product). So L/2 is a half-integer — not an integer.

Iverson's note: "They are symmetrical about 52½" for the {3,5,7} system (lcm=105, midpoint=52.5).

**Even cycle length present**: lcm is even (since it must be divisible by 2). So L/2 is an integer — a proper midpoint.

Example: {2,3,5} system has L=30, midpoint=15.

Verified: 154 all-odd coprime pairs have odd LCM; 153 even-containing pairs have even LCM.

## Theorem NT Note

"Sine wave," "amplitude," "frequency," "quarter-point," "three-quarter point," and "Fig.13/14 illustrations" are observer projections of the underlying LCM arithmetic. The QA layer contains only:
- Integer cycle lengths
- LCM computation via gcd arithmetic
- The coincidence period = LCM of the cycle lengths
- QA sine = F/G (exact rational)

The wave picture in Ch.XIII is an observer projection onto continuous geometry. The discrete structure is the LCM.

**Depends on**: [367] Prime Number Symmetry (coprime-to-30/60 brackets, same 30=lcm(2,3,5) structure); [366] Bead Arithmetic Laws (factor 3 in every bead set — F/G trigonometry grounded in Pythagorean bead arithmetic)
