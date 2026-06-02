# [304] QA Polyphase Sum Structure

**Family**: `qa_polyphase_sum_structure_cert_v1`  
**Depends on**: [291] Fibonacci Matrix Orbit Periods, [296] SL(2,Z) Versor Isomorphism, [298] Orbit Grade Decomposition, [299] Cayley-Hamilton Fibonacci-Lucas, [303] Three-Phase Cosmos Cancellation

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Six-phase (n=6, step=4): M⁰+M⁴+M⁸+M¹²+M¹⁶+M²⁰ ≡ 0 (mod 9); three antipodal pairs (M^k, −M^k) each cancel | PASS |
| C2 | Twelve-phase (n=12, step=2): sum of 12 equidistant matrices ≡ 0 (mod 9); six antipodal pairs cancel | PASS |
| C3 | Universal n-phase theorem: for all n\|24, the sum is I (n=1), 3·I (n=3), or 0 (all other n); proved by grade-inversion pairing for even n, direct computation for n=3 | PASS |
| C4 | T⁴ maps Cosmos→Cosmos; 72 states form 12 disjoint period-6 sextets; T² maps Cosmos→Cosmos; 72 states form 6 disjoint period-12 dodecaplets | PASS |
| C5 | Observer layer (Theorem NT): six-phase rational cosine sum 1+½−½−1−½+½=0 (exact Fraction); discrete and observer results differ for n=3; Theorem NT boundary confirmed | PASS |

## Key result

The **Grade-Inversion Pairing Theorem** unifies all even-n polyphase systems:

```
M¹² ≡ −I (mod 9)   [cert 298: grade-inversion antipodal map]

For any even n|24 with step k=24/n:
  M^(jk) + M^(jk+12) = M^(jk)·(I + M¹²) = M^(jk)·(I − I) = 0

All n/2 pairs cancel → n-phase sum = 0
```

The complete n-phase sum table:

| n | Step | Sum mod 9 | Structure |
|---|------|-----------|-----------|
| 1 | 24 | I | trivial |
| **2** | 12 | **0** | 1 pair: (I, −I) |
| **3** | 8 | **3·I** | odd, no pairing; unique non-zero case |
| **4** | 6 | **0** | 2 pairs |
| **6** | 4 | **0** | 3 pairs |
| **8** | 3 | **0** | 4 pairs |
| **12** | 2 | **0** | 6 pairs |
| **24** | 1 | **0** | 12 pairs |

**Three-phase [303] is the unique non-zero non-trivial case** because n=3 is the only odd non-trivial divisor of 24. For even n, M¹² = −I (cert [298]) guarantees exact cancellation regardless of the step size.

## Orbit partition comparison

| Advance | Period on Cosmos | Orbit type | Count |
|---------|-----------------|------------|-------|
| T⁸ (three-phase) | 3 | triads | 24 |
| T⁴ (six-phase) | 6 | sextets | 12 |
| T² (twelve-phase) | 12 | dodecaplets | 6 |
| T¹ (twenty-four-phase) | 24 | full T-orbit | 3 |

Formula: orbit size = 24/gcd(step, 24); count = 72/orbit\_size.

## Six-phase explicit matrices

```
M⁰  mod 9 = [[1,0],[0,1]]          M¹² mod 9 = [[8,0],[0,8]] = −M⁰
M⁴  mod 9 = [[2,3],[3,5]]          M¹⁶ mod 9 = [[7,6],[6,4]] = −M⁴
M⁸  mod 9 = [[4,3],[3,7]]          M²⁰ mod 9 = [[5,6],[6,2]] = −M⁸
```

All three pairs sum to zero, column-wise: {1+2+4+8+7+5}=27≡0, {0+3+3+0+6+6}=18≡0 (mod 9).

## Why n=3 escapes (algebraic reason)

det(M⁸ − I) = 9 ≡ 0 (mod 9): M⁸ − I is **not** invertible mod 9. This traces to the 3-adic filtration (cert [301]): M⁸ ≡ I (mod 3) (since M has order 8 in GL(2,Z/3Z) and 8|8), so M⁸ − I has all entries divisible by 3, making det ≡ 0 (mod 9).

By contrast: det(M⁴ − I) = −5 ≡ 4 (mod 9) — invertible. Since (A − I)·S(n) = 0 and (A − I)⁻¹ exists, S(6) = 0 is forced.

## Observer layer (Theorem NT)

For n = 6: cos(0°)+cos(60°)+cos(120°)+cos(180°)+cos(240°)+cos(300°) = 1+½−½−1−½+½ = 0 (exact rational, Fraction arithmetic).  

For n = 12: the twelve cosines also sum to 0 (irrational terms ±√3/2 cancel in pairs; rational terms sum as for n=6).

The zero observer sum holds for ALL n ≥ 2. The discrete QA sum differs (3·I for n=3). This is the Theorem NT boundary: the physical "balance" of polyphase power is a continuous observer fact; the underlying QA discrete structure distinguishes three-phase from all others.

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.X
- Wall, D.D. (1960) Fibonacci Primitive Roots, *Amer. Math. Monthly* 67(6):525–532, DOI 10.1080/00029890.1960.11989541
- Hestenes, D. and Sobczyk, G. (1984) *Clifford Algebra to Geometric Calculus*, Reidel, ISBN 978-90-277-1673-6
