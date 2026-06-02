# [301] QA 3-Adic Filtration

**Family**: `qa_3adic_filtration_cert_v1`  
**Depends on**: [291] Fibonacci Matrix Orbit Periods, [298] Orbit Grade Decomposition, [300] SL(2,Z) Equivariance  
**Note**: Works in Z/9Z = {0,...,8} with T₀(b,e)=(e,(b+e) mod 9)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | μ₃(b,e)=(3b mod 9, 3e mod 9) maps all 72 Cosmos states to Satellite states | PASS |
| C2 | μ₃ maps all 8 Satellite states to (0,0) = Singularity | PASS |
| C3 | μ₃ is exactly 9-to-1 on Cosmos→Satellite: each of the 8 Satellite states has exactly 9 Cosmos preimages | PASS |
| C4 | Intertwining: T₀∘μ₃ = μ₃∘T₀ on all 81 states; algebraic proof: T₀ is linear, T₀(3v) = 3·T₀(v) mod 9 | PASS |
| C5 | x²−x−1 irreducible over Z/3Z; M mod 3 has order exactly 8; Cosmos reduces 9-to-1 onto (Z/3Z)²\{(0,0)} | PASS |

## Key result

Multiplication by 3 in Z/9Z gives a **3-adic filtration tower**:

```
Cosmos  →^{×3}→  Satellite  →^{×3}→  Singularity
72 states         8 states            1 state
period 24         period 8            period 1
```

This is a Hensel lifting tower: the Cosmos orbit "covers" the 8-cycle of M mod 3 in (Z/3Z)², and the Satellite is the 3-scaled image of that cover. The key structural facts:

- **C4** (intertwining): μ₃∘T₀ = T₀∘μ₃ because T₀ is Z/9Z-linear — so the three strata are not just stable sets but form an **equivariant filtration**
- **C5** (GF(3) layer): x²−x−1 irreducible over GF(3) means M mod 3 generates the field GF(9) over GF(3); GF(9)* has order 8 = Satellite period; the Cosmos orbit is the full Hensel lift to Z/9Z with multiplicity 9 = 3²

The Hensel ratio: π(9)/π(3) = 24/8 = 3 = p (the prime). This is Wall's formula π(p²) = p·π(p) for non-Wall primes; p=3 is non-Wall since the Pisano period does not divide π(3)=8 for the prime 3.

## Primary sources

- Hardy, G.H. and Wright, E.M. (2008) *An Introduction to the Theory of Numbers*, Oxford, ISBN 978-0-19-921986-5, Ch.IV (p-adic valuation), Ch.VII (Hensel's lemma)
- Wall, D.D. (1960) Fibonacci Primitive Roots, *Amer. Math. Monthly* 67(6):525–532, DOI 10.1080/00029890.1960.11989541 (π(3)=8, π(9)=24)
