# [351] QA Pythagorean G,H,I Exclusion Laws

**Family**: `qa_pythagorean_g_exclusion_cert_v1`  
**Source**: Iverson, B. (1993) *Pythagorean Arithmetic Vol I* Chapter IV pp.39-41

> *(p.39-40)*: "G must always be a 5-par (4n+1) number...it can have no divisor smaller  
>  than 5. For some unknown reason, it seems that G may not have factors of 11, 19, or 43."

> *(p.40)*: "The seven lowest values for G are 5, 13, 17, 25, 29, 37, and 41. Note  
>  specifically the absence of 11, 19, 23, and 31 which are 3-par prime numbers."

> *(p.40)*: "When H and I are not actually prime they may have no factor less than 7,  
>  and also may not have 11, 19, or 43 as a factor."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | G first 7 values = {5,13,17,25,29,37,41}; 3-par primes {11,19,23,31} absent; G≡1 (mod 4) always | PASS |
| C2 | G coprime to {2,3,11,19,43}; all primes p≡3 (mod 4) ≤100 never divide G (13 primes verified) | PASS |
| C3 | H=C+F and I=|C-F| always odd and coprime to {2,3}; 268 valid pairs (b,e)≤25 verified | PASS |
| C4 | Composite H or I has all prime factors ≥ 7; 196 composite cases verified for (b,e)≤25 | PASS |
| C5 | H and I never divisible by 11, 19, or 43; 369 valid pairs (b,e)≤30 verified | PASS |

## Structure

### G Exclusion Law (C1-C2)

**G = d² + e²** where gcd(d,e)=1 with d,e having opposite parity.

The "unknown reason" Iverson noted (why G cannot have factors 11,19,43) is the classical theorem on sums of two squares:

> **Theorem**: A prime p divides a²+b² (with gcd(a,b)=1) if and only if p=2 or p≡1 (mod 4).

Since 11≡3, 19≡3, 43≡3 (mod 4), they can never divide G=d²+e² with gcd(d,e)=1. The general rule covers ALL primes ≡3 (mod 4): {3,7,11,19,23,31,43,47,...}.

| Prime | Residue mod 4 | Divides G? |
|-------|--------------|-----------|
| 2 | 2 | No (G always odd) |
| 3 | 3 | No |
| 5 | 1 | Yes (G=5 for b=1,e=1) |
| 7 | 3 | No |
| 11 | 3 | No (Iverson's example) |
| 13 | 1 | Yes (G=13 for b=1,e=2) |
| 17 | 1 | Yes (G=17 for b=3,e=1) |
| 19 | 3 | No (Iverson's example) |
| 23 | 3 | No |
| 43 | 3 | No (Iverson's example) |

Iverson discovered empirically what is the Fermat-Euler theorem on Gaussian integer factorization. The discovery from pure QA arithmetic uncovered classical number theory independently.

### H,I Exclusion Law (C3-C5)

**H = C+F = 2de + ab** and **I = |C-F| = |2de - ab|**

**C=2de is always even** (has factor 2 from definition). **F=ab is always odd** (a=d+e and b=d-e are both odd when d,e have opposite parity). So:
- H = even + odd = **always odd** ✓
- I = |even - odd| = **always odd** ✓

**H coprime to 3**: algebraic proof via mod-3 case analysis on d,e:
- If 3|d: then 3∤e (gcd=1), b≡-e, a≡e (mod 3) → H = 2de+ab ≡ 0 + (-e²) ≡ -1 ≡ 2 (mod 3) ✓
- If 3|e: 3∤d → H ≡ 0 + d² ≡ 1 (mod 3) ✓  
- If 3∤d,e: ab = (d+e)(d-e) = d²-e² ≡ 1-1 = 0 (mod 3) → H ≡ 2de (mod 3) ∈ {1,2} ✓

In no case is H ≡ 0 (mod 3). The same argument holds for I.

The exclusion of factors {11,19,43} from H and I is verified computationally for all valid pairs with b≤30. Unlike the G exclusion (which follows from Fermat's two-square theorem), the H,I exclusion does not have an obvious algebraic explanation — Iverson's observation applies to all three.

## Observer Projection Note (Theorem NT)

Residue class membership (5-par, 3-par), primeness, and factor exclusions are observer classification labels applied to integer arithmetic outputs. The causal structure: integer parametric definitions (d=b+e, a=b+2e), polynomial evaluations (G=d²+e², H=C+F), modular arithmetic. No continuous functions enter the QA causal layer.

**Depends on**: [342] Pythagorean Divisibility Laws (G always 5-par); [339] H,I Median Identity  
**New insight**: Iverson's "unknown reason" for {11,19,43} exclusion = Fermat's theorem on two squares
