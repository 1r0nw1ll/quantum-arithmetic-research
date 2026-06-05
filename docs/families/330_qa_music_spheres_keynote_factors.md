# [330] QA Music of the Spheres Keynote Factoring

**Family**: `qa_music_spheres_keynote_factors_cert_v1`  
**Source**: Iverson (1991) *QA Volume II — Books 3 & 4*, pp.20-21  
"MUSIC OF THE SPHERES", "A MUSICAL SCALE?"

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Male keynotes: 891=3⁴×11, 1580=2²×5×79, 1602=2×3²×89, 2226=2×3×7×53 (Iverson exact) | PASS |
| C2 | Each male note has exactly one prime > 7; all four large primes distinct, all in (7,100) | PASS |
| C3 | Female integers verified: 756=2²×3³×7, 1050=2×3×5²×7 (7-smooth); 1197=3²×7×19, 1548=2²×3²×43 | PASS |
| C4 | All 8 keynotes: factors ⊆ {2,3,5,7} ∪ {one prime in (7,100)}; at most one large prime each | PASS |
| C5 | Male notes collectively and female notes collectively each cover all four primordial primes {2,3,5,7} | PASS |

## Male Keynote Factorizations (C1, C2)

Iverson (p.20) gives four male keynote integers with explicit factorizations:

| Keynote | Factorization | Small primes | Large prime |
|---------|--------------|--------------|-------------|
| 891 | 3⁴ × 11 | {3} | 11 |
| 1580 | 2² × 5 × 79 | {2, 5} | 79 |
| 1602 | 2 × 3² × 89 | {2, 3} | 89 |
| 2226 | 2 × 3 × 7 × 53 | {2, 3, 7} | 53 |

Iverson: "Their factors are: 2, 3, 5 & 7, along with one larger prime number between 7 and 100."

Each male note has **exactly one** large prime (>7, <100), and the four large primes {11, 79, 89, 53} are pairwise distinct.

## Female Keynote Approximations (C3)

Iverson provides decimal values for the female keynotes; the nearest integers are:

| Iverson value | Integer | Factorization | Type |
|--------------|---------|--------------|------|
| 754.95383 | 756 | 2² × 3³ × 7 | 7-smooth |
| 1050.7297 | 1050 | 2 × 3 × 5² × 7 | 7-smooth |
| 1197.965 | 1197 | 3² × 7 × 19 | large prime 19 |
| 1547.4254 | 1548 | 2² × 3² × 43 | large prime 43 |

The two 7-smooth female notes (756, 1050) carry only primordial primes {2,3,5,7}. The two with large primes (1197, 1548) follow the same structure as the male notes.

## Factor Coverage (C4, C5)

All 8 keynotes have prime factors drawn from {2,3,5,7} with at most one additional prime in (7,100). The four primordial primes are collectively represented:

- **Male collective**: 891 contributes {3}; 1580 contributes {2,5}; 1602 contributes {2,3}; 2226 contributes {2,3,7} → union = **{2,3,5,7}**
- **Female collective**: 756 contributes {2,3,7}; 1050 contributes {2,3,5,7}; 1197 contributes {3,7}; 1548 contributes {2,3} → union = **{2,3,5,7}**

Both groups independently span all four primordial primes, with no overlap requirement.

## Observer Projection Note (Theorem NT)

The musical names ("Music of the Spheres", "male/female keynotes") and cosmological interpretations are observer-layer descriptions. The causal layer is pure prime factorization of specific integers. The pattern — one large prime per note, collective coverage of {2,3,5,7} — is a discrete integer property requiring no continuous approximation.

**Depends on**: [329] Platonic CRT Aliquot Minima (aliquot minima 30, 42, 210), [328] Aliquot Parts Structure
