# [315] QA Rhind 2/n Unit Fraction

**Family**: `qa_rhind_2n_unit_fraction_cert_v1`  
**Depends on**: [314] Egyptian Ennead Orbit Partition (mod-9 3-divisibility)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | All 50 Rhind 2/n entries (n=3,5,...,101) sum to exactly 2/n as Python Fraction; no float arithmetic at any step | PASS |
| C2 | All 17 Satellite/Singularity n (3\|n) have exactly 2 unit fractions in their Rhind decomposition | PASS |
| C3 | For all 17 n=3k, Rhind denominators are exactly (2k, 6k); the formula 2/(3k)=1/(2k)+1/(6k) generates all Satellite entries identically | PASS |
| C4 | 4-term decompositions occur for exactly 8 Cosmos n: {29,43,61,73,79,83,89,101}; zero Satellite or Singularity entry has 4 terms | PASS |
| C5 | Theorem NT: unit fraction notation and hieroglyphic form are observer projections; orbit_family(dr(n),dr(n),9), Fraction sums, term count, and 3-divisor formula are the discrete QA claims | PASS |

## Key result

The Rhind Mathematical Papyrus (c.1550 BCE, Ahmes scribe, Thebes) opens with a table of 50 unit-fraction decompositions of 2/n for odd n = 3 to 101. The **QA mod-9 orbit family** of n partitions the table into two structurally distinct halves:

### The Satellite formula — 17 entries, zero exceptions

For every n with 3|n (orbit_family on (dr(n),dr(n),9) ∈ {satellite, singularity}):

> 2/(3k) = 1/(2k) + 1/(6k)

| n | k | Denominators | Orbit family |
|---|---|---|---|
| 3 | 1 | 2, 6 | satellite |
| 9 | 3 | 6, 18 | singularity |
| 15 | 5 | 10, 30 | satellite |
| 21 | 7 | 14, 42 | satellite |
| 27 | 9 | 18, 54 | satellite |
| 33 | 11 | 22, 66 | satellite |
| 39 | 13 | 26, 78 | satellite |
| 45 | 15 | 30, 90 | satellite |
| 51 | 17 | 34, 102 | satellite |
| 57 | 19 | 38, 114 | satellite |
| 63 | 21 | 42, 126 | satellite |
| 69 | 23 | 46, 138 | satellite |
| 75 | 25 | 50, 150 | satellite |
| 81 | 27 | 54, 162 | satellite |
| 87 | 29 | 58, 174 | satellite |
| 93 | 31 | 62, 186 | satellite |
| 99 | 33 | 66, 198 | satellite |

Every Satellite/Singularity denominator is generated mechanically. No creative scribal choice needed.

### The Cosmos — 33 entries, three tiers

| Term count | n values | Count |
|---|---|---|
| 2-term | 5,7,11,23,25,35,49,55,65,77,85,91 | 12 |
| 3-term | 13,17,19,31,37,41,47,53,59,67,71,95,97 | 13 |
| 4-term | **29,43,61,73,79,83,89,101** | **8** |

4-term decompositions arise exclusively for Cosmos n. The scribes needed their most complex reasoning for the numbers QA classifies as Cosmos — the ones with no 3-divisibility handle.

### Proof that the 3-divisor formula is the ONLY rule needed for Satellite

The formula 2/(3k) = 1/(2k) + 1/(6k) holds because:
- 1/(2k) + 1/(6k) = 3/(6k) + 1/(6k) = 4/(6k) = 2/(3k)

This is a trivial identity in Q, but it requires 3|n to generate **integer** denominators 2k and 6k. When 3∤n, k = n/3 is not an integer, so the formula produces fractional denominators and cannot be used. The boundary between Satellite and Cosmos is the boundary between the mechanical formula and the creative decomposition — which is exactly the QA 3-divisibility boundary.

### The scribes' implicit orbit classification

The Rhind scribe Ahmes (c.1550 BCE) did not know modular arithmetic. But the 2/n table reveals that he **implicitly applied** the 3-divisibility test:
- Numbers divisible by 3: handle mechanically (2k, 6k formula)
- Numbers not divisible by 3: compute case-by-case

This is the oldest recorded instance of what QA formalizes as orbit_family classification. The Satellite orbit (3-divisible states) yields the "easy" decompositions; the Cosmos orbit yields the "hard" ones.

## Full Rhind 2/n table

| n | Orbit | Terms | Denominators |
|---|---|---|---|
| 3 | satellite | 2 | 2, 6 |
| 5 | cosmos | 2 | 3, 15 |
| 7 | cosmos | 2 | 4, 28 |
| 9 | singularity | 2 | 6, 18 |
| 11 | cosmos | 2 | 6, 66 |
| 13 | cosmos | 3 | 8, 52, 104 |
| 15 | satellite | 2 | 10, 30 |
| 17 | cosmos | 3 | 12, 51, 68 |
| 19 | cosmos | 3 | 12, 76, 114 |
| 21 | satellite | 2 | 14, 42 |
| 23 | cosmos | 2 | 12, 276 |
| 25 | cosmos | 2 | 15, 75 |
| 27 | satellite | 2 | 18, 54 |
| 29 | cosmos | **4** | 24, 58, 174, 232 |
| 31 | cosmos | 3 | 20, 124, 155 |
| 33 | satellite | 2 | 22, 66 |
| 35 | cosmos | 2 | 30, 42 |
| 37 | cosmos | 3 | 24, 111, 296 |
| 39 | satellite | 2 | 26, 78 |
| 41 | cosmos | 3 | 24, 246, 328 |
| 43 | cosmos | **4** | 42, 86, 129, 301 |
| 45 | satellite | 2 | 30, 90 |
| 47 | cosmos | 3 | 30, 141, 470 |
| 49 | cosmos | 2 | 28, 196 |
| 51 | satellite | 2 | 34, 102 |
| 53 | cosmos | 3 | 30, 318, 795 |
| 55 | cosmos | 2 | 30, 330 |
| 57 | satellite | 2 | 38, 114 |
| 59 | cosmos | 3 | 36, 236, 531 |
| 61 | cosmos | **4** | 40, 244, 488, 610 |
| 63 | satellite | 2 | 42, 126 |
| 65 | cosmos | 2 | 39, 195 |
| 67 | cosmos | 3 | 40, 335, 536 |
| 69 | satellite | 2 | 46, 138 |
| 71 | cosmos | 3 | 40, 568, 710 |
| 73 | cosmos | **4** | 60, 219, 292, 365 |
| 75 | satellite | 2 | 50, 150 |
| 77 | cosmos | 2 | 44, 308 |
| 79 | cosmos | **4** | 60, 237, 316, 790 |
| 81 | satellite | 2 | 54, 162 |
| 83 | cosmos | **4** | 60, 332, 415, 498 |
| 85 | cosmos | 2 | 51, 255 |
| 87 | satellite | 2 | 58, 174 |
| 89 | cosmos | **4** | 60, 356, 534, 890 |
| 91 | cosmos | 2 | 70, 130 |
| 93 | satellite | 2 | 62, 186 |
| 95 | cosmos | 3 | 60, 380, 570 |
| 97 | cosmos | 3 | 56, 679, 776 |
| 99 | satellite | 2 | 66, 198 |
| 101 | cosmos | **4** | 101, 202, 303, 606 |
