# [291] QA Fibonacci Matrix Orbit Periods Cert

**Family ID**: 291
**Slug**: `qa_fibonacci_matrix_orbit_periods_cert_v1`
**Status**: Active
**Registered**: 2026-06-01

## Claim (narrow, falsifiable)

For QA mod m=9, the **Fibonacci matrix** M = [[0,1],[1,1]] acting on (Z/9Z)² by (b,e) → (e, b+e) mod 9 has **order exactly 24** in GL(2,Z/9Z). This equals the Pisano period π(9). No proper divisor of 24 is the order.

The three orbit types partition {0,...,8}² = (Z/9Z)² exactly:

| Type | Characterization | Count | Period |
|---|---|---|---|
| Singularity | {(0,0)} | 1 | 1 |
| Satellite | {(b,e) : 3\|b AND 3\|e} \\ {(0,0)} | 8 | 8 |
| Cosmos | all other states | 72 | 24 |

Total: 1 + 8 + 72 = 81 = 9².

## Key Matrix Facts (mod 9)

```
M^1  = [[0,1],[1,1]]      (Fibonacci step)
M^8  = [[4,3],[3,7]]      (not I; kernel = satellite states)
M^12 = [[8,0],[0,8]] = -I (order does not divide 12)
M^24 = [[1,0],[0,1]] = I  (Pisano period)
```

M^8 - I = [[3,3],[3,6]] = 3·[[1,1],[1,2]]. The kernel of (M^8 - I) mod 9 is exactly the set {3|b AND 3|e}, confirming the Satellite characterization.

## Why the Order Is Exactly 24

The Pisano period satisfies π(p^k) = p^(k−1)·π(p) for primes p ≠ 2,5. So:

```
π(9) = π(3²) = 3^(2−1) · π(3) = 3 · 8 = 24
```

The Satellite period 8 = π(3) (the prime factor). The Cosmos period 24 = π(9) (the full modulus). The three orbit periods **are the Pisano periods of the prime power factors** of m=9.

## The Satellite as a Subgroup

States with 3|b and 3|e form a subgroup of (Z/9Z)² isomorphic to (Z/3Z)². Under M mod 3 (restricting to this subgroup), M has order π(3) = 8. The Singularity {(0,0)} is the identity element; the remaining 8 form a single orbit of length 8.

## Five-Families Alignment

| Family | Sequence pairs | Period | Orbit type |
|---|---|---|---|
| Fibonacci | (F_n mod 9, F_{n+1} mod 9) | 24 | Cosmos |
| Lucas | (L_n mod 9, L_{n+1} mod 9) | 24 | Cosmos |
| Phibonacci | similar | 24 | Cosmos |
| Tribonacci | (T_n mod 9, T_{n+1} mod 9) | 8 | Satellite |
| Ninbonacci | (9,9) only | 1 | Singularity |

Tribonacci pairs all lie in the {3|b, 3|e} subgroup because the tribonacci recurrence mod 3 has all values ≡ 0 (mod 3) from the tribonacci starting conditions. Ninbonacci (9,9) = (0,0) mod 9 is the fixed point.

## Checks

| ID | Description |
|---|---|
| FMO_PISANO_24 | M^24 = I₂ (mod 9) |
| FMO_PISANO_MIN | No proper divisor k of 24 satisfies M^k = I₂ (mod 9) |
| FMO_SAT_CHAR | {3\|b AND 3\|e} \\ {(0,0)} = exactly 8 states, each period 8 |
| FMO_PARTITION | 1 + 8 + 72 = 81 = 9² |
| FMO_ORBIT | Fixture state has the declared orbit period |
| FMO_TYPE | Fixture state has the declared orbit type |

**Fixtures**: 4 PASS + 2 FAIL
**Self-test**: exhaustive orbit distribution, M^12=-I check, Fibonacci pairs period-24, satellite orbit completeness

## Primary Sources

- Wall, D. D. (1960). Fibonacci primitive roots and the period of the Fibonacci sequence modulo a prime. *American Mathematical Monthly*, 67(6), 525–532. DOI: 10.1080/00029890.1960.11989541. Establishes Pisano period = order of Fibonacci matrix in GL(2,Z/mZ).
- Wildberger, N. J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8. QA T-operator and orbit classification.

## Mechanism Chain

- [289] QA Koenig Pell Ford Circle — SL(2,Z) generators are the two QA moves
- [290] QA Classical Subfamily Ford Cusps — Fibonacci/Tribonacci/Ninbonacci as three cusp families
- [281] QA Pisano-Orbit Correspondence — Pisano period = orbit period (related cert)
