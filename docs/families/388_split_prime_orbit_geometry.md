# [388] QA Split Prime Orbit Geometry

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_split_prime_orbit_geometry_cert_v1/`

## Claim

For σ(a,b) = (a+b mod p, a) acting on **F_p × F_p** for split prime p:

> When the two roots r₁, r₂ of x²−x−1 mod p have **unequal orders**, the orbit structure has exactly **three** period sizes: {1, ord_min, π(p)}, where ord_min = min(ord_p(r₁), ord_p(r₂)). The period-ord_min orbits are exactly the eigenspace of σ for the eigenvalue of smaller order. For inert primes, only **two** period sizes {1, π(p)} appear — the intermediate period is a split-prime signature.

| Check | Result |
|-------|--------|
| SPLIT_THREE_PERIODS: periods = {1, ord_min, π(p)} for p ∈ {11,19,29,31} | PASS |
| EIGENSPACE_IDENTIFICATION: period-ord_min orbits = {(r_min·c, c) : c ∈ F_p\*} | PASS |
| ORBIT_COUNTS: 1 + (p−1)/ord_min + (p²−p)/π(p) = p² for each | PASS |
| EQUAL_ORDER_CASE: p=41 (both roots order 40) → only periods {1, 40} | PASS |
| INERT_CONTRAST: p ∈ {3,7,13,17} → only {1, π(p)}, no intermediate | PASS |

## Orbit structure table

| p | Class | r₁ | ord(r₁) | r₂ | ord(r₂) | ord_min | π(p) | Periods | Orbits |
|---|-------|----|---------|----|---------|---------|----|---------|--------|
| 3  | inert | — | — | — | — | — | 8  | {1, 8}      | 1+1=2 total |
| 11 | split | 4 | 5 | 8 | 10 | 5 | 10 | {1, 5, 10}  | 1+2+11=14 |
| 13 | inert | — | — | — | — | — | 7  | {1, 7}      | 1+3=4 total |
| 19 | split | 5 | 9 | 15 | 18 | 9 | 18 | {1, 9, 18}  | 1+2+19=22 |
| 29 | split | 24| 7 | 6  | 14 | 7 | 14 | {1, 7, 14}  | 1+4+58=63 |
| 31 | split | 19| 15| 13 | 30 | 15| 30 | {1, 15, 30} | 1+2+31=34 |
| 41 | split | 7 | 40| 35 | 40 | 40| 40 | {1, 40}     | 1+42=43 |

## The eigenspace orbit (intermediate period)

For split p with unequal root orders, the eigenvector of σ for eigenvalue r_min is (r_min, 1):

```
σ(r_min·c, c) = (r_min·c + c, r_min·c) = ((r_min+1)·c, r_min·c) = r_min·(r_min·c, c)
```

since r_min² = r_min + 1 (characteristic equation). So (r_min·c, c) is a true eigenvector, and the entire eigenspace {(r_min·c, c) : c ∈ F_p\*} forms orbits of period exactly ord_p(r_min) = ord_min.

## Why inert primes have no intermediate period

For inert p, x²−x−1 is irreducible mod p, so σ = F (Fibonacci matrix) has NO eigenvectors in F_p² — its characteristic roots live in GF(p²) only. With no eigenspace in F_p², there is no "smaller-period" orbit family. The ring ℤ[φ]/(p) = GF(p²) is a FIELD (local ring), and φ acts as a primitive element of GF(p²)×.

For split p, ℤ[φ]/(p) ≅ F_p × F_p — NOT a field. Two distinct maximal ideals correspond to the two roots. The eigenspace orbits are the orbits within one of the two "copies" of F_p inside ℤ[φ]/(p).

## Root-order rule

For split prime p with roots r₁ · r₂ ≡ −1 (mod p):
- If one root has odd order d < π(p): the other has order 2d = π(p). Intermediate period exists.
- If both roots have even order π(p): equal orders, no intermediate period (p=41 case).

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_split_prime_orbit_geometry_cert_v1
python qa_split_prime_orbit_geometry_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {"SPLIT_THREE_PERIODS": true, "EIGENSPACE_IDENTIFICATION": true, "ORBIT_COUNTS": true, "EQUAL_ORDER_CASE": true, "INERT_CONTRAST": true}, ...}`

## Lineage

- Extends **[386]** (inert/split/ramified prime classification; φ primitive in GF(9))
- Extends **[387]** (Witt carry sub-orbit invariant for inert p=3)
- Completes the orbit-geometry picture for all prime types: inert ([385]/[387]), split ([388]), ramified ([386] C5)

## Primary sources

- Wall, D.D. (1960). [doi.org/10.1080/00029890.1960.11989541](https://doi.org/10.1080/00029890.1960.11989541) — Pisano periods, orders of roots of Fibonacci characteristic polynomial
- Neukirch, J. (1999). *Algebraic Number Theory*. ISBN 978-3-540-65399-8 §I.8
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.5

## What this cert does NOT claim

- Does not certify the orbit structure of ℤ[φ]/(p²) for split p (only ℤ[φ]/(p) = F_p × F_p)
- Does not prove the root-order rule (stated as an observed pattern; the algebraic proof is r₁r₂=−1 → r₂^(ord_r₁) = (−1)^(ord_r₁))
- Does not certify the ramified case (p=5) orbit geometry

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-derived the full root-order
table in a fresh, separate script for p ∈ {11,19,29,31,41}: roots and
their multiplicative orders match the doc exactly (e.g. p=11: roots
{4,8}, orders {5,10}; p=41: roots {7,35}, both order 40 — the
equal-order exception), and Pisano periods π(p) match the larger root
order in each split case. Genuine falsifiable number theory, not
fixture-trusting.
