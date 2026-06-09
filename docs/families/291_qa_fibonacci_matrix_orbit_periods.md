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

The v1 cert now also validates the exact theorem-map feature used by
`tools/qa_multibase_workbench.py` for the cross-modulus pressure test. For
the transition matrix T = [[0,1],[1,1]]:

```
T^4 - I = [[1,3],[3,4]]
T^8 - I = [[12,21],[21,33]]
```

These two residual kernels define an exact predicate tree:

```
IF period8_fixed AND NOT period4_fixed -> Satellite
ELSE IF period1_fixed                  -> Singularity
ELSE                                      Cosmos
```

The validator exhaustively checks this rule against integer orbit iteration
for m = {9,12,15,18,21,24,27,30}. Result: 0 classification errors.

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
| FMO_EXACT_PERIOD_KERNELS | T^4-I and T^8-I equal the declared residual matrices |
| FMO_CROSS_MODULUS_RULE | Exact residual-kernel predicate rule matches exhaustive orbit iteration for pressure-test moduli |
| FMO_CROSS_MODULUS_COUNTS | Cross-modulus class counts match declared fixture counts |

**Fixtures**: 5 PASS + 3 FAIL
**Self-test**: exhaustive orbit distribution, M^12=-I check, Fibonacci pairs period-24, satellite orbit completeness, cross-modulus exact predicate theorem

## Primary Sources

- Wall, D. D. (1960). Fibonacci primitive roots and the period of the Fibonacci sequence modulo a prime. *American Mathematical Monthly*, 67(6), 525–532. DOI: 10.1080/00029890.1960.11989541. Establishes Pisano period = order of Fibonacci matrix in GL(2,Z/mZ).
- Wildberger, N. J. (2005). *Divine Proportions*. Wild Egg Books. ISBN 978-0-9757492-0-8. QA T-operator and orbit classification.

## Application Demos

Three standalone tools connect cert [291] to downstream applications:

### A. Multibase Orbit Classifier (`tools/qa_multibase_workbench.py`)
Transfer test: train exact residual-kernel predicate tree on moduli {9,12,15,18},
evaluate on held-out {21,24,27,30}. Result: **0 errors on 2646 held-out states**.
Report: `results/QA_EXACT_ORBIT_THEOREM_DEMO_2026_06_09.md`

### B. Langlands Prime Signatures (`tools/qa_langlands_prime_signatures.py`)
For each prime p ≤ 500 (95 primes): verify Wall's splitting law α(p) | p−1 or 2(p+1);
compute joint Fibonacci clock (F_p mod m, m ∈ {9,12,15,18,21,24}) across the
Chebotarev group structure for the joint period 240.
Result: **95/95 PASS Wall's law; 34 signature groups; 28/34 conjugate pairs (r, 240−r)**.
Report: `results/QA_LANGLANDS_PRIME_SIGNATURES_2026_06_08.md`

### C. Constructive HSI Classifier (`tools/qa_hsi_indian_pines.py`)
Real Indian Pines AVIRIS dataset (10 249 labeled pixels, 1799 integer features, 16 classes).
Features: 200 raw bands + 4 spatial-mean scales (3×3/5×5/9×9/15×15) + 3 spatial-variance
scales (texture) + 199 spectral first-differences (red-edge slope).

Constructive claim: every classification error is a structural diagnosis, not noise.
For each of the 120 class pairs the tree issues a certificate:
- **SEPARABLE** (70/120, 58%): integer threshold exists at some feature → tree branch added
- **INDISTINGUISHABLE** (50/120, 42%): no threshold at any feature → requires new sensor

Test accuracy **91.3%**. Of 223 test errors: **211 are spectral-limit** (diagnosed),
**12 are tree errors** (separable pairs missed by greedy split). Top confused cluster:
Corn/Soy variants (gap −7 to −9), spectrally overlapping even in texture + derivative space.
These require multi-temporal NDVI or LiDAR to resolve.
Report: `results/QA_HSI_INDIAN_PINES_2026_06_09.md`

### D. Houston Multimodal LiDAR Validation (`tools/qa_hsi_houston_lidar.py`)
Houston 2013 GRSS DFC dataset (2817 samples, 15 urban classes, 105 class pairs).
Validates that the constructive certificate is an **actionable sensor guide** — not
merely a description of failure — by running the classifier twice: HSI-only, then
HSI+LiDAR (2 integer features: centre-pixel height in decimetres + patch mean).

| Pass | Separable pairs | Test accuracy | Spectral-limit errors |
|---|---:|---:|---:|
| HSI-only | 45 / 105 | 92.3% | 50 |
| HSI+LiDAR | 57 / 105 | 93.1% | 39 |

**LiDAR promoted 12 pairs** from INDISTINGUISHABLE → SEPARABLE. Every promoted pair
involves Commercial buildings where height is the key: Soil vs Commercial (gap −332 → +38),
Commercial vs Parking-lot-2 (−211 → +36), Commercial vs Running-track (−305 → +36).
No non-LiDAR pair was promoted — the certificate named exactly the right sensor.
Report: `results/QA_HSI_HOUSTON_LIDAR_2026_06_09.md`

## Mechanism Chain

- [289] QA Koenig Pell Ford Circle — SL(2,Z) generators are the two QA moves
- [290] QA Classical Subfamily Ford Cusps — Fibonacci/Tribonacci/Ninbonacci as three cusp families
- [281] QA Pisano-Orbit Correspondence — Pisano period = orbit period (related cert)
