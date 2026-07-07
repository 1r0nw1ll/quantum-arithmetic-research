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

## GCD-3 Structural Requirement

The Satellite characterization {3|b AND 3|e} is only a genuine **subgroup** of (Z/mZ)² when 3|m — adding m never changes a residue's value mod 3, so `b%3==0` is a well-defined quotient predicate. When gcd(m,3)=1, 3 is a unit mod m, and the literal condition on canonical representatives is **not** closed under addition (e.g. m=10: 9%3==0 and 3%3==0, but (9+3)%10=2, and 2%3≠0).

**Claim (narrow, falsifiable, verified not proved-in-general)**: for the 8 tested moduli m ∈ {7,10,11,13,14,16,17,20} (all gcd(m,3)=1), the Satellite class is **empty** — no non-origin state has orbit period 8, and the predicate-tree's `period8_nontrivial` flag never fires. This is consistent with a general theorem but is not a full proof of one: M⁸−I mod 9 = [[12,21],[21,33]] = 3·[[4,7],[7,11]]; the factor of 3 is invertible whenever gcd(m,3)=1, so the `period8_fixed` system reduces to a determinant-(−5) system with only the trivial solution **when gcd(m,5)=1** (m=7,11,13,14,16,17 above). For m=10,20 (both divisible by 5), emptiness is confirmed by exhaustive enumeration only — the closed-form argument for the 5|m case is an open gap, not resolved here.

Verified for all 8 moduli: the orbit-period spectrum is a richer, m-specific divisor lattice of π(m), with no period-8 class anywhere:

| m | period spectrum |
|---|---|
| 7 | {1:1, 16:48} |
| 10 | {1:1, 3:3, 4:4, 12:12, 20:20, 60:60} |
| 11 | {1:1, 5:10, 10:110} |
| 13 | {1:1, 28:168} |
| 14 | {1:1, 3:3, 16:48, 48:144} |
| 16 | {1:1, 3:3, 6:12, 12:48, 24:192} |
| 17 | {1:1, 36:288} |
| 20 | {1:1, 3:3, 4:4, 6:12, 12:60, 20:20, 60:300} |

**Relation to cert [515]** (QA Orbit-Lattice Mod-3 Collapse): [515] independently proves, for its own `qa_step` NTRU-coefficient recursion (not the Fibonacci matrix used here), that reduction mod 3 is only a well-defined quotient map when 3|m, and that this creates extra low-period structure absent when gcd(m,3)=1. This cert's finding is **suggestive of, not proved identical to**, that mechanism — both share the same mod-3-invertibility root cause, but this cert does not establish that its Satellite subgroup literally *is* [515]'s NTRU weakness.

New checks: `FMO_GCD3_SUBGROUP_CLOSURE`, `FMO_SATELLITE_EMPTY_OFF3`, `FMO_NONMULT3_SPECTRUM`.

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
| FMO_GCD3_SUBGROUP_CLOSURE | {3\|b,3\|e} closed under mod-m addition iff 3\|m (true on cross-moduli, false on gcd(m,3)=1 moduli) |
| FMO_SATELLITE_EMPTY_OFF3 | No period-8 states and no nontrivial `period8_fixed` flags for gcd(m,3)=1 moduli |
| FMO_NONMULT3_SPECTRUM | Exact orbit-period distribution matches declared divisor-lattice spectrum for gcd(m,3)=1 moduli |

**Fixtures**: 6 PASS + 4 FAIL
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

**120/120 class pairs SEPARABLE** — every pair has at least one integer feature
with gap ≥ 0. Zero spectral-limit errors. 11 residual errors are pure tree errors
(ensemble vote wrong; separator exists but isn't reliably found in every random subspace).

Feature set (3997 integer features): raw bands + 6 spatial-mean scales (3×3…31×31) +
4 spatial-variance scales + cross-scale contrast + dual-scale directional anisotropy
(1×21, 21×1, 1×31, 31×1 strip means and their H−V differences) + spectral first- and
second-differences.

| Method | Test accuracy | Errors | Sensor-limit errors |
|---|---:|---:|---:|
| Spectral only, single tree | 70.7% | 749 | 745 |
| +spatial features, single tree | 91.3% | 223 | 211 |
| +ensemble (31 trees) | 98.8% | 30 | 30 |
| +larger windows, 101 trees bagged | 99.5% | 14 | 14 |
| **+anisotropy+curvature, 201 trees** | **99.6%** | **11** | **0** |

10 classes achieve 100% test accuracy. Residual 11 errors: barely-separable Corn/Soy
pixels where the ensemble vote goes wrong (gap +1 to +3 after feature addition).
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
- [515] QA Orbit-Lattice Mod-3 Collapse — shares this cert's mod-3-invertibility root cause (proved independently for [515]'s own `qa_step` recursion, not shown identical to the Fibonacci-matrix Satellite subgroup here)
