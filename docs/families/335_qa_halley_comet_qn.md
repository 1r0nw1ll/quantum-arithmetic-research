# [335] QA Halley's Comet Quantum Number

**Family**: `qa_halley_comet_qn_cert_v1`  
**Source**: Iverson (1993) *QA Book 2: Natural Arithmetic*, Ch.1 "Doubling" p.7

> "The Quantum Number for the orbit of Halley's Comet is 1, 29, 30, 59.  
>  Its ratio between the perigee and apogee is 1:59. This is a much elongated elliptical orbit."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | QN=(1,29,30,59): d=b+e=30, a=b+2e=59, gcd(1,29)=1 | PASS |
| C2 | b:a = 1:59; b=1 (unit base); a=59 prime; e=29 prime | PASS |
| C3 | d=30=2×3×5 (5-smooth); d=lcm(2,3,5); maximally composite d for b=1 | PASS |
| C4 | mod-24 orbit: (1,29) → (b24,e24,d24,a24)=(1,5,6,11); Cosmos family | PASS |
| C5 | Among b=1 QNs with a≤60: (1,29,30,59) is maximally elongated (ratio 1:59); next is (1,28,29,57) | PASS |

## Core Structural Result

### The Quantum Number

$$\text{QN} = (b, e, d, a) = (1, 29, 30, 59)$$

- $d = b + e = 1 + 29 = 30$ ✓ (A2 raw, no mod reduction)
- $a = b + 2e = 1 + 58 = 59$ ✓
- $\gcd(1, 29) = 1$ — valid coprime pair

### Perigee:Apogee Ratio

$$b : a = 1 : 59$$

The first and fourth elements of the QN encode the orbital extremes. With b=1, this is the **unit base orbit** — the most fundamental starting point. a=59 is prime, giving an irreducible ratio.

### d=30 Structure

$$d = 30 = 2 \times 3 \times 5 = \text{lcm}(2, 3, 5)$$

The intermediate sum d=30 is the smallest number divisible by 2, 3, and 5. It is the maximally composite value achievable with d = 1 + 29 (b=1, e=29). This connects Halley's QN to the harmonic base of the QA aliquot system.

### Mod-24 Orbit Classification (C4)

| Value | Raw | mod-24 |
|-------|-----|--------|
| b | 1 | 1 |
| e | 29 | 5 |
| d | 30 | 6 |
| a | 59 | 11 |

Orbit family: **Cosmos** (24-cycle under QA mod-24 dynamics).

### Elongation Context (C5)

Among all male QNs with b=1 and a≤60:

| QN (1,e,d,a) | a/b ratio | e prime? |
|--------------|-----------|----------|
| (1,29,30,59) | **59** | ✓ (most elongated) |
| (1,28,29,57) | 57 | ✗ |
| (1,27,28,55) | 55 | ✗ |

Halley's QN uses the largest prime e<30, producing the maximum elongation while staying in the {a≤60} window.

## Observer Projection Note (Theorem NT)

"Comet orbit", "perigee", "apogee", "elliptical" are observer-layer labels. The causal structure: integer QN (1,29,30,59) with its arithmetic properties (d=30=lcm(2,3,5), a=59 prime, b:a=1:59). The physical orbit is an observer projection of this integer structure — not a cause.

**Depends on**: [151] QA Par Numbers; [326] Euclid Four Par Types; [323] Harmonic Chemistry LCM
