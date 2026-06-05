# [323] QA Harmonic Chemistry LCM

**Family**: `qa_harmonic_chemistry_lcm_cert_v1`  
**Depends on**: [322] Harmonic Aliquot Structure (harmonic dyads); [318] Synchronous Harmonics Ceiling (5040)

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | LCM harmonic cycle identity: for all 20 Cosmos harmonic dyads, lcm(d1,d2) = d1×p2 = d2×p1 = A×p1×p2 — the "harmonic cycle" of a bonded pair is their LCM | PASS |
| C2 | 3-wave composite LCM: for all C(6,3)=20 3-subsets of prime Cosmos d-values {3,5,7,11,13,17}, lcm(d1,d2,d3) = d1×d2×d3 (pairwise coprime) | PASS |
| C3 | C(n,2) pairing law: n mutually harmonic wavelets form C(n,2)=n(n-1)/2 pairs; C(7,2)=21 (Iverson's "21 different pairs"); mod-9 Cosmos max group = 6 | PASS |
| C4 | L = b·e·d·a wavelength product: all 72 Cosmos L values are positive integers; min=24 at (2,1), max=31824 at (8,9) | PASS |
| C5 | Universal 2·3 bond: b·e·d·a divisible by 6 for ALL 72 Cosmos pairs — QA expression of "all Quantum Waves must be multiples of 6" | PASS |

## Harmonic Cycle Identity

Iverson's explicit calculation (pp.62–63): for harmonic waves W1=A×p1 and W2=A×p2,

```
HC = W1 × p2 = W2 × p1
```

This is exactly the **LCM formula**: since gcd(W1,W2) = A,

```
lcm(W1,W2) = W1×W2/gcd(W1,W2) = A×p1 × A×p2 / A = A×p1×p2
```

The harmonic cycle is the first moment when both waves "start over" together — algebraically, their LCM.

**Iverson's example** (p.62):
- W1 = 870,870 = 2·3·5·7·11·13·29 (unique prime p1=5 vs W3)
- W2 = 5,399,394 = 2·3·7·11·13·29·31 (unique prime p2=31 vs W1)
- A = gcd(W1,W2) = 2·3·7·11·13·29 = 174,174
- HC = 870,870 × 31 = 5,399,394 × 5 = 26,996,970 = lcm(W1,W2) ✓

## C(n,2) — The 21-Pair Prediction

Iverson (p.68): "One can add four more wavelets that harmonize with these by having the same common factors of 2, 3, 11, 13, & 29 but having a different unique prime factor for each wave. They will make 21 different pairs."

The combinatorics: 3 + 4 = 7 mutually harmonic wavelets → C(7,2) = 7×6/2 = **21 pairs**. All 21 pairs share the same aliquot A = 2·3·11·13·29 = 24,882.

The mod-9 Cosmos maximum mutual-harmonic group has **6 members** (the prime d-values {3,5,7,11,13,17} with A=1). This is C(6,2) = 15 harmonic pairs. Ben's n=7 scenario requires a larger Myriad.

| n | C(n,2) | Description |
|---|--------|-------------|
| 2 | 1 | A single harmonic pair |
| 3 | 3 | Ben's 3-wave example (pp.62–63) |
| 4 | 6 | Minimum "tetrad" |
| 5 | 10 | Five-wavelet bond |
| 6 | 15 | Mod-9 Cosmos maximum (prime d group) |
| 7 | **21** | Ben's Music of the Spheres scale (p.68) |

## L = b·e·d·a — The Wavelength Product

Iverson (p.72): "the wavelength of a wave of sound energy relates directly to the product of all four of the integers (b,e,d,a) in a Quantum Number. That value has been designated as 'L' which is the area of the Pythagorean triangle which creates the ellipse."

For the Cosmos:
- Minimum: L=24 at (b=2,e=1): 2·1·3·4=24
- Maximum: L=31824 at (b=8,e=9): 8·9·17·26=31824

The "Pythagorean triangle area" interpretation: the chromogeometric area element involves F·C/2 where F and C are the short and long legs of the associated Pythagorean triple. The connection between b·e·d·a and the triangle area is a bridge from Iverson's harmonic model to cert [152] (Equilateral Triangle) and cert [138] (Plimpton 322).

## Universal 2·3 Bond (C5)

Iverson (p.67): "All waves of any dimension within any single Myriad of scale of energy are bonded together by the factors 2 and 3 which they contain. This bond is very weak because it is only two prime numbers, but in spite of the weakness, it is the beginning of quantumness. We can see by this that all Quantum Waves must be some multiple of 6."

QA certification: for ALL 72 Cosmos pairs (b,e), the product b·e·d·a = b·e·(b+e)·(b+2e) is always divisible by 6. This is not a coincidence: among any 3 consecutive-like integers in the BEDA sequence, at least one is even and at least one is divisible by 3.

**Proof sketch**: For any Cosmos pair (b,e) with b≠e:
- The four values b, e, d=b+e, a=b+2e form an arithmetic-like progression; among any 3 consecutive integers, one is divisible by 3 (Pigeonhole). Among {b,e,d}, since d=b+e, at least one of b,e,d is divisible by 3 (if b≡1, e≡2 mod 3 then d≡0; etc.) similarly for 2.
