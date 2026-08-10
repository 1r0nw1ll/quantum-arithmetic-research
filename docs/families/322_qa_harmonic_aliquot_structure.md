# [322] QA Harmonic Aliquot Structure

**Family**: `qa_harmonic_aliquot_structure_cert_v1`  
**Depends on**: [318] Synchronous Harmonics Ceiling (5040); [320] Quantize Algorithm (d-value, DO=d²); [321] Quantize-to-ONE (ratio structure)

## Scope boundary

This cert is a finite mod-9 Cosmos test model of Iverson's aliquot/multitude
theory. It does not certify the entire QA-3 energy-transport doctrine or the
full physical claim that higher harmonics literally create lower tones across
all myriads. What it certifies is narrower: within the enumerated d-value
domain `{3,...,17}`, Iverson's aliquot-part/unique-prime structure and
Rayleigh-reversal direction are represented by exact gcd/prime arithmetic.

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | 20 harmonic dyads among the 15 distinct Cosmos d-values {3,...,17}; all aliquot parts are 7-smooth (prime factors ≤ 7); verified by exhaustive enumeration | PASS |
| C2 | Direction law: for every harmonic dyad (d1<d2), the unique prime p2=d2/gcd > p1=d1/gcd — higher d carries the larger unique prime; confirms Ben's reversal "The higher harmonics CREATE the lower tone" (p.43) | PASS |
| C3 | Aliquot spectrum = {1, 2, 3, 5} exactly — no aliquot part > 5 appears among the 20 dyads; the shared aliquot structure lives entirely in the {2,3,5} smooth core | PASS |
| C4 | 5040 = 2⁴·3²·5·7 has ω=4 distinct prime factors; all 15 Cosmos d-values ≤ 17 < 5040 — within Ben's "definite quantum configuration" range (Quantum Flexibility threshold, p.53) | PASS |
| C5 | Exactly 3 Cosmos d-values are primes > 7: {11, 13, 17} (tonal identity carriers); each forms exactly 5 harmonic dyads (3 with {3,5,7} + 2 cross-tonal) | PASS |
| C6 | Source-native wave decomposition: a wave witness W=A*p has aliquot part A equal to the product of all non-unique prime factors, unique prime p not inside A, factor count 5/7 where declared, and A>p | PASS |
| C7 | Source-native harmony: two completed waves with the same aliquot core A and distinct unique primes p1,p2 satisfy gcd(W1,W2)=A and gcd(A,p1*p2)=1 | PASS |

## Theory of Harmony — Ben's Reversal

Iverson (p.52):
> "Harmonics, or harmony occurs between two dissimilar cycles of energy when both can be divided into similar aliquot parts having the same magnitude but different multitudes."

This is the formal QA statement of harmonic identity. Two d-values d1 and d2 are harmonic iff:
- Their common aliquot part A = gcd(d1, d2)
- d1/A = p1 and d2/A = p2 are **distinct primes** (their unique "multitudes")
- gcd(A, p1) = gcd(A, p2) = 1 (the unique primes are not factors of the aliquot part)

The **reversal model** (C2): within the finite d-value domain, higher d carries
the larger unique prime. This matches the direction of Iverson's stated
Rayleigh reversal, but the cert itself proves only the finite arithmetic model,
not the full corpus-level energy-transport claim.

## Harmonic Dyads in mod-9 Cosmos

| Dyad | Aliquot A | p1 | p2 |
|------|-----------|----|----|
| (3,5) | 1 | 3 | 5 |
| (3,7) | 1 | 3 | 7 |
| (3,11) | 1 | 3 | 11 |
| (3,13) | 1 | 3 | 13 |
| (3,17) | 1 | 3 | 17 |
| (5,7) | 1 | 5 | 7 |
| (5,11) | 1 | 5 | 11 |
| (5,13) | 1 | 5 | 13 |
| (5,17) | 1 | 5 | 17 |
| (6,10) | 2 | 3 | 5 |
| (6,14) | 2 | 3 | 7 |
| (6,15) | 3 | 2 | 5 |
| (7,11) | 1 | 7 | 11 |
| (7,13) | 1 | 7 | 13 |
| (7,17) | 1 | 7 | 17 |
| (10,14) | 2 | 5 | 7 |
| (10,15) | 5 | 2 | 3 |
| (11,13) | 1 | 11 | 13 |
| (11,17) | 1 | 11 | 17 |
| (13,17) | 1 | 13 | 17 |

## Gear Analogy (Iverson p.50)

"When two or more waves have the same aliquot parts, these aliquot parts will act in the same way that two or more gears will mesh by having teeth which are the same pitch. The diameter of each gear will represent the unique prime number of the wave."

The aliquot part = gear tooth pitch. The unique prime = gear diameter. Two d-values harmonize exactly when their "gears" have the same tooth pitch — same aliquot part — and different diameters (distinct unique primes).

## Completed-Wave Aliquot Witnesses

The validator now includes source-native completed-wave witnesses in addition
to the finite mod-9 d-value model:

| Wave W | Aliquot A | Unique p | Factor count |
|--------|-----------|----------|--------------|
| 2*3*5*7*11 | 2*3*5*7 | 11 | 5 |
| 2*3*5*7*11*13*17 | 2*3*5*7*11*13 | 17 | 7 |
| 2*3*5*7*11*13*19 | 2*3*5*7*11*13 | 19 | 7 |

The 17 and 19 witnesses share the same aliquot core A=30030 but have different
unique-prime multitudes. This is the direct arithmetic form of Iverson's
"same aliquot parts, different multitudes" claim rather than only a mod-9
analogue.

## 5040 — Quantum Flexibility Threshold

Iverson (p.53): "In the range above 5040 or 10,080, the correlating of a given number in quantum configuration, is less definite. There comes some slippage which is termed 'Quantum Flexibility'."

5040 = 7! = 2⁴ · 3² · 5 · 7: the product involving all four small primes in their first appearance in the factorization sequence. This connects to cert [318] (Synchronous Harmonics Ceiling) where 5040 is the extremal value for τ(n)=60.

## Tonal Identity Primes {11, 13, 17}

Three Cosmos d-values are primes > 7: {11, 13, 17}. These d-values carry "tonal identity" — they cannot be expressed as A×p with A > 1 from the 7-smooth core. Each forms 5 harmonic dyads:
- 3 with the lower primes {3, 5, 7} (aliquot=1)
- 2 cross-tonal with the other two tonal primes (aliquot=1)

"A wave gains its uniqueness through one prime number which is unique to that wave." (Iverson p.50) For d=11, 13, 17, the wave IS its unique prime — the entire d-value is the tonal identity.

## Negative Self-Test Probes

The self-test rejects three source-scope mistakes:

- An aliquot part smaller than the unique prime (`A=6, p=7`)
- A "unique" prime already present in the aliquot part (`A=210, p=7`)
- Two waves with different aliquot cores being misreported as same-aliquot harmony
