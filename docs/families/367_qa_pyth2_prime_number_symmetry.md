# [367] QA Pyth-2 Prime Number Symmetry

**Family**: `qa_pyth2_prime_number_symmetry_cert_v1`  
**Source**: Iverson (1993) *Pythagorean Arithmetic Vol II* Chapter XII pp.28-37

> *(p.29-30)*: "If we start at both ends of 1, 7, 11, 13, 17, 19, 23, and 29 and add the matched numbers together we come up with 1+29=30, 7+23=30, 11+19=30, and 13+17=30."

> *(p.30)*: "3 × 4 × 5 = 60 ... 1+59=60, 7+53=60, 11+49=60, 13+47=60, 17+43=60, 19+41=60, 23+37=60, and 29+31=60."

> *(p.30)*: "The number 49 is included, but as previously explained, it is a prime number because it has a single factor of 7."

## Claims

| ID | Claim | Status |
|----|-------|--------|
| C1 | Coprime-to-30 bracket {n∈[1,29]: gcd(n,30)=1}={1,7,11,13,17,19,23,29}; φ(30)=8; 4 pairs summing to 30 | PASS |
| C2 | Coprime-to-60 bracket {n∈[1,59]: gcd(n,60)=1} has 16 elements in 8 pairs summing to 60; 49=7²∈bracket | PASS |
| C3 | Pairing identity: gcd(M−n,M)=gcd(n,M) for all M,n; n coprime to M iff M−n coprime to M | PASS |
| C4 | All primes p∈(5,30)={7,11,13,17,19,23,29} satisfy gcd(p,30)=1 (all in the 30-bracket) | PASS |
| C5 | I(1,5)=49=7² ∈ coprime-to-60 bracket; gcd(49,60)=1; partner 60−49=11; 11+49=60 | PASS |

## Mathematical Details

### C1: The 30-Bracket (Coprime-to-30 Symmetry)

The modulus M=30=2×3×5. The integers coprime to 30 in [1,29] are exactly those sharing no factor with 2, 3, or 5.

**Bracket**: {1, 7, 11, 13, 17, 19, 23, 29}

**Count**: φ(30) = φ(2)·φ(3)·φ(5) = 1·2·4 = 8

**Pairs summing to 30**:
| n | 30−n | Sum |
|---|------|-----|
| 1 | 29 | 30 |
| 7 | 23 | 30 |
| 11 | 19 | 30 |
| 13 | 17 | 30 |

**Why the pairing works**: gcd(30−n, 30) = gcd(−n, 30) = gcd(n, 30). So if n is coprime to 30, then 30−n is also coprime to 30. Since no element equals 15 (gcd(15,30)=15≠1), every element has a distinct partner: n ≠ 30−n. So the 8 elements form exactly 4 pairs.

### C2: The 60-Bracket (Coprime-to-60 Symmetry)

The modulus M=60=2²×3×5=3×4×5. The integers coprime to 60 in [1,59]:

**Bracket**: {1, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 49, 53, 59}

**Count**: φ(60) = φ(4)·φ(3)·φ(5) = 2·2·4 = 16

**Pairs summing to 60**:
| n | 60−n | Sum | Notes |
|---|------|-----|-------|
| 1 | 59 | 60 | both prime |
| 7 | 53 | 60 | both prime |
| 11 | 49 | 60 | 49=7² (functionally prime) |
| 13 | 47 | 60 | both prime |
| 17 | 43 | 60 | both prime |
| 19 | 41 | 60 | both prime |
| 23 | 37 | 60 | both prime |
| 29 | 31 | 60 | both prime (twin primes!) |

**49 in the bracket**: gcd(49, 60) = gcd(7², 2²·3·5) = 1 since 7 is coprime to 2, 3, and 5. So 49 is genuinely in the coprime-to-60 bracket — not by a special exception but by the standard coprimality criterion.

### C3: The Pairing Identity

**Theorem**: For any integer M≥2 and any n∈[1,M−1]: gcd(M−n, M) = gcd(n, M).

**Proof**: gcd(M−n, M) = gcd(−n mod M, M) = gcd(n, M) (since gcd is invariant under negation mod M).

**Corollary**: n is coprime to M if and only if M−n is coprime to M. This is the foundational reason why all coprime-to-M brackets exhibit pairing symmetry.

Verified for all M∈[2,99] and all n∈[1,M−1] — 4851 cases total.

### C4: All Primes Above 5 Are in the 30-Bracket

Any prime p>5 satisfies p∉{2,3,5}, so p shares no prime factor with 30=2·3·5, giving gcd(p,30)=1.

The primes in (5,30) = {7, 11, 13, 17, 19, 23, 29} — all 7 of them appear in the coprime-to-30 bracket {1,7,11,13,17,19,23,29}. The only non-prime in this bracket is 1 (which is coprime to everything).

**Converse**: Every element of the 30-bracket except 1 is prime or a prime power (since each must have all prime factors ∉{2,3,5}, and elements <30 with all prime factors >5 can only be 7, 11, 13, 17, 19, 23, or 29 — all prime — and 49>30 so doesn't appear).

### C5: Koenig Value 49=7² Bridging Koenig Series and Prime Symmetry

From cert [364] (Koenig Series): I(1,5) = 2·5²−1 = 49 = 7².

This same value 49 appears in the coprime-to-60 bracket:
- gcd(49, 60) = 1 ✓
- Bracket partner: 60−49 = 11 (prime, gcd(11,60)=1) ✓
- Iverson's third pair in the 60-bracket: 11+49=60 ✓

This connects Ch.VII (Koenig Series, where 49 was the first composite I-value in the b=1 branch) with Ch.XII (Prime Number Count, where 49 appears in the symmetric structure of the 60-bracket). The shared structure: 49=7² is coprime to 60=2²·3·5 because gcd(7,2)=gcd(7,3)=gcd(7,5)=1.

## Theorem NT Note

The "Sieve of Eratosthenes" described in Ch.XII is an observer projection — a mechanical procedure for finding primes. The certifiable QA content is the coprimality structure itself: the coprime-to-M brackets and their pairing symmetry are purely about integer gcd arithmetic. The sieve is one way to *observe* which integers are prime; the symmetry is a *structural* property of coprimality.

**Depends on**: [364] Koenig Series (49=I(1,5)=7² first composite); [366] Bead Arithmetic Laws (factor 3 in every bead set, underpinning the 30=2·3·5 structure)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently reproduced the coprime-to-30 and
coprime-to-60 brackets exactly ({1,7,11,13,17,19,23,29} and its 16-element
superset), matching the doc's tables precisely. The remaining claims
(pairing identity gcd(M-n,M)=gcd(n,M) over 4851 cases, prime-in-30-bracket
inclusion, and the 49=I(1,5) cross-reference to cert [364]) were
confirmed by running the validator itself, which genuinely recomputes
every case — no fixture-trusting gap.
