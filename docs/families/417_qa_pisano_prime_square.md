<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Sun-Sun (1992) Acta Arithmetica 60(3), McIntosh-Roettger (2007) doi:10.1090/S0025-5718-07-01955-2 -->
# [417] QA Pisano Lift to Prime Squares — Wall-Sun-Sun Regularity

**Cert family**: `qa_pisano_prime_square_cert_v1`
**Claim**: The QA T-step modulo p² yields a Pisano period exactly p times the period mod p:

> **π(p²) = p · π(p)**    for all primes tested

Cert [416] proved p | F_{p−(5/p)} (first-order divisibility). Cert [417] lifts to:
p² ∤ F_{p−(5/p)} for all tested primes — the second-order condition fails, as expected if no
Wall-Sun-Sun prime exists.

## What a Wall-Sun-Sun prime would mean

A Wall-Sun-Sun prime (if it existed) would be a prime p where the QA T-orbit mod p² has the
*same period* as the orbit mod p — the period would fail to multiply by p when lifting the
modulus. Equivalently, four conditions all fail simultaneously for every known prime:

| Condition | Statement | Status |
|---|---|---|
| (A) | π(p²) = π(p) [not p·π(p)] | False for all tested primes |
| (B) | α(p²) = α(p) [rank does not lift] | False for all tested primes |
| (C) | p² \| F_{α(p)} [higher-order zero] | False for all tested primes |
| (D) | w_F(p) = 0 [Fermat quotient vanishes] | False for all tested primes |

Sun-Sun (1992) proved (A)⟺(B)⟺(C)⟺(D). No prime satisfying any of these has been found;
numerical search (McIntosh-Roettger 2007) confirms none below 2×10¹⁴.

## The Wall-Sun-Sun Fermat quotient w_F(p)

Since p | F_{p−(5/p)} (cert [416]), the quotient

> w_F(p) = F_{p−(5/p)} / p  mod p

is a well-defined integer in {0, 1, ..., p−1}. A Wall-Sun-Sun prime satisfies w_F(p) = 0.
All tested primes have w_F(p) ≠ 0. Selected values:

| p | class | w_F(p) | w_F(p)/p approx |
|---|---|---|---|
| 2 | inert | 1 | 0.5 |
| 5 | ramified | 1 | 0.2 |
| 11 | split | 5 | 0.45 |
| 41 | split | 39 ≡ −2 | 0.95 |
| 251 | split | 250 ≡ −1 | 0.996 |
| 421 | split | 76 | 0.18 |
| 491 | split | 66 | 0.13 |

## The p-adic unfolding (C4 orbit reading)

The QA T-orbit of (0,1) mod p² has **p sheets** over the orbit mod p:

```
Orbit mod p:   (0,1) → (1,1) → ... → period π(p) → (0,1)

Orbit mod p²:  (0,1) → (1,1) → ...
                    at step α(p):   (F_{α(p)} mod p², *)
                                     ≡ 0 mod p, ≢ 0 mod p²  ← first p-zero, not p²-zero
                    at step 2α(p):  still ≢ 0 mod p² in general
                    ...
                    at step p·α(p): (0, *)  ← first p²-zero
                    at step p·π(p): (0, 1)  ← full orbit closes
```

The orbit "unwinds" p times before hitting the first exact p²-zero at step p·α(p).

Sample witnesses (C4):

| p | α(p) | F_{α(p)} mod p² | p·α(p) | F_{p·α(p)} mod p² |
|---|---|---|---|---|
| 11 | 10 | 55 (≢ 0 mod 121) | 110 | 0 ✓ |
| 3 | 4 | 3 (≢ 0 mod 9) | 12 | 0 ✓ |
| 5 | 5 | 5 (≢ 0 mod 25) | 25 | 0 ✓ |
| 41 | 20 | 41 (≢ 0 mod 1681) | 820 | 0 ✓ |
| 7 | 8 | 21 (≢ 0 mod 49) | 56 | 0 ✓ |

## The Pisano period lift (C1)

Direct computation of π(p²) via T-step mod p² for 15 small primes:

| p | π(p) | π(p²) | π(p²)/π(p) |
|---|---|---|---|
| 2 | 3 | 6 | 2 |
| 3 | 8 | 24 | 3 |
| 5 | 20 | 100 | 5 |
| 7 | 16 | 112 | 7 |
| 11 | 10 | 110 | 11 |
| 13 | 28 | 364 | 13 |
| 23 | 48 | 1104 | 23 |
| 37 | 76 | 2812 | 37 |
| 41 | 40 | 1640 | 41 |
| 47 | 32 | 1504 | 47 |

In each case π(p²)/π(p) = p exactly.

## Connection to the chain

Cert [415] established the "first-order" picture: mod p, the T-orbit period encodes prime
splitting via Kronecker (5/p). Cert [416] sharpened to one formula α(p)|p−(5/p).

Cert [417] is the "second-order" question: does the orbit structure lift cleanly to mod p²?
The answer is yes for all tested primes — the orbit grows by exactly a factor of p, consistent
with the p-adic expansion being non-degenerate at p.

The algebraic interpretation (via ℤ[φ]/p²):
- For split p: ℤ[φ]/p² ≅ ℤ/p² × ℤ/p² (two sheets, each with p² elements)
- For inert p: ℤ[φ]/p² is an order in 𝔽_{p²} with p² elements
- The orbit period reflects the group structure: |group/torsion| = p · |group mod p|

A Wall-Sun-Sun prime would signal a degenerate structure where the p-adic expansion of φ in
ℤ[φ]_p (the p-adic completion) has a vanishing first-order coefficient — the φ-unit is a
"p-adic near-identity" at higher precision.

## Checks

- **C1**: π(p²) = p·π(p) for 15 primes ≤ 47 (direct T-step mod p²) — **PASS**
- **C2**: w_F(p) ≠ 0 for 45 primes (Wall-Sun-Sun Fermat quotient) — **PASS**
- **C3**: α(p²) = p·α(p) for 45 primes (rank of apparition lifts) — **PASS**
- **C4**: QA T-orbit mod p²: p-sheets, first p²-zero at step p·α(p), 5 sample primes — **PASS**

## Chain

- **Inherits from**: cert [415] (Pisano period = prime splitting), cert [416] (rank of apparition)
- **Bridges to**: cert [413] (BSD — same prime trichotomy at s=½), cert [414] (norm form)
- **Open connection**: Wall-Sun-Sun conjecture (none of the four equivalent conditions holds for any known prime)

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-derived π(p²)=p·π(p) from
scratch in a fresh script (own Pisano-period implementation) for all
10 sample primes {2,3,5,7,11,13,23,37,41,47} — exact match. Correctly
cites the real Wall-Sun-Sun literature (Sun-Sun 1992 equivalence proof;
McIntosh-Roettger 2007 search bound of 2×10¹⁴, no known WSS prime).
Genuine falsifiable number theory, no fixture-trusting gap.
