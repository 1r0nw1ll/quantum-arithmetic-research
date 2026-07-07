<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Cassini (1680), Wall (1960) doi:10.2307/2309169, Lehmer (1930) doi:10.2307/1968235 -->
# [418] QA Cassini Alpha Parity Gate

**Cert family**: `qa_cassini_alpha_parity_cert_v1`
**Claim**: The parity of α(p) — the QA T-step first-zero time — is constrained by whether
−1 is a quadratic residue mod p, via the Cassini identity evaluated at the first-zero step.

> **F_{α(p)−1}² ≡ (−1)^{α(p)} mod p**    [Cassini Gate identity]
>
> Consequence: **p ≡ 3 mod 4  ⟹  α(p) is even**

## The Cassini identity at the first-zero step

Cassini (1680): F_{n−1}·F_{n+1} − F_n² = (−1)^n

At n = α(p), where F_{α(p)} ≡ 0 mod p:
- F_{α(p)+1} = F_{α(p)} + F_{α(p)−1} ≡ F_{α(p)−1} mod p
- Substituting: **F_{α(p)−1}² ≡ (−1)^{α(p)} mod p**  [the Cassini Gate]

If α(p) is **odd**: F_{α(p)−1}² ≡ −1 mod p → −1 must be a quadratic residue → p ≡ 1 mod 4.

Contrapositive: **p ≡ 3 mod 4  ⟹  α(p) is even**. (No prime ≡ 3 mod 4 can have an odd rank.)

## Why 193 has w_F = 1 (the question that sparked this cert)

Cert [417] observed that w_F(193) = δ(193) = F_{97}/193 mod 193 = 1. In ℤ[φ]/193²:

```
φ^{97} ≡ F_{96} + 193·φ   (mod 193²)
```

Since α(193) = 97 is odd and 193 ≡ 1 mod 4, C3 guarantees F_{96} is a sqrt(−1) mod 193:

```
F_{96}² ≡ −1 ≡ 192  (mod 193)
```

The coefficient of φ in the p²-expansion is δ(193) = 1 — the minimum non-zero lift depth.
So φ^{97} takes the form i + 193φ where i² ≡ −1, sitting at the minimum possible p-adic distance
from being a zero of the p²-rank-of-apparition.

This happens for Fibonacci primes (F_{α} = p exactly) and certain non-Fibonacci primes like
193 and 41. The deeper characterization of the "δ = 1 class" remains open.

## Explicit sqrt(−1) witnesses (C3)

The QA T-step, iterated α(p)−1 times, produces a square root of −1 mod p for all primes with odd α. No separate square-root algorithm needed.

| p | p mod 4 | α(p) | F_{α−1} mod p | (F_{α−1})² mod p |
|---|---|---|---|---|
| 5 | 1 | 5 | F₄=3 | 9≡4≡−1 ✓ |
| 13 | 1 | 7 | F₆=8 | 64≡12≡−1 ✓ |
| 17 | 1 | 9 | F₈=21≡4 | 16≡−1 ✓ |
| 37 | 1 | 19 | F₁₈ mod 37 | ≡−1 ✓ |
| 61 | 1 | 15 | F₁₄ mod 61 | ≡−1 ✓ |
| 97 | 1 | 49 | F₄₈ mod 97 | ≡−1 ✓ |
| 193 | 1 | 97 | F₉₆≡√(−1) | ≡192≡−1 ✓ |
| 421 | 1 | 21 | F₂₀ mod 421 | ≡−1 ✓ |

## Even-α primes: sqrt(+1) instead (C1 for even α)

For even α(p): F_{α(p)−1}² ≡ +1 mod p, so F_{α(p)−1} ≡ ±1 mod p. These primes can be ≡ 1 or 3 mod 4.

## Orbit parity readout (C4)

At step α(p), the QA T-orbit of (0,1) is (F_{α(p)}, F_{α(p)+1}) ≡ (0, F_{α(p)−1}) mod p.
Squaring the second component immediately tells you:
- Second component² ≡ p−1: α(p) is ODD → p ≡ 1 mod 4
- Second component² ≡ 1: α(p) is EVEN → p can be any class

The parity of α is *embedded in the orbit state* — no factorization needed.

## The p ≡ 3 mod 4 constraint (C2)

All 24 primes ≡ 3 mod 4 in the test set have even α:

| p | p mod 4 | α(p) | parity |
|---|---|---|---|
| 3 | 3 | 4 | even |
| 7 | 3 | 8 | even |
| 11 | 3 | 10 | even |
| 23 | 3 | 24 | even |
| 31 | 3 | 30 | even |
| 43 | 3 | 44 | even |
| 47 | 3 | 16 | even |
| 67 | 3 | 68 | even |
| 71 | 3 | 70 | even |
| 83 | 3 | 84 | even |
| ... | 3 | (even) | even |

Falsifiable: one prime p ≡ 3 mod 4 with odd α(p) would refute.

## Checks

- **C1**: F_{α−1}² ≡ (−1)^α mod p for all 45 primes — **PASS**
- **C2**: α(p) even for all 24 primes ≡ 3 mod 4 — **PASS**
- **C3**: F_{α−1} is explicit sqrt(−1) for all 13 odd-α primes — **PASS**
- **C4**: Orbit parity readout consistent for 5 sample primes — **PASS**

## Chain

- **Inherits from**: cert [416] (rank of apparition), cert [417] (Wall-Sun-Sun lift)
- **Connection**: Cassini identity (1680) + Euler's criterion (QR theory) meet QA T-orbit dynamics
- **Open**: Characterize the "δ(p) = 1" primes (Fibonacci primes ∪ {41, 193, ...})

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-derived the Cassini gate
identity F_{α−1}²≡(−1)^α mod p and the p≡3 mod 4 ⟹ α even consequence
in a fresh script (own rank-of-apparition and Fibonacci implementations)
for all 95 primes ≤500 — zero failures. Genuine falsifiable number
theory, no fixture-trusting gap.
