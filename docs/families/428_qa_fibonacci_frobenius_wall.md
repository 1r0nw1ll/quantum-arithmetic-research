<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Lucas (1878) doi:10.2307/2369308, Legendre (1798) -->
# [428] QA Fibonacci Frobenius Congruences & Wall Divisibility

**Cert family**: `qa_fibonacci_frobenius_wall_cert_v1`
**Claim**: The Frobenius element Frob_p acting on the Fibonacci matrix's eigenvalues
yields four congruences: F_p ≡ (5/p) mod p; L_p ≡ 1 mod p; F_{p−(5/p)} ≡ 0 mod p;
and α(p) ∣ p − (5/p). These connect the GL₂ Frobenius (certs [424]–[427]) to
Wall's divisibility theorem (Wall 1960).

## Setup: Frobenius on Eigenvalues

The Fibonacci recurrence matrix M = [[1,1],[1,0]] has eigenvalues φ, ψ
(roots of x² − x − 1, with φ = (1+√5)/2 and φ+ψ=1, φψ=−1).

The Legendre symbol (5/p) determines which field contains these eigenvalues:

| Class | p mod 5 | (5/p) | Eigenvalues | Frob_p action |
|-------|---------|-------|-------------|----------------|
| Split | ±1 mod 5 | +1 | φ, ψ ∈ 𝔽_p | Frob_p fixes both |
| Inert | ±2 mod 5 | −1 | φ, ψ ∈ 𝔽_{p²} ∖ 𝔽_p | Frob_p swaps φ ↔ ψ |
| Ramified | p = 5 | 0 | φ = ψ in 𝔽_5 | — |

## C1: Fibonacci-Legendre Congruence

**Theorem**: F_p ≡ (5/p) mod p.

**Proof** (pure integer): F_p = (φ^p − ψ^p)/(φ − ψ).
- Split: By FLT, φ^p = φ and ψ^p = ψ in 𝔽_p. So F_p = (φ−ψ)/(φ−ψ) = 1 = (5/p).
- Inert: Frobenius swaps: φ^p = ψ and ψ^p = φ. So F_p = (ψ−φ)/(φ−ψ) = −1 = (5/p).
- Ramified: F_5 = 5 ≡ 0 = (5/5).

Verified: 166/166 primes in [5, 1000].

## C2: Lucas-Fermat Congruence

**Theorem**: L_p ≡ 1 mod p for all odd primes p.

**Proof**: L_p = φ^p + ψ^p.
- Split: φ^p = φ, ψ^p = ψ → φ+ψ = 1 (sum of roots of x²−x−1).
- Inert: φ^p = ψ, ψ^p = φ → ψ+φ = 1.

Both splitting types give L_p = 1 mod p. Note: C1 gives the **difference** φ^p−ψ^p;
C2 gives the **sum** φ^p+ψ^p. Together they pin down both φ^p and ψ^p individually mod p.

Verified: 167/167 primes in [3, 1000].

## C3: Wall Zero

**Theorem**: F_{p − (5/p)} ≡ 0 mod p.

**Proof**:
- Split: (5/p)=+1, so target index = p−1. Eigenvalues φ,ψ ∈ 𝔽_p; FLT gives
  φ^{p−1} = ψ^{p−1} = 1. So M^{p−1} = I mod p. The off-diagonal entry is F_{p−1} = 0.
- Inert: (5/p)=−1, so target = p+1. From the Frobenius swap: φ^{p+1} = φ·φ^p = φ·ψ = φψ = −1.
  Similarly ψ^{p+1} = −1. So F_{p+1} = (φ^{p+1}−ψ^{p+1})/(φ−ψ) = (−1−(−1))/(φ−ψ) = 0.
- Ramified: F_5 = 5 ≡ 0 mod 5.

Verified: 166/166 primes in [5, 1000].

## C4: Wall Divisibility

**Theorem**: α(p) ∣ p − (5/p).

**Proof**: Immediate from C3. The function k ↦ F_k mod p has zeros forming a subgroup
of ℤ (Wall 1960, Theorem 2). The smallest positive zero is α(p). Since p−(5/p) is a
zero by C3, it must be divisible by α(p).

Verified: 166/166 primes in [5, 1000].

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 | F_p ≡ (5/p) mod p; 166 primes in [5, 1000] | **PASS** |
| C2 | L_p ≡ 1 mod p; 167 primes in [3, 1000] | **PASS** |
| C3 | F_{p−(5/p)} ≡ 0 mod p; 166 primes in [5, 1000] | **PASS** |
| C4 | α(p) ∣ p−(5/p); 166 primes in [5, 1000] | **PASS** |

## Theorem NT Factorisation

```
QA layer (pure integer):
  fib_fast(n, m)    — fast doubling, returns F_n mod m
  lucas_fast(n, m)  — L_n = F_{n+1} + F_{n-1} mod m
  legendre5(p)      — pow(5, (p-1)//2, p); maps p-1 → -1
  rank_of_apparition(p) — linear walk, first k with F_k ≡ 0

Observer layer: none (no chi-squared, no floats in this cert)
```

## Langlands Ladder

| Cert | Rung |
|------|------|
| [424] | p splits in ℚ(√5) iff p ≡ ±1 mod 5 |
| [425] | Chebotarev: density(split) = density(inert) = 1/2 |
| [426] | M^{α(p)} = ε(p)·I₂; T(p) = α·ord(ε) |
| [427] | L_{α(p)} = 2ε(p); equal-thirds type distribution |
| **[428]** | **F_p = (5/p); L_p = 1; F_{p−(5/p)} = 0; α(p) ∣ p−(5/p)** |

Cert [428] connects the GL₂ Frobenius (C1, C2) to Wall's divisibility (C4) via
the Wall zero (C3). Together with [427], this means α(p) is sandwiched:
α(p) ∣ p−(5/p) from above, and α(p) ∣ T(p) = α(p)·ord(ε) from below
(the Pisano period is an exact multiple of α). The next natural rung ([429])
is lifting to ℤ/p²ℤ: does α(p²) equal p·α(p) (the generic case) or α(p)
(a Wall-Sun-Sun prime)?

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified all four
congruences (F_p≡(5/p), L_p≡1, Wall zero, α(p)|p−(5/p)) in a fresh
script for all primes in [3,1000] — zero failures. Genuine falsifiable
number theory, no fixture-trusting gap.
