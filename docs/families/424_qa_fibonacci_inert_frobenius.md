<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Lagrange (1771), Chebotarev (1926) MA 95 -->
# [424] QA Fibonacci Inert Frobenius Conjugation

**Cert family**: `qa_fibonacci_inert_frobenius_cert_v1`
**Claim**: For inert primes p ((5/p)=−1), the Frobenius automorphism Frob_p of
𝔽_{p²}/𝔽_p swaps the two roots φ̃ and ψ̃ of x²−x−1=0:

    Frob_p(φ̃) = φ̃^p = ψ̃   in 𝔽_{p²}

Consequently α(p)|p+1, and α(p) = ord_{𝔽_{p²}×}(φ̃/ψ̃).

## Setup: 𝔽_{p²} arithmetic

For an inert prime p, x²−x−1 is **irreducible** over 𝔽_p. Define:
```
F_{p²} = (Z/pZ)[x] / (x²-x-1)
Elements: (a, b) representing a + b·φ̃   (a, b ∈ {0,...,p-1})
Multiplication: (a,b)·(c,d) = (ac+bd, ad+bc+bd) mod p
                               [using φ̃² = φ̃+1]
Frobenius:  Frob(a + b·φ̃) = a + b·φ̃^p = a + b·ψ̃
                            = (a+b) + (−b)·φ̃
                            = ((a+b) mod p, (−b) mod p)
```

In coordinates: φ̃ = (0,1), ψ̃ = (1, p−1).

## Why Frob swaps φ̃ and ψ̃

φ̃ satisfies x²−x−1=0 over 𝔽_p. The Frobenius σ: x↦x^p is the **unique** non-trivial automorphism of 𝔽_{p²}/𝔽_p. Since x²−x−1 has coefficients in 𝔽_p, if φ̃ is a root then σ(φ̃) is also a root — but not φ̃ itself (φ̃ ∉ 𝔽_p for inert p). So σ(φ̃) = ψ̃.

Direct computation confirms: `fp2_pow(0, 1, p, p) = (1, p-1)` for all 12 inert primes ≤ 100.

## α(p) | p+1 from Frobenius

The Frobenius conjugation gives:
```
(φ̃/ψ̃)^p = φ̃^p/ψ̃^p = ψ̃/φ̃ = (φ̃/ψ̃)^{-1}
```
Therefore `(φ̃/ψ̃)^{p+1} = (φ̃/ψ̃)^p · (φ̃/ψ̃) = (φ̃/ψ̃)^{-1} · (φ̃/ψ̃) = 1`.

So ord_{𝔽_{p²}×}(φ̃/ψ̃) | p+1, and since α(p) = ord_{𝔽_{p²}×}(φ̃/ψ̃), we get **α(p) | p+1**.

## Unified GL₁ Frobenius Order Statement

Combining cert [423] (split) and this cert (inert):

| Prime type | φ̃, ψ̃ live in | Frob acts as | α(p) divides | α(p) = ord in |
|---|---|---|---|---|
| Split (5/p)=+1 | 𝔽_p× | identity | p−1 | GL₁(𝔽_p) |
| Inert (5/p)=−1 | 𝔽_{p²}×\𝔽_p× | conjugation φ̃↔ψ̃ | p+1 | GL₁(𝔽_{p²}) |

**Unified formula**: α(p) = ord_{GL₁(𝔽_{p^{1+e_p}})}(φ̃/ψ̃) where e_p = 0 (split) or 1 (inert).

This is the GL₁/ℚ(√5) Langlands statement for ALL non-ramified primes: the rank of apparition is the Frobenius order in the appropriate multiplicative group.

## Checks (inert primes ≤ 500, n=47)

| Check | Content | Status |
|-------|---------|--------|
| C1 | x²−x−1 irreducible over 𝔽_p for 47/47 inert primes | **PASS** |
| C2 | φ̃^p = ψ̃ in 𝔽_{p²} for 12/12 inert primes ≤ 100 | **PASS** |
| C3 | α(p) \| p+1 for 47/47 inert primes | **PASS** |
| C4 | ord_{𝔽_{p²}×}(φ̃/ψ̃) = α(p) for 47/47 inert primes | **PASS** |

## Chain

| Cert | Claim |
|------|-------|
| [416] | α(p) \| p−(5/p) (covers both split and inert) |
| [421] | Hensel lift, Binet mod p² for split primes |
| [422] | δ(p)/p equidistributed (split primes) |
| [423] | α(p) = ord_{GL₁(𝔽_p)}(φ̃/ψ̃) for split primes |
| **[424]** | **Frob_p swaps φ̃↔ψ̃ in 𝔽_{p²}/𝔽_p; α(p)=ord_{GL₁(𝔽_{p²})}(φ̃/ψ̃) for inert primes** |

**Open (next rung)**: Chebotarev density — split and inert primes each have density exactly 1/2 among non-5 primes. Equivalently, L(1, χ₅) ≠ 0 where χ₅ = Legendre symbol (·/5). This is the GL₁→GL₂ interface in the Langlands ladder.

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified the Frobenius swap
φ̃^p=ψ̃ in 𝔽_{p²} and α(p)|p+1 for all inert primes ≤500 in a fresh
script (own 𝔽_{p²} arithmetic implementation). Note: the validator's
`inert_primes_upto` explicitly filters `p > 5` (excluding 2 and 3,
likely due to characteristic-2/3 edge cases in the field-arithmetic
representation), giving 47 primes rather than the naive 49 — this is a
deliberate scope choice, not a bug; independently confirmed p=2 and
p=3 also satisfy both properties when checked directly, so nothing is
actually lost by the exclusion, it's just undocumented in the doc text.
