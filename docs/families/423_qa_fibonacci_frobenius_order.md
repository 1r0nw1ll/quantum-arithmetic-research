<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Lagrange (1771), Hecke (1920) MZ 6 -->
# [423] QA Fibonacci Frobenius Order Identity

**Cert family**: `qa_fibonacci_frobenius_order_cert_v1`
**Claim**: For split primes p ((5/p)=+1), the rank of apparition α(p) equals the
multiplicative order of the Frobenius eigenvalue ratio φ̃/ψ̃ in (ℤ/p)×:

    α(p)  =  ord_{(ℤ/p)×}(φ̃/ψ̃)

This is the GL₁ Frobenius order statement connecting QA T-step dynamics to
multiplicative group structure in 𝔽_p.

## Algebraic Setup

For a split prime p, φ̃ and ψ̃ are the two roots of x²−x−1=0 in 𝔽_p (cert [421]).
Key identities (all mod p):
```
φ̃ + ψ̃ = 1          (Vieta: sum of roots = 1)
φ̃ · ψ̃ = -1         (Vieta: product of roots = -1)
ψ̃ = -φ̃⁻¹          (since φ̃·ψ̃ = -1)
φ̃/ψ̃ = -φ̃²        (since ψ̃⁻¹ = -φ̃, so φ̃/ψ̃ = -φ̃² = -(φ̃+1))
```

## The Frobenius Order Identity

**Proof:**  
F_n ≡ 0 (mod p)  
&emsp;iff (by Binet mod p, cert [421]) φ̃ⁿ ≡ ψ̃ⁿ (mod p)  
&emsp;iff (φ̃/ψ̃)ⁿ ≡ 1 (mod p)  
&emsp;iff ρⁿ ≡ 1 (mod p) where ρ = φ̃/ψ̃

Therefore α(p) = min{n≥1: ρⁿ=1} = ord_{(ℤ/p)×}(ρ).

## Langlands Interpretation

In the GL₁/ℚ(√5) picture:
- The Hecke eigenvalue at 𝔭|p (the prime above p in ℤ[φ]) is φ̃ ∈ 𝔽_p×
- The Frobenius Frob_𝔭 is the **identity** (p splits, so Gal acts trivially)
- The ratio ρ = φ̃/ψ̃ = −φ̃² measures how the two Hecke eigenvalues differ
- Its multiplicative order in GL₁(𝔽_p) is α(p)

This is the GL₁ rung of the ℚ(√5) Langlands ladder: QA T-step dynamics (Fibonacci recurrence mod p) is exactly the Frobenius orbit in GL₁(𝔽_p).

## Primitive Primes and Artin's Conjecture

A split prime p is **primitive** if α(p) = p−1 (i.e. ρ = φ̃/ψ̃ is a primitive root mod p).

Among 609 split primes ≤ 10,000: **216 are primitive** (fraction 0.355).

Artin's conjecture predicts this fraction → A ≈ 0.3739 (Artin constant) as N → ∞.
The empirical 35.5% vs theoretical 37.4% is consistent with the slow logarithmic convergence.

## Checks (split primes ≤ 500, n=45)

| Check | Content | Status |
|-------|---------|--------|
| C1 | φ̃·ψ̃ ≡ −1 (mod p) for 45/45 split primes | **PASS** |
| C2 | φ̃/ψ̃ ≡ −φ̃² (mod p) for 45/45 split primes | **PASS** |
| C3 | ord_{(ℤ/p)×}(φ̃/ψ̃) = α(p) for 45/45 split primes | **PASS** |
| C4 | Primitive fraction = 0.355 ∈ (0.25, 0.50); Artin ≈ 0.374 | **PASS** |

## Chain

| Cert | Claim |
|------|-------|
| [416] | α(p) \| p−(5/p) |
| [417] | δ(p) = F_α/p mod p defined |
| [418] | α(p) parity via Cassini |
| [419] | δ=1 census |
| [420] | δ=0 iff WSS |
| [421] | δ(p) = φ-slope in ℤ/p²ℤ |
| [422] | δ(p)/p equidistributed (Hecke) |
| **[423]** | **α(p) = ord_{GL₁(𝔽_p)}(φ̃/ψ̃) for split primes** |
| [424] | Inert Frobenius: Frob_p swaps φ̃↔ψ̃ in 𝔽_{p²}; α(p)\|p+1 |

**Open**: Artin's conjecture — the precise density A of primitive primes (for φ̃/ψ̃ as a primitive root generator) requires GRH.
