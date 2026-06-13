<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Hecke (1920) doi:10.1007/BF01458074, Shimura (1971) ISBN 978-0-691-08092-5 -->
# [403] QA Langlands Cap — CM Frobenius Ramanujan Equality

**Cert family**: `qa_langlands_cap_frobenius_cert_v1`
**Claim**: For the CM Hilbert modular form f = 2.2.5.1-125.1-a over F = ℚ(√5), the Hecke character factorization L(s,f) = L(s,ψ)·L(s,ψ̄) implies **Ramanujan EQUALITY** at every split prime: the Frobenius discriminant Δ = a𝔭² − 4N(𝔭) is strictly negative at both real embeddings of ℚ(√5)/ℚ, and a𝔭 = 0 for all inert primes.

## The Langlands Ladder (Complete)

| Rung | Cert | Content |
|---|---|---|
| GL₁ Frobenius | [394] | Char. poly. det(I − Frob_p·t) for p=5 (ramified prime) |
| GL₂ Weil bound | [395] | \|a_𝔭\| ≤ 2√N(𝔭) for Hilbert modular forms |
| ℤ[ζ₅] infrastructure | [396] | Ring of integers, norm, Galois action |
| CM relative norm | [397] | N_{K/F}: ℚ(ζ₅) → ℚ(√5) |
| CM form identification | [399] | 2.2.5.1-125.1-a identified via LMFDB + disc. resonance |
| **CM Frobenius equality** | **[403]** | **Ramanujan equality: \|α\| = \|ᾱ\| = √p exactly** |

## Statement

For f = 2.2.5.1-125.1-a (CM by K = ℚ(ζ₅) over F = ℚ(√5)):

**CM zero pattern**: a_𝔭 = 0 whenever p ≢ 1 (mod 5) (inert or ramified primes).

**Frobenius char. poly.**: At split primes 𝔭 (p ≡ 1 mod 5), the eigenvalues α, ᾱ satisfy:
```
X² − a_𝔭·X + N(𝔭) = 0,   α·ᾱ = N(𝔭) = p,   |α| = |ᾱ| = √p
```

**Ramanujan equality** (not merely bound): Δ = a_𝔭² − 4p < 0 at both real embeddings σ₁, σ₂ of ℚ(√5)/ℚ. The roots are genuinely complex conjugates with modulus exactly √p.

This is equality (not bound) because f is CM: the Hecke character ψ: K^× → ℂ^× satisfies ψ·ψ̄ = N_{K/F}, so ψ(𝔓)·ψ̄(𝔓) = N_{K/F}(𝔓) = p exactly.

## Discriminant Resonance

The eigenvalue e = −3 + 5φ (φ = (1+√5)/2) satisfies:

```
e² + e − 31 = 0
disc(e/ℚ) = Tr(e)² − 4·N(e) = (−1)² − 4·(−31) = 1 + 124 = 125
```

**125 = 5³ = level norm = disc(ℚ(ζ₅)/ℚ)**

This threefold coincidence (level, eigenvalue discriminant, cyclotomic discriminant) is the algebraic fingerprint of the CM factorization.

## Frobenius Discriminants at Split Primes

Eigenvalues derived from explicit ℤ[ζ₅] Frobenius search: N_K/Q(π)=p², |σk(π)|=√p, v_{(ζ₅−1)}(π−1)≥3.

| p | π ∈ ℤ[ζ₅] | a_𝔭 = Tr_{K/F}(π) | Δ = a_𝔭² − 4p | σ₁(Δ) | σ₂(Δ) | N(a_𝔭) |
|---|---|---|---|---|---|---|
| 11 | −1+ζ₅−ζ₅²−3ζ₅³ | −3 + 5φ | −10 − 5φ | −18.09 | −6.91 | −31 |
| 31 | −6−4ζ₅−6ζ₅²−3ζ₅³ | −8 + 5φ | −35 − 55φ | −124.0 | −1.01 | −1 |
| 41 | 2−3ζ₅−2ζ₅²+4ζ₅³ | 7 − 5φ | −90 − 45φ | −162.8 | −62.2 | −11 |
| 61 | −2ζ₅−3ζ₅²+6ζ₅³ | 2 − 5φ | −215 + 5φ | −206.9 | −218.1 | −31 |
| 71 | 3−ζ₅+ζ₅²−7ζ₅³ | 7 + 5φ | −210 + 95φ | −56.3 | −268.7 | 59 |

All discriminants negative at both embeddings — Ramanujan equality confirmed.

**Algebraic orbit structure**: The CM involution c: ζ₅↦ζ₅⁴ maps each π to its conjugate c(π),
giving the eigenvalue at the complementary prime 𝔭₂: {−3+5φ, 2−5φ} are Galois conjugates
(σ_F swaps them), as are {7−5φ, 2+5φ}. Notably σ_F(a₁₁) = σ_F(−3+5φ) = 2−5φ = a₆₁:
the generator eigenvalue's Galois conjugate reappears as the eigenvalue at p=61.

## Checks

- **C1**: a_𝔭 = 0 for all p ≢ 1 (mod 5) — pure CM zero pattern — PASS
- **C2**: a_𝔭 ∈ ℤ[φ] (both coordinates integer) for all 5 split primes — PASS
- **C3**: Δ = a_𝔭² − 4p < 0 at both real embeddings σ₁, σ₂ — Ramanujan equality — PASS
- **C4**: N_{ℚ(√5)/ℚ}(a_𝔭) = {−31,−1,−11,−31,59} for p={11,31,41,61,71}; |N| ≤ 4p — PASS
- **C5**: e² + e − 31 = 0; disc(e/ℚ) = 125 = level norm = disc(ℚ(ζ₅)/ℚ) — PASS

## Chain

- Caps [394] → [395] → [396] → [397] → [399] (the Langlands ladder over ℚ(√5))
- Connected to [398] (Five Families partition — Cosmos/Satellite orbits live in ℤ[φ])
- Connected to [281] (Pisano periods — ℤ[ζ₅] and mod-5 cyclotomic structure)
