<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Arthur-Clozel (1989) ISBN 978-0-691-08517-3, Langlands (1980) ISBN 978-0-691-08258-5 -->
# [404] QA Langlands GL₄ Induction — AI_{ℚ(√5)/ℚ}(f) Euler Factors

**Cert family**: `qa_gl4_induction_euler_cert_v1`
**Claim**: For f = 2.2.5.1-125.1-a (GL₂/ℚ(√5), CM by ℚ(ζ₅)), the GL₄/ℚ automorphic induction AI_{ℚ(√5)/ℚ}(f) has Euler polynomial at each split prime p (p ≡ 1 mod 5):

```
P_p(Y) = (1 − a_𝔭·Y + p·Y²)(1 − σ_F(a_𝔭)·Y + p·Y²)
       = 1 − T·Y + (N+2p)·Y² − p·T·Y³ + p²·Y⁴
```

where T = Tr_{ℚ(√5)/ℚ}(a_p) = 2u+v and N = N_{ℚ(√5)/ℚ}(a_p) = u²+uv−v².

## The Langlands Ladder (Complete through [404])

| Rung | Cert | Content |
|---|---|---|
| GL₁ Frobenius | [394] | Char. poly. det(I − Frob_p·t) for p=5 (ramified prime) |
| GL₂ Weil bound | [395] | \|a_𝔭\| ≤ 2√N(𝔭) for Hilbert modular forms |
| ℤ[ζ₅] infrastructure | [396] | Ring of integers, norm, Galois action |
| CM relative norm | [397] | N_{K/F}: ℚ(ζ₅) → ℚ(√5) |
| CM form identification | [399] | 2.2.5.1-125.1-a identified via LMFDB + disc. resonance |
| CM Frobenius equality | [403] | Ramanujan equality + Universal Pell for all p≤500 |
| **GL₄ Euler factors** | **[404]** | **THIS — AI_{ℚ(√5)/ℚ}(f) palindrome + Ramanujan** |

## Statement

**Automorphic induction**: By Arthur–Clozel (1989), for a cuspidal automorphic representation π of GL₂/F with F/ℚ quadratic, there exists a cuspidal automorphic representation AI_{F/ℚ}(π) of GL₄/ℚ. At split primes (p splits in F), the Euler factors of AI_{F/ℚ}(π) factor as a product over the primes 𝔭 | p.

**For our CM form**: Since F = ℚ(√5) and p ≡ 1 mod 5 splits as p = 𝔭·𝔭̄ in ℤ[φ], the GL₄ Euler factor at p is:

```
L_p(s, AI(f))⁻¹ = (1 − a_𝔭·p⁻ˢ + p·p⁻²ˢ)(1 − a_{𝔭̄}·p⁻ˢ + p·p⁻²ˢ)
```

where a_{𝔭̄} = σ_F(a_𝔭) (the Galois conjugate under σ_F: (u,v) ↦ (u+v,−v)).

Substituting Y = p⁻ˢ gives P_p(Y) as stated.

## The T ≡ −1 mod 5 Theorem (C1)

For every split prime p ≡ 1 mod 5 of 2.2.5.1-125.1-a, the rational trace T = Tr_{ℚ(√5)/ℚ}(a_p) satisfies:

```
T ≡ 4 ≡ −1 (mod 5)
```

**Proof**: The CM Frobenius π ∈ ℤ[ζ₅] satisfies π ≡ 1 mod λ³ where λ = (1−ζ₅) is the prime above 5 in K = ℚ(ζ₅). In particular π ≡ 1 mod λ. The four Galois conjugates σ^i(π) each satisfy σ^i(π) ≡ 1 mod σ^i(λ). Since all σ^i(λ) generate the unique prime above 5 in K, each σ^i(π) ≡ 1 mod (prime above 5). Therefore:

```
Tr_{K/ℚ}(π) = σ(π) + σ²(π) + σ³(π) + σ⁴(π) ≡ 4·1 = 4 ≡ −1 (mod 5)
```

And T = Tr_{ℚ(√5)/ℚ}(a_p) = Tr_{K/ℚ}(π) ≡ −1 mod 5. ∎

This is verified for all 22 split primes p ≤ 500.

## ℤ[φ] Factorization (C2)

The product over ℤ[φ][Y] expands term-by-term:

| Degree | Factor | Value |
|---|---|---|
| Y⁰ | 1 | 1 |
| Y¹ | −(a_p + σ_F(a_p)) | −T ∈ ℤ |
| Y² | a_p·σ_F(a_p) + 2p | N + 2p ∈ ℤ |
| Y³ | −p·(a_p + σ_F(a_p)) | −pT ∈ ℤ |
| Y⁴ | p² | p² ∈ ℤ |

All coefficients are rational integers because sum and product of a_p and σ_F(a_p) are both rational (T and N). Verified for all 22 primes.

## Palindrome / Functional Equation Purity (C3)

The polynomial P_p(Y) = 1 − TY + (N+2p)Y² − pTY³ + p²Y⁴ satisfies:

```
p²Y⁴ · P_p(1/(pY)) = p²Y⁴ · [1 − T/(pY) + (N+2p)/(pY)² − pT/(pY)³ + p²/(pY)⁴]
                    = p²Y⁴ − p·TY³ + (N+2p)Y² − TY + 1 = P_p(Y)
```

Equivalently: a₄ = p²·a₀ = p² and a₃ = p·a₁ = −pT. This is the Weil polynomial palindrome, reflecting the functional equation s ↦ 1−s for a pure-weight-1 L-function. Verified for all 22 primes.

## GL₄ Ramanujan (C4)

All 4 roots of P_p(Y) have |root| = p^{−1/2}.

**Proof**: The roots come from the two GL₂ factors:
- Factor 1: X² − σ₁(a_p)·X + p = 0 gives roots α, p/α with |α| = √p (from [403])
- Factor 2: X² − σ₂(a_p)·X + p = 0 gives roots β, p/β with |β| = √p (from [403])

In Y = 1/X coordinates: |Y| = 1/√p = p^{−1/2} for all 4 roots. The discriminant disc_i = σᵢ(a_p)² − 4p < 0 at both real embeddings σ₁, σ₂ (verified in [403] for all 22 primes), confirming complex roots with product p.

## GL₄ Euler Coefficients for All 22 Split Primes

| p | T | N | N+2p | Coefficients [1, −T, N+2p, −pT, p²] |
|---|---|---|---|---|
| 11 | −1 | −31 | −9 | [1, 1, −9, 11, 121] |
| 31 | −11 | −1 | 61 | [1, 11, 61, 341, 961] |
| 41 | 9 | −11 | 71 | [1, −9, 71, −369, 1681] |
| 61 | −1 | −31 | 91 | [1, 1, 91, 61, 3721] |
| 71 | 19 | 59 | 201 | [1, −19, 201, −1349, 5041] |
| 101 | 29 | 179 | 381 | [1, −29, 381, −2929, 10201] |
| 131 | −11 | −1 | 261 | [1, 11, 261, 1441, 17161] |
| 151 | 4 | −496 | −194 | [1, −4, −194, −604, 22801] |
| 181 | −11 | −1 | 361 | [1, 11, 361, 1991, 32761] |
| 191 | −41 | 389 | 771 | [1, 41, 771, 7831, 36481] |
| 211 | −1 | −781 | −359 | [1, 1, −359, 211, 44521] |
| 241 | −16 | −436 | 46 | [1, 16, 46, 3856, 58081] |
| 251 | 4 | −496 | 6 | [1, −4, 6, −1004, 63001] |
| 271 | −31 | 209 | 751 | [1, 31, 751, 8401, 73441] |
| 281 | −11 | −751 | −189 | [1, 11, −189, 3091, 78961] |
| 311 | 49 | 569 | 1191 | [1, −49, 1191, −15239, 96721] |
| 331 | −61 | 899 | 1561 | [1, 61, 1561, 20191, 109561] |
| 401 | 29 | 179 | 981 | [1, −29, 981, −11629, 160801] |
| 421 | 19 | −691 | 151 | [1, −19, 151, −7999, 177241] |
| 431 | −36 | −176 | 686 | [1, 36, 686, 15516, 185761] |
| 461 | −1 | −781 | 141 | [1, 1, 141, 461, 212521] |
| 491 | 9 | −11 | 971 | [1, −9, 971, −4419, 241081] |

## Checks

- **C1**: T ≡ −1 ≡ 4 mod 5 for all 22 split primes p ≤ 500 — PASS (22/22)
- **C2**: ℤ[φ]-factorization gives integer GL₄ coefficients [1,−T,N+2p,−pT,p²] — PASS (22/22)
- **C3**: Palindrome a₄ = p²·a₀ and a₃ = p·a₁ for all 22 primes — PASS (22/22)
- **C4**: GL₄ Ramanujan |root| = p^{−1/2} (disc₁ < 0 and disc₂ < 0) for all 22 primes — PASS (22/22)

## Chain

- Caps [403] (CM Frobenius Ramanujan + Universal Pell — the direct GL₂ predecessor)
- Uses eigenvalue table from [403] C6 (all 22 corrected split primes p ≤ 500)
- Connected to [396] (ℤ[ζ₅] infrastructure — T ≡ −1 mod 5 theorem uses conductor λ³)
- Connected to [398] (Five Families — ℤ[φ] orbits underlie the automorphic induction)
- Next rung: GL₄ × GL₄ Rankin–Selberg (if needed) or Sym² / ∧² transfers
