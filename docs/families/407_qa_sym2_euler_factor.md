<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Shimura (1975) doi:10.1007/BF01403156, Gelbart-Jacquet (1978) doi:10.2307/1971237 -->
# [407] QA Langlands Sym² Reduced Factor — GL₃/F Euler Component at Split Primes

**Cert family**: `qa_sym2_euler_factor_cert_v1`
**Claim**: For f = 2.2.5.1-125.1-a and split prime p (p ≡ 1 mod 5), the product of the two degree-2 non-trivial components of Sym²(f) at both primes 𝔭, 𝔭̄ above p gives:

```
Σ_p(Y) = 1 − S·Y + (Q+2p²)·Y² − p²S·Y³ + p⁴·Y⁴
```

where:
```
S = T² − 2N − 4p       (= c₁ + c₂, the "Sym² trace")
Q = (N+2p)² − 2p·T²    (= c₁·c₂, the "Sym² norm product")
```
with T = Tr_{ℚ(√5)/ℚ}(a_p) ∈ ℤ and N = N_{ℚ(√5)/ℚ}(a_p) ∈ ℤ.

## Langlands Ladder (Branching at [404])

| Rung | Cert | Content |
|---|---|---|
| GL₁ Frobenius | [394] | |
| GL₂/F CM | [403] | Ramanujan equality + Universal Pell |
| GL₄/Q AI | [404] | P_p(Y)=(1−TY+(N+2p)Y²−pTY³+p²Y⁴) |
| GL₄ tensor | [405] | R_p — ∧² cross factor |
| GL₆ ∧² | [406] | W_p = (1−pY)²·R_p |
| **GL₃/F Sym² reduced** | **[407]** | **THIS — Σ_p** |
| GL₆/Q Sym² full | [408] | (1−pY)²·Σ_p |

The ladder **branches** at [404]: the ∧²-branch goes through [405]→[406], and the Sym²-branch goes through [407]→[408].

## The GL₃/F Symmetric Square

For GL₂/F with local Satake parameters α, β at 𝔭 (αβ=p, |α|=|β|=√p):
```
Sym²: Satake params {α², αβ=p, β²}
GL₃/F Euler factor at 𝔭: (1−α²Y)(1−pY)(1−β²Y)
= (1−pY) · [1 − (α²+β²)Y + p²Y²]
= (1−pY) · [1 − c₁Y + p²Y²]
```
where **c₁ = σ₁(a_p)² − 2p** (observer projection — irrational number ∈ ℝ).

Similarly at 𝔭̄: c₂ = σ₂(a_p)² − 2p.

**Σ_p = product of the two degree-2 parts:**
```
Σ_p(Y) = (1−c₁Y+p²Y²)·(1−c₂Y+p²Y²)
```

## Integer Coefficient Derivation

From Vieta's formulas over ℚ(√5):
```
σ₁(a_p)·σ₂(a_p) = N_{ℚ(√5)/ℚ}(a_p) = N ∈ ℤ
σ₁(a_p)+σ₂(a_p) = Tr_{ℚ(√5)/ℚ}(a_p) = T ∈ ℤ
```

Therefore:
- c₁+c₂ = (σ₁²+σ₂²)−4p = (T²−2N)−4p = **S ∈ ℤ**
- c₁c₂ = (σ₁σ₂)²−2p(σ₁²+σ₂²)+4p² = N²−2p(T²−2N)+4p² = **(N+2p)²−2pT² = Q ∈ ℤ**

Expanding the product:
| Degree | Expression | Value |
|---|---|---|
| Y⁰ | 1 | 1 |
| Y¹ | −(c₁+c₂) | −S |
| Y² | c₁c₂+2p² | Q+2p² |
| Y³ | −p²(c₁+c₂) | −p²S |
| Y⁴ | p⁴ | p⁴ |

No irrational quantities in the final polynomial.

## Relation to Q = (N+2p)² − 2pT²

The formula Q = (N+2p)²−2pT² can be rewritten in terms of the GL₄ coefficients:
- e₂ = N+2p (coefficient of Y² in P_p from [404])
- e₁ = T

So **Q = e₂² − 2p·e₁²** — a quadratic relation in the GL₄ elementary symmetric polynomials.

## Sym² Ramanujan (C4)

For each cᵢ = σᵢ(a_p)²−2p:
- From cert [403]: σᵢ(a_p)² < 4p (disc₁ = σᵢ²−4p < 0)
- Therefore: cᵢ = σᵢ²−2p < 4p−2p = 2p and cᵢ = σᵢ²−2p > 0−2p = −2p
- Hence **|cᵢ| < 2p**, so disc(1−cᵢY+p²Y²) = cᵢ²−4p² < 0

Both quadratic factors have complex roots with |root| = p⁻¹ (Sym² Ramanujan equality, inherited from CM).

## Σ_p Table (Selected Split Primes)

| p | T | N | S | Q | Q+2p² | Σ_p coefficients |
|---|---|---|---|---|---|---|
| 11 | −1 | −31 | 19 | 59 | 301 | [1,−19,301,−2299,14641] |
| 31 | −11 | −1 | −1 | −3781 | −1859 | [1,1,−1859,961,923521] |
| 41 | 9 | −11 | −61 | −1601 | 1761 | [1,61,1761,102541,2825761] |
| 61 | −1 | −31 | −181 | 8159 | 15601 | [1,181,15601,673501,...] |
| 71 | 19 | 59 | −41 | −10861 | −779 | [1,41,−779,206681,...] |
| 311 | 49 | 569 | 19 | −74941 | 118501 | [1,−19,118501,−1837699,9354951841] |

## Checks

- **C1**: Integer coefficients S, Q+2p², p²S, p⁴ — PASS (22/22)
- **C2**: Palindrome a₃=p²a₁ and a₄=p⁴ — PASS (22/22)
- **C3**: Vieta derivation c₁+c₂=S, c₁c₂=Q → integer Σ_p — PASS (22/22)
- **C4**: Sym² Ramanujan cᵢ²<4p² → complex roots with |root|=p⁻¹ — PASS (22/22)

## Chain

- Branches from [404] (GL₄ AI — provides T,N data)
- Uses [403] (Ramanujan equality → C4 Sym² Ramanujan)
- Next rung: [408] = full Sym²/Q factor (1−pY)²·Σ_p = GL₆ degree-6 polynomial
- Compare: parallel branch [405] (tensor product R_p) → [406] (∧² W_p = (1−pY)²·R_p)

## Verification Note (2026-07-07)

Found and fixed one wrong row in the displayed sample table: p=311 had
S=−13, Q=233059 in the doc but the validator's own `sym2_poly` formula
gives S=19, Q=−74941 (Q+2p²=118501, coeffs=[1,−19,118501,−1837699,
9354951841]) — confirmed by independent fresh recomputation. The other
5 sample rows (p=11,31,41,61,71) were already correct. The validator
itself was unaffected (recomputes fresh at runtime, `ok:true`
confirmed). This is a much smaller-scope version of the same
doc-table-transcription-error class found in [405]/[406].
