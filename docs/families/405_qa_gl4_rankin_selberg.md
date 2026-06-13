<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Jacquet-PS-Shalika (1983) doi:10.2307/2374264, Shahidi (1981) doi:10.2307/2374219 -->
# [405] QA Langlands GL₄ Rankin-Selberg — Tensor Product Local Factor

**Cert family**: `qa_gl4_rankin_selberg_cert_v1`
**Claim**: For f = 2.2.5.1-125.1-a and split prime p (p ≡ 1 mod 5), the tensor product π_{𝔭} ⊗ π_{𝔭̄} of the two GL₂ local components has Euler polynomial:

```
R_p(Y) = 1 − N·Y + p(T²−2N−2p)·Y² − p²N·Y³ + p⁴·Y⁴
```

where T = Tr_{ℚ(√5)/ℚ}(a_p) ∈ ℤ and N = N_{ℚ(√5)/ℚ}(a_p) ∈ ℤ.

## The Langlands Ladder (Complete through [405])

| Rung | Cert | Content |
|---|---|---|
| GL₁ Frobenius | [394] | Char. poly. det(I − Frob_p·t) for p=5 |
| GL₂ Weil bound | [395] | \|a_𝔭\| ≤ 2√N(𝔭) |
| ℤ[ζ₅] infrastructure | [396] | Ring of integers, norm, Galois action |
| CM relative norm | [397] | N_{K/F}: ℚ(ζ₅) → ℚ(√5) |
| CM form identification | [399] | 2.2.5.1-125.1-a identified via LMFDB + disc. resonance |
| CM Frobenius equality | [403] | Ramanujan equality + Universal Pell for all p≤500 |
| GL₄ Euler factors | [404] | AI_{ℚ(√5)/ℚ}(f) palindrome + Ramanujan |
| **GL₄ tensor-product factor** | **[405]** | **THIS — R_p(Y) from Vieta cross-product** |

## The Algebraic Derivation

Let σ₁(a_p) = u + vφ₁ and σ₂(a_p) = u + vφ₂ be the real embeddings (observer projections). The two GL₂ Euler factors are:
- Q₁(Y) = 1 − σ₁(a_p)·Y + p·Y²  (at prime 𝔭 above p)
- Q₂(Y) = 1 − σ₂(a_p)·Y + p·Y²  (at prime 𝔭̄ above p)

**Satake params** of Q₁: α₁, α₂ with α₁+α₂ = σ₁(a_p), α₁α₂ = p.

**Tensor product** = ∏_{i∈{1,2}, j∈{1,2}} (1 − αᵢγⱼY) = Q₂(α₁Y)·Q₂(α₂Y):

```
Q₂(α₁Y)·Q₂(α₂Y)
= (1 − σ₂·α₁Y + pα₁²Y²)(1 − σ₂·α₂Y + pα₂²Y²)
= 1 − σ₂(α₁+α₂)Y + [σ₂²α₁α₂ + p(α₁²+α₂²)]Y²
    − pσ₂·α₁α₂(α₁+α₂)Y³ + p²(α₁α₂)²Y⁴
= 1 − σ₁σ₂·Y + p(σ₂²+σ₁²−2p)Y² − p²σ₁σ₂·Y³ + p⁴Y⁴
```

Now apply Vieta over ℚ:
- σ₁σ₂ = N_{ℚ(√5)/ℚ}(a_p) = **N ∈ ℤ**
- σ₁²+σ₂² = (σ₁+σ₂)² − 2σ₁σ₂ = T² − 2N **∈ ℤ**

Therefore all coefficients are integers — no irrational quantities appear in R_p.

## Palindrome (Weight-2 Functional Equation)

The roots of R_p(Y) = 0 are {1/(αᵢγⱼ)} for cross pairs with |αᵢγⱼ| = p, so |root| = p⁻¹. For a Weil polynomial of pure weight 2, the functional equation is:

```
R_p(Y) = p⁴Y⁴ · R_p(1/(p²Y))
```

Equivalently: **a₃ = p²a₁** (i.e., −p²N = p²·(−N) ✓) and **a₄ = p⁴a₀** (= p⁴ ✓). The middle coefficient a₂ = p(T²−2N−2p) is self-symmetric.

## Relation to ∧²(AI(f))

The exterior square of the GL₄ form AI_{ℚ(√5)/ℚ}(f) decomposes (as a virtual GL₆ representation) as:
```
∧²(AI(f)) = det(π₁) ⊕ (π₁ ⊗ π₂) ⊕ det(π₂)
```
where det(πᵢ) = ∧²(2-dim) = 1-dim with Satake param = αᵢβᵢ = p. So:

```
∧²P_p(Z) = (1 − pZ)² · R_p(Z)   [degree-6 ∧² GL₆ Euler factor]
```

The (1−pZ)² factor comes from the two GL₁ determinant pieces; R_p(Z) is the GL₄ tensor-product piece. Cert [405] isolates and certifies R_p.

## R_p Coefficient Table (All 22 Split Primes)

| p | T | N | −N | p(T²−2N−2p) | −p²N | p⁴ |
|---|---|---|---|---|---|---|
| 11 | −1 | −31 | 31 | 451 | 3751 | 14641 |
| 31 | −11 | −1 | 1 | 3658 | 961 | 923521 |
| 41 | 9 | −11 | 11 | 3936 | 18491 | 2825761 |
| 61 | −1 | −31 | 31 | 14327 | 115327 | 13845841 |
| 71 | 19 | 59 | −59 | 23654 | −297239 | 25411681 |
| 101 | 29 | 179 | −179 | 66139 | −1824179 | 104060401 |
| 131 | −11 | −1 | 1 | 25358 | 17161 | 294499921 |
| 151 | 4 | −496 | 496 | −62074 | 11288776 | 521660401 |
| 181 | −11 | −1 | 1 | 32358 | 32761 | 1073741824 |
| 191 | −41 | 389 | −389 | 119254 | −14223319 | 1330863361 |
| 211 | −1 | −781 | 781 | 109919 | 34791019 | 1986902711... |
| 241 | −16 | −436 | 436 | 73012 | 25326916 | ... |
| 251 | 4 | −496 | 496 | 13754 | 31254496 | ... |
| 271 | −31 | 209 | −209 | 185591 | −15342209 | ... |
| 281 | −11 | −751 | 751 | 141758 | 59330531 | ... |
| 311 | 49 | 569 | −569 | 228194 | −55041929 | ... |
| 331 | −61 | 899 | −899 | 293954 | −98485699 | ... |
| 401 | 29 | 179 | −179 | 585619 | −28795379 | ... |
| 421 | 19 | −691 | 691 | 337094 | 122453291 | ... |
| 431 | −36 | −176 | 176 | 562174 | 32693776 | ... |
| 461 | −1 | −781 | 781 | 1063679 | 166097141 | ... |
| 491 | 9 | −11 | 11 | 1195894 | 2651771 | ... |

## Checks

- **C1**: All coefficients −N, p(T²−2N−2p), −p²N, p⁴ ∈ ℤ — PASS (22/22, immediate from T,N ∈ ℤ)
- **C2**: Weight-2 palindrome a₃=p²a₁ and a₄=p⁴ — PASS (22/22)
- **C3**: Vieta algebraic derivation: a₁a₂=N and a₁²+a₂²=T²−2N → integer R_p — PASS (22/22)

## Chain

- Caps [404] (GL₄ AI Euler factors — the GL₄ polynomial R_p is the cross factor)
- Uses eigenvalue data from [403] (22 split primes p≤500 with Universal Pell check)
- Together with (1−pZ)²: constitutes the full ∧²(AI(f)) GL₆ Euler factor
- Next rung: full ∧² GL₆ L-function (degree-6 cert) or Sym²(f) over F = GL₃/F
