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
| 31 | −11 | −1 | 1 | 1891 | 961 | 923521 |
| 41 | 9 | −11 | 11 | 861 | 18491 | 2825761 |
| 61 | −1 | −31 | 31 | −3599 | 115351 | 13845841 |
| 71 | 19 | 59 | −59 | 7171 | −297419 | 25411681 |
| 101 | 29 | 179 | −179 | 28381 | −1825979 | 104060401 |
| 131 | −11 | −1 | 1 | −18209 | 17161 | 294499921 |
| 151 | 4 | −496 | 496 | 106606 | 11309296 | 519885601 |
| 181 | −11 | −1 | 1 | −43259 | 32761 | 1073283121 |
| 191 | −41 | 389 | −389 | 99511 | −14191109 | 1330863361 |
| 211 | −1 | −781 | 781 | 240751 | 34770901 | 1982119441 |
| 241 | −16 | −436 | 436 | 155686 | 25323316 | 3373402561 |
| 251 | 4 | −496 | 496 | 127006 | 31248496 | 3969126001 |
| 271 | −31 | 209 | −209 | 271 | −15349169 | 5393580481 |
| 281 | −11 | −751 | 751 | 298141 | 59299711 | 6234839521 |
| 311 | 49 | 569 | −569 | 199351 | −55034249 | 9354951841 |
| 331 | −61 | 899 | −899 | 417391 | −98495339 | 12003612721 |
| 401 | 29 | 179 | −179 | −127919 | −28783379 | 25856961601 |
| 421 | 19 | −691 | 691 | 379321 | 122473531 | 31414372081 |
| 431 | −36 | −176 | 176 | 338766 | 32693936 | 34507149121 |
| 461 | −1 | −781 | 781 | 295501 | 165978901 | 45165175441 |
| 491 | 9 | −11 | 11 | −431589 | 2651891 | 58120048561 |

## Checks

- **C1**: All coefficients −N, p(T²−2N−2p), −p²N, p⁴ ∈ ℤ — PASS (22/22, immediate from T,N ∈ ℤ)
- **C2**: Weight-2 palindrome a₃=p²a₁ and a₄=p⁴ — PASS (22/22)
- **C3**: Vieta algebraic derivation: a₁a₂=N and a₁²+a₂²=T²−2N → integer R_p — PASS (22/22)

## Chain

- Caps [404] (GL₄ AI Euler factors — the GL₄ polynomial R_p is the cross factor)
- Uses eigenvalue data from [403] (22 split primes p≤500 with Universal Pell check)
- Together with (1−pZ)²: constitutes the full ∧²(AI(f)) GL₆ Euler factor
- Next rung: full ∧² GL₆ L-function (degree-6 cert) or Sym²(f) over F = GL₃/F

## Verification Note (2026-07-07)

**Found and fixed a real documentation-table bug — 21 of 22 rows in the
displayed coefficient table above were wrong**, though the validator
itself was never affected. The `(u,v)` eigenvalue data duplicated from
cert [403] is byte-identical and correct (no repeat of the [390]/[395]
indexing bug). But independently recomputing R_p(Y)'s coefficients
directly from the validator's own formula (`rankin_selberg_poly`, which
the validator applies correctly at runtime) revealed the *displayed*
markdown table had wrong values in the "p(T²−2N−2p)" column for 21/22
rows (only p=11 was correct) and wrong values in the "−p²N" column for
most rows past p=41, plus two outright wrong p⁴ values (p=151, and
p=181 — the latter showing 1073741824 = 2³⁰, an apparent copy-paste
artifact rather than 181⁴=1073283121). The automated C1-C3 checks all
still passed because they compute R_p(Y) from `(T, N)` fresh at
validator runtime rather than reading the markdown table — the doc
table was purely decorative and never fed into the actual pass/fail
determination. Regenerated the entire 22-row table from the validator's
own formula and replaced it above. This is a genuine, if
consequence-free, finding: a human reader trusting this doc's numbers
before this fix would have gotten the wrong coefficients for nearly
every prime beyond the first.
