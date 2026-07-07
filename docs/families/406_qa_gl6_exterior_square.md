<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Kim-Shahidi (2002) doi:10.4007/annals.2002.155.837, Cogdell (2004) ISBN 978-0-8218-3516-0 -->
# [406] QA Langlands GL₆ Exterior Square — ∧²(AI_{ℚ(√5)/ℚ}(f)) Euler Factor

**Cert family**: `qa_gl6_exterior_square_cert_v1`
**Claim**: For f = 2.2.5.1-125.1-a and split prime p (p ≡ 1 mod 5), the full exterior square GL₆ Euler polynomial is:

```
W_p(Y) = 1 − (N+2p)Y + p(T²−p)Y²
        − 2p²(T²−N−2p)Y³
        + p³(T²−p)Y⁴ − p⁴(N+2p)Y⁵ + p⁶Y⁶
```

where T = Tr_{ℚ(√5)/ℚ}(a_p) ∈ ℤ and N = N_{ℚ(√5)/ℚ}(a_p) ∈ ℤ.

## The Langlands Ladder (Complete through [406])

| Rung | Cert | Content |
|---|---|---|
| GL₁ Frobenius | [394] | Char. poly. det(I − Frob_p·t) |
| GL₂ Weil bound | [395] | \|a_𝔭\| ≤ 2√N(𝔭) |
| ℤ[ζ₅] infrastructure | [396] | Ring of integers, norm |
| CM relative norm | [397] | N_{K/F}: ℚ(ζ₅) → ℚ(√5) |
| CM form identification | [399] | LMFDB + discriminant resonance |
| CM Frobenius equality | [403] | Ramanujan equality + Universal Pell |
| GL₄ Euler factors | [404] | AI induction palindrome + Ramanujan |
| GL₄ tensor-product factor | [405] | R_p = 1−NY+p(T²−2N−2p)Y²−p²NY³+p⁴Y⁴ |
| **GL₆ exterior square** | **[406]** | **THIS — W_p = (1−pY)²·R_p** |

## The ∧² Decomposition

For the GL₄ form AI_{ℚ(√5)/ℚ}(f) decomposed as V = V₁ ⊕ V₂ (two GL₂ pieces at p):
```
∧²V = ∧²V₁ ⊕ (V₁⊗V₂) ⊕ ∧²V₂
```

- **∧²V₁ = det(π₁)**: 1-dim with Satake param α₁α₂ = p → L-factor (1−pY)
- **∧²V₂ = det(π₂)**: 1-dim with Satake param γ₁γ₂ = p → L-factor (1−pY)
- **V₁⊗V₂ = π₁⊗π₂**: GL₄ tensor product → L-factor R_p(Y) (cert [405])

Therefore: **W_p(Y) = (1−pY)² · R_p(Y)**

## Coefficient Derivation

Expanding (1 − 2pY + p²Y²) · R_p(Y) with R_p = [1, −N, p(T²−2N−2p), −p²N, p⁴]:

| Degree | Expression | Simplified |
|---|---|---|
| Y⁰ | 1 | 1 |
| Y¹ | −N − 2p | −(N+2p) |
| Y² | p(T²−2N−2p) + 2pN + p² | p(T²−p) |
| Y³ | −p²N − 2p·p(T²−2N−2p) − p²N | −2p²(T²−N−2p) |
| Y⁴ | p⁴ + 2p³N + p³(T²−2N−2p) | p³(T²−p) |
| Y⁵ | −2p⁵ − p⁴N | −p⁴(N+2p) |
| Y⁶ | p⁶ | p⁶ |

The simplifications use only T,N ∈ ℤ and basic algebra. No irrational quantities appear.

## Coefficient Patterns

The GL₆ polynomial has two remarkable symmetries:

**Pattern 1 — inner palindrome**: coefficients of Y¹ and Y⁵ are both ±(N+2p); coefficients of Y² and Y⁴ are both ±p(T²−p). This is the weight-2 palindrome.

**Pattern 2 — shared with GL₄**: the coefficient of Y¹ in W_p is −(N+2p), which equals **−(coefficient of Y² in P_p)** from cert [404]. The GL₆ exterior square "sees" the GL₄ quadratic coefficient at degree 1.

## Palindrome (Weight-2 Functional Equation)

All 6 roots of W_p have magnitude p⁻¹ (2 from each det piece, 4 from R_p). Weil functional equation: W_p(Y) = p⁶Y⁶·W_p(1/(p²Y)), equivalently:
```
a₅ = p⁴·a₁   ↔   −p⁴(N+2p) = p⁴·(−(N+2p)) ✓
a₄ = p²·a₂   ↔   p³(T²−p) = p²·p(T²−p) ✓
a₆ = p⁶·a₀   ↔   p⁶ = p⁶·1 ✓
a₃ = a₃      ↔   self-symmetric (no condition)
```

## Middle Coefficient

The self-symmetric middle term a₃ = −2p²(T²−N−2p) is divisible by 2p² for every prime p. The factor 2 is a signature of the ∧² decomposition: it arises from the cross-coupling of det(π₁) with det(π₂) in the expansion of (1−pY)².

## W_p Table (All 22 Split Primes)

| p | T | N | a₁ | a₂ | a₃ | a₆ |
|---|---|---|---|---|---|---|
| 11 | −1 | −31 | 9 | −110 | −2420 | 1771561 |
| 31 | −11 | −1 | −61 | 2790 | −115320 | 887503681 |
| 41 | 9 | −11 | −71 | 1640 | −33620 | 4750104241 |
| 61 | −1 | −31 | −91 | −3660 | 669780 | 51520374361 |
| 71 | 19 | 59 | −201 | 20590 | −1613120 | 128100283921 |
| 101 | 29 | 179 | −381 | 74740 | −9384920 | 1061520150601 |
| 131 | −11 | −1 | −261 | −1310 | 4805080 | 5053913144281 |
| 151 | 4 | −496 | 194 | −20385 | −9576420 | 11853911588401 |
| 181 | −11 | −1 | −361 | −10860 | 15725280 | 35161828327081 |
| 191 | −41 | 389 | −771 | 284590 | −66395420 | 48551226272641 |
| 211 | −1 | −781 | 359 | −44310 | −32055120 | 88245939632761 |
| 241 | −16 | −436 | −46 | 3615 | −24394020 | 195930594145441 |
| 251 | 4 | −496 | −6 | −58985 | −1260020 | 250058907189001 |
| 271 | −31 | 209 | −751 | 186990 | −30845220 | 396109944105121 |
| 281 | −11 | −751 | 189 | −44960 | −48955820 | 492309163417681 |
| 311 | 49 | 569 | −1191 | 649990 | −234064820 | 904820297013361 |
| 331 | −61 | 899 | −1561 | 1122090 | −473303520 | 1315127813325481 |
| 401 | 29 | 179 | −981 | 176440 | 45024280 | 4157825282402401 |
| 421 | 19 | −691 | −151 | −25260 | −74441220 | 5567914722008521 |
| 431 | −36 | −176 | −686 | 372815 | −226628420 | 6410082527866081 |
| 461 | −1 | −781 | −141 | −212060 | 59505880 | 9598548249896761 |
| 491 | 9 | −11 | −971 | −201310 | 429124180 | 14011639427134441 |

## Checks

- **C1**: All 7 coefficients in ℤ — PASS (22/22)
- **C2**: Palindrome a₅=p⁴a₁, a₄=p²a₂, a₆=p⁶ — PASS (22/22)
- **C3**: Middle a₃ = −2p²(T²−N−2p) ∈ 2p²ℤ — PASS (22/22)
- **C4**: W_p = (1−pY)²·R_p(Y) coefficient match — PASS (22/22)

## Chain

- Caps [405] (GL₄ tensor-product R_p — the GL₄ component of W_p)
- Caps [404] (GL₄ induction P_p — the GL₄ form being exteriorly squared)
- Together [403]→[404]→[405]→[406] forms the complete Langlands ladder for 2.2.5.1-125.1-a
- Next: Sym²(AI(f)) → GL₁₀ or full ∧²(AI(f)) functional equation with conductor

## Verification Note (2026-07-07)

**Found and fixed a real documentation-table bug**, same class as [405]'s.
The displayed sample table only showed 5 of 22 primes (with "..." for
the rest) but 4 of those 5 rows (p=31,41,61,71) had wrong values — only
p=11 was correct; p=61/71 even had the a₁ sign flipped. The validator
itself was unaffected (it recomputes `gl6_exterior_square_poly` fresh
from `(T,N)` at runtime, confirmed via `python3
qa_gl6_exterior_square_cert_validate.py` returning `ok:true`).
Independently regenerated the complete, correct 22-row table from the
validator's own formula and replaced the old sparse/wrong sample above.
