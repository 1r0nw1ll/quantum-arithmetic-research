<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Washington (1997) ISBN 978-0-387-94762-4, Ireland & Rosen (1990) ISBN 978-0-387-97329-6 Ch.14, Lang (1994) ISBN 978-1-4612-0853-2 -->
# [396] QA Degree-4 Cyclotomic Orbit over ℤ[ζ₅]

**Cert family**: `qa_cyclotomic_orbit_cert_v1`
**Claim**: `C^5 = I`, `G^4 = I`, quartic norm form, 5-orbit structure, partial trace to ℤ[φ]

## Statement

The companion matrix `C` of `Φ₅(x) = x⁴+x³+x²+x+1` defines a degree-4
σ₅-operator on ℤ⁴ (representing ℤ[ζ₅]) that is the natural degree-4 extension
of the QA σ-operator (certs [394]/[391]).

Four structural theorems, proved by exact integer arithmetic:

### A — 5-Periodicity
```
C^5 = I₄   (multiplication by ζ₅ has exact order 5 in GL₄(ℤ))
```
Since `Φ₅(x) | x⁵−1`, we have `ζ₅⁵=1` in ℤ[ζ₅]. This is the degree-4 analog
of `det(M^p)=(−1)^p` (Cassini invariant, cert [391]) — the canonical
periodicity condition at each Langlands rung.

### B — Galois Order 4
```
G^4 = I₄   where G encodes σ: ζ₅ → ζ₅²
```
`Gal(ℚ(ζ₅)/ℚ) = (ℤ/5ℤ)* ≅ ℤ/4ℤ` acts faithfully on ℤ⁴ coordinates.

### C — Quartic Norm Form
For `π = a+bζ₅ ∈ ℤ[ζ₅]`:
```
N_{ℚ(ζ₅)/ℚ}(a+bζ₅) = a⁴ − a³b + a²b² − ab³ + b⁴
```
Prime generators for `p ≡ 1 (mod 5)`: `N(2+ζ₅)=11`, `N(2−ζ₅)=31`, `N(3+ζ₅)=61`, `N(4+3ζ₅)=181`.

### D — Partial Trace (Bridge to ℤ[φ])
```
Tr_{ℚ(ζ₅)/ℚ(√5)}(a+bζ₅) = (2a−b) + b·φ   ∈ ℤ[φ]
```
`ζ₅+ζ₅⁴ = φ−1` (Euler identity: `2cos(2π/5) = (√5−1)/2`).
This projects the degree-4 arithmetic back to the degree-2 QA layer.

## Matrices

### C — multiplication by ζ₅ on {1, ζ₅, ζ₅², ζ₅³} basis
```
σ₅(a,b,c,d) = (−d, a−d, b−d, c−d)
```

### G — Galois action σ: ζ₅ → ζ₅²
```
G(a,b,c,d) = (a−c, d−c, b−c, −c)
```

## Langlands Position

| Level | Object | Cert |
|---|---|---|
| GL_1 | Frobenius character `(5/p)` | [394] — degree-2 orbit |
| GL_2 | Weil bound `\|a_f\| ≤ 2√N` | [395] — discriminant sign flip |
| GL_2 | Exact eigenvalue `a_f` | [390] — LMFDB verification |
| **CM** | **Degree-4 orbit infrastructure** | **[396] THIS** |
| CM | `a_f` from Hecke character `ψ(𝔭)` | [397] FUTURE — needs LMFDB CM data |

## Degree-2 vs Degree-4 Comparison

| | Degree-2 (ℤ[φ]) | Degree-4 (ℤ[ζ₅]) |
|---|---|---|
| Periodicity | `det(M^k) = (−1)^k` | `C^5 = I` |
| Operator | σ (Fibonacci Frobenius) | σ₅ (companion C) |
| Base ring | ℤ[φ] = ℤ[x]/(x²−x−1) | ℤ[ζ₅] = ℤ[x]/(x⁴+x³+x²+x+1) |
| Galois group | ℤ/2ℤ | ℤ/4ℤ |
| Primes | p ≡ ±1 (mod 5) | p ≡ 1 (mod 5) |
| Cert | [391] | **[396]** |

## Checks

- **C1**: `C^5 = I₄` (exact integer arithmetic; exact order 5) — PASS
- **C2**: `G^4 = I₄` (Galois order 4, exact; G≠I, G²≠I) — PASS
- **C3**: Norm form verified for 8 cases at `p ∈ {11, 31, 61, 181}` — PASS
- **C4**: C-orbit of each prime generator has period exactly 5 (5 distinct elements) — PASS
- **C5**: Partial trace `(2,1)→(3,1)`, `(2,−1)→(5,−1)`, `(3,1)→(5,1)`, `(4,3)→(5,3)` — PASS

## Chain Position

Extends: [391] (Cassini/σ on ℤ[φ]), [394] (GL₁ Frobenius from orbit), [395] (Weil sign flip), [390] (LMFDB data).

Infrastructure for: [397] FUTURE — CM Hecke character `ψ(𝔭) ∈ ℤ[ζ₅]` with
`N_{ℚ(ζ₅)/ℚ(√5)}(ψ(𝔭)) = N(𝔭) = p`; requires CM form label from LMFDB.
