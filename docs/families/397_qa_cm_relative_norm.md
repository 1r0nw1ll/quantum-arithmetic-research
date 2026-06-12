<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Neukirch (1999) ISBN 978-3-540-65399-8 §II.10, Washington (1997) ISBN 978-0-387-94762-4, Silverman (1994) ISBN 978-0-387-94328-2 §II.10 -->
# [397] QA CM Relative Norm and Tower Law

**Cert family**: `qa_cm_relative_norm_cert_v1`
**Claim**: For π=a+bζ₅ with N_{K/ℚ}(π)=p, the CM structure of ℚ(ζ₅)/ℚ(√5) gives N_{K/F}(π)=(a²-ab+b²)+ab·φ, tower law, CM factoring, and the totally-negative discriminant −b²(2+φ) — the degree-4/CM analog of the Weil sign flip.

## Statement

For K = ℚ(ζ₅), F = ℚ(√5), and π = a+bζ₅ ∈ ℤ[ζ₅] with N_{K/ℚ}(π) = p (prime, p ≡ 1 mod 5):

### A — Relative Norm Formula
```
N_{K/F}(a+bζ₅) = (a²−ab+b²) + ab·φ   ∈ ℤ[φ]
```
Derivation: N_{K/F}(π) = π·σ_K(π) = (a+bζ₅)(a+bζ₅⁴) = a²+b² + ab(ζ₅+ζ₅⁴) = (a²−ab+b²) + ab·φ
since ζ₅+ζ₅⁴ = φ−1 (Euler: 2cos(2π/5) = (√5−1)/2).

### B — Tower Law
```
N_{F/ℚ}(N_{K/F}(π)) = N_{K/ℚ}(π) = p
```
Algebraic identity: (a²−ab+b²)² + (a²−ab+b²)(ab) − (ab)² = a⁴−a³b+a²b²−ab³+b⁴ ✓

### C — CM Norm Factorization
```
N_{K/F}(π) · σ_F(N_{K/F}(π)) = p
```
where σ_F: φ → 1−φ. The relative norm and its F/ℚ-conjugate are the two "halves" of p.

Examples: `(3+2φ)(5−2φ)=11`, `(7−2φ)(5+2φ)=31`, `(7+3φ)(10−3φ)=61`, `(13+12φ)(25−12φ)=181`.

### D — CM Minimal Polynomial
```
X² − Tr_{K/F}(π)·X + N_{K/F}(π)   ∈ ℤ[φ][X]
```
where Tr_{K/F}(π) = (2a−b)+bφ (= the partial trace from cert [396]).

### E — Totally Negative Discriminant
```
Δ = Tr² − 4·N_{K/F} = −b²(2+φ)   TOTALLY NEGATIVE for b≠0
```
- Component form: Δ = (−2b²) + (−b²)·φ, both parts negative.
- N_{F/ℚ}(Δ) = 5b⁴ > 0 (product of two negative conjugates = positive).
- 2+φ ≈ 3.618 > 0 and 2+(1−φ) ≈ 1.382 > 0 in both real embeddings.
- Therefore: Δ negative in BOTH real embeddings of F = "totally negative."
- Totally negative Δ ⟺ CM min poly has no real roots in either embedding ⟺ Weil holds.

## The Degree-4 Discriminant Comparison

| Cert | Level | Discriminant | Sign | Consequence |
|---|---|---|---|---|
| [395] | GL₂ over ℚ | Δ_{GL2} = a_f² − 4p | < 0 (1 place) | Weil bound |
| [397] | CM over ℚ(√5) | Δ_{CM} = −b²(2+φ) | << 0 (TOTALLY negative, 2 places) | Weil at every place |

The extra strength comes from working over the totally real field F = ℚ(√5) instead of ℚ.

## Checks

- **C1**: N_{K/F}(a+bζ₅) = (a²−ab+b²)+ab·φ for 4 generators — PASS
- **C2**: N_{F/ℚ}(N_{K/F}(π)) = N_{K/ℚ}(π) = p, tower law verified — PASS
- **C3**: N_{K/F}(π)·σ_F(N_{K/F}(π)) = p, CM norm factoring — PASS
- **C4**: CM min poly: X² − [(2a−b)+bφ]·X + [(a²−ab+b²)+abφ] ∈ ℤ[φ][X] — PASS
- **C5**: disc = −b²(2+φ), totally negative, N_{F/ℚ}(disc) = 5b⁴ — PASS

## Chain Position

Extends: [396] (ℤ[ζ₅] infrastructure, partial trace = Tr_{K/F}), [394] (GL₁), [395] (Weil).

The partial trace from cert [396] appears here as the CM Hecke eigenvalue candidate π+σ_K(π).
The totally-negative discriminant closes the Langlands ladder at the CM/degree-4 level.

## Langlands Ladder — Complete

| Step | Cert | Object |
|---|---|---|
| GL₁ | [394] | Frobenius character (5/p) from σ-orbit |
| GL₂ Weil | [395] | \|a_f\|² < 4N — discriminant sign flip |
| GL₂ exact | [390] | a_f exact (LMFDB data) |
| K/F infra | [396] | ℤ[ζ₅]: C^5=I, G^4=I, partial trace |
| CM | **[397]** | **Relative norm + totally-negative discriminant** |
