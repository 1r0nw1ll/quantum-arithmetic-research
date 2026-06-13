<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Hecke (1920) doi.org/10.1007/BF01458074, Shimura (1971) ISBN 978-0-691-08092-5, LMFDB (2024) label 2.2.5.1-125.1-a -->
# [399] QA CM Form Identification

**Cert family**: `qa_cm_form_identification_cert_v1`
**Claim**: LMFDB 2.2.5.1-125.1-a is the CM Hilbert modular form over F=ℚ(√5) induced by K=ℚ(ζ₅)/F; certified by five algebraic checks using exact ℤ[φ] arithmetic.

## Statement

The Langlands ladder over ℚ(√5) built in certs [394]–[397] predicts a specific CM Hilbert modular form. This cert identifies it: **LMFDB label 2.2.5.1-125.1-a** (CM: yes, level norm 125).

Five algebraic certificates:

| Check | Statement | Value |
|---|---|---|
| C1 | Level norm = 5³ = disc(ℚ(ζ₅)/ℚ) | 125 = 125 ✓ |
| C2 | e = 5φ−3 ∈ ℤ[φ] satisfies e²+e−31=0 | (34,−5)+(−3,5) = (31,0) ✓ |
| C3 | e + σ_F(e) = −1 ∈ ℤ (rational trace) | (−3,5)+(2,−5) = (−1,0) ✓ |
| C4 | a_𝔭 = 0 iff p ≢ 1 (mod 5) | 7 primes verified ✓ |
| C5 | disc(e/ℚ) = Tr²−4·N = 1+124 = 125 = level | 125 = 125 ✓ |

## Eigenvalue Element e = 5φ−3

The LMFDB eigenvalue field is ℚ(e) with **e² + e − 31 = 0**. This field equals ℚ(√5) = F because:

```
disc(e² + e − 31) = 1 + 4·31 = 125 = 5³ → √125 = 5√5 → ℚ(√125) = ℚ(√5) = F
```

In ℤ[φ]-coordinates: **e = (−3, 5)** meaning −3 + 5φ = 5φ − 3.

Verification: e² + e = (34,−5) + (−3,5) = (31, 0) = 31 ∈ ℤ. So e² + e − 31 = 0. ✓

## Zero Pattern (C4)

CM forces a_𝔭 = 0 when the prime 𝔭 does not split completely in K = ℚ(ζ₅)/F:

| Norm | p | p mod 5 | Status in K/F | a_𝔭 |
|---|---|---|---|---|
| 4 | 2 | 2 | inert in F (norm 4) | 0 |
| 5 | 5 | 0 | ramified | 0 |
| 9 | 3 | 3 | inert in F (norm 9) | 0 |
| 11 | 11 | 1 | **splits completely** | e = (−3,5) and σ_F(e) = (2,−5) |
| 19 | 19 | 4 | splits in F, inert in K/F | 0 |
| 29 | 29 | 4 | splits in F, inert in K/F | 0 |
| 31 | 31 | 1 | **splits completely** | e−5 = (−8,5) |

## Discriminant Resonance (C5)

```
Tr(e) = e + σ_F(e) = (−3,5) + (2,−5) = (−1, 0) = −1
N(e)  = u² + uv − v² = 9 − 15 − 25 = −31
disc(e/ℚ) = (−1)² − 4·(−31) = 1 + 124 = 125 = level norm
```

The eigenvalue generator witnesses the level: **disc(e) = level norm = disc(ℚ(ζ₅)/ℚ)**.

## Langlands Ladder Closure

| Cert | Rung | Content |
|---|---|---|
| [394] | GL₁/F | Fibonacci Frobenius character |
| [395] | GL₂ Weil | Cassini determinant bound |
| [396] | GL₂ ℤ[ζ₅] | Degree-4 cyclotomic orbit infrastructure |
| [397] | GL₂ CM | Relative norm, tower law, totally-negative discriminant |
| **[399]** | **GL₂ LMFDB** | **CM form identification: 2.2.5.1-125.1-a** |

Cert [398] (Five Families partition) is the shadow of this tower at the mod-9 level.

## Checks

- **C1**: Level 125 = 5³ = disc(ℚ(ζ₅)/ℚ) — PASS
- **C2**: e = (−3,5) ∈ ℤ[φ] satisfies e²+e−31=0 — PASS
- **C3**: Conjugate e + σ_F(e) = −1 rational; matches LMFDB norm-11 eigenvalues — PASS
- **C4**: 7 primes: zero/non-zero pattern matches CM prediction — PASS
- **C5**: disc(e/ℚ) = 125 = level norm — PASS
