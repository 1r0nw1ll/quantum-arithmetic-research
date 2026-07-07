<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Weil (1949) doi.org/10.1090/S0002-9904-1949-09219-4, Deligne (1974) doi.org/10.1007/BF02684373, Diamond & Shurman (2005) ISBN 978-0-387-27226-9 -->
# [395] QA Weil Bound from Cassini

**Cert family**: `qa_weil_bound_cassini_cert_v1`
**Claim**: `|a_f(p)|^2 < 4*N(p)` for all 34 prime ideals with N(p) ≤ 151

## Statement

For the Hilbert modular form 2.2.5.1-31.1-a over ℚ(√5), weight [2,2],
the Hecke eigenvalues satisfy the Weil bound at all tested prime ideals:

```
|a_f(𝔭)|² < 4·N(𝔭)   (equivalently: Δ_GL2 = a_f² − 4N < 0)
```

## The Cassini → Weil Sign Flip

The Fibonacci Frobenius (cert [394]) and the GL₂ Frobenius have
structurally opposite discriminant signs:

| Object | det | Char poly | Discriminant | Eigenvalues |
|---|---|---|---|---|
| Fibonacci Frobenius (weight-0) | (-1)^p = −1 | x²−L_p·x−1 | L_p²+4 **> 0** | Real |
| GL₂ Frobenius (weight-2) | N(𝔭) = p | x²−a_f·x+p | a_f²−4p **< 0** | Complex |

The weight shift **det=−1 → det=p** flips the discriminant sign.
The Cassini invariant det(M^p)=−1 (cert [391]) is the weight-0 analogue of
the Weil bound: Fibonacci eigenvalues have product −1; GL₂ eigenvalues have
product p and each has absolute value √p.

## Langlands Ladder Position

| Level | Object | Source |
|---|---|---|
| GL_1 | Frobenius character (5/p) ∈ {±1} | Cert [394] — orbit iteration |
| GL_2 | Weil bound \|a_f(𝔭)\| ≤ 2√N(𝔭) | **Cert [395] — Cassini sign flip** |
| GL_2 | Exact eigenvalue a_f(𝔭) ∈ ℤ | Cert [390] — LMFDB verification |
| Full | L-function factorization | Open (6–12 month scope) |

## Checks

- **C1**: `|a_f(𝔭)|² ≤ 4·N(𝔭)` for all 34 prime ideals with N(𝔭) ≤ 151 — PASS
- **C2**: Strict inequality (no equality) — consistent with non-CM form — PASS
- **C3**: Fibonacci discriminant L_p²+4 > 0 for 10 tested primes — PASS
- **C4**: GL₂ discriminant a_f²−4N < 0 for all 34 prime ideals — PASS
- **C5**: Cassini: det(M^p) = −1 for spot-check primes {11,19,41,59,71} — PASS

## Chain Position

Extends: [391] (Cassini identity and sigma=phi-multiplication on ℤ[φ]),
[394] (GL₁ Frobenius character from orbit iteration),
[390] (LMFDB HMF eigenvalue data)

The discriminant sign flip is the structural bridge from the Cassini chain
([391]→[392]→[394]) to the GL₂ Weil theory.

## Verification Note (2026-07-07)

**Found and fixed a real bug propagated sideways from cert [390].** This
cert independently duplicates the same hardcoded `EIGS_31_1` LMFDB
eigenvalue array as [390], and had the identical index-misalignment bug
(the level-31 Atkin-Lehner values were never stripped, shifting every
prime index ≥9 by 2 slots — see [390]'s Verification Note for the full
diagnosis). Applied the identical fix here (deleted the 2 level-31
entries). Re-ran the validator after the fix — C1-C5 all still pass,
and an independent fresh re-check of the Weil bound (|a|²<4N) across all
34 corrected prime-ideal entries confirms zero violations. The
underlying Cassini→Weil sign-flip theorem is genuinely true and
unaffected by the bug (the Weil bound is a much looser inequality than
the exact-eigenvalue-equality checks in [390], so it happened to hold
either way) — but the specific (prime, eigenvalue) pairs reported for
p≥41 were wrong before this fix, same as in [390].
