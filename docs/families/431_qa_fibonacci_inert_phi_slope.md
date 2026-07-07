<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Wall (1960) doi:10.2307/2309169, Sun & Sun (1992) doi:10.4064/aa-60-4-371-388 -->
# [431] QA Fibonacci Inert phi-Slope Formula mod p²

**Cert family**: `qa_fibonacci_inert_phi_slope_cert_v1`
**Claim**: The Hensel-lifted Binet formula and the phi-slope reformulation of
the Wall-Sun-Sun (WSS) criterion — built for **split** primes only in cert
[421] — also hold for **inert** primes, via the depth-2 Galois ring
R_p = (ℤ/p²ℤ)[φ]/(φ²−φ−1). This completes the Galois/Frobenius picture of
δ(p) (and hence the WSS criterion of [429]) for **all** primes.

## The Gap in [421]

Cert [421]'s own docstring is explicit about its scope: "the cert covers the
split case where the formula is ℤ/p²ℤ-valued and directly computable." For
inert primes, φ and ψ (roots of x²−x−1) don't live in ℤ/pℤ at all (cert
[424]) — they live in the quadratic extension 𝔽_{p²}. So the depth-2 lift
can't just Hensel-lift an integer square root of 5 mod p²; it has to lift the
*extension ring itself*.

## Construction

For inert prime p, define
```
R_p = (Z/p^2 Z)[phi] / (phi^2 - phi - 1)
```
— the literal mod-p² lift of [424]'s `F_{p²} = (Z/pZ)[φ]/(φ²−φ−1)`. Elements
are pairs `(a,b) ↔ a+bφ`, `a,b ∈ Z/p²Z`, with multiplication
`(a+bφ)(c+dφ) = (ac+bd)+(ad+bc+bd)φ` (using `φ²=φ+1`).

Set `ψ̃ = 1−φ̃` and `s̃ = φ̃−ψ̃ = 2φ̃−1`. Then `s̃² = 5·1` — a scalar — and since
p≠5 is inert, 5 is a unit mod p², so `s̃⁻¹ = (5⁻¹ mod p²)·s̃`. **No separate
Hensel lift of √5 is needed**: unlike the split case, R_p is a free rank-2
ℤ/p²ℤ-module by construction, so φ̃ and ψ̃ already live there exactly — the
extension is built, not approximated.

## C1: Ring Correctness

**Claim**: `φ̃²=φ̃+1`, `ψ̃²=ψ̃+1`, `φ̃ψ̃=−1`, `φ̃+ψ̃=1`, `s̃²=5`, `s̃` invertible
— all mod p² in R_p — for 20 inert primes ≤ 200.

Verified: 20/20.

## C2: Binet Formula mod p²

**Claim**: `F_n = (φ̃ⁿ−ψ̃ⁿ)·s̃⁻¹ (mod p²)` in R_p, for n=1..15, 15 inert primes
≤ 150. Note: the Binet expression must land in the scalar (b=0) part of R_p
— this is checked explicitly, not assumed.

Verified: 225/225 cases.

## C3: δ(p) Agreement

**Claim**: δ(p) (the depth invariant from [429]/[430], δ(p) = F_{α(p)}/p mod p)
computed via the R_p-Binet path equals δ(p) computed via the direct integer
Fibonacci recurrence — for all inert primes ≤ 500.

Verified: 47/47.

## C4: WSS via Frobenius

**Claim**: δ(p)=0 iff `φ̃^{α(p)} = ψ̃^{α(p)}` in R_p, for all inert primes
≤ 500 (none found — consistent with [429]'s combined split+inert sweep, and
with no WSS prime known below 9.7×10¹⁴).

Verified: 47/47.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 | Ring correctness; 20 inert primes ≤200 | **PASS** |
| C2 | Binet mod p² in R_p; 225 cases (15 primes × 15 n-values) | **PASS** |
| C3 | δ(p): R_p-Binet = recurrence; 47 inert primes ≤500 | **PASS** |
| C4 | WSS iff φ̃^α=ψ̃^α in R_p; 47 inert primes ≤500 | **PASS** |

## Theorem NT Factorisation

```
QA layer (pure integer):
  ring_mul/ring_add/ring_sub/ring_pow — pair arithmetic in R_p = (Z/p^2Z)[phi]
  fib_fast(n, p*p) — fast doubling mod p^2, as in [428]/[429]/[430]
  rank_of_apparition(p) — linear walk mod p

Observer layer: none (no floats, no statistics, direct equality checks only)
```

## Langlands Ladder

| Cert | Rung |
|------|------|
| [423] | α(p) = ord_{𝔽_p×}(φ̃/ψ̃) for split primes — GL₁ Frobenius order, mod p |
| [424] | inert extension of [423]: α(p) | p+1, Frobenius swaps φ̃↔ψ̃ in 𝔽_{p²} |
| [421] | δ(p) via Hensel-lifted Binet mod p², **split primes only** |
| **[431]** | **δ(p) via Galois ring R_p mod p², inert primes — completes the split/inert pairing at depth p²** |

[431] closes the split/inert pairing pattern at depth p², matching the
mod-p pairing [423]→[424]. Together [421]+[431] give the full Galois/Frobenius
interpretation of δ(p) for all primes, underlying the general WSS criterion
established (without the Galois interpretation) in [429].

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-implemented the Galois ring
R_p = (ℤ/p²ℤ)[φ]/(φ²−φ−1) from scratch and verified all ring identities
(φ̃²=φ̃+1, ψ̃²=ψ̃+1, φ̃ψ̃=−1, s̃²=5) for all inert primes ≤200, plus the
Binet-mod-p² formula for 225 (prime, n) cases — all exact matches.
Genuine falsifiable algebra, no fixture-trusting gap. This closes the
independent audit of the entire orbit-theory/Galois-representation
cluster [384]-[431] (48 certs): two real bugs found and fixed
([390]/[395]'s LMFDB index misalignment), four doc-table transcription
errors found and fixed ([405]-[408]), one large hardcoded dataset
verified genuine via brute-force search ([403]), and every other cert
independently reproduced clean.
