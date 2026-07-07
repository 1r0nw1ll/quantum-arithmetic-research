<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Wall (1960) doi.org/10.1080/00029890.1960.11989541, Neukirch (1999) ISBN 978-3-540-65399-8, Serre (1973) ISBN 978-0-387-90041-7 -->
# [394] QA Fibonacci Frobenius Character

**Cert family**: `qa_fibonacci_frobenius_character_cert_v1`
**Claim**: `sigma^p(1,0)[e] == (5/p) mod p` for all primes p ≠ 5

## Statement

For every prime p ≠ 5, iterating the QA sigma-operator p times from state (b,e) = (1,0)
yields an e-component equal to the Legendre symbol (5/p) modulo p:

```
sigma^p(1, 0) = (F_{p+1}, F_p)
F_p ≡ (5/p)  (mod p)
```

where (5/p) = +1 if p splits in Z[phi], −1 if inert, 0 if ramified (p=5).

## Langlands Significance

This is the **GL_1 base case** of the Langlands ladder for Q(sqrt(5)):

| Level | Object | Computed by |
|---|---|---|
| GL_1 | Frobenius character (5/p) ∈ {±1} | **This cert [394]** — orbit iteration |
| GL_2 | Hecke eigenvalue a_p ∈ Z, \|a_p\| ≤ 2√p | Cert [390] — LMFDB verification |
| Full | L-function factorization | Open (6–12 month scope) |

The key distinction from cert [386] (which classifies primes by ring-theoretic
root counts): **this cert computes (5/p) as an orbit value**, not as a congruence.
The QA sigma-orbit IS the Frobenius evaluation.

## Checks

- **C1**: F_p ≡ (5/p) mod p for all 94 primes ≤ 500 — PASS
- **C2**: 45 split primes give F_p ≡ +1 — PASS
- **C3**: 49 inert primes give F_p ≡ −1 — PASS
- **C4**: Ramified prime p=5 gives F_5 ≡ 0 — PASS
- **C5**: Fast-path spot-check at p ∈ {1009, 2003, 4999, 7919, 9973} — PASS

## Chain Position

Extends: [385] (Nikolaev/Z[phi] Hecke structure), [386] (prime classification),
[391] (sigma = phi-multiplication on Z[phi])

Bridges to: [390] (GL_2 HMF Galois symmetry using LMFDB)

The chain from [385]→[394]→[390] is: *structural identification* → *GL_1 orbit
computation* → *GL_2 eigenvalue symmetry*. Cert [394] is the missing oracle step
that makes the QA Frobenius computation explicit.

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified F_p ≡ (5/p) mod p in a
fresh, separate script for all odd primes ≤500 (Legendre symbol via
Euler's criterion, Fibonacci via direct integer recurrence) — zero
mismatches. This cert does not touch the LMFDB eigenvalue arrays (unlike
[390]/[395]) so it was unaffected by the index-misalignment bug found in
that pair. Genuine falsifiable number theory, no fixture-trusting gap.
