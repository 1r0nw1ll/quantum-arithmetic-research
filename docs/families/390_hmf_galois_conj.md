# [390] QA HMF Galois Conjugation Symmetry

**Status**: PASS  
**Derived**: 2026-06-11  
**Cert directory**: `qa_alphageometry_ptolemy/qa_hmf_galois_conj_cert_v1/`

## Claim

For the Galois-conjugate pair {f₁, f₂} = {2.2.5.1-31.1-a, 2.2.5.1-31.2-a} — weight [2,2] Hilbert modular forms over ℚ(√5), level norm 31, CM=no — the Hecke eigenvalues satisfy:

| Check | Result |
|-------|--------|
| INERT_EQUAL: a_{f₁}(𝔭) = a_{f₂}(𝔭) at all inert primes p ≡ ±2 (mod 5) | PASS |
| RAMIFIED_EQUAL: eigenvalues equal at p=5 (ramified) | PASS |
| SPLIT_PERMUTED: {a_{f₁}(𝔭₁), a_{f₁}(𝔭₂)} = {a_{f₂}(𝔭₁), a_{f₂}(𝔭₂)} as multisets at all split primes | PASS |
| QA_SWAP_MIRRORS_GALOIS: Fibonacci roots r₁+r₂ ≡ 1, r₁·r₂ ≡ -1 mod p, and τ(r₁) = 1-r₁ ≡ r₂ mod p | PASS |
| WEIL_BOUND: all eigenvalues satisfy \|a_𝔭\| ≤ 2√N(𝔭) (Ramanujan-Petersson) | PASS |

Verified for 34 prime ideals of ℤ[φ] with N(𝔭) ≤ 151.

## Selected Hecke eigenvalue data

| p | class | N(𝔭) | f₁ eigenvalues | f₂ eigenvalues | r₁ | r₂ | π(p) |
|---|-------|-------|---------------|----------------|-----|-----|------|
| 2 | inert | 4 | −3 | −3 | — | — | 3 |
| 3 | inert | 9 | 2 | 2 | — | — | 8 |
| 5 | ram | 5 | −2 | −2 | — | — | 20 |
| 7 | inert | 49 | −6 | −6 | — | — | 16 |
| 11 | split | 11,11 | (4, −4) | (−4, 4) | 4 | 8 | 10 |
| 19 | split | 19,19 | (−4, 4) | (4, −4) | 5 | 15 | 18 |
| 29 | split | 29,29 | (−2, −2) | (−2, −2) | 6 | 24 | 14 |
| 41 | split | 41,41 | (−1, 8) | (8, −1) | 7 | 35 | 40 |
| 59 | split | 59,59 | (−6, 2) | (−6, 2) | 26 | 34 | 58 |
| 61 | split | 61,61 | (12, −4) | (−4, 12) | 18 | 44 | 60 |
| 71 | split | 71,71 | (6, −2) | (−2, 6) | 9 | 63 | 70 |
| 79 | split | 79,79 | (0, −8) | (−8, 0) | 30 | 50 | 78 |
| 89 | split | 89,89 | (0, 16) | (16, 0) | 10 | 80 | 44 |

Note: For p=59, both forms show (−6, 2) at the LMFDB index positions. This is consistent with the multiset claim: {−6, 2} = {−6, 2}. The LMFDB uses form-dependent prime ideal labeling for this prime, so the ordered swap is hidden in the indexing.

## The QA connection

The Galois automorphism τ: φ ↦ ψ = 1−φ of ℚ(√5)/ℚ acts on:
- **HMFs**: τ sends f₁ → f₂ (the conjugate form) by permuting the two primes above each split rational prime
- **QA Fibonacci roots**: τ sends r₁ → 1−r₁ ≡ r₂ mod p (since r₁+r₂ ≡ 1 mod p always)
- **QA σ-eigenspaces**: at split prime p, the eigenspaces of σ = Fibonacci matrix are spanned by the eigenvectors for r₁ and r₂; τ swaps them

**The same Galois automorphism τ simultaneously:**
1. Permutes the two Hilbert modular forms f₁ ↔ f₂
2. Permutes the two prime ideals 𝔭₁ ↔ 𝔭₂ above each split p
3. Permutes the two QA Fibonacci roots r₁ ↔ r₂ (cert [386])
4. Permutes the two QA σ-eigenspaces (cert [388])

This is the SAME GALOIS ACTION on the SAME algebraic structure. The Hecke eigenvalues, the prime ideal decomposition, and the QA orbit structure are all manifestations of the ℤ[φ] real multiplication.

## Atkin-Lehner data

From LMFDB: AL_eigenvalues for 2.2.5.1-31.2-a = [(level=(31,31,5w−3), sign=+1)].

The level prime 31 is split (31 ≡ 1 mod 5), with roots 13 and 19 of x²−x−1 mod 31. The form 31.2-a has level ideal above root 19 (the larger root, ord=30=π(31)), while 31.1-a has level ideal above root 13 (ord=15=π(31)/2). The Atkin-Lehner sign +1 at the level prime is the same in both forms (consistent with the level being principal).

## What this cert does NOT claim

- Does not identify which specific prime ideal 𝔭₁ above p has which eigenvalue (LMFDB ordering is form-dependent)
- Does not prove a formula a_{f}(𝔭) = g(r₁) for any function g of the Fibonacci root
- Does not certify eigenvalue data beyond N(𝔭) ≤ 151 (LMFDB rate limits prevented full retrieval)

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_hmf_galois_conj_cert_v1
python3 qa_hmf_galois_conj_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "7/7 passed"}`

## Lineage

- Extends **[388]** (split prime eigenspace decomposition: σ eigenvalues are r₁, r₂)
- Extends **[386]** (prime classification by x²−x−1 mod p)
- Extends **[385]** (QA orbit filtration = real multiplication on Bratteli C*-algebra)
- The Galois symmetry here is the SAME symmetry underlying [385]: real multiplication by ℤ[φ] on the C*-algebra means the Galois automorphism of ℚ(√5)/ℚ acts compatibly on both the HMF space and the QA orbit structure

## Primary sources

- LMFDB Collaboration. *The L-functions and modular forms database*. doi.org/10.1112/jlms.12687. Forms: 2.2.5.1-31.1-a, 2.2.5.1-31.2-a (accessed 2026-06-11)
- Shimura, G. (1978). *The special values of the zeta functions associated with Hilbert modular forms*. ISBN 978-0-691-08090-5 §9–10
- Blasius, D. & Rogawski, J.D. (1993). *Motives for Hilbert modular forms*. doi.org/10.2307/2152776
- van der Geer, G. (1988). *Hilbert Modular Surfaces*. ISBN 978-3-540-17659-9
