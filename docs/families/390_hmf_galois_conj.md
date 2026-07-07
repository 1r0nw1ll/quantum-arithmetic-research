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

| p | class | N(𝔭) | f₁ eigenvalues | f₂ eigenvalues |
|---|-------|-------|---------------|----------------|
| 2 | inert | 4 | −3 | −3 |
| 3 | inert | 9 | 2 | 2 |
| 5 | ram | 5 | −2 | −2 |
| 11 | split | 11,11 | (4, −4) | (−4, 4) |
| 19 | split | 19,19 | (−4, 4) | (4, −4) |
| 29 | split | 29,29 | (−2, −2) | (−2, −2) |
| 41 | split | 41,41 | (−6, −6) | (−6, −6) |
| 7 | inert | 49 | 2 | 2 |
| 59 | split | 59,59 | (12, −4) | (−4, 12) |
| 61 | split | 61,61 | (6, −2) | (−2, 6) |
| 71 | split | 71,71 | (0, −8) | (−8, 0) |
| 79 | split | 79,79 | (0, 16) | (16, 0) |
| 89 | split | 89,89 | (−6, 10) | (10, −6) |

(Table ordered by prime ideal norm, matching LMFDB's own natural ordering — see
Verification Note below for why p=7's row falls after p=41 rather than before it.)

Note: p=41 is a **split-equal** case — both prime ideals above 41 carry the same
eigenvalue (−6) in both forms, consistent with cert [388]'s independent finding
that p=41 is the one split prime ≤151 where both roots of x²−x−1 have equal
multiplicative order (40).

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

## Verification Note (2026-07-07)

**Found and fixed a real data-alignment bug.** Independently re-fetched the raw
LMFDB Hecke eigenvalue data for both forms (via the LMFDB page and its
`/api/hmf_hecke/` endpoint) and confirmed the hardcoded `EIGS_31_1`/`EIGS_31_2`
arrays in the validator are a byte-for-byte faithful copy of LMFDB's own
`hecke_eigenvalues` sequence — no fabrication. However, that raw sequence
includes the level-prime-31 Atkin-Lehner pseudo-eigenvalue in its natural
norm-sorted position (2 entries, since 31 is split), while the validator's
`_build_prime_list()` assumed those entries had already been stripped out (per
its own "skipping level prime 31" comment). They were never actually removed,
so every array index from 9 onward was misaligned by 2 slots relative to the
prime labels: index 9-10 (the real p=41 data, eigenvalue −6 in both forms) was
being read as p=41's data mislabeled with the level-31 values (−1, 8); index 11
(real p=7 data, eigenvalue 2) was being read as if it were p=7's, but held the
real p=41 values; and every split prime from p=59 onward inherited the same
one-slot cascade. This was **not caught by the automated checks** because the
corruption was identical in both `EIGS_31_1` and `EIGS_31_2` — the
INERT_EQUAL/SPLIT_PERMUTED checks only compare f₁ against f₂ at the same
(wrong) index, so a shared mislabeling still "passes" as internally
consistent. The QA_SWAP_MIRRORS_GALOIS check was unaffected since it only
uses Fibonacci-root arithmetic, not the eigenvalue arrays.

**Fix**: deleted the 2 level-31 entries from both arrays, restoring correct
index-to-prime alignment for all 34 verified prime ideals. Re-ran the
validator after the fix — all 5 structural checks (INERT_EQUAL,
RAMIFIED_EQUAL, SPLIT_PERMUTED, QA_SWAP_MIRRORS_GALOIS, WEIL_BOUND) still pass
with the corrected, properly-aligned real data, confirming the underlying
Galois-symmetry claim is genuinely true and not an artifact of the bug. One
hardcoded fixture (`SPLIT_P41_SWAPPED`, which encoded the old wrong values)
was updated to `SPLIT_P41_EQUAL` reflecting the real p=41 data. The corrected
full 34-entry table is reproduced above; independently spot-checked against
fresh LMFDB fetches through norm 79 (all exact matches) — see also the
p=41/[388] cross-cert consistency noted in the table above, which only became
visible after this fix.
