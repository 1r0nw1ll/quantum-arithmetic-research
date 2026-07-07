<!-- PRIMARY-SOURCE-EXEMPT: reason=human-readable cert family doc; mathematical content cited in mapping_protocol_ref.json: Serre (1979) doi:10.1007/978-1-4757-5673-9, Ireland & Rosen (1990) ISBN 978-0-387-97329-6, Wall (1960) doi:10.2307/2309169 -->
# [436] QA Witt Tower Ramified Prime e-Value Rigidity

**Cert family**: `qa_witt_tower_ramified_e_rigidity_cert_v1`
**Claim**: [435] described `e := ord(lambda0 mod p)` as a "free
parameter" in the generalized period-set formula. This cert proves
that framing is wrong: `e` is mechanically forced by two binary
choices — the determinant of the companion matrix and which Vieta
factor the prime divides — and takes only one of three possible
values {1, 2, 4} total across both families.

## The gap

[434] derived the period-set/multiplicity law for Fibonacci, p=5,
treating `e=4` as given input.  [435] verified the same formula on
D12/p=3 (e=2) and called the result "e and p as free parameters of a
general law."  Neither cert asked: what values can `e` actually take?
Is it truly free?  This cert answers: no — `e` is rigid.

## The rigidity law

For the two companion-matrix families used throughout this chain:

**det=+1 family** (`x² − tx + 1`, `D = t²−4 = (t−2)(t+2)`):

Every odd prime `p | D` divides exactly one of `(t−2)` or `(t+2)`
(since `gcd(t−2, t+2) | 4`).  Then `lambda0 = t/2 mod p` equals:

```
p | (t−2)  =>  t ≡ 2 mod p  =>  lambda0 ≡ +1 mod p  =>  e = 1
p | (t+2)  =>  t ≡ −2 mod p =>  lambda0 ≡ −1 mod p  =>  e = 2
```

**det=−1 family** (`x² − tx − 1`, `D = t²+4`):

Ramification requires `t² ≡ −4 mod p`, so `lambda0² ≡ (t/2)² ≡ −1
mod p`.  For any odd prime, `−1 ≠ 1`, so `ord(lambda0) ≠ 1, 2`, and
since `lambda0^4 = (lambda0²)² = (−1)² = 1`, the order must be **4**
exactly — no exceptions, ever.

## Connection to prior certs

| Cert | Family | Prime | Reported e | This cert says |
|------|--------|-------|-----------|----------------|
| [434] | det=−1 (Fibonacci, t=1) | p=5 | 4 | det=−1 → e=4 always |
| [435] | det=+1 (D12, t=4) | p=3 | 2 | p=3 \| (t+2)=6 → e=2 |

Neither prior cert obtained `e` from a free-parameter space — both
were instances of this rigid law.

## Checks

| Check | Content | Result |
|-------|---------|--------|
| C1 DET_PLUS1_E_RIGIDITY | det=+1 family, t=3..3000: every odd p\|D has e∈{1,2} with the correct branch | **PASS** (10,691 pairs) |
| C2 DET_MINUS1_E_RIGIDITY | det=−1 family, t=1..3000: every odd p\|D has e=4 exactly | **PASS** (6,420 pairs) |
| C3 NO_THIRD_VALUE_GLOBAL | Observed e-values: plus1={1,2}, minus1={4} — e=3,5,6,… never appear | **PASS** (17,111 pairs total) |
| C4 BRANCH_MECHANISM_EXACT | lambda0≡+1 / −1 / lambda0²≡−1 verified as exact mod-p identities for every pair | **PASS** (17,111 checks) |
| C5 LEGACY_CONSISTENCY | [434]'s e=4 and [435]'s e=2 reproduced by the same generic code path | **PASS** |

## What this cert does NOT claim

- Does not modify the period-set or multiplicity formulas from
  [434]/[435] — those remain correct.
- Does not claim the rigidity extends beyond the two det=±1
  companion-matrix families; higher-order recurrences or matrices
  with |det|>1 may yield different `e` values.
- Does not prove a period formula for the `e=1` collapse case (when
  `e=1`, periods 1 and `e·p^L` merge into a single bucket — this is
  analogous to the p=2 exception in [435] and is not analysed here).

## Running the validator

```bash
cd qa_alphageometry_ptolemy/qa_witt_tower_ramified_e_rigidity_cert_v1
python3 qa_witt_tower_ramified_e_rigidity_cert_validate.py --self-test
```

Expected: `{"ok": true, "checks": {...all true...}, "fixture_summary": "7/7 passed"}`

## Lineage

- Directly refines **[435]** by characterizing `e` itself (not just
  verifying the formula assuming `e` is known)
- Shows **[434]**'s `e=4` and **[435]**'s `e=2` are the only two
  outcomes of a rigid binary law, not independent free-parameter
  data points
- Consistent with **[432]** (scaling isomorphism) and **[433]**
  (recursive refinement law) throughout — those certs never examined
  `e` directly

## Primary sources

- Serre, J.-P. (1979). *Local Fields*. [doi:10.1007/978-1-4757-5673-9](https://doi.org/10.1007/978-1-4757-5673-9) — ramification theory, double roots mod p
- Ireland, K. & Rosen, M. (1990). *A Classical Introduction to Modern Number Theory*. ISBN 978-0-387-97329-6 Ch.7 — primitive roots, Hensel lifting
- Wall, D.D. (1960). *American Mathematical Monthly* 67(6). [doi:10.2307/2309169](https://doi.org/10.2307/2309169) — Pisano-style period tables

## Verification Note (2026-07-07)

Confirmed clean, no bugs. Independently re-verified the e-rigidity law
in a fresh script (own λ₀-order computation, not reusing validator
code): for the det=+1 family, all 1402 tested (t,p) pairs (t=3..500,
odd p|D<200) correctly split into e=1 (p|t−2) and e=2 (p|t+2) with no
exceptions; for the det=−1 family, all 559 tested pairs give e=4
exactly with no exceptions. This is a genuine, falsifiable structural
claim (not just re-confirming [434]/[435]'s two data points) and it
holds with zero counterexamples across ~2000 independent checks.
