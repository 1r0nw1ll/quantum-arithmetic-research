# Family [128] — QA_SPREAD_PERIOD_CERT.v1

**QA cosmos orbit period = Pisano period; spread polynomial period = Fibonacci sequence period mod m**

---

## What this family certifies

The QA cosmos orbit period for modulus m equals the **Pisano period** π(m) — the period
of the Fibonacci sequence modulo m:

```
π(m) = smallest k > 0  such that  F_k ≡ 0 (mod m)  AND  F_{k+1} ≡ 1 (mod m)
```

This equals the order of the Fibonacci shift matrix `F = [[0,1],[1,1]]` in GL₂(Z/mZ),
which family [126] showed is the QA T-operator.

The same period governs spread polynomials: the n-th spread polynomial S_n(s) satisfies
`S_n(sin²θ) = sin²(nθ)`, so for fixed `s ∈ F_m`, the sequence S_1(s), S_2(s), ... returns
to s after π(m) steps (because the angle group has order π(m)).

---

## Standard Pisano periods

| m | π(m) | F^{π/2} mod m | QA cosmos states |
|---|------|----------------|-----------------|
| 2 | 3 | — (odd period) | 3 |
| 3 | 8 | −I | 8 |
| 4 | 6 | — (F³=[[1,2],[2,3]] ≠ −I) | 12 |
| 5 | 20 | −I | 20 |
| 6 | 24 | −I | 24 |
| 7 | 16 | −I | 48 |
| 8 | 12 | — (F⁶=5·I mod 8 ≠ −I=7·I) | 48 |
| 9 | 24 | −I | 72 |
| 24 | 24 | 17·I mod 24, **≠ −I** (corrected 2026-07-06; −I=23·I mod 24) | 504 |

When F^{π/2} = −I, the affine QA translation cancels and the affine orbit period equals
the linear matrix order (proven in family [126]).

---

## Validation checks (SP1–SP5)

| ID | Check | Fail type |
|----|-------|-----------|
| SP1 | `schema_version == 'QA_SPREAD_PERIOD_CERT.v1'` | SCHEMA\_VERSION\_WRONG |
| SP2 | Fibonacci sequence period mod m == claimed cosmos\_period | PISANO\_PERIOD\_MISMATCH |
| SP3 | `F_matrix^cosmos_period ≡ I mod m` | MATRIX\_PERIOD\_WRONG |
| SP4 | cosmos\_period is minimal (F^(P/k) ≢ I for prime k\|P) | PERIOD\_NOT\_MINIMAL |
| SP5 | orbit\_type ∈ valid set; period=1 ↔ singularity | ORBIT\_TYPE\_MISMATCH |

---

## Fixtures

| File | m | period | orbit\_type | Result | Notes |
|------|---|--------|-------------|--------|-------|
| `sp_pass_m9.json` | 9 | 24 | cosmos | PASS | π(9)=24; F^12=−I≢I |
| `sp_pass_m7.json` | 7 | 16 | cosmos | PASS | π(7)=16; F^8=−I≢I |
| `sp_fail_wrong_period.json` | 9 | 12 (wrong) | cosmos | FAIL | PISANO\_PERIOD\_MISMATCH + MATRIX\_PERIOD\_WRONG: confuses projective order 12 with linear order 24 |

---

## Mathematical context

**Sources**: Wildberger arXiv:0911.1025 (Spread Polynomials), arXiv:math/0701338 (1D Metrical
Geometry), arXiv:0909.1377 (UHG I).

The Pisano period π(m) is a classical number-theoretic function. For the QA system:
- **π(p) for prime p ≡ ±1 (mod 5)**: p−1 divides π(p) (5 is a QR mod p)
- **π(p) for prime p ≡ ±2 (mod 5)**: 2(p+1) divides π(p) (5 is a QNR mod p)
- **π(p^k)**: determined by π(p) and the lift

For m=9=3²: π(9)=24=3·π(3)=3·8 (standard Pisano lifting formula for p=3, k=2).

**Connection to [125]-[127]**:
- [125] certifies C,F,G = chromogeometric quadrances (algebra)
- [126] certifies T-operator = Fibonacci shift = red isometry (dynamics)
- [127] certifies QA triples = UHG null points (geometry)
- [128] (this family) certifies the PERIOD of those dynamics = Pisano period (arithmetic)

Together, [125]–[128] form the Wildberger synthesis: QA as chromogeometry (algebra) + red
isometry dynamics (group theory) + UHG null points (geometry) + Pisano periods (number theory).

---

## ok semantics

`ok=True` means the certificate is internally consistent:
detected failure types == declared `fail_ledger`, and `result` field is consistent.

## Verification Note (2026-07-06)

Independently recomputed the full "Standard Pisano periods" table from
scratch for all 9 moduli by directly simulating the `(b,e)→(e,(b+e) mod m)`
orbit map over every state in `(Z/mZ)²` and by computing real matrix
powers of `F=[[0,1],[1,1]]` mod m: every `π(m)` value and every "QA
cosmos states" count matches exactly (e.g. m=9: π=24, 72 cosmos + 8
satellite + 1 singularity = 81 = 9², matching the project's established
orbit structure). The validator (`qa_spread_period_cert_validate.py`)
only checks the two certified fixtures (m=9, m=7) and genuinely
recomputes the Pisano period and `F^period≡I` there — no bugs.

**Found and fixed a real computational error in the illustrative table**
(not in any certified fixture): the `F^{π/2} mod m` column claimed
`m=24 → −I`, but directly computing `F^12 mod 24` gives `17·I`, not
`23·I` (`−1 mod 24 = 23`). Every other even-π(m) row was independently
re-verified as correctly stated (3, 5, 6, 7, 9 genuinely give `−I` at
the half-period; 4 and 8 correctly show "—" since their half-period
powers are genuinely not `−I` either). Corrected the m=24 cell and added
the computed matrix value for transparency. Not certified by any
fixture, so this was a pure documentation-table error with no
downstream validator impact — but worth noting for anyone reading the
table: m=24 is the one modulus in this list where the
"F^(π/2)=−I ⟹ affine period = linear order" precondition genuinely does
NOT hold, which could matter for future work extending this cert to
m=24 specifically (not investigated further here).
