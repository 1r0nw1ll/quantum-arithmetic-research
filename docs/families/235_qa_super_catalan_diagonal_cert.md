# [235] QA Super Catalan Diagonal Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-13
**Source:** Will Dale + Claude; Limanta + Wildberger, *Super Catalan Numbers and Fourier Summations over Finite Fields*, arXiv 2108.10191, Bull. Austral. Math. Soc. 2022

## Claim

Under the QA identification `(b,e)=(m,n)`, the Super Catalan number
`S(m,n) = (2m)!(2n)!/(m!n!(m+n)!)` has `d=b+e`, so the denominator factor
`(m+n)!` is exactly `d!`.

On `D_1`, `S(b,b)` matches OEIS A000984 central binomials for `b=0..10`.
The fixture also certifies swap symmetry on `[0..7]^2`, the recurrence
`4*S(b,e)=S(b+1,e)+S(b,e+1)` on `[0..7]^2`, and `S(1,n)=2*Catalan(n)` for
`n=0..9`.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_super_catalan_diagonal_cert_v1/qa_super_catalan_diagonal_cert_validate.py`
- Fixtures: `scd_pass_super_catalan_diagonal.json` (PASS), `scd_fail_wrong_a000984.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_SUPER_CATALAN_DIAGONAL_CERT.v1` |
| `d1_a000984_values` | `S(b,b)` values for `b=0..10` and the matching A000984 values |
| `swap_symmetry_range` | Exhaustive range marker for `S(b,e)=S(e,b)` |
| `recurrence_range` | Exhaustive range marker for the four-term recurrence |
| `s1n_catalan_values` | `S(1,n)`, `Catalan(n)`, and `2*Catalan(n)` for `n=0..9` |
| `qa_identification` | `(b,e)=(m,n)`, `d=b+e`, and `(m+n)! = d!` |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| SCD_1 | schema version matches |
| SCD_D1_A000984 | `S(b,b)` for `b=0..10` recomputes and matches A000984 |
| SCD_SYMMETRY | swap symmetry has zero failures on `[0..7]^2` |
| SCD_RECURRENCE | recurrence has zero failures on `[0..7]^2` |
| SCD_CATALAN | `S(1,n)=2*Catalan(n)` for `n=0..9` |
| SCD_QA_IDENT | QA identification fields match the cert contract |
| SCD_SRC | source attribution includes Limanta, Wildberger, and Super Catalan |
| SCD_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `scd_pass_super_catalan_diagonal.json` | PASS | Packages the verified diagonal, symmetry, recurrence, Catalan, and QA-identification witnesses |
| `scd_fail_wrong_a000984.json` | FAIL | Changes the `b=10` diagonal value to prove the A000984 check is active |

## Family Relationships

- Complements [231] by filling the `D_1` central-binomial diagonal.
- Extends the [234] Wildberger batch through the Super Catalan paper's chromogeometric finite-field context.
- Source theory note: `docs/theory/QA_WILDBERGER_TIER1_FOLLOWUPS.md` section 1.
