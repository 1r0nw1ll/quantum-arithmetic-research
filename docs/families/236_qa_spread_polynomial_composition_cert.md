# [236] QA Spread Polynomial Composition Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-13
**Source:** Will Dale + Claude; Goh + Wildberger, *Spread polynomials, rotations and the butterfly effect*, arXiv 0911.1025, 2009

## Claim

Spread polynomials are defined by `S_0=0`, `S_1=s`, and
`S_{n+1}=2(1-2s)*S_n-S_{n-1}+2s`. They satisfy the exact composition law
`S_n` composed with `S_m` equals `S_{n*m}`.

The cert verifies the requested pairs `(2,3)`, `(3,2)`, `(2,4)`, `(4,3)`,
`(3,3)`, and `(2,5)` by exact symbolic composition. It also certifies
`S_2=4*s*(1-s)` as the logistic map and checks the integer coefficient forms
for `S_2`, `S_3`, and `S_4`. The rational-trig sine-square identity is recorded
as a note but skipped because this cert excludes float-dependent checks.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_spread_polynomial_composition_cert_v1/qa_spread_polynomial_composition_cert_validate.py`
- Fixtures: `spc_pass_composition.json` (PASS), `spc_fail_wrong_composition_coeff.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_SPREAD_POLYNOMIAL_COMPOSITION_CERT.v1` |
| `composition_witnesses` | Exact coefficient maps for `S_n` composed with `S_m` and `S_{n*m}` |
| `logistic_map` | Factored and expanded exact `S_2` witness |
| `rational_trig_identity_note` | Non-executed note for the float-dependent trig identity |
| `integer_closed_forms` | Exact coefficient maps for `S_2`, `S_3`, and `S_4` |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| SPC_1 | schema version matches |
| SPC_COMPOSITION | exact SymPy composition matches `S_{n*m}` for all requested pairs |
| SPC_CLOSED_FORMS | `S_2`, `S_3`, and `S_4` coefficient maps match recomputation |
| SPC_LOGISTIC | `S_2` matches `4*s*(1-s)` |
| SPC_TRIG_NOTE | trig identity is explicitly marked skipped for float-dependence |
| SPC_SRC | source attribution includes Goh, Wildberger, and Spread polynomials |
| SPC_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `spc_pass_composition.json` | PASS | Packages the six composition witnesses, logistic map, closed forms, and trig skip note |
| `spc_fail_wrong_composition_coeff.json` | FAIL | Changes one coefficient in the `(2,5)` composition witness |

## Family Relationships

- Extends [128] spread-period work from modular cycling into exact composition algebra.
- Connects to [216] through exact discrete dynamics rather than continuous logistic-map chaos.
- Source theory note: `docs/theory/QA_WILDBERGER_TIER1_FOLLOWUPS.md` section 2.
