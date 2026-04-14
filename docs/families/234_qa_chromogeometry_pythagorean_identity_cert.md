# [234] QA Chromogeometry Pythagorean Identity Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-13
**Source:** Will Dale + Claude; Wildberger, *Chromogeometry* (2008)

## Claim

Under QA coordinates \((b,e)\), Wildberger's chromogeometry quadrances are:

- \(Q_b=b*b+e*e\)
- \(Q_r=b*b-e*e\)
- \(Q_g=2*b*e\)

The Pythagorean identity \(Q_b\) square \(= Q_r\) square \(+ Q_g\) square
holds exhaustively for \((b,e)\in[1,19]^2\) with zero failures. The fixture
also verifies the QA coordinate forms \(Q_r=(b-e)d\), \(Q_g=2be\), and
\(Q_b=b*b+e*e\), plus five QA-generated Pythagorean triples.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_chromogeometry_pythagorean_identity_cert_v1/qa_chromogeometry_pythagorean_identity_cert_validate.py`
- Fixtures: `cpi_pass_exhaustive_1to19.json` (PASS), `cpi_fail_wrong_sign.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| CPI_1 | schema version matches |
| CPI_SAMPLES | 20 sample pairs recompute \(Q_b,Q_r,Q_g\), both sides, and difference |
| CPI_RANGE | exhaustive range marker and zero-failure count match the recompute |
| CPI_FORMULAS | QA coordinate formula strings match the cert contract |
| CPI_PLIMPTON | five Pythagorean triples recompute from QA pairs |
| CPI_SRC | source attribution present |
| CPI_F | fail ledger well-formed |

## Cross-references

- Spec: `docs/specs/WILDBERGER_CERT_BATCH_SPEC.md`
- Theory: `docs/theory/QA_WILDBERGER_G2_INTEGER_CONSTRUCTION.md`
- Template family: `docs/families/217_qa_fuller_ve_diagonal_decomposition_cert.md`

## Open questions (not in this cert)

- This cert packages the exact polynomial identity and finite witness range; it does not make a statistical or empirical claim.
