# [245] QA SL3 Hexagonal Ring Identity Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-14
**Source:** Will Dale + Claude; Wildberger, *Quarks, diamonds, and representations of sl_3*, UNSW preprint, 2003

## Claim

For `a,b >= 1`, the outer hexagonal ring size in the sl(3) diamond model is

`ring(a,b) = dim pi[a,b] - dim pi[a-1,b-1]`.

Using the standard sl(3) dimension formula,

`dim pi[a,b] = (a+1)*(b+1)*(a+b+2)//2`,

the ring identity becomes

`ring(a,b) = T(d+1) + a*b`, where `d=a+b` and `T(n)=n*(n+1)//2`.

Under QA coordinates `(b_QA,e_QA)=(a,b)`, this is the integer polynomial

`ring = T(d+1) + b_QA*e_QA`.

The validator checks the cleared-denominator algebraic expansion and all 196
entries on `(a,b) in [1..14]^2`.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_sl3_hexagonal_ring_identity_cert_v1/qa_sl3_hexagonal_ring_identity_cert_validate.py`
- Fixtures: `shr_pass_ring_identity.json` (PASS), `shr_fail_wrong_closed_form.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_SL3_HEXAGONAL_RING_IDENTITY_CERT.v1` |
| `algebraic_expansion` | Cleared-denominator symbolic identity and SymPy zero-difference witness |
| `exhaustive_grid` | Full 196-entry table over `[1..14]^2` with dimension difference and closed form |
| `qa_coord_form` | QA identification `(b_QA,e_QA)=(a,b)` and formula marker |
| `known_multiplicities` | Literature-facing checks for `ring(1,1)`, `ring(2,1)`, and `ring(2,2)` |
| `witnesses` | Required witness kinds |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| SHR_1 | schema version matches |
| SHR_ALGEBRAIC_EXPANSION | cleared-denominator expansion reduces identically to zero |
| SHR_EXHAUSTIVE | all 196 rows match `ring(a,b)=T(d+1)+a*b` |
| SHR_QA_COORD_FORM | QA coordinate formula is present and recomputes as `T(d+1)+b_QA*e_QA` |
| SHR_KNOWN_MULTIPLICITIES | `ring(1,1)=7`, `ring(2,1)=12`, `ring(2,2)=19` |
| SHR_SRC | Wildberger, Quarks/diamonds, and `sl_3` source markers are present |
| SHR_WITNESS | all four required witness kinds are present |
| SHR_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `shr_pass_ring_identity.json` | PASS | Packages the symbolic witness, full 196-entry grid, QA coordinate form, and known multiplicities |
| `shr_fail_wrong_closed_form.json` | FAIL | Changes the first closed-form row to the synthetic wrong formula `T_d+a*b` |

## Family Relationships

- Structural follow-up to [240], refining the sl(3) dimension formula into a consecutive-ring identity.
- Adds the QA decomposition `T(d+1)+b*e`: a triangular `d` component plus a bilinear cross term.
- Source theory note: `docs/theory/QA_WILDBERGER_STRUCTURAL_FOLLOWUPS.md` section 1.
