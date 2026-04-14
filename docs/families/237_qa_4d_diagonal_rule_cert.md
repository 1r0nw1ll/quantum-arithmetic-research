# [237] QA 4D Diagonal Rule Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-13
**Source:** Will Dale + Claude; Wildberger, *Rational Trigonometry in Higher Dimensions and a Diagonal Rule for 2-planes in Four-dimensional Space*, KoG 21:47-54, 2017

## Claim

The QA tuple `(b,e,d,a)` with `d=b+e` and `a=b+2e` is exactly the point
`b*v1+e*v2` in the 2-plane of `R^4` spanned by `v1=(1,0,1,1)` and
`v2=(0,1,1,2)`.

The Gram matrix is `[[3,3],[3,6]]`, with determinant `9`, matching the QA
canonical modulus `m=9`. The fixture also includes two concrete perpendicular
QA-tuple pairs satisfying Wildberger's Diagonal Rule `Q1+Q2=Q3`.

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_4d_diagonal_rule_cert_v1/qa_4d_diagonal_rule_cert_validate.py`
- Fixtures: `q4d_pass_diagonal_rule.json` (PASS), `q4d_fail_wrong_det.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Schema

| Field | Meaning |
|-------|---------|
| `schema_version` | Must be `QA_4D_DIAGONAL_RULE_CERT.v1` |
| `embedding_range` | Exhaustive integer range for checking `(b,e,b+e,b+2e)=b*v1+e*v2` |
| `embedding_failures` | Claimed number of embedding failures |
| `basis` | `v1` and `v2` spanning vectors |
| `gram_matrix` | Recomputed basis Gram matrix |
| `gram_det` | Recomputed Gram determinant |
| `canonical_modulus` | QA modulus expected to equal the determinant |
| `diagonal_rule_witnesses` | Concrete perpendicular QA-tuple witnesses with `Q1+Q2=Q3` |
| `source_attribution` | Primary-source attribution |
| `fail_ledger` | Negative-fixture explanation |

## Checks

| Check | Meaning |
|-------|---------|
| Q4D_1 | schema version matches |
| Q4D_EMBED | embedding is exhaustive and failure-free on `[-5..5]^2` |
| Q4D_GRAM | Gram matrix and determinant recompute to `[[3,3],[3,6]]` and `9` |
| Q4D_MODULUS | determinant equals canonical modulus `9` |
| Q4D_DIAGONAL_RULE | two perpendicular witnesses satisfy `Q1+Q2=Q3` |
| Q4D_SRC | source attribution includes Wildberger, Diagonal Rule, and KoG |
| Q4D_F | fail ledger is well-formed |

## Fixtures

| Fixture | Expected | Purpose |
|---------|----------|---------|
| `q4d_pass_diagonal_rule.json` | PASS | Packages the embedding, Gram determinant, modulus, and two diagonal-rule witnesses |
| `q4d_fail_wrong_det.json` | FAIL | Changes determinant `9` to `8` to prove the Gram/modulus checks are active |

## Family Relationships

- Gives QA's `(b,e,d,a)` tuple a native 4D rational-trigonometry interpretation.
- Complements [234] by moving from chromogeometry quadrance identity to a 4D diagonal-rule setting.
- Source theory note: `docs/theory/QA_WILDBERGER_TIER1_FOLLOWUPS.md` section 3.
