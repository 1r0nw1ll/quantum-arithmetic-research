# [232] QA UHG Diagonal Coincidence Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-13
**Source:** Will Dale + Claude; Wildberger, *Universal Hyperbolic Geometry I: Trigonometry* (2013)

## Claim

On the \(m=9\) grid \(\{1,\ldots,9\}^2\), the UHG bilinear form
\(\langle a,b\rangle=-(b_1e_2+e_1b_2)\) and projective quadrance identify
the same pairs as QA gcd-reduced diagonal classes.

The exhaustive witness contains:

- 64 unordered zero-quadrance pairs
- 64 unordered same-diagonal pairs
- intersection size 64
- zero counterexamples in either direction

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_uhg_diagonal_coincidence_cert_v1/qa_uhg_diagonal_coincidence_cert_validate.py`
- Fixtures: `udc_pass_m9_exhaustive.json` (PASS), `udc_fail_spoofed_coincidence.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| UDC_1 | schema version matches |
| UDC_M | modulus is 9 |
| UDC_COUNTS | zero-quadrance and same-diagonal counts both recompute to 64 |
| UDC_INTERSECTION | intersection recomputes to 64 |
| UDC_COUNTEREXAMPLES | both counterexample directions are empty |
| UDC_WITNESS | sample witnesses recompute their quadrance and gcd class labels |
| UDC_SRC | source attribution present |
| UDC_F | fail ledger well-formed |

## Cross-references

- Spec: `docs/specs/WILDBERGER_CERT_BATCH_SPEC.md`
- Theory: `docs/theory/QA_UHG_PROJECTIVE_COORDINATES.md`
- Template family: `docs/families/217_qa_fuller_ve_diagonal_decomposition_cert.md`

## Open questions (not in this cert)

- This cert fixes the witness at \(m=9\); it does not claim a new sweep over arbitrary \(m\).
