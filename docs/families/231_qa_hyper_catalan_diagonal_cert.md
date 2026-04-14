# [231] QA Hyper-Catalan Diagonal Cert

**Status:** PASS (self-test; meta-validator registered)
**Created:** 2026-04-13
**Source:** Will Dale + Claude; Wildberger and Rubine, *A Hyper-Catalan Series Solution to Polynomial Equations, and the Geode* (2025)

## Claim

For hyper-Catalan multi-index \(m=[m_2,m_3,m_4,\ldots]\), use
\(b := V_m - 1\) and \(e := F_m\). The QA derived coordinate satisfies
\(d=b+e=E_m\), so Euler \(V_m-E_m+F_m=1\) becomes automatic.

Single-type cases \(m_k=n\) sit on sibling diagonals:

- \(b=(k-1)e+1\)
- Catalan/Fuss values match OEIS A000108, A001764, A002293, and A002294
- no single-type case in \(k \in [2,7]\), \(n \in [0,9]\) sits on \(D_1\)

## Artifacts

- Validator: `qa_alphageometry_ptolemy/qa_hyper_catalan_diagonal_cert_v1/qa_hyper_catalan_diagonal_cert_validate.py`
- Fixtures: `hcd_pass_euler_and_oeis.json` (PASS), `hcd_fail_wrong_formula.json` (FAIL)
- Protocol ref: `mapping_protocol_ref.json`

## Checks

| Check | Meaning |
|-------|---------|
| HCD_1 | schema version matches |
| HCD_EULER | fixture rows recompute \(V,E,F,b,e,d\), with \(d=E\) and Euler value 1 |
| HCD_OEIS | Catalan \(k=2\) row matches A000108 |
| HCD_FUSS | Fuss \(k=3,4,5\) rows match A001764, A002293, A002294 |
| HCD_SINGLE_DIAGONAL | single-type rows satisfy \(b=(k-1)e+1\) |
| HCD_D1_DISJOINT | exhaustive single-type sweep has no \(b=e\) hit |
| HCD_SRC | source attribution present |
| HCD_WITNESS | witness kinds cover Euler, OEIS, and D1-disjoint checks |
| HCD_F | fail ledger well-formed |

## Cross-references

- Spec: `docs/specs/WILDBERGER_CERT_BATCH_SPEC.md`
- Theory: `docs/theory/QA_HYPER_CATALAN_DIAGONAL.md`
- Template family: `docs/families/217_qa_fuller_ve_diagonal_decomposition_cert.md`

## Open questions (not in this cert)

- No new mathematical claim is introduced here; this cert packages the pre-verified mapping and OEIS witnesses.
