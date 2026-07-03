# [510] QA Hodge Constitutive Boundary Certificate

**Schema**: `QA_HODGE_CONSTITUTIVE_BOUNDARY_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_hodge_constitutive_boundary_cert_v1/`
**Status**: PASS
**Added**: 2026-07-03

## Purpose

This is M2 of the QA Maxwell derivation program
(`docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`). It certifies the boundary
where the Hodge star / constitutive relation enters the chain.

The cert classifies a declared `star_QA` as one of:

- `OBSERVER_BOUNDARY`: metric, orientation, units, and medium parameters are
  explicitly imported observer-side structure.
- `QA_NATIVE`: all required evidence is exact QA-native integer/rational data.
- `INVALID`: hidden imports, missing provenance, unsupported native evidence,
  or overclaiming.

The v1 PASS fixture is intentionally `OBSERVER_BOUNDARY`: it declares exact
rational matrix entries over finite primal/dual face bases, but it also
declares metric signature, orientation, units, and medium parameters as
observer imports. That means a later inhomogeneous Maxwell cert may be a
conditional recovery, not a QA-native full derivation.

This cert does not derive full Maxwell, does not prove inhomogeneous Maxwell,
does not construct a Hodge star, does not derive source laws, does not prove
electromagnetism, and does not claim physical fields.

## Source Anchor

Primary mathematical anchors:

- Allen Hatcher, *Algebraic Topology* (2002), Ch. 2, boundary operator,
  ISBN 978-0-521-79540-1.
- Alain Bossavit, *Computational Electromagnetism* (1998),
  ISBN 978-0-12-118710-1.

QA context:

- `[508]` QA Discrete Exterior Nilpotency Cert.
- `[509]` QA Field 2-Form Bianchi Cert.
- `docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`, M2.

## Validator Checks

| Check | Meaning |
| --- | --- |
| `HCB_1` | Claim policy allows only Hodge boundary classification and rejects full-Maxwell, inhomogeneous, constructed-Hodge, electromagnetism, and physical-field overclaims. |
| `HCB_2` | Dependencies cite `[508]` and `[509]`; source attribution cites Hatcher and Bossavit. |
| `HCB_3` | Primal and dual face bases are non-empty and have unique labels. |
| `HCB_4` | `star_operator.matrix_entries` are exact rational entries from declared primal faces to declared dual faces. |
| `HCB_5` | Classification is one of `OBSERVER_BOUNDARY`, `QA_NATIVE`, `INVALID` and matches the declared boundary verdict. |
| `HCB_6` | `OBSERVER_BOUNDARY` requires explicit metric/orientation/units/medium imports and blocks full-Maxwell derivation downstream. |
| `HCB_7` | `QA_NATIVE` requires exact native evidence and no observer imports. |
| `HCB_8` | `INVALID` requires explicit invalid reasons. |

## Fixtures

| Fixture | Expected | Purpose |
| --- | --- | --- |
| `pass_hcb_observer_boundary.json` | PASS | Honest observer-boundary Hodge declaration with exact rational matrix entries and explicit imported metric/medium data. |
| `fail_hcb_full_maxwell_overclaim.json` | FAIL `HCB_1` | Rejects claiming full Maxwell derivation at M2. |
| `fail_hcb_missing_observer_import.json` | FAIL `HCB_6` | Rejects an observer-boundary Hodge declaration missing medium parameters. |
| `fail_hcb_bad_matrix_basis.json` | FAIL `HCB_4` | Rejects a star matrix entry referencing an undeclared dual face. |
| `fail_hcb_qa_native_missing_evidence.json` | FAIL `HCB_7` | Rejects a QA-native Hodge declaration without complete native evidence. |
| `fail_hcb_invalid_without_reason.json` | FAIL `HCB_8` | Rejects `INVALID` classification without reasons. |

## Family Relationships

- Builds on `[508]` and `[509]`.
- Establishes that the current Hodge/constitutive object is observer-side in
  the v1 passing witness.
- Leaves full Maxwell blocked. The next valid step is M3/M4 conditional source
  and inhomogeneous recovery under the declared observer-boundary Hodge data.

## Verification

Standalone:

```bash
python3 qa_alphageometry_ptolemy/qa_hodge_constitutive_boundary_cert_v1/qa_hodge_constitutive_boundary_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":1,"n_fail":5}
```
