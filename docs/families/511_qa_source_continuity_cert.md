# [511] QA Source Continuity Certificate

**Schema**: `QA_SOURCE_CONTINUITY_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_source_continuity_cert_v1/`
**Status**: PASS
**Added**: 2026-07-03

## Purpose

This is M3 of the QA Maxwell derivation program
(`docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`). It certifies source/current
continuity as an exact finite cochain identity:

```text
J = delta(starF)
delta(J) = 0
```

The cert now has two positive paths:

- `OBSERVER_BOUNDARY`: consumes `[510]`'s observer-boundary Hodge verdict; this
  remains conditional recovery scaffolding.
- `QA_NATIVE`: consumes `[510]`'s QA-native Hodge verdict and requires explicit
  QA source-carrier evidence (`J` is an exact cochain, no observer source
  imports). This fixes the native source-carrier side of the blocker, while
  still not claiming physical charge/current generation.

This cert does not derive sources, does not prove inhomogeneous Maxwell, does
not derive full Maxwell, does not prove electromagnetism, does not claim
physical charge/current generation, and does not claim physical fields.

## Source Anchor

Primary mathematical anchors:

- Allen Hatcher, *Algebraic Topology* (2002), Ch. 2, boundary operator,
  ISBN 978-0-521-79540-1.
- Alain Bossavit, *Computational Electromagnetism* (1998),
  ISBN 978-0-12-118710-1.

QA context:

- `[508]` QA Discrete Exterior Nilpotency Cert.
- `[509]` QA Field 2-Form Bianchi Cert.
- `[510]` QA Hodge Constitutive Boundary Cert.
- `docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`, M3.

## Validator Checks

| Check | Meaning |
| --- | --- |
| `SRC_1` | Claim policy allows only source continuity; rejects source-generation, inhomogeneous-Maxwell, full-Maxwell, electromagnetism, physical-current, and physical-field overclaims; permits `claims_qa_native_hodge` only on the `QA_NATIVE` dependency branch. |
| `SRC_2` | Dependencies cite `[508]`, `[509]`, `[510]`; accept `[510]`'s `OBSERVER_BOUNDARY` or `QA_NATIVE` verdict; require source-carrier evidence on the native branch. |
| `SRC_3` | Declared 3-cells are unique positive integer labels. |
| `SRC_4` | Declared 4-cells have signed boundaries using declared 3-cells. |
| `SRC_5` | Declared 5-cells have signed boundaries using declared 4-cells. |
| `SRC_6` | `starF_3cochain` assigns every 3-cell exactly once with exact rational values. |
| `SRC_7` | Optional `declared_source_J` assigns every 4-cell exactly once and matches recomputed `delta(starF)`. |
| `SRC_8` | `delta(J)` is exactly zero on every declared 5-cell. |

## Fixtures

| Fixture | Expected | Purpose |
| --- | --- | --- |
| `pass_src_square_boundary.json` | PASS | Closed 5-cell boundary; recomputed `J=delta(starF)` satisfies `delta(J)=0`. |
| `pass_src_qa_native_carrier.json` | PASS | Native Hodge branch with exact QA source-carrier evidence; recomputed `J=delta(starF)` satisfies `delta(J)=0`. |
| `fail_src_full_maxwell_overclaim.json` | FAIL `SRC_1` | Rejects claiming full Maxwell derivation at M3. |
| `fail_src_wrong_hodge_verdict.json` | FAIL `SRC_1` | Rejects a mismatch between a `QA_NATIVE` dependency and claim policy denying QA-native Hodge. |
| `fail_src_bad_declared_j.json` | FAIL `SRC_7` | Rejects declared source current that does not match `delta(starF)`. |
| `fail_src_missing_starf_assignment.json` | FAIL `SRC_6` | Rejects incomplete `starF` cochain assignment. |
| `fail_src_nonzero_continuity.json` | FAIL `SRC_8` | Rejects nonzero `delta(J)` on a declared 5-cell. |

## Family Relationships

- Builds on `[508]`, `[509]`, and `[510]`.
- Establishes continuity after either `[510]`'s observer-boundary gate or its
  QA-native Hodge seed.
- Fixes the native source-carrier evidence side of the blocker. What remains is
  M4/M5: inhomogeneous recovery/assembly with sign/unit/projection conventions.

## Verification

Standalone:

```bash
python3 qa_alphageometry_ptolemy/qa_source_continuity_cert_v1/qa_source_continuity_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":2,"n_fail":5}
```
