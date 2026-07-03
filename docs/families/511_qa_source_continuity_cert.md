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

The v1 path consumes `[510]`'s `OBSERVER_BOUNDARY` Hodge verdict. Therefore
`starF` is treated as a declared exact cochain after an observer-boundary Hodge
step, not as a QA-native derived physical field.

This cert does not derive sources, does not prove inhomogeneous Maxwell, does
not derive full Maxwell, does not prove electromagnetism, does not claim
physical charge/current generation, does not claim physical fields, and does
not claim a QA-native Hodge star.

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
| `SRC_1` | Claim policy allows only source continuity and rejects source-generation, inhomogeneous-Maxwell, full-Maxwell, electromagnetism, physical-current, physical-field, and QA-native-Hodge overclaims. |
| `SRC_2` | Dependencies cite `[508]`, `[509]`, `[510]`, and require `[510]`'s `OBSERVER_BOUNDARY` verdict. |
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
| `fail_src_full_maxwell_overclaim.json` | FAIL `SRC_1` | Rejects claiming full Maxwell derivation at M3. |
| `fail_src_wrong_hodge_verdict.json` | FAIL `SRC_2` | Rejects pretending the v1 path consumes a QA-native Hodge verdict. |
| `fail_src_bad_declared_j.json` | FAIL `SRC_7` | Rejects declared source current that does not match `delta(starF)`. |
| `fail_src_missing_starf_assignment.json` | FAIL `SRC_6` | Rejects incomplete `starF` cochain assignment. |
| `fail_src_nonzero_continuity.json` | FAIL `SRC_8` | Rejects nonzero `delta(J)` on a declared 5-cell. |

## Family Relationships

- Builds on `[508]`, `[509]`, and `[510]`.
- Establishes continuity only after `[510]`'s observer-boundary Hodge gate.
- Leaves inhomogeneous Maxwell recovery to M4, and only as conditional recovery
  unless a future cert supplies QA-native Hodge/source generation evidence.

## Verification

Standalone:

```bash
python3 qa_alphageometry_ptolemy/qa_source_continuity_cert_v1/qa_source_continuity_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":1,"n_fail":5}
```
