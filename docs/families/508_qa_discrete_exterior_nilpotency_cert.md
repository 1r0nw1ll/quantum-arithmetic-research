# [508] QA Discrete Exterior Nilpotency Certificate

**Schema**: `QA_DISCRETE_EXTERIOR_NILPOTENCY_CERT.v1`  
**Family dir**: `qa_alphageometry_ptolemy/qa_discrete_exterior_nilpotency_cert_v1/`  
**Status**: PASS  
**Added**: 2026-07-03

## Purpose

This is M0 of the QA Maxwell derivation program
(`docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`). It certifies the exact finite
chain/cochain identity needed before any honest Maxwell-derivation work can
start:

```text
boundary(boundary(face)) = 0
delta(delta(f)) = 0
```

The cert is intentionally non-physical. It does not derive Maxwell equations,
does not prove electromagnetism, does not claim physical fields, does not use
Whittaker operators, and does not cross the observer boundary.

## Source Anchor

Primary mathematical anchor:

- Allen Hatcher, *Algebraic Topology* (2002), Ch. 2, boundary operator,
  ISBN 978-0-521-79540-1.

QA context:

- `docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`, M0.

## Validator Checks

| Check | Meaning |
| --- | --- |
| `DN_1` | Claim policy rejects Maxwell/electromagnetism/physical-field overclaims and allows only discrete nilpotency. |
| `DN_2` | Vertex labels are positive integer QA labels; zero is rejected. |
| `DN_3` | Edges are positive-labeled oriented pairs of declared vertices, with no loops. |
| `DN_4` | Faces have positive labels and signed boundary terms using declared edges. |
| `DN_5` | `boundary(boundary(face))` cancels exactly at every vertex for every face. |
| `DN_6` | Declared 0-cochain witnesses assign every vertex exactly once with integer values. |
| `DN_7` | `delta(delta(f))` is exactly zero on every face for every 0-cochain witness. |

## Fixtures

| Fixture | Expected | Purpose |
| --- | --- | --- |
| `pass_dn_two_triangles_square.json` | PASS | Square split into two oriented triangles; boundary and coboundary nilpotency both cancel exactly. |
| `fail_dn_overclaims_maxwell.json` | FAIL `DN_1` | Rejects claiming Maxwell derivation from this combinatorial cert. |
| `fail_dn_zero_vertex_label.json` | FAIL `DN_2` | Rejects a zero vertex label. |
| `fail_dn_open_boundary.json` | FAIL `DN_5` | Rejects a face whose signed edge boundary does not close. |
| `fail_dn_incomplete_cochain.json` | FAIL `DN_6` | Rejects a 0-cochain witness missing a vertex assignment. |

## Family Relationships

- First concrete cert in `docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`.
- Precedes any future M1 Bianchi/homogeneous-Maxwell cert.
- Independent of `[507]` Whittaker two-scalar-potential bridge. `[507]`
  remains a compatibility bridge, not a Maxwell derivation root.

## Verification

Standalone:

```bash
python3 qa_alphageometry_ptolemy/qa_discrete_exterior_nilpotency_cert_v1/qa_discrete_exterior_nilpotency_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":1,"n_fail":4}
```
