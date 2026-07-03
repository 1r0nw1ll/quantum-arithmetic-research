# [509] QA Field 2-Form Bianchi Certificate

**Schema**: `QA_FIELD_2FORM_BIANCHI_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_field_2form_bianchi_cert_v1/`
**Status**: PASS
**Added**: 2026-07-03

## Purpose

This is M1 of the QA Maxwell derivation program
(`docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`). It certifies the first
Maxwell-shaped identity over the exact QA carrier substrate:

```text
F = delta(A)
delta(F) = 0
```

Here `A` is a declared integer edge-potential 1-cochain, `F` is the recomputed
integer face-field 2-cochain, and `delta(F)` is checked exactly on declared
volumes.

This cert allows the narrow wording "QA derives the homogeneous Maxwell/Bianchi
identity for exact finite field carriers." It does not derive full Maxwell,
does not prove inhomogeneous Maxwell, does not construct a Hodge star or
constitutive law, does not derive source laws, does not prove electromagnetism,
does not claim physical fields, and does not import Whittaker operators.

## Source Anchor

Primary mathematical anchor:

- Allen Hatcher, *Algebraic Topology* (2002), Ch. 2, boundary operator,
  ISBN 978-0-521-79540-1.

QA context:

- `[508]` QA Discrete Exterior Nilpotency Cert.
- `docs/specs/QA_MAXWELL_DERIVATION_PROGRAM.md`, M1.

## Validator Checks

| Check | Meaning |
| --- | --- |
| `BIA_1` | Claim policy allows exact field 2-form Bianchi/homogeneous Maxwell wording and rejects full-Maxwell, source, Hodge, Whittaker, physical-field, and scalar-wave-energy overclaims. |
| `BIA_2` | Vertex labels are positive integer QA labels. |
| `BIA_3` | Edges are positive-labeled oriented pairs of declared vertices, with no loops. |
| `BIA_4` | Faces have positive labels and signed boundary terms using declared edges. |
| `BIA_5` | Volumes have positive labels and signed boundary terms using declared faces. |
| `BIA_6` | `edge_potential_A` assigns every edge exactly once with integer values. |
| `BIA_7` | Optional `declared_field_F` assigns every face exactly once and matches the recomputed `delta(A)`. |
| `BIA_8` | `delta(F)` is exactly zero on every declared volume. |

## Fixtures

| Fixture | Expected | Purpose |
| --- | --- | --- |
| `pass_bia_tetrahedron_field.json` | PASS | Tetrahedron boundary with integer edge potential; recomputed face field satisfies `delta(F)=0`. |
| `fail_bia_full_maxwell_overclaim.json` | FAIL `BIA_1` | Rejects claiming full Maxwell derivation. |
| `fail_bia_bad_declared_field.json` | FAIL `BIA_7` | Rejects a declared face field that does not equal `delta(A)`. |
| `fail_bia_open_volume_boundary.json` | FAIL `BIA_8` | Rejects a volume boundary whose signed field sum does not vanish. |
| `fail_bia_missing_edge_potential.json` | FAIL `BIA_6` | Rejects an edge-potential cochain missing an edge assignment. |

## Family Relationships

- Builds directly on `[508]`, which certified the finite chain/cochain
  nilpotency substrate.
- Establishes the homogeneous/Bianchi side only.
- Leaves the inhomogeneous side blocked on the future M2 Hodge/constitutive
  boundary and M3/M4 source-recovery steps.
- Independent of `[507]` Whittaker two-scalar-potential bridge, which remains
  a compatibility result rather than a Maxwell derivation root.

## Verification

Standalone:

```bash
python3 qa_alphageometry_ptolemy/qa_field_2form_bianchi_cert_v1/qa_field_2form_bianchi_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":1,"n_fail":4}
```
