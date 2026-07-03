# [514] QA Longitudinal Scalar EM Boundary Certificate

**Schema**: `QA_LONGITUDINAL_SCALAR_EM_BOUNDARY_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_longitudinal_scalar_em_boundary_cert_v1/`
**Status**: PASS
**Added**: 2026-07-03

## Purpose

This is a representation-boundary cert after `[513]`. It answers the
longitudinal/scalar EM question without taking Heaviside, Hertz, or Gibbs
reductions as premises.

Required statement:

```text
QA does not use Heaviside-Hertz-Gibbs vector reduction as a premise for the longitudinal/scalar EM question; scalar and longitudinal terms are admitted only as certified carrier, source, boundary, media, gauge, or observer-projection structure.
```

Allowed: scalar-potential carriers; longitudinal source components;
boundary-constrained longitudinal components; media/constitutive longitudinal
components; gauge or observer-projection longitudinal terms.

Rejected: an extra source-free vacuum scalar radiation mode; longitudinal/free
energy; Bearden/Pond/SVP scalar energy; scalar-potential equals physical-field
collapse; transverse-only Maxwell premise; physical electromagnetism beyond
`[513]`.

## Source Anchor

- James Clerk Maxwell (1865), "A Dynamical Theory of the Electromagnetic
  Field," *Philosophical Transactions of the Royal Society* 155:459-512.
- Allen Hatcher, *Algebraic Topology* (2002), Ch. 2,
  ISBN 978-0-521-79540-1.
- Alain Bossavit, *Computational Electromagnetism* (1998),
  ISBN 978-0-12-118710-1.

## Validator Checks

| Check | Meaning |
| --- | --- |
| `LSE_1` | Depends on `[513]`, `[510]`, `[511]`, and `[507]` context, while preserving `[513]`'s exact bounded Maxwell phrase and rejecting Whittaker as derivation premise. |
| `LSE_2` | Source lineage is Maxwell 1865 plus QA finite cochain exterior calculus; Heaviside/Hertz/Gibbs reductions must not be premises. |
| `LSE_3` | Representation-boundary statement matches the required phrase exactly. |
| `LSE_4` | Allows only constrained scalar/longitudinal roles and rejects free vacuum scalar-mode and energy overclaims. |
| `LSE_5` | Claim policy is representation-boundary only, with no new physical radiation mode or unbounded Maxwell claim. |
| `LSE_6` | Negative evidence explicitly rejects transverse-only, Heaviside, Hertz, Gibbs, hidden medium, free scalar energy, and potential-field equivalence assumptions. |
| `LSE_7` | Floats are rejected anywhere in the payload. |

## Fixtures

| Fixture | Expected | Purpose |
| --- | --- | --- |
| `pass_lse_representation_boundary.json` | PASS | Correct representation-boundary stance. |
| `fail_lse_wrong_bounded_statement.json` | FAIL `LSE_1` | Rejects drift from `[513]`'s bounded Maxwell phrase. |
| `fail_lse_heaviside_premise.json` | FAIL `LSE_2` | Rejects Heaviside vector reduction as a premise. |
| `fail_lse_missing_boundary_phrase.json` | FAIL `LSE_3` | Rejects vague scalar-longitudinal approval. |
| `fail_lse_free_vacuum_scalar_mode.json` | FAIL `LSE_4` | Rejects extra source-free vacuum scalar mode. |
| `fail_lse_physical_radiation_overclaim.json` | FAIL `LSE_5` | Rejects new physical radiation-mode overclaim. |
| `fail_lse_missing_negative_evidence.json` | FAIL `LSE_6` | Rejects missing transverse-only guardrail. |
| `fail_lse_float_leak.json` | FAIL `LSE_7` | Rejects observer numeric leakage. |

## Family Relationships

- Sits after `[513]` bounded Maxwell derivation assembly.
- References `[507]` Whittaker two-scalar-potential bridge only as compatibility
  context, never as derivation premise.
- Clarifies that vector formulations may appear as observer notation but are
  not allowed to decide the scalar/longitudinal question.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_longitudinal_scalar_em_boundary_cert_v1/qa_longitudinal_scalar_em_boundary_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":1,"n_fail":7}
```
