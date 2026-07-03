# [513] QA Maxwell Derivation Certificate

**Schema**: `QA_MAXWELL_DERIVATION_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_maxwell_derivation_cert_v1/`
**Status**: PASS
**Added**: 2026-07-03

## Purpose

This is M5 of the QA Maxwell derivation program. It assembles the certified
homogeneous half `[509]` and inhomogeneous half `[512]` under `[510]`'s
`QA_NATIVE` Hodge branch and `[511]`'s source-continuity/source-carrier
evidence.

Required bounded claim:

```text
QA derives Maxwell only within the stated carrier, boundary, source, metric, unit, and observer-projection conventions certified here.
```

This cert does not claim unbounded Maxwell, physical electromagnetism, physical
fields, physical source generation, Whittaker-derived Maxwell, free energy, or
scalar-wave-energy physics.

## Source Anchor

- Allen Hatcher, *Algebraic Topology* (2002), Ch. 2,
  ISBN 978-0-521-79540-1.
- Alain Bossavit, *Computational Electromagnetism* (1998),
  ISBN 978-0-12-118710-1.
- James Clerk Maxwell (1865), "A Dynamical Theory of the Electromagnetic
  Field," *Philosophical Transactions of the Royal Society* 155:459-512.

## Validator Checks

| Check | Meaning |
| --- | --- |
| `MXD_1` | Dependencies require `[509]`, `[510]`, `[511]`, `[512]`, with `QA_NATIVE` Hodge and `QA_NATIVE` inhomogeneous recovery branches. |
| `MXD_2` | Claim policy allows only bounded full-Maxwell assembly and rejects unbounded/physical/Whittaker/free-energy overclaims. |
| `MXD_3` | Bounded derivation statement matches the required phrase exactly. |
| `MXD_4` | Component claims tie homogeneous, Hodge, source continuity, and inhomogeneous halves to the registered certs. |
| `MXD_5` | Carrier, boundary, source, metric, unit, and observer-projection conventions are explicit and within Theorem NT crossing budget. |
| `MXD_6` | Negative evidence rejects hidden Maxwell assumptions, Whittaker premise, hidden Hodge, premature floats/trig, physical overextension, and scalar-wave-energy claims. |
| `MXD_7` | Floats are rejected anywhere in the payload. |

## Fixtures

| Fixture | Expected | Purpose |
| --- | --- | --- |
| `pass_mxd_bounded_native.json` | PASS | Bounded full-Maxwell assembly over the QA-native branch. |
| `fail_mxd_observer_boundary_branch.json` | FAIL `MXD_1` | Rejects full derivation when Hodge branch is observer-boundary. |
| `fail_mxd_unbounded_overclaim.json` | FAIL `MXD_2` | Rejects unbounded Maxwell claim. |
| `fail_mxd_whittaker_premise.json` | FAIL `MXD_2` | Rejects using Whittaker as the derivation premise. |
| `fail_mxd_missing_exact_phrase.json` | FAIL `MXD_3` | Rejects missing bounded derivation wording. |
| `fail_mxd_missing_metric_convention.json` | FAIL `MXD_5` | Rejects missing metric convention. |
| `fail_mxd_float_leak.json` | FAIL `MXD_7` | Rejects hidden float leakage. |

## Family Relationships

- Assembles `[509]` and `[512]`.
- Requires `[510]` `QA_NATIVE` and `[511]` source-carrier continuity evidence.
- Closes the M0-M5 Maxwell derivation program only in the bounded certified
  sense stated above.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_maxwell_derivation_cert_v1/qa_maxwell_derivation_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":1,"n_fail":6}
```
