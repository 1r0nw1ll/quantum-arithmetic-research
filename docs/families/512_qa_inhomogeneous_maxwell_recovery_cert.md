# [512] QA Inhomogeneous Maxwell Recovery Certificate

**Schema**: `QA_INHOMOGENEOUS_MAXWELL_RECOVERY_CERT.v1`
**Family dir**: `qa_alphageometry_ptolemy/qa_inhomogeneous_maxwell_recovery_cert_v1/`
**Status**: PASS
**Added**: 2026-07-03

## Purpose

This is M4 of the QA Maxwell derivation program. It certifies the exact finite
inhomogeneous equation

```text
delta(starF) = J
```

under declared Hodge verdict, sign convention, unit scale, and observer
projection status.

It has two positive branches:

- `OBSERVER_BOUNDARY`: conditional observer recovery under imported
  constitutive assumptions from `[510]`.
- `QA_NATIVE`: symbolic QA-native inhomogeneous recovery using `[510]`'s native
  Hodge seed and `[511]`'s source-carrier path.

This still does not claim full Maxwell, physical electromagnetism, physical
fields, physical source generation, free energy, or scalar-wave-energy physics.

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
| `IMR_1` | Claim policy allows inhomogeneous recovery while rejecting full Maxwell, electromagnetism, physical fields, source generation, and scalar/free-energy overclaims. |
| `IMR_2` | Dependencies cite `[508]`, `[509]`, `[510]`, `[511]`, and the Hodge verdict is recognized. |
| `IMR_3` | Declared 2/3/4-cell bases are non-empty with unique labels. |
| `IMR_4` | `star_operator` matrix entries are exact rational maps from declared 2-cells to declared 3-cells. |
| `IMR_5` | Every declared 4-cell has a signed boundary over declared 3-cells. |
| `IMR_6` | Recomputed `starF = star_QA(F)` matches declared `starF`. |
| `IMR_7` | Recomputed `J = delta(starF)` matches declared source `J`. |
| `IMR_8` | Sign convention, unit scale, and observer projection status are explicit and branch-consistent. |
| `IMR_9` | Floats are rejected anywhere in the payload. |

## Fixtures

| Fixture | Expected | Purpose |
| --- | --- | --- |
| `pass_imr_observer_conditional.json` | PASS | Conditional observer-boundary recovery of `delta(starF)=J`. |
| `pass_imr_qa_native.json` | PASS | QA-native symbolic recovery of `delta(starF)=J`. |
| `fail_imr_full_maxwell_overclaim.json` | FAIL `IMR_1` | Rejects claiming full Maxwell at M4. |
| `fail_imr_bad_starf.json` | FAIL `IMR_6` | Rejects declared `starF` that does not equal `star_QA(F)`. |
| `fail_imr_bad_j.json` | FAIL `IMR_7` | Rejects declared `J` that does not equal `delta(starF)`. |
| `fail_imr_missing_projection.json` | FAIL `IMR_8` | Rejects missing observer projection conventions. |
| `fail_imr_float_leak.json` | FAIL `IMR_9` | Rejects hidden float leakage. |

## Family Relationships

- Builds on `[508]` nilpotency, `[509]` homogeneous/Bianchi carrier, `[510]`
  Hodge boundary/native seed, and `[511]` source continuity.
- Closes M4 recovery/assembly at exact finite cochain level.
- Leaves M5 as the only remaining place where a full-Maxwell claim can be
  assembled, with explicit boundary/source/metric/unit/projection conventions.

## Verification

```bash
python3 qa_alphageometry_ptolemy/qa_inhomogeneous_maxwell_recovery_cert_v1/qa_inhomogeneous_maxwell_recovery_cert_validate.py --self-test
```

Expected summary:

```json
{"ok":true,"n_pass":2,"n_fail":5}
```
