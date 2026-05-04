# QA Whittaker Phase-Packet Algebra Cert v1

Candidate family ID: `[275]`

Status: standalone unregistered artifact. Do not add to
`qa_meta_validator.py` until this artifact has passed standalone validation
and review.

## Purpose

This cert freezes the exact phase-packet algebra substrate for the Whittaker
ladder.

The v1 claim is narrow:

```text
phase_arg = k * (omega dot x - v*t)
```

where `omega` is an exact rational S2 direction supplied by registered `[273]`,
and `x`, `t`, `k`, `v`, and packet weights are exact rational values.

The validator uses `fractions.Fraction` only for pass/fail arithmetic.

## Dependencies

- Hard dependency: registered `[273]`
  `qa_whittaker_rational_direction_s2_cert_v1`.
- Lineage context: registered `[274]`
  `qa_whittaker_scalar_angular_kernel_sampling_cert_v1`.

`[274]` is recorded as context for the Layer 3 progression, but is not a hard
validator dependency in v1 because this cert does not perform scalar averaging.

## Packet Families

Allowed symbolic packet families:

```text
phase_arg
phase_pair
formal_cos_symbol
formal_sin_symbol
```

`formal_cos_symbol` and `formal_sin_symbol` are labels only. They do not
evaluate trigonometric functions.

## Gates

| Gate | Check |
| --- | --- |
| `WPPA_1` | dependency provenance: `[273]` present/registered, correct slug/chart |
| `WPPA_2` | packet declarations complete |
| `WPPA_3` | `omega dot x` and `phase_arg` recomputed exactly |
| `WPPA_4` | weights are exact rational values |
| `WPPA_5` | target composition references declared packet IDs only |
| `WPPA_6` | heldout packet identities and phase witnesses match |
| `WPPA_7` | rejects trig evaluation, numerical approximation, fitted coefficients |
| `WPPA_8` | rejects Maxwell/EM/scalar-potential/full-Whittaker overclaims |

## Fixtures

PASS:

- `fixtures/pass_wppa_single_phase_arg_m3.json`
- `fixtures/pass_wppa_two_phase_packet_composition_m5.json`
- `fixtures/pass_wppa_formal_cos_sin_pair_symbolic_m5.json`

FAIL:

- `fixtures/fail_wppa_float_trig_in_v1.json`
- `fixtures/fail_wppa_missing_phase_arg.json`
- `fixtures/fail_wppa_wrong_phase_arg_fraction.json`
- `fixtures/fail_wppa_hidden_packet_generator.json`
- `fixtures/fail_wppa_numeric_approximation_claim_in_v1.json`
- `fixtures/fail_wppa_overclaimed_maxwell_em.json`
- `fixtures/fail_wppa_overclaimed_full_whittaker_theorem.json`

## Non-Claims

This cert does not claim numerical approximation, trigonometric evaluation,
spherical quadrature, Whittaker 1903 theorem proof, Maxwell/EM derivation,
scalar-potential physics, or physical field reconstruction.

## Run

```bash
python3 qa_alphageometry_ptolemy/qa_whittaker_phase_packet_algebra_cert_v1/qa_whittaker_phase_packet_algebra_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`.
