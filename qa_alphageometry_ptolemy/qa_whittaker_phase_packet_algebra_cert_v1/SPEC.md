# QA Whittaker Phase-Packet Algebra Cert v1 Spec

Candidate family ID: `[275]`

## Claim

For declared finite symbolic phase packets over registered `[273]` exact S2
directions, the validator recomputes:

```text
omega dot x
phase_arg = k * (omega dot x - v*t)
```

as exact `fractions.Fraction` values and checks declared witnesses.

## Rational Encoding

Rational scalar values are JSON objects:

```json
{"num": 1, "den": 2}
```

The denominator must be a positive integer. The pair must be reduced.

Rational vectors are length-3 arrays of rational scalar objects.

## Dependency

`[273]` supplies exact rational S2 packets under:

```text
chart = inverse_stereographic_excluding_south_pole
```

Each declared `omega_packet = [x_num, y_num, z_num, den]` must be present in
the recomputed `[273]` `D_m^(2)` set for the fixture's `m`.

`[274]` is lineage context only in v1.

## Packet Schema

Each packet declares:

```text
packet_id
packet_family
omega_packet
k
v
weight
```

Allowed `packet_family` values:

```text
phase_arg
phase_pair
formal_cos_symbol
formal_sin_symbol
```

`formal_cos_symbol` and `formal_sin_symbol` are symbolic labels only.

## Point Schema

Each evaluation or heldout point declares:

```text
point_id
x
t
split_label
target_packet_ids
phase_witnesses
```

For every `packet_id` in `target_packet_ids`, `phase_witnesses[packet_id]`
must equal the exact recomputed `phase_arg`.

## Rejections

v1 rejects:

- floating trigonometric evaluation;
- numerical approximation pass/fail;
- fitted coefficients;
- missing phase witnesses;
- wrong phase witnesses;
- hidden packet generators;
- Maxwell/EM overclaims;
- scalar-potential physics overclaims;
- full Whittaker theorem overclaims.

## Future Layers

```text
Layer 3.1 v1   exact finite phase-packet algebra
Layer 3.1 v1.1 QA-native discrete/rational trig surrogate
Layer 3.1 v2   observer-side numerical wave-kernel approximation
Layer 4        Whittaker 1904 scalar-potential bridge
Layer 5        Maxwell/scalar-pair reconstruction
```
