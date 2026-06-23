# [498] QA Whittaker Phase-Packet Algebra Cert

## What this is

A Layer 3.1 cert in the Whittaker → QA development ladder. Freezes the exact phase-packet algebra substrate for the Whittaker ladder, using `fractions.Fraction` arithmetic only.

The v1 claim is narrow:

```
phase_arg = k * (omega · x − v·t)
```

where `omega` is an exact rational S2 direction from registered cert **[273]**, and `x`, `t`, `k`, `v`, and packet weights are all exact rational values.

## Claim (narrow)

For declared finite symbolic phase packets with exact rational inputs:
- The validator recomputes `omega · x` and `phase_arg = k * (omega · x − v*t)` as `fractions.Fraction` values
- It checks declared witnesses exactly (no floating-point)
- All composition targets reference only declared packet IDs

Allowed symbolic packet families:

| Family | Description |
|--------|-------------|
| `phase_arg` | Linear phase argument |
| `phase_pair` | Pair of phase arguments |
| `formal_cos_symbol` | Label only — no trig evaluation |
| `formal_sin_symbol` | Label only — no trig evaluation |

## Dependencies

| Cert | Role |
|------|------|
| **[273]** `qa_whittaker_rational_direction_s2_cert_v1` | **Hard dependency** — supplies exact rational S2 directions |
| **[274]** `qa_whittaker_scalar_angular_kernel_sampling_cert_v1` | Lineage context only (Layer 3 progression) |

## Artifacts

| Artifact | Path |
|----------|------|
| Validator | `qa_whittaker_phase_packet_algebra_cert_v1/qa_whittaker_phase_packet_algebra_cert_validate.py` |
| Mapping ref | `qa_whittaker_phase_packet_algebra_cert_v1/mapping_protocol_ref.json` |
| PASS fixture 1 | `fixtures/pass_wppa_single_phase_arg_m3.json` |
| PASS fixture 2 | `fixtures/pass_wppa_two_phase_packet_composition_m5.json` |
| PASS fixture 3 | `fixtures/pass_wppa_formal_cos_sin_pair_symbolic_m5.json` |
| FAIL fixtures | `fixtures/fail_wppa_*.json` (7 cases) |
| Spec | `qa_whittaker_phase_packet_algebra_cert_v1/SPEC.md` |

## How to run

```bash
cd qa_alphageometry_ptolemy/qa_whittaker_phase_packet_algebra_cert_v1
python3 qa_whittaker_phase_packet_algebra_cert_validate.py --self-test
```

Expected: JSON with `"ok": true`.

## Gates

| Gate | Check |
|------|-------|
| `WPPA_1` | Dependency provenance: [273] present/registered, correct slug/chart; lineage context [274] recorded |
| `WPPA_2` | Packet declarations complete (id, family, declared witnesses) |
| `WPPA_3` | `omega · x` and `phase_arg` recomputed exactly as Fraction values; witness match |
| `WPPA_4` | Packet weights are exact rational values (Fraction or int) |
| `WPPA_5` | Target composition references only declared packet IDs |
| `WPPA_6` | Heldout packet identities and phase witnesses match exactly |
| `WPPA_7` | Rejects trig evaluation, numerical approximation, fitted coefficients, forbidden operations |
| `WPPA_8` | Rejects Maxwell/EM/scalar-potential/full-Whittaker theorem overclaims |

## Non-Claims

This cert does **not** claim:
- Numerical approximation of trigonometric functions
- Trig evaluation at runtime
- Spherical quadrature
- Whittaker 1903 theorem proof
- Maxwell/EM derivation
- Scalar-potential physics
- Physical field reconstruction

## QA Axiom Compliance

- **A1**: omega directions are exact rational; state variables are exact rational (Fraction), never 0.0
- **A2**: derived quantities recomputed from (b,e) if present; no independent assignment
- **T2**: all arithmetic is exact Fraction; no float state; trig labels (formal_cos_symbol, formal_sin_symbol) are symbolic, not evaluated — Theorem NT enforced
- **S1**: no `**2` — squared terms written as products
- **S2**: all numeric state is `fractions.Fraction` or `int`; `np.zeros` / `np.random.rand` absent from QA logic
- **T1**: `t` is an exact rational scalar, not a continuous time variable; phase_arg is an algebraic expression

## Primary Sources

- Whittaker, E.T. (1903). On the partial differential equations of mathematical physics. *Math. Annalen* 57:333–355. DOI 10.1007/BF01444290.
- Registered cert [273] `qa_whittaker_rational_direction_s2_cert_v1` — exact rational S2 directions.

## Relation to other certs

- **[273] `qa_whittaker_rational_direction_s2_cert_v1`** — hard dependency; supplies the exact rational omega directions used in WPPA_1 and WPPA_3
- **[274] `qa_whittaker_scalar_angular_kernel_sampling_cert_v1`** — lineage context only; Layer 3.0 of the Whittaker ladder
- **[497] `qa_steinmetz_whittaker_bridge_cert_v1`** — companion cert on the bridge operator side; [498] freezes the phase-packet algebra that the Whittaker ↔ QA bridge operates over

## Scope boundary

**The cert does NOT:**
- Prove the Whittaker 1903 general integral theorem
- Evaluate trigonometric functions numerically
- Perform spherical surface quadrature
- Make any claim about electromagnetic field reconstruction

**The cert DOES:**
- Verify exact rational arithmetic for declared phase packets
- Enforce that trig labels are symbolic only
- Check that composition targets are closed within the declared packet set
- Reject overclaims via WPPA_7 and WPPA_8 gates
