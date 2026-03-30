# [121] QA Engineering Core Cert

**Family ID**: 121
**Schema**: `QA_ENGINEERING_CORE_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]
**Directory**: `qa_alphageometry_ptolemy/qa_engineering_core_cert/`
**Intended audience**: engineers mapping classical control/dynamical-systems analysis to QA

---

## Purpose

[121] certifies that a **classical engineering system** (state-space model, stability conditions,
controllability claim) maps validly to a QA specification and that **arithmetic obstructions are
not silently ignored** by classical controllability analysis.

The key contribution (EC11) is the arithmetic override: even when the Kalman rank condition
certifies full controllability, a target state with `v_p(r)=1` for an inert prime `p` is
arithmetically unreachable. Classical linear algebra cannot detect this. QA can.

---

## Validator Checks

### Inheritance checks (IH)

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from == 'QA_CORE_SPEC.v1'` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope == 'family_extension'` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `gate_policy_respected == [0,1,2,3,4,5]` | `GATE_POLICY_INCOMPATIBLE` |

### Engineering core checks (EC)

| Check | Description | Fail Type |
|-------|-------------|-----------|
| EC1 | All `state_encoding` entries have `1 ≤ b,e ≤ modulus` | `STATE_ENCODING_INVALID` |
| EC2 | All transitions have a non-empty generator name | `TRANSITION_NOT_GENERATOR` |
| EC3 | All failure modes map to a QA canonical fail type | `FAILURE_TAXONOMY_INCOMPLETE` |
| EC4 | `target_condition.orbit_family ∈ {singularity, satellite, cosmos}` | `TARGET_NOT_ORBIT_FAMILY` |
| EC5 | Declared `orbit_family` for each state matches recomputed `f(b,e)` valuation | `ORBIT_FAMILY_CLASSIFICATION_FAILURE` |
| EC6 | `stability_claim.lyapunov_function` is non-empty and mentions a QA invariant | `LYAPUNOV_QA_MISMATCH` |
| EC7 | `stability_claim.orbit_contraction_factor < 1.0` | `LYAPUNOV_QA_MISMATCH` |
| EC8 | `equilibrium_state` resolves to a state with `orbit_family == singularity` | `LYAPUNOV_QA_MISMATCH` |
| EC9 | `reachability_witness` present when `classical_controllability == full_rank` | `CONTROLLABILITY_QA_MISMATCH` |
| EC10 | `minimality_witness` present when `optimization_claim` is present | `CONTROLLABILITY_QA_MISMATCH` |
| EC11 | `obstruction_check.obstructed` agrees with recomputed `v_p(target_r)` for inert primes | `ARITHMETIC_OBSTRUCTION_IGNORED` |

---

## Certified Fixtures

### `engineering_core_pass_spring_mass.json`

Spring-mass-damper mapped to mod-9 QA. Three distinguishable regimes:

| State | `(b, e)` | Orbit family | Physical regime |
|-------|----------|--------------|-----------------|
| `still` | (9, 9) | singularity | no oscillation (fixed point) |
| `transient` | (3, 6) | satellite | damped startup |
| `steady_oscillation` | (1, 2) | cosmos | limit cycle |

Generators `excite` (inject energy) and `tune` (adjust frequency) drive the canonical path
`still → transient → steady_oscillation` in k=2 steps. Classical Lyapunov V decreasing maps to
orbit contraction `ρ = 0.001582 < 1`. Target `r = b·e = 1·2 = 2`, `v₃(2) = 0` — not obstructed.
All EC1–EC11 pass. **PASS.**

### `engineering_core_fail_arithmetic_obstruction.json`

Same system, but target `steady_oscillation` encoded at `(b=1, e=3)`, giving `r = 1·3 = 3`.
Since 3 is inert in Z[φ] and `v₃(3) = 1`, the target is arithmetically forbidden — no generator
sequence can reach it. The cert incorrectly declares `obstructed: false`. The validator
recomputes `v₃(3) = 1` and flags `ARITHMETIC_OBSTRUCTION_IGNORED`.

Classical Kalman rank analysis certifies full controllability; EC11 catches the obstruction
invisible to that analysis. **FAIL.**

### `engineering_core_fail_invalid_encoding.json`

State `still` encoded as `(b=0, e=9)`. Zero is excluded from the QA domain `{1,...,9}`.
A common mistake when translating classical state-space models: C/Python arrays index from 0,
but QA uses `{1,...,N}`. The validator recomputes `b=0 < 1` and flags `STATE_ENCODING_INVALID`.
**FAIL.**

---

## Architecture Position

```
[107] QA_CORE_SPEC.v1 (kernel)
  ├── [108] QA_AREA_QUANTIZATION_CERT.v1    ← mod-9 forbidden quadreas {3,6}
  ├── [111] QA_INERT_PRIME_AREA_QUANT       ← v_p(r)=1 forbidden theorem
  │     └── ... obstruction spine [112]→[115]→[116]
  ├── [106] QA_PLAN_CONTROL_COMPILER_CERT.v1
  │     └── ... control spine [105],[110]→[117]→[118]→[119]→[120]
  └── [121] QA_ENGINEERING_CORE_CERT.v1     ← classical engineering → QA mapping
```

[121] is a **sibling** to the obstruction and control spines, not a leaf of either. It shows
that classical engineering concepts (Lyapunov stability, Kalman controllability, state-space
models) each have QA counterparts, and that the arithmetic obstruction (EC11) is the novel
insight that classical analysis misses.

---

## The EC11 Arithmetic Override

The key theorem informally:

> Classical Kalman rank = full does NOT imply QA reachability.
> If `v_p(b·e) = 1` for any inert prime `p` of the modulus, the target is arithmetically
> unreachable regardless of the classical controllability certificate.

This is not a numerical precision issue or an edge case — it is a structural consequence of the
inert prime factorisation of Z[φ]/mZ[φ]. The orbit family check (EC5) may pass (the target is a
valid cosmos state), yet EC11 overrides reachability.

Inert primes by modulus:
- mod 9: `{3}`
- mod 24: `{3, 7}`

---

## Running

```bash
# Self-test (JSON output, 3/3 fixtures)
python qa_engineering_core_cert/qa_engineering_core_cert_validate.py --self-test

# Demo (human-readable)
python qa_engineering_core_cert/qa_engineering_core_cert_validate.py --demo

# Single cert
python qa_engineering_core_cert/qa_engineering_core_cert_validate.py \
  --cert qa_engineering_core_cert/fixtures/engineering_core_pass_spring_mass.json

# Full meta-validator
python qa_meta_validator.py
```
