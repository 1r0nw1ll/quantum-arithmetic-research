# [112] QA Obstruction-Compiler Bridge

**Family ID**: 112
**Schema**: `QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; cross-references [111] and [106]
**Directory**: `qa_alphageometry_ptolemy/qa_obstruction_compiler_bridge/`

---

## Purpose

[112] is the first **bridge family** in the QA certificate ecosystem. It connects the two previously parallel halves of the theory:

- **Arithmetic side**: [111] proved that for inert prime p, residues with v_p=1 are forbidden by the norm form.
- **Control side**: [106] defines a generic plan→control compilation law; [105] and [110] are its physical instances.

The bridge theorem states:

> **Obstruction-Compiler Bridge Theorem**: If a target arithmetic class r satisfies v_p(r) = 1 (forbidden by [111]), then no valid PASS cert from the plan-control compiler family [106] — or its domain instances — may assert r as a reachable control target.

This is not merely a conjunction of two independent facts. It is a **causal** connection: arithmetic impossibility from the norm-form structure *directly prevents* control-theoretic reachability.

---

## What "Arithmetic Class of a Target" Means

In the QA framework, every state has an arithmetic fingerprint: the residue class of f(b,e) = b²+be−e² mod p^k for the representative state pair (b,e). If that fingerprint falls in the forbidden zone (v_p=1), no physical state with that fingerprint can exist as a valid QA orbit point. Therefore no control sequence — however cleverly designed — can drive a system to it.

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from` is `'QA_CORE_SPEC.v1'` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope` is `'family_extension'` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `gate_policy_respected` ⊇ [0..5] | `GATE_POLICY_INCOMPATIBLE` |
| B1 | `arithmetic_family_ref.schema_version` is `'QA_AREA_QUANTIZATION_PK_CERT.v1'` | `ARITHMETIC_FAMILY_MISMATCH` |
| B2 | `control_family_ref.schema_version` is `'QA_PLAN_CONTROL_COMPILER_CERT.v1'` | `CONTROL_FAMILY_MISMATCH` |
| B3 | `modulus == p^k` | `MODULUS_MISMATCH` |
| B4 | p is inert: Legendre(5,p) = −1 | `PRIME_NOT_INERT` |
| B5 | `target_arithmetic_class` in {0,...,modulus−1} | `TARGET_OUT_OF_RANGE` |
| B6 | `obstruction_verdict` matches computed v_p(r) | `OBSTRUCTION_VERDICT_WRONG` |
| B7 | `claimed_reachable` consistent with `obstruction_verdict` | `FORBIDDEN_TARGET_REACHABILITY_CLAIM` |

B7 is the key check: if `obstruction_verdict = OBSTRUCTION_PRESENT` and `claimed_reachable = true`, the cert is in direct violation of the bridge theorem.

---

## Certified Fixtures

### `bridge_pass_forbidden_class_3.json`

r=3 (mod 9), p=3, k=2. v₃(3)=1 → OBSTRUCTION_PRESENT. claimed_reachable=false.

The presenting party correctly recognizes the obstruction and suppresses the reachability claim. Cert confirms this is consistent. **PASS.**

### `bridge_pass_valid_class_4.json`

r=4 (mod 9), p=3, k=2. v₃(4)=0 → OBSTRUCTION_ABSENT. claimed_reachable=true.

No arithmetic barrier. Arithmetic does not block a control cert from targeting class 4. (Actual reachability still requires a valid [106] PASS cert with a witnessed plan.) **PASS.**

### `bridge_fail_reachability_claim_ignored.json`

r=6 (mod 9), p=3, k=2. v₃(6)=1 → OBSTRUCTION_PRESENT. claimed_reachable=true.

The presenting party ignores the arithmetic obstruction and asserts reachability. The bridge cert catches this:

**Fail type**: `FORBIDDEN_TARGET_REACHABILITY_CLAIM`. No element of Z[φ]/9Z[φ] has norm ≡ 6 mod 9. Any plan/control cert claiming otherwise would be structurally invalid.

---

## Architecture Position

```
[107] QA_CORE_SPEC.v1 (kernel)
    │
    ├── [108] QA_AREA_QUANTIZATION_CERT.v1      (arithmetic, mod-9 specific)
    │
    ├── [111] QA_AREA_QUANTIZATION_PK_CERT.v1   (arithmetic, general inert p^k)
    │                   ↓ references
    │
    ├── [112] QA_OBSTRUCTION_COMPILER_BRIDGE     ← THIS FAMILY
    │                   ↑ references
    ├── [106] QA_PLAN_CONTROL_COMPILER_CERT.v1  (control, generic)
    │         ├── [105] cymatics domain instance
    │         └── [110] seismology domain instance
    │
    └── [109] QA_INHERITANCE_COMPAT_CERT.v1     (certifies all edges above)
```

[112] is the first family in the graph that has two domain-authority references ([111] and [106]) rather than a single parent. This "join" structure is unique: [112] does not inherit from [111] or [106] — it *cross-references* them as domain authorities while inheriting gate compliance from [107].

---

## Running

```bash
# Self-test (3 fixtures)
python qa_obstruction_compiler_bridge/qa_obstruction_compiler_bridge_validate.py --self-test

# Single cert
python qa_obstruction_compiler_bridge/qa_obstruction_compiler_bridge_validate.py \
  --file qa_obstruction_compiler_bridge/fixtures/bridge_pass_forbidden_class_3.json

# Full meta-validator
python qa_meta_validator.py
```
