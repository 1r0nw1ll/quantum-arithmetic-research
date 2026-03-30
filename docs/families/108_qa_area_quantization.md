# [108] QA Area Quantization

**Family ID**: 108
**Schema**: `QA_AREA_QUANTIZATION_CERT.v1`
**Scope**: `family_extension` (extends `QA_CORE_SPEC.v1` — [107])
**Directory**: `qa_alphageometry_ptolemy/qa_area_quantization/`

---

## Purpose

This family is the **first downstream inheritance proof** of [107] QA Core Spec Kernel. It certifies the discrete quadrea spectrum of the Q(√5) norm form over the kernel's state space — a concrete number-theoretic result that cannot be stated without the kernel's generator/invariant framework.

**The core result**: the norm form `f(b,e) = b² + be - e²` over Z/9Z takes only the values `{0,1,2,4,5,7,8}`. The values `{3,6}` are structurally forbidden.

**Why**: 3 is *inert* in Z[φ] (the Legendre symbol (5/3) = (2/3) = −1). For b,e not both divisible by 3, checking all residue pairs shows f(b,e) ≢ 0 mod 3. Therefore f(b,e) ≡ 0 mod 3 iff 3|b and 3|e, which forces f ≡ 0 mod 9. The values 3 and 6 (divisible by 3 but not 9) are unreachable.

This is a genuine theorem: the quadrea spectrum of the QA tuple system is governed by the splitting behaviour of 3 in the ring of integers of Q(√5).

---

## Directory Layout

```
qa_area_quantization/
  mapping_protocol_ref.json                     Gate 0 protocol reference
  schemas/
    qa_area_quantization_cert_v1.schema.json    JSON Schema (draft-07)
  fixtures/
    area_quant_pass_mod9.json                   PASS: mod-9 spectrum {0,1,2,4,5,7,8}
    area_quant_fail_wrong_kernel_ref.json        FAIL: inherits_from='QA_CORE_SPEC.v0'
  qa_area_quantization_validate.py              Validator + self-test
```

---

## Validator Checks

### Inheritance Checks (IH1–IH4)

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from` == `"QA_CORE_SPEC.v1"` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope` == `"family_extension"` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `core_kernel_compatibility.gate_policy_inherited` == `[0,1,2,3,4,5]` | `GATE_POLICY_INCOMPATIBLE` |
| IH4 | `failure_algebra.types` ⊇ kernel types `{OUT_OF_BOUNDS, PARITY, INVARIANT_BREAK, ZERO_DENOMINATOR}` | `FAILURE_ALGEBRA_BREAKS_KERNEL` |

### Domain Checks (AQ1–AQ2)

| Check | Description | Fail Type |
|-------|-------------|-----------|
| AQ1 | `quadrea_claim.spectrum` matches computed `{f(b,e) mod m : b,e ∈ Z/mZ}` | `QUADREA_MISMATCH` |
| AQ2 | `quadrea_claim.forbidden_values` == complement of spectrum in `{0..m-1}` | `FORBIDDEN_QUADREA_INCORRECT` |

The validator computes the quadrea spectrum directly (O(m²) loop, ≤1ms for m≤100).

---

## Failure Algebra

| Fail Type | Meaning |
|-----------|---------|
| `INVALID_KERNEL_REFERENCE` | `inherits_from` points to a non-existent or superseded kernel version |
| `SPEC_SCOPE_MISMATCH` | `spec_scope` is not `"family_extension"` |
| `GATE_POLICY_INCOMPATIBLE` | `gate_policy_inherited` ≠ `[0,1,2,3,4,5]` |
| `FAILURE_ALGEBRA_BREAKS_KERNEL` | `failure_algebra.types` does not include all four kernel fail types |
| `QUADREA_MISMATCH` | Claimed spectrum ≠ computed spectrum |
| `FORBIDDEN_QUADREA_INCORRECT` | Claimed forbidden values ≠ complement of actual spectrum |

---

## The Mod-9 Quadrea Theorem

**Theorem** (witnessed by `area_quant_pass_mod9.json`):

Let `f: Z/9Z × Z/9Z → Z/9Z` be defined by `f(b,e) = b² + be - e² mod 9`. Then:

```
Im(f) = {0, 1, 2, 4, 5, 7, 8}
```

The values `{3, 6}` are forbidden. Equivalently: `f(b,e) ≡ 0 (mod 3)` if and only if `3 | b` and `3 | e`.

**Proof sketch**: For `b, e ∈ {0,1,2} mod 3` with not both zero:
- b≡1, e≡0: f≡1; b≡2, e≡0: f≡1; b≡0, e≡1: f≡−1≡2; b≡0, e≡2: f≡−4≡2
- b≡1, e≡1: f≡1; b≡1, e≡2: f≡−1≡2; b≡2, e≡1: f≡5≡2; b≡2, e≡2: f≡4≡1

In all cases, `f ≢ 0 mod 3`. Therefore `f ≡ 0 mod 3` requires `3|b` and `3|e`, forcing `f = 9(b'²+b'e'−e'²) ≡ 0 mod 9`. QED.

**Number-theoretic interpretation**: 3 is inert in Z[φ] = Z[(1+√5)/2] because the minimal polynomial x²−x−1 is irreducible mod 3 (discriminant 5 ≡ 2 mod 3 is a non-residue). The norm form N(b+eφ) = b²+be−e² is the algebraic norm in Q(√5). Inertness of 3 means the norm cannot take values that are multiples of 3 but not 9 — exactly {3,6}.

---

## Inheritance Pattern

This family demonstrates the **two-layer cert model**:

```
QA_CORE_SPEC.v1  [107]         ← kernel: state_space + generators + invariants
        ↓
QA_AREA_QUANTIZATION_CERT.v1 [108]  ← extension: inherits_from + compatibility block + domain claim
```

Every family_extension cert must carry:
- `inherits_from: "QA_CORE_SPEC.v1"`
- `spec_scope: "family_extension"`
- `core_kernel_compatibility` block (6 fields)
- Full gate sequence `[0,1,2,3,4,5]`
- Failure algebra that extends (not replaces) the kernel's four fail types

---

## Running

```bash
# Self-test (2 fixtures)
python qa_area_quantization/qa_area_quantization_validate.py --self-test

# Single cert
python qa_area_quantization/qa_area_quantization_validate.py \
  --file qa_area_quantization/fixtures/area_quant_pass_mod9.json

# Full meta-validator
python qa_meta_validator.py
```
