# [109] QA Inheritance Compat

**Family ID**: 109
**Schema**: `QA_INHERITANCE_COMPAT_CERT.v1`
**Scope**: Meta-family — certifies edges in the QA spec inheritance graph
**Directory**: `qa_alphageometry_ptolemy/qa_inheritance_compat/`

---

## Purpose

Makes inheritance a **first-class certified object** in the QA formal ecosystem.

Before [109], the statement "[108] extends [107]" was a validator convention — enforced inside downstream validators, but not represented as its own verifiable artifact. After [109], each inheritance edge is a cert:

```
parent_family: { family_id: 107, schema_version: "QA_CORE_SPEC.v1", spec_scope: "kernel" }
child_family:  { family_id: 108, schema_version: "QA_AREA_QUANTIZATION_CERT.v1", spec_scope: "family_extension" }
```

The cert carries extracted compatibility evidence (gate policies, failure algebras, invariant names, logging contracts) from both parent and child, and documents all 8 compatibility checks.

---

## Directory Layout

```
qa_inheritance_compat/
  mapping_protocol_ref.json                          Gate 0 protocol reference
  schemas/
    qa_inheritance_compat_cert_v1.schema.json         JSON Schema (draft-07)
  fixtures/
    inherit_pass_107_to_108.json     PASS: [107] kernel → [108] area quantization
    inherit_pass_106_to_105.json     PASS: [106] compiler → [105] cymatics control
    inherit_fail_gate_policy_deleted.json  FAIL: child removes Gate 5 (GATE_POLICY_INCOMPATIBLE)
  qa_inheritance_compat_validate.py  Validator + self-test
```

---

## Validator Checks (IC1–IC8)

| Check | Description | Fail Type | Scope |
|-------|-------------|-----------|-------|
| IC1 | `parent_family.schema_version` is a known QA cert family | `PARENT_CERT_MISSING` | Always |
| IC2 | `child_family.schema_version` is a known QA cert family | `CHILD_CERT_MISSING` | Always |
| IC3 | `declared_inherits_from` == `parent_family.schema_version` | `INVALID_INHERITANCE_EDGE` | Always |
| IC4 | `extracted_child.gate_policy` ⊇ `extracted_parent.gate_policy` | `GATE_POLICY_INCOMPATIBLE` | Always |
| IC5 | `extracted_child.failure_algebra_types` ⊇ `extracted_parent.failure_algebra_types` | `FAILURE_ALGEBRA_BREAKS_PARENT` | When both non-empty |
| IC6 | `extracted_child.logging_required_fields` ⊇ `{move, fail_type, invariant_diff}` | `LOGGING_CONTRACT_INCOMPATIBLE` | When child logging non-empty |
| IC7 | `extracted_child.preserves_invariants_refs` ⊆ `extracted_parent.invariant_names` | `INVARIANT_REFERENCE_UNRESOLVED` | When both non-empty |
| IC8 | `scope_transition` is in `VALID_SCOPE_TRANSITIONS` | `SCOPE_TRANSITION_INVALID` | Always |

IC5–IC7 are **conditionally applied**: domain instances have their own physical fail algebras, logging, and invariants that are not required to mirror the parent.

---

## Valid Scope Transitions

```
kernel            → family_extension
kernel            → domain_kernel
domain_kernel     → family_extension
domain_kernel     → domain_instance
family_extension  → family_extension
family_extension  → domain_instance
```

---

## Certified Edges

### `inherit_pass_107_to_108.json`

**Edge**: [107] `QA_CORE_SPEC.v1` (kernel) → [108] `QA_AREA_QUANTIZATION_CERT.v1` (family_extension)

| Check | Result | Evidence |
|-------|--------|---------|
| IC1 | PASS | `QA_CORE_SPEC.v1` registered as kernel |
| IC2 | PASS | `QA_AREA_QUANTIZATION_CERT.v1` registered as family_extension |
| IC3 | PASS | `declared_inherits_from='QA_CORE_SPEC.v1'` matches parent |
| IC4 | PASS | child `[0..5]` ⊇ parent `[0..5]` |
| IC5 | PASS | child adds `QUADREA_MISMATCH`, `FORBIDDEN_QUADREA_INCORRECT`; parent types all preserved |
| IC6 | PASS | `[move, fail_type, invariant_diff]` present in child |
| IC7 | PASS | `{QA_relation_1, QA_relation_2}` ⊆ parent invariant names |
| IC8 | PASS | `"kernel -> family_extension"` valid |

### `inherit_pass_106_to_105.json`

**Edge**: [106] `QA_PLAN_CONTROL_COMPILER_CERT.v1` (family_extension) → [105] `QA_CYMATIC_CONTROL_CERT.v1` (domain_instance)

| Check | Result | Evidence |
|-------|--------|---------|
| IC1 | PASS | `QA_PLAN_CONTROL_COMPILER_CERT.v1` registered as family_extension |
| IC2 | PASS | `QA_CYMATIC_CONTROL_CERT.v1` registered as domain_instance |
| IC3 | PASS | `declared_inherits_from` matches parent |
| IC4 | PASS | gate policies both `[0..5]` |
| IC5–IC7 | PASS (skipped) | domain_instance has physical fail algebra; parent/child extracted fields empty |
| IC8 | PASS | `"family_extension -> domain_instance"` valid |

### `inherit_fail_gate_policy_deleted.json`

**Edge**: [107] kernel → hypothetical bad version of [108] that strips Gate 5

| Check | Result | Evidence |
|-------|--------|---------|
| IC4 | **FAIL** | child `[0,1,2,3,4]` is missing Gate 5 from parent `[0,1,2,3,4,5]` |

**Fail type**: `GATE_POLICY_INCOMPATIBLE`. Gate 5 closes the replay chain (Merkle hash integrity). Removing it means certs from this hypothetical family cannot be independently verified as unmodified — the inheritance edge is structurally invalid.

---

## The QA Spec Graph

After [109], the inheritance stack can be drawn as certified edges:

```
[107] QA_CORE_SPEC.v1 (kernel)
    │
    ├─── [108] QA_AREA_QUANTIZATION_CERT.v1          ← inherit_pass_107_to_108.json
    │         (family_extension, arithmetic domain)
    │
    └─── [106] QA_PLAN_CONTROL_COMPILER_CERT.v1      ← (inherits_from [107])
              (family_extension, compilation law)
                   │
                   └─── [105] QA_CYMATIC_CONTROL_CERT.v1   ← inherit_pass_106_to_105.json
                             (domain_instance, physical cymatics)
```

Each arrow is a certified `QA_INHERITANCE_COMPAT_CERT.v1` instance that can be independently audited.

---

## Running

```bash
# Self-test (3 fixtures)
python qa_inheritance_compat/qa_inheritance_compat_validate.py --self-test

# Single cert
python qa_inheritance_compat/qa_inheritance_compat_validate.py \
  --file qa_inheritance_compat/fixtures/inherit_pass_107_to_108.json

# Full meta-validator
python qa_meta_validator.py
```
