# [129] QA Projection Obstruction Cert

**Family ID**: 129  
**Schema**: `QA_PROJECTION_OBSTRUCTION_CERT.v1`  
**Scope**: Family extension of `QA_ENGINEERING_CORE_CERT.v1` [121]  
**Directory**: `qa_alphageometry_ptolemy/qa_projection_obstruction_cert/`  
**Intended audience**: QA researchers analyzing when a lawful native system accumulates debt under representation or realization

---

## Purpose

[129] certifies a distinction that matters for electronic logic and other engineered symbolic systems:

1. **Native symbolic closure**  
   The law may hold exactly in its own native state space.
2. **Representation-basis mismatch**  
   The law may incur debt when rewritten into a different discrete basis such as display codes, Boolean rails, selector covers, or lookup-style topologies.
3. **Physical device realization**  
   Actual device performance is a separate layer. Lack of physical measurements is not the same thing as device failure.

This family exists to stop a common analytical mistake:

> Representation debt is **not** itself proof that the underlying electronic logic is physically bad.

The family makes that distinction machine-checkable.

---

## Schema fields

| Field | Type | Required | Notes |
|---|---|---|---|
| `schema_version` | string | yes | must be `"QA_PROJECTION_OBSTRUCTION_CERT.v1"` |
| `cert_type` | string | yes | must be `"projection_obstruction"` |
| `inherits_from` | string | yes | must be `"QA_ENGINEERING_CORE_CERT.v1"` |
| `spec_scope` | string | yes | must be `"family_extension"` |
| `core_kernel_compatibility.gate_policy_respected` | list[int] | yes | must equal `[0,1,2,3,4,5]` |
| `engineering_context.native_invariants` | list | yes | non-empty invariant packet |
| `native_witness.lawful_invariants_supported` | list[str] | yes | each id must resolve to a declared native invariant |
| `native_witness.verdict` | string | yes | one of `CONSISTENT/PARTIAL/CONTRADICTS/INCONCLUSIVE` |
| `representation_layers` | list | yes | one or more discrete representation / observation layers |
| `physical_realization` | object | yes | separate device layer |
| `overall_verdict` | string | yes | derived from native, representation, and physical layers |
| `fail_ledger` | list | yes | must contain any recomputed obstruction tags |
| `result` | string | yes | validator outcome: `PASS` or `FAIL` |

---

## Validator checks

### Inheritance checks

| Check | Description | Fail type |
|---|---|---|
| IH1 | `inherits_from == 'QA_ENGINEERING_CORE_CERT.v1'` | `INVALID_PARENT_ENGINEERING_REFERENCE` |
| IH2 | `spec_scope == 'family_extension'` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `gate_policy_respected == [0,1,2,3,4,5]` | `GATE_POLICY_INCOMPATIBLE` |

### Projection obstruction checks

| Check | Description | Fail type |
|---|---|---|
| PO1 | `engineering_context.native_invariants` is non-empty | `EMPTY_NATIVE_INVARIANTS` |
| PO2 | native witness references resolve to declared native invariants | `NATIVE_WITNESS_INVALID`, `INVARIANT_REFERENCE_UNRESOLVED` |
| PO3 | each `representation_layer` has valid structure and witnesses | `REPRESENTATION_LAYER_INVALID` |
| PO4 | declared `obstruction_tags` match recomputed representation debt tags | `REPRESENTATION_LEDGER_MISMATCH` |
| PO5 | declared representation-layer verdict matches recomputed debt class | `REPRESENTATION_VERDICT_MISMATCH` |
| PO6 | `physical_realization` has valid structure | `PHYSICAL_LAYER_INVALID` |
| PO7 | physical verdict and device tags match the assessment status / device metrics | `PHYSICAL_LEDGER_MISMATCH`, `PHYSICAL_VERDICT_MISMATCH` |
| PO8 | all recomputed obstruction tags appear in `fail_ledger` | `OBSTRUCTION_LEDGER_REQUIRED` |
| PO9 | `overall_verdict` matches native support plus representation/physical outcomes | `OVERALL_VERDICT_MISMATCH` |

---

## Obstruction vocabulary

### Representation-basis debt

| Tag | Meaning |
|---|---|
| `STATE_SPACE_RESIDUAL` | The observed image uses only a strict subset of native states |
| `COST_INFLATION` | The chosen representation costs more than the stated baseline |
| `SELECTOR_AND_MERGE_DEBT` | The representation requires explicit selectors or merges |
| `TOPOLOGY_PART_COUNT_DEBT` | The realized topology has raw part-count overhead |

### Physical-device debt

| Tag | Meaning |
|---|---|
| `INSUFFICIENT_STABLE_STATES` | The device does not realize enough stable digital states |
| `THRESHOLD_MARGIN_WEAK` | Measured threshold margin is below adequacy threshold |
| `NOISE_MARGIN_WEAK` | Measured noise margin is below adequacy threshold |
| `FANOUT_LIMITED` | Supported fanout is below required fanout |
| `TIMING_UNVERIFIED` | Timing has not been characterized in the measured layer |

Important: a cert may show heavy representation debt while the physical layer remains `INCONCLUSIVE`. That is valid and often the correct conclusion.

---

## Certified fixtures

| File | Result | Purpose |
|---|---|---|
| `ppo_pass_arto_ternary.json` | PASS | Canonical pass case: native ternary arithmetic is lawful, representation debt is explicit, physical realization remains unassessed and therefore `INCONCLUSIVE` |
| `ppo_fail_physical_conflation.json` | FAIL | Negative case: an `UNASSESSED` physical layer is incorrectly declared `CONTRADICTS` with device-failure tags |
| `ppo_fail_bad_invariant_ref.json` | FAIL | Negative case: native witness references an undeclared invariant |

### `ppo_pass_arto_ternary.json`

This fixture formalizes the distinction that motivated the family:

- native layer: lawful
- representation layers: debt-bearing
- physical layer: not yet measured

The result is **overall `PARTIAL`**, not because the native law failed, but because the law is preserved only in its native symbolic space while non-native representations accumulate debt.

---

## Family relationships

```text
[107] QA_CORE_SPEC.v1
  -> [121] QA_ENGINEERING_CORE_CERT.v1
     -> [129] QA_PROJECTION_OBSTRUCTION_CERT.v1
```

- **[107]** provides the core ontology: state space, generators, invariants, failure algebra.
- **[121]** brings engineering mapping and obstruction-aware interpretation into the cert ecosystem.
- **[129]** refines that engineering layer by separating:
  - symbolic/native law,
  - representation-basis debt,
  - physical-device evidence.

This makes [129] a natural family for ternary logic, display encodings, mixed-radix realizations, and any system where a law survives natively but is expensive or distorted in a chosen implementation basis.

---

## Running

```bash
python qa_alphageometry_ptolemy/qa_projection_obstruction_cert/qa_projection_obstruction_cert_validate.py --self-test

python qa_alphageometry_ptolemy/qa_projection_obstruction_cert/qa_projection_obstruction_cert_validate.py \
  --file qa_alphageometry_ptolemy/qa_projection_obstruction_cert/fixtures/ppo_pass_arto_ternary.json
```
