# [119] QA Dual Spine Unification Report

**Family ID**: 119
**Schema**: `QA_DUAL_SPINE_UNIFICATION_REPORT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; top-level overview of [116] and [118]
**Directory**: `qa_alphageometry_ptolemy/qa_dual_spine_unification_report/`
**Intended audience**: mathematicians, systems researchers, reviewers encountering the QA repo for the first time

---

## Purpose

[119] is the first artifact a reviewer should read. It places the two public spines — obstruction ([116]) and control ([118]) — side by side and states the unified QA claim:

> QA is a kernel-governed system with two interacting public spines: an obstruction spine and a control spine.

The validator checks that the report faithfully summarises both spine entry points, both theorem statements, and the canonical witness values.

---

## Unified Claim

| Spine | Entry point | Main chain | Canonical theorem |
|-------|-------------|------------|-------------------|
| Obstruction | [116] QA_OBSTRUCTION_STACK_REPORT | [111]→[112]→[113]→[114]→[115]→[116] | v_p(r)=1 → unreachable → pruned → ratio=1.0 |
| Control | [118] QA_CONTROL_STACK_REPORT | [105],[110]→[106]→[117]→[118] | orbit singularity→satellite→cosmos and k=2 preserved across domains |

Both spines are rooted in `QA_CORE_SPEC.v1` [107].

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1–IH3 | Kernel inheritance | `INVALID_KERNEL_REFERENCE` etc. |
| DU1 | `obstruction_spine_ref` is `QA_OBSTRUCTION_STACK_REPORT.v1` | `OBSTRUCTION_SPINE_REF_MISMATCH` |
| DU2 | `control_spine_ref` is `QA_CONTROL_STACK_REPORT.v1` | `CONTROL_SPINE_REF_MISMATCH` |
| DU3 | `obstruction_theorem` non-empty (≥20 chars) | `OBSTRUCTION_THEOREM_MISSING` |
| DU4 | `control_theorem` non-empty (≥20 chars) | `CONTROL_THEOREM_MISSING` |
| DU5 | `comparison_table` has exactly 2 rows | `COMPARISON_TABLE_INCOMPLETE` |
| DU6 | One row `entry_point` is `QA_OBSTRUCTION_STACK_REPORT.v1` | `COMPARISON_TABLE_MISMATCH` |
| DU7 | Other row `entry_point` is `QA_CONTROL_STACK_REPORT.v1` | `COMPARISON_TABLE_MISMATCH` |
| DU8 | Obstruction row `canonical_theorem` contains `v_p` | `COMPARISON_TABLE_MISMATCH` |
| DU9 | Control row `canonical_theorem` contains `singularity` or `cosmos` | `COMPARISON_TABLE_MISMATCH` |
| DU10 | `synthesis_statement` non-empty (≥20 chars) | `SYNTHESIS_STATEMENT_MISSING` |
| DU11 | `synthesis_statement` mentions both `obstruction` and `control` | `SYNTHESIS_STATEMENT_INCOMPLETE` |
| DU12 | `canonical_pass_witness.obstruction_spine_result` is `'obstruction_spine_verified'` | `WITNESS_MISMATCH` |
| DU13 | `canonical_pass_witness.control_spine_result` is `'control_spine_verified'` | `WITNESS_MISMATCH` |
| DU14 | `canonical_fail_witness.fail_types` non-empty | `WITNESS_MISMATCH` |

---

## Certified Fixtures

### `unification_pass_canonical.json`

Two-row comparison table (obstruction spine, control spine). Both spine refs correct. Theorem paragraphs present for both. Synthesis statement mentions both spines. Canonical witnesses present. **PASS.**

### `unification_fail_spine_ref_mismatch.json`

`obstruction_spine_ref` is `QA_OBSTRUCTION_STACK_CERT.v1` (the synthesis cert [115]) instead of `QA_OBSTRUCTION_STACK_REPORT.v1` (the validated report [116]). A realistic mistake — confusing the cert layer with the report layer.

**Fail type**: `OBSTRUCTION_SPINE_REF_MISMATCH`. **FAIL.**

---

## Architecture Position

```
[107] QA_CORE_SPEC.v1 (kernel)
  └── [116] QA_OBSTRUCTION_STACK_REPORT.v1  ← obstruction public spine
  └── [118] QA_CONTROL_STACK_REPORT.v1      ← control public spine
        └── [119] QA_DUAL_SPINE_UNIFICATION_REPORT.v1  ← top-level map
```

---

## Synthesis Statement

The obstruction spine governs **impossibility / pruning / efficiency**: arithmetic structure (v_p) propagates upward through control reachability, planner correctness, and efficiency savings to yield a zero-cost pruning guarantee. The control spine governs **cross-domain compilation / orbit preservation**: the same compiler law governs structurally distinct physical domains. Both spines are rooted in QA_CORE_SPEC.v1 and certified independently.

---

## Running

```bash
python qa_dual_spine_unification_report/qa_dual_spine_unification_report_validate.py --self-test
python qa_meta_validator.py
```
