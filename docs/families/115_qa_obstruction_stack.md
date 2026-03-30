# [115] QA Obstruction Stack

**Family ID**: 115
**Schema**: `QA_OBSTRUCTION_STACK_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; synthesis of [111]–[114]
**Directory**: `qa_alphageometry_ptolemy/qa_obstruction_stack/`

---

## Purpose

[115] is the public spine of the obstruction chain. Where [111]–[114] each certify one layer in isolation, [115] compresses all four into a single theorem-bearing artifact.

> **Obstruction Stack Theorem**: For any prime p inert in Z[φ] and any r with v_p(r)=1, arithmetic impossibility deterministically induces control impossibility, planner pruning, and guaranteed search-cost savings. The implication chain is:
>
> `v_p(r)=1` (arithmetic) → `r` unreachable (control) → `nodes_expanded=0` (planner) → `pruning_ratio=1.0` (efficiency)

Each step is independently recomputed — no layer trusts the adjacent layer's claimed outputs.

---

## The Four Layers

| Layer | Family | Claim for v_p(r)=1 |
|-------|--------|---------------------|
| **Arithmetic** | [111] | r ∉ Im(f); OBSTRUCTION_PRESENT |
| **Control** | [112] | claimed_reachable=false; UNREACHABLE |
| **Planner** | [113] | pruned_before_search=true; nodes_expanded=0 |
| **Efficiency** | [114] | saved_nodes=N; pruning_ratio=1.0 |

For valid targets (v_p(r)≠1): control allows reachability, planner does not prune, no false savings.

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from` is `'QA_CORE_SPEC.v1'` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope` is `'family_extension'` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `gate_policy_respected` ⊇ [0..5] | `GATE_POLICY_INCOMPATIBLE` |
| OS1 | `arithmetic_ref` is `QA_AREA_QUANTIZATION_PK_CERT.v1` | `ARITHMETIC_REF_MISMATCH` |
| OS2 | `control_ref` is `QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1` | `CONTROL_REF_MISMATCH` |
| OS3 | `planner_ref` is `QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1` | `PLANNER_REF_MISMATCH` |
| OS4 | `efficiency_ref` is `QA_OBSTRUCTION_EFFICIENCY_CERT.v1` | `EFFICIENCY_REF_MISMATCH` |
| OS5 | `modulus == p^k` | `MODULUS_MISMATCH` |
| OS6 | p is inert: Legendre(5,p) = −1 | `PRIME_NOT_INERT` |
| OS7 | `target` in {0,...,modulus−1} | `TARGET_OUT_OF_RANGE` |
| OS8 | `arithmetic_layer.obstruction_verdict` matches independently recomputed v_p(r) | `OBSTRUCTION_VERDICT_WRONG` |
| OS9 | OBSTRUCTION_PRESENT → `control_layer.claimed_reachable=false`, `control_verdict=UNREACHABLE` | `STACK_INCONSISTENCY` |
| OS10 | OBSTRUCTION_PRESENT → `planner.pruned_before_search=true`, `nodes_expanded=0` | `PRUNING_CONCLUSION_MISMATCH` |
| OS11 | Recompute `saved_nodes` and `pruning_ratio`; OBSTRUCTION_PRESENT → `aware_nodes=0`, `ratio=1.0` | `EFFICIENCY_CONCLUSION_MISMATCH` |
| OS12 | `stack_conclusion.full_chain_holds` consistent with all four layers | `STACK_INCONSISTENCY` |

---

## Certified Fixtures

### `stack_pass_forbidden_class_6.json`

Canonical forbidden case: p=3, k=2, r=6 (mod 9), v₃(6)=1.

- Arithmetic: v₃(6)=1 → OBSTRUCTION_PRESENT
- Control: claimed_reachable=false, control_verdict=UNREACHABLE
- Planner: pruned_before_search=true, nodes_expanded=0
- Efficiency: baseline=47, aware=0, saved_nodes=47, pruning_ratio=1.0
- stack_conclusion.full_chain_holds=true

All four layers self-consistent and independently confirmed. **PASS.**

### `stack_fail_inconsistent_class_6.json`

Same target r=6 with OBSTRUCTION_PRESENT correctly declared — but planner and efficiency layers are broken:

- Planner: pruned_before_search=**false**, nodes_expanded=**12** (should be 0)
- Efficiency: aware_nodes=12, saved_nodes=**35**, pruning_ratio=**0.74** (should be 0, 47, 1.0)
- stack_conclusion.full_chain_holds=**true** (lie — chain is broken)

**Fail types**: `PRUNING_CONCLUSION_MISMATCH` + `EFFICIENCY_CONCLUSION_MISMATCH` + `STACK_INCONSISTENCY`. **FAIL.**

---

## Architecture Position

```
[107] QA_CORE_SPEC.v1 (kernel)
  │
  ├── [111] Inert Prime Area Quantization    ← arithmetic obstruction theory
  ├── [112] Obstruction-Compiler Bridge      ← connects arithmetic → control
  ├── [113] Obstruction-Aware Planner        ← correctness: 0 nodes on forbidden
  ├── [114] Obstruction Efficiency           ← efficiency: pruning_ratio=1.0
  └── [115] Obstruction Stack               ← synthesis: full chain in one cert
```

[115] is the single artifact a reviewer needs to understand the complete story. The four component families remain independently certifiable; [115] asserts their joint consequence.

---

## Running

```bash
# Self-test (2 fixtures)
python qa_obstruction_stack/qa_obstruction_stack_validate.py --self-test

# Full meta-validator
python qa_meta_validator.py
```
