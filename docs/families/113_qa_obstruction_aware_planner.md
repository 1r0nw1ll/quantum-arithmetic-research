# [113] QA Obstruction-Aware Planner

**Family ID**: 113
**Schema**: `QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; cross-references [112] bridge
**Directory**: `qa_alphageometry_ptolemy/qa_obstruction_aware_planner/`

---

## Purpose

[113] is the operational consequence of the obstruction bridge [112]. Where [112] proves *impossibility*, [113] proves *algorithmic correctness*:

> **Obstruction-Aware Planner Theorem**: A planner certified by [113] must consult the [112] obstruction bridge before any search expansion. If `v_p(target_arithmetic_class) = 1`, the planner must prune immediately (`pruned_before_search=true`, `nodes_expanded=0`). A planner that expands any nodes on a forbidden target fails with `OBSTRUCTION_NOT_APPLIED`.

This closes the loop from theory to algorithm:
- [111]: these residues are arithmetically impossible
- [112]: impossible residues cannot be claimed reachable
- [113]: a correct planner never wastes search effort on impossible targets

---

## Planner Correctness Contract

| Condition | Required behavior | Fail if violated |
|-----------|-------------------|-----------------|
| `v_p(r) = 1` (forbidden) | `pruned_before_search=true`, `nodes_expanded=0`, `plan_found=null` | `OBSTRUCTION_NOT_APPLIED` |
| `v_p(r) ≠ 1` (valid) | `pruned_before_search=false` | `PRUNE_DECISION_INCONSISTENT` |

Note: for valid targets, the planner may still fail to find a plan within its search bound — that is a normal exploratory failure, not a cert failure. Only arithmetic pruning of a valid target is invalid.

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from` is `'QA_CORE_SPEC.v1'` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope` is `'family_extension'` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `gate_policy_respected` ⊇ [0..5] | `GATE_POLICY_INCOMPATIBLE` |
| BR1 | `obstruction_ref` is `'QA_OBSTRUCTION_COMPILER_BRIDGE_CERT.v1'` | `OBSTRUCTION_REF_MISMATCH` |
| BR2 | `modulus == p^k` | `MODULUS_MISMATCH` |
| BR3 | p is inert: Legendre(5,p) = −1 | `PRIME_NOT_INERT` |
| BR4 | `target_arithmetic_class` in {0,...,modulus−1} | `TARGET_OUT_OF_RANGE` |
| BR5 | `obstruction_verdict` matches computed v_p(r) | `OBSTRUCTION_VERDICT_WRONG` |
| PA1 | OBSTRUCTION_PRESENT → `pruned_before_search=true` | `OBSTRUCTION_NOT_APPLIED` |
| PA2 | OBSTRUCTION_PRESENT → `nodes_expanded=0` | `OBSTRUCTION_NOT_APPLIED` |
| PA3 | OBSTRUCTION_ABSENT → `pruned_before_search=false` | `PRUNE_DECISION_INCONSISTENT` |

---

## Certified Fixtures

### `planner_pass_pruned_class_3.json`

Target r=3 (mod 9), v₃(3)=1 → OBSTRUCTION_PRESENT.
Planner: `pruned_before_search=true`, `nodes_expanded=0`, `plan_found=null`.
Correct behavior — zero search effort wasted. **PASS.**

### `planner_pass_search_class_4.json`

Target r=4 (mod 9), v₃(4)=0 → OBSTRUCTION_ABSENT.
Planner: `pruned_before_search=false`, `nodes_expanded=3`, `plan_found=true` (2-move plan: increase_gain → apply_lowpass, reaching state pair (b=2,e=0) with f(2,0)=4).
Correct behavior — search proceeds, plan found. **PASS.**

### `planner_fail_obstruction_not_applied.json`

Target r=6 (mod 9), v₃(6)=1 → OBSTRUCTION_PRESENT.
Planner: `pruned_before_search=false`, `nodes_expanded=47`, `plan_found=false`.
47 nodes wasted on an arithmetically impossible target.
**Fail type**: `OBSTRUCTION_NOT_APPLIED`. **FAIL.**

---

## The Chain

```
[111] norm form omits v_p=1 residues         (arithmetic impossibility)
  ↓ via [112] bridge
[112] v_p(r)=1 → no control cert claims r    (control impossibility)
  ↓ via [113] planner
[113] v_p(r)=1 → planner prunes at frontier  (algorithmic correctness)
```

Obstruction theory is not decorative. It governs computation.

---

## Running

```bash
# Self-test (3 fixtures)
python qa_obstruction_aware_planner/qa_obstruction_aware_planner_validate.py --self-test

# Full meta-validator
python qa_meta_validator.py
```
