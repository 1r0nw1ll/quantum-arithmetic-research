# [114] QA Obstruction Efficiency

**Family ID**: 114
**Schema**: `QA_OBSTRUCTION_EFFICIENCY_CERT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; cross-references [113] planner
**Directory**: `qa_alphageometry_ptolemy/qa_obstruction_efficiency/`

---

## Purpose

[114] completes the obstruction chain by quantifying *how much computation is saved* when the obstruction-aware planner ([113]) is used instead of a naive planner.

> **Obstruction Efficiency Theorem**: For any forbidden target (v_p(r)=1), the obstruction-aware planner expands 0 nodes while a naive planner expands N>0 nodes before exhaustion. `saved_nodes = N`, `pruning_ratio = 1.0`. The savings are **guaranteed** — not probabilistic — because v_p(r)=1 is a structural fact about Z[φ]/p^k Z[φ], not a heuristic estimate. For valid targets (v_p(r)≠1), no false pruning occurs (`false_pruning=false`, `pruning_ratio=0.0`).

The chain in full:

```
[111] v_p(r)=1 residues are arithmetically impossible          (theory)
  ↓
[112] impossible residues cannot be claimed reachable           (control law)
  ↓
[113] planner prunes forbidden targets before search            (correctness)
  ↓
[114] pruning saves all N baseline nodes for forbidden targets  (efficiency)
```

---

## Efficiency Semantics

| Target class | Verdict | `aware.nodes_expanded` | `saved_nodes` | `pruning_ratio` | `false_pruning` |
|-------------|---------|------------------------|---------------|-----------------|-----------------|
| v_p(r) = 1 (forbidden) | OBSTRUCTION_PRESENT | 0 | baseline N | 1.0 | false |
| v_p(r) ≠ 1 (valid) | OBSTRUCTION_ABSENT | same as baseline | 0 | 0.0 | false |

The validator **recomputes** `saved_nodes` and `pruning_ratio` from raw traces — it does not trust the cert values. This is the "recompute-not-trust" principle that runs throughout the QA cert ecosystem.

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1 | `inherits_from` is `'QA_CORE_SPEC.v1'` | `INVALID_KERNEL_REFERENCE` |
| IH2 | `spec_scope` is `'family_extension'` | `SPEC_SCOPE_MISMATCH` |
| IH3 | `gate_policy_respected` ⊇ [0..5] | `GATE_POLICY_INCOMPATIBLE` |
| EF1 | `planner_ref` is `'QA_OBSTRUCTION_AWARE_PLANNER_CERT.v1'` | `PLANNER_REF_MISMATCH` |
| EF2 | `modulus == p^k` | `MODULUS_MISMATCH` |
| EF3 | p is inert: Legendre(5,p) = −1 | `PRIME_NOT_INERT` |
| EF4 | `target_arithmetic_class` in {0,...,modulus−1} | `TARGET_OUT_OF_RANGE` |
| EF5 | `obstruction_verdict` matches computed v_p(target) | `OBSTRUCTION_VERDICT_WRONG` |
| EF6 | `saved_nodes == baseline_nodes - aware_nodes` (recomputed) | `EFFICIENCY_CLAIM_INCORRECT` |
| EF7 | `pruning_ratio == saved_nodes / baseline` (recomputed; 0.0 if baseline=0) | `EFFICIENCY_CLAIM_INCORRECT` |
| EF8 | OBSTRUCTION_PRESENT → `aware.pruned_before_search=true`, `aware.nodes_expanded=0` | `AWARE_TRACE_MISMATCH` |
| EF9 | OBSTRUCTION_ABSENT → `aware.pruned_before_search=false`; `false_pruning=false` | `FALSE_PRUNING_EFFICIENCY` + `AWARE_TRACE_MISMATCH` |

---

## Certified Fixtures

### `efficiency_pass_forbidden_class_6.json`

Target r=6 (mod 9), v₃(6)=1 → OBSTRUCTION_PRESENT.
Baseline naive planner: 47 nodes expanded, no plan found (class 6 is arithmetically impossible — exhaustion is inevitable).
Obstruction-aware planner: 0 nodes expanded (pruned immediately).
`saved_nodes=47`, `pruning_ratio=1.0`, `false_pruning=false`.
100% of search cost eliminated. **PASS.**

### `efficiency_pass_valid_class_4.json`

Target r=4 (mod 9), v₃(4)=0 → OBSTRUCTION_ABSENT.
Both planners: 3 nodes expanded, plan found.
`saved_nodes=0`, `pruning_ratio=0.0`, `false_pruning=false`.
Zero overhead on valid target — correct abstention. **PASS.**

### `efficiency_fail_false_pruning.json`

Target r=4 (mod 9), v₃(4)=0 → OBSTRUCTION_ABSENT.
Aware planner: `pruned_before_search=true`, `nodes_expanded=0` — incorrectly pruned a valid target.
`efficiency_claim.false_pruning=true`, `pruning_ratio=1.0` (spurious).
**Fail types**: `AWARE_TRACE_MISMATCH` (pruned on OBSTRUCTION_ABSENT) + `FALSE_PRUNING_EFFICIENCY` (valid target blocked). **FAIL.**

---

## Design Note: Why Recompute?

The validator recomputes `saved_nodes = baseline.nodes_expanded - aware.nodes_expanded` and `pruning_ratio = saved_nodes / baseline` rather than reading them from the cert. This guards against:

1. A cert that inflates savings to look better than it is
2. A cert that hides false pruning by adjusting the efficiency claim to look consistent

Arithmetic consistency is necessary but not sufficient — EF8/EF9 check the raw `pruned_before_search` flag independently.

---

## Running

```bash
# Self-test (3 fixtures)
python qa_obstruction_efficiency/qa_obstruction_efficiency_validate.py --self-test

# Full meta-validator
python qa_meta_validator.py
```
