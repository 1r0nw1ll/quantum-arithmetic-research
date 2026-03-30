# [116] QA Obstruction Stack Report

**Family ID**: 116
**Schema**: `QA_OBSTRUCTION_STACK_REPORT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; packages [115] for external readers
**Directory**: `qa_alphageometry_ptolemy/qa_obstruction_stack_report/`
**Intended audience**: mathematicians, systems researchers, AI/planning reviewers

---

## Purpose

[116] is the public-facing layer of the obstruction spine. Where [115] is machine-perfect, [116] is reader-ready.

A reviewer who will not inspect five validator files can read [116] and understand the complete story: what the theorem says, what each layer contributes, what the numbers are, and what a broken cert looks like.

The validator enforces that the report is **faithful** — every value in the summary table is independently recomputed from arithmetic params and checked against the claimed values. A report that inflates savings or misstates pruning behavior fails with `SUMMARY_TABLE_MISMATCH`.

---

## What the Report Contains

| Section | Content |
|---------|---------|
| `theorem_statement` | Human-readable statement of the full obstruction theorem |
| `layer_summaries` | One-line meaning of each of [111]–[115] |
| `summary_table` | Recomputed per-target table: v_p, forbidden, reachable, pruned, baseline/aware/saved nodes, pruning_ratio |
| `canonical_pass_witness` | The PASS case (r=6, full chain holds) |
| `canonical_fail_witness` | The FAIL case (broken planner layer, 3 fail types) |
| `source_refs` | Paths to the five underlying cert family directories |

---

## Summary Table (canonical)

| r | v₃(r) | forbidden | reachable | pruned | baseline | aware | saved | ratio |
|---|-------|-----------|-----------|--------|----------|-------|-------|-------|
| 6 | 1 | ✓ | ✗ | ✓ | 47 | 0 | 47 | 1.0 |
| 4 | 0 | ✗ | ✓ | ✗ | 3 | 3 | 0 | 0.0 |

r=6: every search node saved, structurally guaranteed.
r=4: zero overhead — no false pruning, no interference.

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1–IH3 | Kernel inheritance (inherits_from, scope, gates) | `INVALID_KERNEL_REFERENCE` etc. |
| RP1 | `stack_ref` is `QA_OBSTRUCTION_STACK_CERT.v1` | `STACK_REF_MISMATCH` |
| RP2 | `modulus == p^k` | `MODULUS_MISMATCH` |
| RP3 | p is inert: Legendre(5,p) = −1 | `PRIME_NOT_INERT` |
| RP4 | `theorem_statement` non-empty (≥20 chars) | `THEOREM_STATEMENT_MISSING` |
| RP5 | `layer_summaries` covers all five families [111]–[115] | `LAYER_SUMMARY_INCOMPLETE` |
| RP6 | Each summary_table row: recompute v_p, forbidden, reachable, pruned, saved_nodes, pruning_ratio | `SUMMARY_TABLE_MISMATCH` |
| RP7 | All forbidden rows: pruned=true, aware_nodes=0, pruning_ratio=1.0 | `SUMMARY_TABLE_MISMATCH` |
| RP8 | `canonical_pass_witness.target_r` is a forbidden row in the table | `WITNESS_MISMATCH` |
| RP9 | `canonical_fail_witness.fail_types` non-empty | `WITNESS_MISMATCH` |

---

## Certified Fixtures

### `report_pass_canonical_r6.json`

Two-row summary table (r=6 forbidden, r=4 valid). All values independently verified. Theorem statement, five one-line layer summaries, canonical witnesses, source refs all present. **PASS.**

### `report_fail_inconsistent_summary.json`

Summary table claims r=6 (forbidden) with `pruned=false`, `aware_nodes=12`, `pruning_ratio=0.74`. Validator recomputes: for v₃(6)=1, aware_nodes must be 0, saved_nodes must be 47, ratio must be 1.0. The report misrepresents the machine-verified results.

**Fail type**: `SUMMARY_TABLE_MISMATCH`. **FAIL.**

---

## Relationship to [115]

```
[115] QA_OBSTRUCTION_STACK_CERT.v1   ← machine-verifiable, all four layers recomputed
  └── [116] QA_OBSTRUCTION_STACK_REPORT.v1  ← reader-ready, summary table checked for faithfulness
```

[115] is the ground truth. [116] certifies that a human-readable representation of [115] is accurate.

---

## Running

```bash
# Self-test (2 fixtures)
python qa_obstruction_stack_report/qa_obstruction_stack_report_validate.py --self-test

# Full meta-validator
python qa_meta_validator.py
```
