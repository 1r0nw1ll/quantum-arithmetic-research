# [118] QA Control Stack Report

**Family ID**: 118
**Schema**: `QA_CONTROL_STACK_REPORT.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; packages [117] for external readers
**Directory**: `qa_alphageometry_ptolemy/qa_control_stack_report/`
**Intended audience**: mathematicians, systems researchers, AI/planning reviewers

---

## Purpose

[118] is the reader-facing layer of the control spine — the control-side analogue of [116]. A reviewer who will not inspect four validator files can read [118] and understand:

- what the compiler-genericity theorem says
- what each of the three component families contributes
- the concrete comparison between cymatics and seismology
- what a broken report looks like

The validator checks that the comparison table is **faithful**: orbit paths and path lengths must be cross-domain consistent. A report claiming seismology reaches a different final orbit fails with `COMPARISON_TABLE_MISMATCH`.

---

## Comparison Table (canonical)

| Domain | Initial | Intermediate | Target | Orbit path | k | Moves |
|--------|---------|--------------|--------|------------|---|-------|
| Cymatics | flat | stripes | hexagons | singularity→satellite→cosmos | 2 | increase_amplitude, set_frequency |
| Seismology | quiet | p_wave | surface_wave | singularity→satellite→cosmos | 2 | increase_gain, apply_lowpass |

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1–IH3 | Kernel inheritance | `INVALID_KERNEL_REFERENCE` etc. |
| CR1 | `control_stack_ref` is `QA_CONTROL_STACK_CERT.v1` | `CONTROL_STACK_REF_MISMATCH` |
| CR2 | `theorem_statement` non-empty (≥20 chars) | `THEOREM_STATEMENT_MISSING` |
| CR3 | `domain_summaries` covers [106], [105], [110] | `DOMAIN_SUMMARY_INCOMPLETE` |
| CR4 | All `comparison_table` rows have consistent `orbit_path` | `COMPARISON_TABLE_MISMATCH` |
| CR5 | All `comparison_table` rows have equal `path_length_k` | `COMPARISON_TABLE_MISMATCH` |
| CR6 | `canonical_pass_witness.result` is `'cross_domain_equivalence_holds'` | `WITNESS_MISMATCH` |
| CR7 | `canonical_fail_witness.fail_types` non-empty | `WITNESS_MISMATCH` |

---

## Certified Fixtures

### `report_pass_cross_domain.json`

Two-row comparison table (cymatics, seismology). Both rows have `orbit_path=[singularity,satellite,cosmos]`, `path_length_k=2`. Theorem statement, three domain summaries, canonical witnesses, source refs all present. **PASS.**

### `report_fail_table_mismatch.json`

Seismology row claims `orbit_path=[singularity,satellite,satellite]` and `path_length_k=3`. Validator checks cross-row consistency: orbit_paths differ and path lengths differ.

**Fail type**: `COMPARISON_TABLE_MISMATCH`. **FAIL.**

---

## Relationship to [117]

```
[117] QA_CONTROL_STACK_CERT.v1     ← machine-verifiable synthesis cert
  └── [118] QA_CONTROL_STACK_REPORT.v1  ← validated public report
```

---

## Two Public Spines

With [116] and [118] both complete, the repo has two polished, validated public entry points:

| Spine | Entry point | Theorem |
|-------|-------------|---------|
| Obstruction | [116] QA_OBSTRUCTION_STACK_REPORT | v_p(r)=1 → unreachable → pruned → ratio=1.0 |
| Control | [118] QA_CONTROL_STACK_REPORT | compiler law preserved: orbit ≡ singularity→satellite→cosmos across domains |

---

## Running

```bash
python qa_control_stack_report/qa_control_stack_report_validate.py --self-test
python qa_meta_validator.py
```
