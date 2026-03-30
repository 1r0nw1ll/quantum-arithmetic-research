# [120] QA Public Overview Doc

**Family ID**: 120
**Schema**: `QA_PUBLIC_OVERVIEW_DOC.v1`
**Scope**: Family extension of `QA_CORE_SPEC.v1` [107]; derived from `QA_DUAL_SPINE_UNIFICATION_REPORT.v1` [119]
**Directory**: `qa_alphageometry_ptolemy/qa_public_overview_doc/`
**Intended audience**: any reviewer encountering QA for the first time

---

## Purpose

[120] is the presentation-grade artifact derived from [119]. It is designed to be handed to a reviewer as the **first thing they read** before diving into any family.

It packages the architecture into:
- a one-paragraph executive summary
- a textual two-spine diagram (chain + theorem per spine)
- one canonical obstruction example (r=6, p=3, k=2)
- one canonical cross-domain control example (cymatics + seismology)
- a short "why it matters" section
- pointers to the two public spine reports [116] and [118]

---

## Validator Checks

| Check | Description | Fail Type |
|-------|-------------|-----------|
| IH1–IH3 | Kernel inheritance | `INVALID_KERNEL_REFERENCE` etc. |
| PO1 | `overview_ref` is `QA_DUAL_SPINE_UNIFICATION_REPORT.v1` | `OVERVIEW_REF_MISMATCH` |
| PO2 | `executive_summary` non-empty (≥20 chars) | `EXECUTIVE_SUMMARY_MISSING` |
| PO3 | `spine_diagram` has both spines, each with `chain` + `theorem` (≥10 chars) | `SPINE_DIAGRAM_MISSING` |
| PO4 | `obstruction_example` has `canonical_r/p/k` integers | `OBSTRUCTION_EXAMPLE_MISSING` |
| PO4b | `obstruction_example` description mentions `v_p`, `ratio`, or `pruning` | `OBSTRUCTION_EXAMPLE_INCOMPLETE` |
| PO5 | `control_example.domains` has ≥2 entries | `CONTROL_EXAMPLE_MISSING` |
| PO5b | Domain names include both `cymatics` and `seismology` | `CONTROL_EXAMPLE_INCOMPLETE` |
| PO6 | `why_it_matters` non-empty (≥20 chars) | `WHY_IT_MATTERS_MISSING` |
| PO7 | `spine_entry_points` contains both `QA_OBSTRUCTION_STACK_REPORT.v1` and `QA_CONTROL_STACK_REPORT.v1` | `SPINE_ENTRY_POINTS_INCOMPLETE` |
| PO8 | `canonical_pass_witness.overview_result` is `'public_overview_verified'` | `WITNESS_MISMATCH` |
| PO9 | `canonical_fail_witness.fail_types` non-empty | `WITNESS_MISMATCH` |

---

## Certified Fixtures

### `overview_pass_canonical.json`

All sections present and substantively correct. Executive summary covers both spines. Spine diagram has full chain + theorem for both spines. Obstruction example: r=6, p=3, k=2, pruning_ratio=1.0. Control example: cymatics + seismology, shared orbit `[singularity,satellite,cosmos]`, k=2. Both spine entry points listed. **PASS.**

### `overview_fail_entry_points_incomplete.json`

`spine_entry_points` lists only `QA_OBSTRUCTION_STACK_REPORT.v1` — the control spine entry point is missing. A report that directs the reviewer to only one spine is not a faithful overview of [119].

**Fail type**: `SPINE_ENTRY_POINTS_INCOMPLETE`. **FAIL.**

---

## Architecture Position

```
[107] QA_CORE_SPEC.v1 (kernel)
  └── [119] QA_DUAL_SPINE_UNIFICATION_REPORT.v1  ← formal top-level map
        └── [120] QA_PUBLIC_OVERVIEW_DOC.v1       ← presentation-grade export
                                                    (hand to reviewer first)
```

---

## The Two Canonical Examples

### Obstruction example (r=6, p=3, k=2)

Since 3 is inert in Z[φ] and v_3(6)=1, the target r=6 is arithmetically forbidden. An obstruction-aware planner prunes it before any search node is expanded: `nodes_expanded=0`, `pruning_ratio=1.0`. Full chain: [111]→[114].

### Cross-domain control example

| Domain | Initial | Target | Orbit | k |
|--------|---------|--------|-------|---|
| Cymatics | flat | hexagons | singularity→satellite→cosmos | 2 |
| Seismology | quiet | surface_wave | singularity→satellite→cosmos | 2 |

Different physics. Same orbit trajectory. The compiler law is structural.

---

## Running

```bash
python qa_public_overview_doc/qa_public_overview_doc_validate.py --self-test
python qa_meta_validator.py
```
