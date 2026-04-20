# Prospective A/B — Triple Table Arm B (mapping-first) raw output

**Concept**: `qa_triple_table_cert_v1` — validate list of (b,e,d,a) 4-tuples with d=b+e, a=b+2e raw.
**Instruction**: author `mapping_protocol_ref.json` FIRST, then DERIVE `validator.py`.
**Subagent**: general-purpose, 30.9k tokens, 44s duration.

## Artifact 1: mapping_protocol_ref.json

```json
{
  "cert_family": "qa_triple_table_cert_v1",
  "version": "1.0",
  "schema_ref": "qa_mapping_protocol_ref/schema.v1.json",
  "observer_projection": {
    "input_type": "list_of_tuples",
    "input_description": "Unordered multiset of 4-tuples (b, e, d, a) with integer entries; row order carries no semantics.",
    "boundary_crossing": "input_only",
    "continuous_inputs": false,
    "rationale": "Table rows are already QA-discrete. No continuous-to-discrete cast. Firewall trivially satisfied."
  },
  "state_space": {
    "base_pair": {"b": {"domain": "{1..9}", "type": "int"}, "e": {"domain": "{1..9}", "type": "int"}},
    "derived": {"d": {"formula": "b + e", "reduction": "raw_unreduced"}, "a": {"formula": "b + 2*e", "reduction": "raw_unreduced"}},
    "modulus_m": 9,
    "raw_vs_mod_note": "Per HARD rule (2026-04-09): elements/tuples use d=b+e and a=b+2e RAW. Mod reduction is a T-operator.",
    "ordering": "none",
    "repetition_allowed": true
  },
  "applicable_axioms": {
    "A1_no_zero": {"applies": true, "reason": "b,e must lie in {1..9}.", "check_ids": ["A1_b_range", "A1_e_range"]},
    "A2_derived_coords": {"applies": true, "reason": "d=b+e and a=b+2*e (raw); validator recomputes and compares.", "check_ids": ["A2_d_equals_b_plus_e", "A2_a_equals_b_plus_2e"]},
    "T2_firewall": {"applies": true, "reason": "Float in any field = T2-b leak.", "check_ids": ["T2_int_only"]},
    "S1_no_power_star_star": {"applies": true, "reason": "Validator must not use **2; a computed as b+e+e.", "check_ids": ["S1_implementation_audit"], "note": "Structural self-declaration."},
    "S2_no_float_state": {"applies": true, "reason": "State entries must be int.", "check_ids": ["S2_int_state"], "note": "Subsumed by T2_int_only."},
    "T1_path_time": {"applies": false, "reason": "Table has no temporal semantics, no path. Row index is not a time step; rows unordered.", "check_ids": []}
  },
  "non_applicability_decisions": [
    {"axiom": "T1_path_time", "decision": "not_applicable", "justification": "Unordered multiset; no k-indexed evolution."},
    {"concept": "mod_reduction", "decision": "not_applied", "justification": "Per 2026-04-09 Hard Rule."}
  ]
}
```

## Artifact 2: validator.py (derived from mapping)

```python
"""Validator for qa_triple_table_cert_v1.

QA_COMPLIANCE: A1 (b,e in {1..9}), A2 (d=b+e, a=b+2*e raw), T2 (int-only),
S1 (no ** in implementation; a computed via b+e+e), S2 (int state only),
T1 (N/A — unordered table).

Each check_id traces to a field in mapping_protocol_ref.json.
"""

from fractions import Fraction


QA_COMPLIANCE = {
    "A1": "no-zero: b,e in {1..9}",
    "A2": "derived: d=b+e, a=b+2*e (raw, unreduced)",
    "T2": "firewall: int entries only, no float cast into QA state",
    "S1": "no **2 in implementation (a via b+e+e)",
    "S2": "no float state (int only; Fraction unused)",
    "T1": "N/A — table is unordered, no path-time semantics",
}

_CHECK_IDS = [
    "T2_int_only",
    "A1_b_range",
    "A1_e_range",
    "A2_d_equals_b_plus_e",
    "A2_a_equals_b_plus_2e",
    "S1_implementation_audit",
    "S2_int_state",
]


def _is_qa_int(x):
    if isinstance(x, bool):
        return False
    return isinstance(x, int)


def validate(rows):
    """Validate an unordered multiset of (b, e, d, a) rows."""
    violations = []

    if rows is None:
        return {
            "pass": False,
            "violations": [{
                "row_index": None,
                "row": None,
                "check_id": "input_shape",
                "detail": "rows is None",
            }],
            "checks_run": list(_CHECK_IDS),
        }

    for idx, row in enumerate(rows):
        if not (isinstance(row, (tuple, list)) and len(row) == 4):
            violations.append({
                "row_index": idx,
                "row": row,
                "check_id": "input_shape",
                "detail": "row must be a 4-tuple (b, e, d, a)",
            })
            continue

        b, e, d, a = row[0], row[1], row[2], row[3]

        bad_types = []
        for name, val in (("b", b), ("e", e), ("d", d), ("a", a)):
            if not _is_qa_int(val):
                bad_types.append((name, type(val).__name__))
        if bad_types:
            violations.append({
                "row_index": idx,
                "row": row,
                "check_id": "T2_int_only",
                "detail": "non-int entries: " + ", ".join(
                    f"{n}:{t}" for n, t in bad_types
                ),
            })
            continue

        if not (1 <= b <= 9):
            violations.append({
                "row_index": idx,
                "row": row,
                "check_id": "A1_b_range",
                "detail": f"b={b} outside {{1..9}}",
            })
        if not (1 <= e <= 9):
            violations.append({
                "row_index": idx,
                "row": row,
                "check_id": "A1_e_range",
                "detail": f"e={e} outside {{1..9}}",
            })

        d_expected = b + e
        if d != d_expected:
            violations.append({
                "row_index": idx,
                "row": row,
                "check_id": "A2_d_equals_b_plus_e",
                "detail": f"d={d}, expected b+e={d_expected}",
            })

        a_expected = b + e + e
        if a != a_expected:
            violations.append({
                "row_index": idx,
                "row": row,
                "check_id": "A2_a_equals_b_plus_2e",
                "detail": f"a={a}, expected b+2*e={a_expected}",
            })

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "checks_run": list(_CHECK_IDS),
    }


if __name__ == "__main__":
    good = [(1, 1, 2, 3), (9, 9, 18, 27), (3, 5, 8, 13), (3, 5, 8, 13)]
    bad = [(0, 1, 1, 2), (1, 1, 99, 3), (1.0, 1, 2, 3)]
    print("good:", validate(good))
    print("bad:", validate(bad))
```

**Axioms (one sentence):** A1+A2+T2+S1+S2 enforced; T1 declared NA (unordered table, no path-time). Each check_id traces to mapping field.
