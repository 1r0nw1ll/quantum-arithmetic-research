# Prospective A/B — Triple Table Arm A (rubric-first) raw output

**Concept**: `qa_triple_table_cert_v1` — validate list of (b,e,d,a) 4-tuples with d=b+e, a=b+2e raw.
**Instruction**: author `validator.py` FIRST, then write `mapping_protocol_ref.json` as documentation.
**Subagent**: general-purpose, 30.8k tokens, 42s duration.

## Artifact 1: validator.py

```python
"""
QA Triple Table Certificate — Validator

QA_COMPLIANCE:
  family: qa_triple_table_cert_v1
  axioms: [A1, A2, S1, S2, T2]
  notes:
    - A1 (no-zero): b, e in {1..9}; derived d, a are raw (unreduced).
    - A2 (derived): d == b+e, a == b+2e, always derived, never assigned.
    - S1 (no **2): no squaring anywhere; only linear combinations.
    - S2 (no float state): rows must be int-typed (bool rejected); no floats.
    - T2 (firewall): reject any float input in the QA tuple (no float->int cast).
  not_applicable:
    - T1 (path time): this is an unordered table; no step index exists.
"""

from fractions import Fraction

M = 9  # modulus for A1 range check on (b, e)


def _is_qa_int(x):
    """
    S2/T2: QA state must be int or Fraction-with-denominator-1.
    Reject bool (a Python int subclass) and any float.
    """
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator == 1
    return False


def _as_int(x):
    """Coerce a validated QA scalar to Python int."""
    if isinstance(x, Fraction):
        return int(x)
    return x


def validate(rows):
    """
    Validate a QA triple table.

    Input: rows — iterable of 4-tuples (b, e, d, a).
    No ordering between rows is assumed; rows may repeat.

    Returns: {"pass": bool, "violations": [...], "checks_run": [...]}
    """
    violations = []
    checks_run = [
        "input_is_iterable",
        "row_is_4tuple",
        "row_elements_are_qa_int (S2, T2)",
        "b_in_range_1_to_M (A1)",
        "e_in_range_1_to_M (A1)",
        "d_equals_b_plus_e_raw (A2)",
        "a_equals_b_plus_2e_raw (A2)",
        "no_squaring_used (S1, structural)",
    ]

    try:
        rows_list = list(rows)
    except TypeError:
        return {
            "pass": False,
            "violations": [{"code": "INPUT-1", "msg": "rows is not iterable"}],
            "checks_run": checks_run,
        }

    for idx, row in enumerate(rows_list):
        if not isinstance(row, (tuple, list)):
            violations.append({
                "code": "SHAPE-1",
                "row_index": idx,
                "msg": f"row is not a tuple/list: {type(row).__name__}",
            })
            continue
        if len(row) != 4:
            violations.append({
                "code": "SHAPE-2",
                "row_index": idx,
                "msg": f"row has {len(row)} elements, expected 4",
            })
            continue

        b, e, d, a = row

        typed_ok = True
        for name, val in (("b", b), ("e", e), ("d", d), ("a", a)):
            if not _is_qa_int(val):
                violations.append({
                    "code": "T2-1" if isinstance(val, float) else "S2-1",
                    "row_index": idx,
                    "field": name,
                    "msg": (
                        f"{name} must be int or integer Fraction "
                        f"(got {type(val).__name__}={val!r})"
                    ),
                })
                typed_ok = False
        if not typed_ok:
            continue

        bi = _as_int(b)
        ei = _as_int(e)
        di = _as_int(d)
        ai = _as_int(a)

        if not (1 <= bi <= M):
            violations.append({
                "code": "A1-1",
                "row_index": idx,
                "field": "b",
                "msg": f"b={bi} not in [1..{M}]",
            })
        if not (1 <= ei <= M):
            violations.append({
                "code": "A1-2",
                "row_index": idx,
                "field": "e",
                "msg": f"e={ei} not in [1..{M}]",
            })

        d_expected = bi + ei
        a_expected = bi + 2 * ei
        if di != d_expected:
            violations.append({
                "code": "A2-1",
                "row_index": idx,
                "field": "d",
                "msg": f"d={di} != b+e={d_expected} (raw, unreduced)",
            })
        if ai != a_expected:
            violations.append({
                "code": "A2-2",
                "row_index": idx,
                "field": "a",
                "msg": f"a={ai} != b+2e={a_expected} (raw, unreduced)",
            })

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "checks_run": checks_run,
    }


if __name__ == "__main__":
    good = [(1, 1, 2, 3), (9, 9, 18, 27), (3, 4, 7, 11), (3, 4, 7, 11)]
    bad_a2 = [(1, 1, 2, 4)]
    bad_a1 = [(0, 1, 1, 2)]
    bad_t2 = [(1.0, 1, 2, 3)]
    bad_s1 = [(1, 1, 2, 3.0)]

    for label, rows in [
        ("good", good),
        ("bad_a2", bad_a2),
        ("bad_a1", bad_a1),
        ("bad_t2_float_b", bad_t2),
        ("bad_float_a", bad_s1),
    ]:
        r = validate(rows)
        print(label, r["pass"], len(r["violations"]))
```

## Artifact 2: mapping_protocol_ref.json (written after validator)

```json
{
  "schema": "qa_mapping_protocol_ref.v1",
  "family": "qa_triple_table_cert_v1",
  "title": "QA Triple Table — raw (b,e,d,a) row-level consistency",
  "description": "Unordered table of 4-tuples (b, e, d, a) with b, e in {1..9} and raw (unreduced) derived coordinates d = b+e, a = b+2e.",
  "observer_projection": {"direction": "input", "boundary_crossings": [{"side": "observer->qa", "description": "Integer rows; validator rejects floats."}]},
  "state_space": {"primary": "(b, e) with b, e in {1,...,9}", "derived_raw": {"d": "b + e", "a": "b + 2*e"}, "row_structure": "(b, e, d, a)", "collection_structure": "unordered multiset"},
  "applicable_axioms": {
    "A1": {"applies": true, "codes": ["A1-1", "A1-2"]},
    "A2": {"applies": true, "codes": ["A2-1", "A2-2"]},
    "T2": {"applies": true, "codes": ["T2-1"]},
    "S1": {"applies": true, "codes": [], "enforcement": "Structural: never squares a state variable."},
    "S2": {"applies": true, "codes": ["S2-1"]}
  },
  "not_applicable_axioms": {
    "T1": {"applies": false, "reason": "The table is unordered. There is no path, step index, or integer time variable to constrain."}
  },
  "validator": {"entrypoint": "validator.validate", "signature": "validate(rows) -> {'pass': bool, 'violations': list, 'checks_run': list}"}
}
```

**Axioms (one sentence):** A1+A2+T2+S1+S2 enforced; T1 declared NA (unordered table has no step index).
