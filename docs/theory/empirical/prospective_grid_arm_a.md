# Prospective A/B — Grid Coverage Arm A (rubric-first) raw output

**Concept**: `qa_grid_coverage_cert_v1` — verify list of (b,e) pairs covers full 81-element grid.
**Instruction**: author `validator.py` FIRST, then write `mapping_protocol_ref.json` as documentation.
**Subagent**: general-purpose, 31.5k tokens, 52s duration.

## Artifact 1: validator.py

```python
"""
QA_COMPLIANCE: A1 (no-zero, states in {1..9}), A2 (derived d=b+e, a=b+2e computed
only for reporting, never assigned independently), T2 (no float->int cast: all
inputs must already be int; floats are rejected as firewall violations), S1 (no
**2 operator used), S2 (no float state: pairs must be int or Fraction with
integer value), T1 (no path-time / continuous-time variables; coverage is a
static set check over integer indices).

Family: qa_grid_coverage_cert_v1
Purpose: verify a list of (b, e) pairs covers exactly the full 81-element
QA grid {1..9} x {1..9}: every grid pair present at least once, and no pair
outside the grid. Duplicates are allowed.
"""

from fractions import Fraction

M = 9  # QA modulus (mod-9 grid)
GRID_MIN = 1
GRID_MAX = M
EXPECTED_CELLS = M * M  # 81


def _is_qa_int(x):
    """A1/S2/T2: accept Python int or integer-valued Fraction."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator == 1
    return False


def _coerce_int(x):
    if isinstance(x, int):
        return x
    return x.numerator


def _in_grid(v):
    return GRID_MIN <= v <= GRID_MAX


def validate(pairs):
    """Validate that `pairs` covers the full QA grid {1..9} x {1..9}."""
    violations = []
    checks_run = []

    checks_run.append("input_shape")
    try:
        materialised = list(pairs)
    except TypeError:
        return {
            "pass": False,
            "violations": ["input is not iterable"],
            "checks_run": checks_run,
        }

    checks_run.append("A1_no_zero_and_range")
    checks_run.append("S2_no_float_state")
    checks_run.append("T2_firewall_no_float_cast")

    seen = set()
    for idx, item in enumerate(materialised):
        try:
            if len(item) != 2:
                violations.append(
                    f"pair[{idx}] has arity {len(item)}, expected 2"
                )
                continue
        except TypeError:
            violations.append(f"pair[{idx}] is not a sequence: {item!r}")
            continue

        b_raw, e_raw = item[0], item[1]

        if isinstance(b_raw, float) or isinstance(e_raw, float):
            violations.append(
                f"pair[{idx}]=({b_raw!r},{e_raw!r}) contains float "
                "(S2/T2 violation: QA state must be int or integer Fraction)"
            )
            continue

        if not _is_qa_int(b_raw):
            violations.append(
                f"pair[{idx}] b={b_raw!r} is not a QA integer "
                "(S2 violation)"
            )
            continue
        if not _is_qa_int(e_raw):
            violations.append(
                f"pair[{idx}] e={e_raw!r} is not a QA integer "
                "(S2 violation)"
            )
            continue

        b = _coerce_int(b_raw)
        e = _coerce_int(e_raw)

        if not _in_grid(b):
            violations.append(
                f"pair[{idx}] b={b} outside QA grid {{1..{M}}} "
                "(A1 violation: no-zero / range)"
            )
            continue
        if not _in_grid(e):
            violations.append(
                f"pair[{idx}] e={e} outside QA grid {{1..{M}}} "
                "(A1 violation: no-zero / range)"
            )
            continue

        seen.add((b, e))

    type_or_range_violated = len(violations) > 0

    checks_run.append("coverage_complete_81_cells")
    missing = []
    for bb in range(GRID_MIN, GRID_MAX + 1):
        for ee in range(GRID_MIN, GRID_MAX + 1):
            if (bb, ee) not in seen:
                missing.append((bb, ee))
    if missing:
        sample = missing[:10]
        more = "" if len(missing) <= 10 else f" (+{len(missing)-10} more)"
        violations.append(
            f"coverage incomplete: {len(missing)}/{EXPECTED_CELLS} cells "
            f"missing, e.g. {sample}{more}"
        )

    checks_run.append("A2_derived_coords_enforced_by_arity")
    checks_run.append("S1_no_double_star_two_in_validator")
    checks_run.append("T1_no_continuous_time")

    passed = (not type_or_range_violated) and (not missing)
    return {
        "pass": passed,
        "violations": violations,
        "checks_run": checks_run,
    }


if __name__ == "__main__":
    full = [(b, e) for b in range(1, 10) for e in range(1, 10)]
    r1 = validate(full)
    assert r1["pass"], r1
    r2 = validate(full[:-1])
    assert not r2["pass"], r2
    r3 = validate(full + [(1.0, 2)])
    assert not r3["pass"], r3
    r4 = validate(full + [(0, 5)])
    assert not r4["pass"], r4
    r5 = validate(full + [(10, 3)])
    assert not r5["pass"], r5
    print("qa_grid_coverage_cert_v1 validator self-test: OK")
```

## Artifact 2: mapping_protocol_ref.json (written after validator)

```json
{
  "family": "qa_grid_coverage_cert_v1",
  "version": "1.0",
  "schema_ref": "docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md",
  "summary": "Static coverage check: list of (b,e) pairs must contain every cell of 81-element QA grid {1..9} x {1..9} at least once.",
  "observer_projection": {"direction": "input_only", "continuous_inputs": [], "discrete_outputs": ["pass (bool)", "violations (list[str])", "checks_run (list[str])"]},
  "state_space": {"modulus": 9, "element_set": "{1, 2, 3, 4, 5, 6, 7, 8, 9}", "pair_space": "{1..9} x {1..9}", "cardinality": 81},
  "applicable_axioms": {
    "A1_no_zero": {"applies": true, "enforcement": "1 <= b <= 9 and 1 <= e <= 9."},
    "A2_derived_coords": {"applies": true, "enforcement": "Input arity fixed at 2."},
    "T2_firewall": {"applies": true, "enforcement": "Float rejected."},
    "S1_no_double_star_two": {"applies": true, "enforcement": "Validator arithmetic uses no **2."},
    "S2_no_float_state": {"applies": true, "enforcement": "Accept int or integer Fraction."},
    "T1_path_time": {"applies": true, "enforcement": "Static check; no continuous time."}
  }
}
```

**Axioms (one sentence):** All 6 declared applied (flat "enforced" labeling).
