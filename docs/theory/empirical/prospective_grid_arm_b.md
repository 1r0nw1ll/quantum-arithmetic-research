# Prospective A/B — Grid Coverage Arm B (mapping-first) raw output

**Concept**: `qa_grid_coverage_cert_v1` — verify list of (b,e) pairs covers full 81-element grid.
**Instruction**: author `mapping_protocol_ref.json` FIRST, then DERIVE `validator.py`.
**Subagent**: general-purpose, 30.8k tokens, 43s duration.

## Artifact 1: mapping_protocol_ref.json

```json
{
  "schema": "qa_mapping_protocol_ref/v1",
  "family": "qa_grid_coverage_cert_v1",
  "purpose": "Verify a list of (b,e) integer pairs covers exactly the 81-element QA grid {1..9} x {1..9}. Duplicates allowed.",
  "observer_projection": {
    "input_type": "list of (b, e) integer pairs",
    "boundary_crossings": [
      "input -> observer layer: accept caller's pair list",
      "QA layer -> output: emit pass/fail + violation list"
    ],
    "qa_discrete_core": "set membership over finite lattice {1..9} x {1..9}",
    "non_qa_inputs": "none"
  },
  "state_space": {
    "modulus": 9,
    "b_range": "{1,...,9}",
    "e_range": "{1,...,9}",
    "grid_size": 81,
    "representation": "Python tuple (int, int); NOT derived into (b,e,d,a)"
  },
  "axioms_applied": {
    "A1_no_zero": {"applies": true, "reason": "b,e must live in {1..9}."},
    "A2_derived": {"applies": "declarative", "reason": "Claim confined to (b,e) plane. d=b+e and a=b+2e NOT materialized because coverage is about (b,e) alone. A2 honored by NOT assigning d or a independently."},
    "T2_firewall": {"applies": true, "reason": "No float inputs accepted."},
    "S1_no_power_two": {"applies": "vacuous", "reason": "No squaring in coverage logic."},
    "S2_no_float_state": {"applies": true, "reason": "Int only; bool excluded."},
    "T1_path_time": {"applies": "vacuous", "reason": "Single set-membership test; no iteration index / continuous time."}
  },
  "checks": [
    {"id": "C1_types", "binds": "T2_firewall, S2_no_float_state"},
    {"id": "C2_range", "binds": "A1_no_zero"},
    {"id": "C3_coverage", "binds": "state_space.grid_size"},
    {"id": "C4_no_extraneous", "binds": "state_space.b_range, e_range"}
  ],
  "pass_condition": "All checks C1-C4 return no violations.",
  "duplicates_policy": "allowed; coverage is set-membership"
}
```

**Key non-applicability taxonomy** (unique to mapping-first frame):
- `"applies": true` — enforced with a runtime check
- `"applies": "declarative"` — honored structurally, not runtime-checked
- `"applies": "vacuous"` — trivially satisfied (no arithmetic / no path to violate)

## Artifact 2: validator.py (derived from mapping)

```python
"""QA Grid Coverage Cert v1 validator.

QA_COMPLIANCE:
  A1_no_zero: enforced (C2_range rejects any component not in {1..9})
  A2_derived: declarative (claim confined to (b,e) plane; d,a not materialized)
  T2_firewall: enforced (C1_types rejects non-int, float, numpy float)
  S1_no_power_two: vacuous (no squaring in coverage logic)
  S2_no_float_state: enforced (C1_types requires Python int, excludes bool)
  T1_path_time: vacuous (single set-membership test, no path index)

Mapping: see mapping_protocol_ref.json; each check Ck traces to an axiom binding.
"""

from fractions import Fraction

MODULUS = 9
GRID_SIZE = MODULUS * MODULUS  # 81


def _is_qa_int(x):
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator == 1 and 1 <= int(x) <= MODULUS
    return False


def _in_range(x):
    if isinstance(x, Fraction):
        x = int(x)
    return 1 <= x <= MODULUS


def validate(pairs):
    violations = []
    checks_run = ["C1_types", "C2_range", "C3_coverage", "C4_no_extraneous"]

    try:
        pair_list = list(pairs)
    except TypeError:
        return {
            "pass": False,
            "violations": [{"check": "C1_types", "detail": "input is not iterable"}],
            "checks_run": checks_run,
        }

    observed = set()
    extraneous = []

    for idx, item in enumerate(pair_list):
        if not isinstance(item, (tuple, list)):
            violations.append({
                "check": "C1_types",
                "index": idx,
                "detail": f"entry is not tuple/list: {type(item).__name__}",
            })
            continue
        if len(item) != 2:
            violations.append({
                "check": "C1_types",
                "index": idx,
                "detail": f"pair length {len(item)} != 2",
            })
            continue

        b, e = item[0], item[1]

        if not _is_qa_int(b):
            violations.append({
                "check": "C1_types",
                "index": idx,
                "detail": f"b component not QA-int (T2/S2): {type(b).__name__}={b!r}",
            })
            continue
        if not _is_qa_int(e):
            violations.append({
                "check": "C1_types",
                "index": idx,
                "detail": f"e component not QA-int (T2/S2): {type(e).__name__}={e!r}",
            })
            continue

        b_i = int(b) if isinstance(b, Fraction) else b
        e_i = int(e) if isinstance(e, Fraction) else e

        if not (_in_range(b_i) and _in_range(e_i)):
            violations.append({
                "check": "C2_range",
                "index": idx,
                "detail": f"pair ({b_i},{e_i}) outside {{1..{MODULUS}}} x {{1..{MODULUS}}} (A1)",
            })
            extraneous.append((b_i, e_i))
            continue

        observed.add((b_i, e_i))

    if extraneous:
        violations.append({
            "check": "C4_no_extraneous",
            "detail": f"{len(extraneous)} pair(s) outside grid",
            "examples": extraneous[:5],
        })

    full_grid = {(b, e) for b in range(1, MODULUS + 1) for e in range(1, MODULUS + 1)}
    missing = sorted(full_grid - observed)
    if missing:
        violations.append({
            "check": "C3_coverage",
            "detail": f"{len(missing)} of {GRID_SIZE} grid cells missing",
            "examples": missing[:5],
        })

    assert observed.issubset(full_grid), "internal: observed escaped grid after C2 gate"

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "checks_run": checks_run,
    }


if __name__ == "__main__":
    full = [(b, e) for b in range(1, 10) for e in range(1, 10)]
    assert validate(full)["pass"] is True
    assert validate(full[1:])["pass"] is False
    assert validate(full + [(0, 5)])["pass"] is False
    assert validate(full + [(1.0, 2)])["pass"] is False
    print("qa_grid_coverage_cert_v1 self-check: OK")
```

**Axioms (one sentence):** 3-tier taxonomy: A1+T2+S2 enforced; A2 "declarative" (d,a never materialized); S1+T1 "vacuous" (no arithmetic / no path).
