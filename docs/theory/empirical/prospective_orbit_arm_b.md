# Prospective A/B — Orbit Membership Arm B (mapping-first) raw output

**Concept**: `qa_orbit_membership_cert_v1` — verify list of (b,e) pairs forms one closed orbit under qa_step.
**Instruction**: author `mapping_protocol_ref.json` FIRST, then DERIVE `validator.py`.
**Subagent**: general-purpose, 30.9k tokens, 44s duration.

## Artifact 1: mapping_protocol_ref.json

```json
{
  "schema_version": "1.0",
  "cert_family": "qa_orbit_membership_cert_v1",
  "concept": "Given a set of (b,e) integer pairs, verify they form exactly one closed orbit under qa_step(b,e) = (((b+e-1) % 9) + 1, e).",
  "observer_projection": {
    "boundary_crossings": [
      {"direction": "input_to_observer_layer", "description": "Caller supplies a list of integer pairs. Type/range validation is observer-layer."},
      {"direction": "qa_layer_to_output", "description": "Validator emits pass/fail dict."}
    ],
    "continuous_inputs": "none",
    "notes": "All dynamics integer-modular. Theorem NT compliance."
  },
  "state_space": {
    "modulus": 9,
    "b_range": [1, 9],
    "e_range": [1, 9],
    "step_operator": "qa_step(b, e) = (((b + e - 1) % 9) + 1, e)",
    "time_model": "integer path-length k in {0, 1, ..., |S|}"
  },
  "axioms_applicable": {
    "A1_no_zero": {"applies": true, "reason": "Orbit states must lie in {1..9}."},
    "A2_derived": {"applies": true, "reason": "d=b+e, a=b+2e must be derivable; validator enforces 2-tuple shape (no d,a stored independently)."},
    "T2_firewall": {"applies": true, "reason": "No float->int cast inside orbit iteration."},
    "S1_no_pow2": {"applies": true, "reason": "Module-level ban on **2."},
    "S2_no_float_state": {"applies": true, "reason": "b, e must be Python int."},
    "T1_path_time": {"applies": true, "reason": "Orbit length counted by integer step index k."}
  },
  "axioms_non_applicable": {},
  "check_mapping": {
    "input_shape": "Each element is a 2-tuple/list of ints — enforces A2 and S2.",
    "input_range": "Each b,e in {1..9} — enforces A1.",
    "input_type": "isinstance(x, int) and not isinstance(x, bool); reject float — enforces T2 and S2.",
    "non_empty": "At least one pair.",
    "uniqueness": "Pairs as set — no duplicates.",
    "e_invariant": "All pairs share same e — follows from qa_step leaving e fixed.",
    "closure_and_coverage": "Starting from pairs[0], applying qa_step exactly len(pairs) times returns to pairs[0], AND visited multiset equals input set — enforces T1."
  }
}
```

## Artifact 2: validator.py (derived from mapping)

```python
"""Validator for qa_orbit_membership_cert_v1.

QA_COMPLIANCE:
  A1 no-zero: states in {1..9}; step uses ((x-1)%m)+1.
  A2 derived: pairs are (b,e) only; d,a never stored — derived if needed.
  T2 firewall: int-only inputs; float rejected pre-iteration.
  S1 no **2: no squaring performed; linter-safe.
  S2 no float state: b,e must be Python int (bool excluded).
  T1 path time: orbit length counted by integer step index k.
"""

MODULUS = 9


def qa_step(b, e):
    """A1-compliant step: b' = ((b+e-1) % 9) + 1, e invariant."""
    return (((b + e - 1) % MODULUS) + 1, e)


def _is_plain_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate(pairs):
    violations = []
    checks_run = []

    checks_run.append("container_nonempty")
    if not isinstance(pairs, (list, tuple)):
        violations.append("input is not a list or tuple")
        return {"pass": False, "violations": violations, "checks_run": checks_run}
    if len(pairs) == 0:
        violations.append("input is empty; orbit of empty set is undefined")
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("pair_shape_2tuple")
    for i, p in enumerate(pairs):
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            violations.append(
                f"pairs[{i}] is not a 2-element sequence (A2: state is (b,e) only)"
            )
    if violations:
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("int_only_T2_S2")
    for i, (b, e) in enumerate(pairs):
        if not _is_plain_int(b):
            violations.append(f"pairs[{i}].b is not a plain int (T2/S2)")
        if not _is_plain_int(e):
            violations.append(f"pairs[{i}].e is not a plain int (T2/S2)")
    if violations:
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("A1_range_1_to_9")
    for i, (b, e) in enumerate(pairs):
        if not (1 <= b <= MODULUS):
            violations.append(f"pairs[{i}].b={b} out of range (A1: {{1..9}})")
        if not (1 <= e <= MODULUS):
            violations.append(f"pairs[{i}].e={e} out of range (A1: {{1..9}})")
    if violations:
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("set_uniqueness")
    as_tuples = [tuple(p) for p in pairs]
    pair_set = set(as_tuples)
    if len(pair_set) != len(as_tuples):
        violations.append("duplicate pair(s) in input")
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("e_invariant")
    e0 = as_tuples[0][1]
    mismatched = [p for p in as_tuples if p[1] != e0]
    if mismatched:
        violations.append(
            f"pairs have mixed e values ({sorted({p[1] for p in as_tuples})})"
        )
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("closure_and_coverage_T1")
    n = len(as_tuples)
    start = as_tuples[0]
    visited = []
    cur = start
    for k in range(n):
        visited.append(cur)
        cur = qa_step(*cur)

    if cur != start:
        violations.append(
            f"after {n} steps did not return to start {start}; got {cur}"
        )
    visited_set = set(visited)
    if visited_set != pair_set:
        missing = pair_set - visited_set
        extra = visited_set - pair_set
        if missing:
            violations.append(f"orbit from {start} misses input pairs: {sorted(missing)}")
        if extra:
            violations.append(
                f"orbit from {start} visits pairs not in input: {sorted(extra)}"
            )
    if len(visited) != len(visited_set):
        violations.append(
            "orbit revisits a state before completing length n"
        )

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "checks_run": checks_run,
    }
```

**Axioms (one sentence):** All 6 (A1, A2, T2, S1, S2, T1) applied; orbit naturally requires each.
