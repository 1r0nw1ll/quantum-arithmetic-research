# Prospective A/B — Orbit Membership Arm A (rubric-first) raw output

**Concept**: `qa_orbit_membership_cert_v1` — verify list of (b,e) pairs forms one closed orbit under qa_step.
**Instruction**: author `validator.py` FIRST, then write `mapping_protocol_ref.json` as documentation.
**Subagent**: general-purpose, 32.4k tokens, 60s duration.

## Artifact 1: validator.py

```python
"""
QA Orbit Membership Certificate v1 — validator.

Verifies a list of (b, e) integer pairs forms exactly one orbit under
  qa_step(b, e) = (((b + e - 1) % 9) + 1, e)
i.e. starting from the first pair, repeated application of qa_step visits
every other pair exactly once and returns to the first pair.

QA_COMPLIANCE:
  axioms: [A1, A2, T2, S1, S2, T1]
  - A1 (no-zero): states in {1..9}; step uses ((x-1) % 9) + 1.
  - A2 (derived): d = b + e, a = b + 2e are derived, never assigned.
  - T2 (firewall): pairs must be ints; float inputs rejected (no float->int cast).
  - S1 (no **2): no squaring used; arithmetic is +, *, %.
  - S2 (no float state): b, e are Python ints or fractions.Fraction.
  - T1 (path time): iteration index k is an integer step count, bounded by len(pairs).
  observer_projection: none — the validator operates purely on discrete state.
  state_space: {1..9} x Z  (b in {1..9}, e an integer; e is preserved by qa_step).
"""

from fractions import Fraction

MOD = 9


def _is_qa_scalar(x):
    """A scalar is QA-admissible iff it is int or Fraction (no bool, no float)."""
    if isinstance(x, bool):
        return False
    return isinstance(x, int) or isinstance(x, Fraction)


def _qa_step(b, e):
    """A1/A2 compliant step: b_next = ((b + e - 1) % 9) + 1, e preserved."""
    return (((b + e - 1) % MOD) + 1, e)


def validate(pairs):
    """
    Validate that `pairs` is exactly one closed orbit under qa_step.
    """
    violations = []
    checks_run = []

    checks_run.append("input_shape")
    try:
        pair_list = list(pairs)
    except TypeError:
        return {
            "pass": False,
            "violations": ["input is not iterable"],
            "checks_run": checks_run,
        }

    if len(pair_list) == 0:
        violations.append("empty pair list: orbit must have at least one element")
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    normalized = []
    for i, p in enumerate(pair_list):
        if not (isinstance(p, (tuple, list)) and len(p) == 2):
            violations.append(f"pairs[{i}] is not a 2-tuple: {p!r}")
            return {"pass": False, "violations": violations, "checks_run": checks_run}
        b, e = p[0], p[1]
        if isinstance(b, float) or isinstance(e, float):
            violations.append(
                f"pairs[{i}] contains a float (T2/S2 violation): {(b, e)!r}"
            )
            return {"pass": False, "violations": violations, "checks_run": checks_run}
        if not (_is_qa_scalar(b) and _is_qa_scalar(e)):
            violations.append(
                f"pairs[{i}] has non-int/Fraction scalar (S2 violation): {(b, e)!r}"
            )
            return {"pass": False, "violations": violations, "checks_run": checks_run}
        normalized.append((b, e))

    checks_run.append("A1_no_zero_b")
    for i, (b, e) in enumerate(normalized):
        if isinstance(b, Fraction):
            if b.denominator != 1:
                violations.append(
                    f"pairs[{i}].b is a non-integer Fraction (A1/S2): {b}"
                )
                return {"pass": False, "violations": violations, "checks_run": checks_run}
            b_int = b.numerator
        else:
            b_int = b
        if not (1 <= b_int <= MOD):
            violations.append(
                f"pairs[{i}].b = {b_int} is outside {{1..{MOD}}} (A1 violation)"
            )
            return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("input_uniqueness")
    def _key(p):
        b, e = p
        if isinstance(b, Fraction) and b.denominator == 1:
            b = b.numerator
        if isinstance(e, Fraction) and e.denominator == 1:
            e = e.numerator
        return (b, e)

    keys = [_key(p) for p in normalized]
    seen = set()
    for i, k in enumerate(keys):
        if k in seen:
            violations.append(f"duplicate pair in input at index {i}: {k!r}")
            return {"pass": False, "violations": violations, "checks_run": checks_run}
        seen.add(k)

    pair_set = set(keys)
    n = len(pair_set)

    checks_run.append("e_constant_across_orbit")
    e_values = {k[1] for k in keys}
    if len(e_values) != 1:
        violations.append(
            f"e is not constant across pairs (qa_step preserves e): got {sorted(e_values)}"
        )
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("A2_derived_computable")
    for (b, e) in normalized:
        d = b + e
        a = b + 2 * e
        if not (_is_qa_scalar(d) and _is_qa_scalar(a)):
            violations.append("derived d or a is non-scalar (A2/S2 violation)")
            return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("orbit_generation_matches_set")
    start = _key(normalized[0])
    b0, e0 = start
    visited = [start]
    cur = start
    for k in range(1, n + 1):
        nb, ne = _qa_step(cur[0], cur[1])
        nxt = (nb, ne)
        if k < n:
            if nxt not in pair_set:
                violations.append(
                    f"step {k}: qa_step({cur}) = {nxt} not in input pair set"
                )
                return {"pass": False, "violations": violations, "checks_run": checks_run}
            if nxt in set(visited):
                violations.append(
                    f"step {k}: orbit revisits {nxt} before covering all pairs"
                )
                return {"pass": False, "violations": violations, "checks_run": checks_run}
            visited.append(nxt)
            cur = nxt
        else:
            if nxt != start:
                violations.append(
                    f"orbit does not close after {n} steps: qa_step({cur}) = {nxt}, expected {start}"
                )
                return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("orbit_covers_pair_set")
    if set(visited) != pair_set:
        missing = pair_set - set(visited)
        extra = set(visited) - pair_set
        violations.append(
            f"orbit set mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    return {"pass": True, "violations": [], "checks_run": checks_run}


if __name__ == "__main__":
    pairs = [(b, 1) for b in range(1, 10)]
    out = validate(pairs)
    print(out)
    assert out["pass"], out

    bad = [(b, 1) for b in range(1, 9)]
    out = validate(bad)
    print(out)
    assert not out["pass"]
```

## Artifact 2: mapping_protocol_ref.json (written after validator)

```json
{
  "cert_family": "qa_orbit_membership_cert_v1",
  "schema_version": "mapping_protocol_ref.v1",
  "description": "Verifies that a set of (b, e) integer pairs forms exactly one closed orbit under qa_step.",
  "observer_projection": {"present": false, "rationale": "Validator operates on discrete QA state. No continuous signal; no float->int cast. Firewall trivially holds."},
  "state_space": {"pair": "(b, e)", "b_domain": "{1..9}", "e_domain": "Z", "modulus": 9, "derived_coords": {"d": "b + e", "a": "b + 2*e"}},
  "dynamics": {"step_operator": "qa_step(b, e) = (((b + e - 1) mod 9) + 1, e)", "time": "T1 — integer path index k"},
  "applicable_axioms": {
    "A1_no_zero": {"enforced": true, "how": "Each b in {1..9}; step uses ((x-1) mod 9) + 1."},
    "A2_derived": {"enforced": true, "how": "d, a computed as derived quantities for sanity."},
    "T2_firewall": {"enforced": true, "how": "Floats rejected before arithmetic."},
    "S1_no_pow2": {"enforced": true, "how": "All arithmetic +, *, mod; no **2."},
    "S2_no_float_state": {"enforced": true, "how": "b, e must be int or Fraction."},
    "T1_path_time": {"enforced": true, "how": "Iteration index k is Python int."}
  }
}
```

**Axioms (one sentence):** All 6 (A1, A2, T2, S1, S2, T1) applied for orbit-membership check.
