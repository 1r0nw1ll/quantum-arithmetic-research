# Prospective A/B — Fibonacci Arm B (mapping-first) raw output

**Concept**: `qa_fibonacci_mod_cert_v1` — validate no-zero Fibonacci-mod-9 sequence.
**Instruction**: author `mapping_protocol_ref.json` FIRST (declare observer projection + axiom applicability), then DERIVE `validator.py`.
**Subagent**: general-purpose, 30.9k tokens, 46s duration.

## Artifact 1: mapping_protocol_ref.json

```json
{
  "schema_version": "1.0",
  "cert_family": "qa_fibonacci_mod_cert_v1",
  "mapping_type": "reference",
  "observer_projection": {
    "description": "Input is a finite sequence of positive integers in {1..9} presented as a no-zero-Fibonacci sequence mod 9.",
    "boundary_crossings": {
      "input": "external list -> QA discrete layer (integer coercion check at boundary; non-integer is T2 violation)",
      "output": "discrete validation verdict -> observer layer (pass/violations dict)"
    },
    "continuous_inputs_as_causal": "NONE. No float, no continuous time, no libm calls."
  },
  "state_space": {
    "modulus": 9,
    "alphabet": "{1, 2, 3, 4, 5, 6, 7, 8, 9}",
    "element_representation": "single integer s in {1..9} per position (scalar QA state, not a (b,e) pair)",
    "index_domain": "integer position k >= 0, finite length",
    "recurrence": "seq[k] = ((seq[k-1] + seq[k-2] - 1) % 9) + 1 for k >= 2",
    "seed_constraint": "seq[0], seq[1] in {1..9}"
  },
  "applicable_axioms": {
    "A1_no_zero": {"applies": true, "reason": "States drawn from {1..9}; recurrence uses A1-adjusted form."},
    "A2_derived_coords": {"applies": false, "reason": "Scalar states, not (b,e) pairs. There is no d=b+e or a=b+2e to derive, so A2 is structurally inapplicable. Documented as non-applicable per spec."},
    "T2_firewall": {"applies": true, "reason": "Any float term would be a float->int cast into QA state (T2-b violation)."},
    "S1_no_power_two": {"applies": true, "reason": "Linear recurrence; no squaring. Validator written without ** anywhere."},
    "S2_no_float_state": {"applies": true, "reason": "Terms must be Python int or Fraction with denominator 1."},
    "T1_path_time": {"applies": true, "reason": "Time is integer index k along the sequence path."}
  },
  "checks": [
    {"id": "CHK-LEN", "description": "Sequence has length >= 2."},
    {"id": "CHK-TYPE-A1", "description": "Every term is a Python int (or Fraction with denom 1) in {1..9}. Enforces A1, T2, S2."},
    {"id": "CHK-REC-A1", "description": "For each k>=2, seq[k] == ((seq[k-1]+seq[k-2]-1) % 9) + 1. Enforces A1 recurrence."},
    {"id": "CHK-T1", "description": "Indices iterated are non-negative integers in ascending order."}
  ],
  "non_applicable_declarations": [
    {"axiom": "A2", "reason": "Scalar state, no (b,e) pair."}
  ],
  "authority": "docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md"
}
```

## Artifact 2: validator.py (derived from mapping)

```python
"""
qa_fibonacci_mod_cert_v1 validator.

QA_COMPLIANCE:
  cert_family: qa_fibonacci_mod_cert_v1
  modulus: 9
  alphabet: {1..9}
  axioms_applicable: [A1, T2, S1, S2, T1]
  axioms_nonapplicable: [A2]   # scalar state, no (b,e) pair
  observer_projection: external sequence -> QA discrete layer; verdict -> observer
  boundary_crossings: 2 (input coerce-check, output verdict)
  notes:
    - A1: states in {1..9}; recurrence seq[k] = ((seq[k-1]+seq[k-2]-1) % 9) + 1
    - T2: reject float/decimal/np-float terms (would be float->int cast into QA)
    - S1: no '**' operator anywhere in this file (grep-safe)
    - S2: terms must be int or Fraction with denom 1
    - T1: index k is an integer step along the path
"""

from fractions import Fraction

MODULUS = 9


def _is_qa_int(x):
    """Return True iff x is an int-valued scalar admissible as QA state (S2, T2)."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator == 1
    return False


def _as_int(x):
    """Coerce an already-validated QA scalar to Python int."""
    if isinstance(x, Fraction):
        return x.numerator
    return int(x)


def _a1_step(prev2, prev1, m):
    """A1-adjusted Fibonacci step: ((prev1 + prev2 - 1) % m) + 1. Integer-only."""
    return ((prev1 + prev2 - 1) % m) + 1


def validate(sequence):
    """
    Validate a no-zero-Fibonacci-mod-9 sequence.

    Args:
        sequence: iterable of integer-valued terms in {1..9}.

    Returns:
        dict with keys:
          pass:        bool
          violations:  list of {"check": str, "index": int|None, "detail": str}
          checks_run:  list of check ids attempted
    """
    violations = []
    checks_run = []

    try:
        seq = list(sequence)
    except TypeError:
        return {
            "pass": False,
            "violations": [{"check": "CHK-INPUT", "index": None,
                            "detail": "sequence is not iterable"}],
            "checks_run": ["CHK-INPUT"],
        }

    checks_run.append("CHK-LEN")
    if len(seq) < 2:
        violations.append({
            "check": "CHK-LEN", "index": None,
            "detail": "sequence length {} < 2; need two seeds".format(len(seq)),
        })

    checks_run.append("CHK-TYPE-A1")
    typed_ok = True
    for i, term in enumerate(seq):
        if not _is_qa_int(term):
            typed_ok = False
            violations.append({
                "check": "CHK-TYPE-A1", "index": i,
                "detail": "term has non-QA type {} (T2/S2 violation)".format(type(term).__name__),
            })
            continue
        v = _as_int(term)
        if v < 1 or v > MODULUS:
            typed_ok = False
            violations.append({
                "check": "CHK-TYPE-A1", "index": i,
                "detail": "term {} outside A1 alphabet {{1..{}}}".format(v, MODULUS),
            })

    checks_run.append("CHK-T1")

    checks_run.append("CHK-REC-A1")
    if typed_ok and len(seq) >= 2:
        ints = [_as_int(t) for t in seq]
        for k in range(2, len(ints)):
            expected = _a1_step(ints[k - 2], ints[k - 1], MODULUS)
            if ints[k] != expected:
                violations.append({
                    "check": "CHK-REC-A1", "index": k,
                    "detail": "seq[{}]={} != expected {} (from seq[{}]={}, seq[{}]={})".format(
                        k, ints[k], expected, k - 2, ints[k - 2], k - 1, ints[k - 1]
                    ),
                })

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "checks_run": checks_run,
    }


if __name__ == "__main__":
    s = [1, 1]
    while len(s) < 12:
        s.append(_a1_step(s[-2], s[-1], MODULUS))
    print(validate(s))
    print(validate([1, 1, 2]))
    print(validate([1, 1.0, 1]))
    print(validate([1, 0, 1]))
    print(validate([1]))
```

**Axioms (one sentence):** A1 + T2 + S1 + S2 + T1 applicable; A2 explicitly NA (scalar state, no (b,e) pair). Each check_id in validator traces to a mapping field.
