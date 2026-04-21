# Prospective A/B — Fibonacci Arm A (rubric-first) raw output

**Concept**: `qa_fibonacci_mod_cert_v1` — validate no-zero Fibonacci-mod-9 sequence.
**Instruction**: author `validator.py` FIRST, then write `mapping_protocol_ref.json` as documentation.
**Subagent**: general-purpose, 31.2k tokens, 48s duration.

## Artifact 1: validator.py

{% raw %}
```python
"""QA Fibonacci Mod Cert v1 — validator.

Verifies a no-zero Fibonacci-like sequence mod 9 is well-formed under the
A1-adjusted recurrence:

    seq[k] = ((seq[k-1] + seq[k-2] - 1) % 9) + 1   for k >= 2

All terms must lie in the no-zero state space {1, ..., 9}.

QA_COMPLIANCE:
    axioms: [A1, T2, S1, S2, T1]
    A1: states drawn from {1..9}; recurrence uses ((x-1)%9)+1 form.
    T2: no float->int casts; inputs rejected if non-integer.
    S1: arithmetic uses explicit + and *, never ** (no squaring needed here).
    S2: state is Python int only; Fraction tolerated but coerced-check'd.
    T1: index k is an integer path-length; no continuous time.
    A2: NOT APPLICABLE — this cert operates on scalar sequence terms, not
        (b, e) pairs, so derived coordinates d=b+e, a=b+2e do not arise.
"""

from fractions import Fraction
from typing import Any, Dict, List

MODULUS = 9
MIN_STATE = 1
MAX_STATE = MODULUS  # inclusive, no-zero


def _is_qa_int(x: Any) -> bool:
    """Accept Python int or integer-valued Fraction. Reject float/bool-as-trap."""
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator == 1
    return False


def _as_int(x: Any) -> int:
    if isinstance(x, Fraction):
        return x.numerator
    return int(x)


def _qa_step(prev2: int, prev1: int) -> int:
    """A1-adjusted Fibonacci recurrence on {1..MODULUS}.

    Formula: ((prev1 + prev2 - 1) % MODULUS) + 1
    No floats. No **. Pure integer arithmetic.
    """
    return ((prev1 + prev2 - 1) % MODULUS) + 1


def validate(sequence) -> Dict[str, Any]:
    """Validate a no-zero Fibonacci-mod-9 sequence.

    Parameters
    ----------
    sequence : list/tuple of positive ints in {1..9}

    Returns
    -------
    dict with keys:
        pass        : bool, True iff all checks pass
        violations  : list of human-readable violation strings
        checks_run  : list of check-name strings actually executed
    """
    violations: List[str] = []
    checks_run: List[str] = []

    checks_run.append("container_is_sequence")
    if not isinstance(sequence, (list, tuple)):
        violations.append(
            f"container_is_sequence: expected list/tuple, got {type(sequence).__name__}"
        )
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("length_at_least_2")
    if len(sequence) < 2:
        violations.append(
            f"length_at_least_2: sequence length {len(sequence)} < 2; "
            "need two seeds before recurrence applies"
        )

    checks_run.append("S2_T2_integer_state")
    typed: List[int] = []
    all_int = True
    for i, x in enumerate(sequence):
        if not _is_qa_int(x):
            all_int = False
            violations.append(
                f"S2_T2_integer_state: term at index {i} has non-integer "
                f"type/value (type={type(x).__name__}, value={x!r}); "
                "QA state must be int (floats violate T2 firewall)"
            )
        else:
            typed.append(_as_int(x))

    if not all_int:
        return {"pass": False, "violations": violations, "checks_run": checks_run}

    checks_run.append("A1_no_zero_range")
    for i, v in enumerate(typed):
        if v < MIN_STATE or v > MAX_STATE:
            violations.append(
                f"A1_no_zero_range: term at index {i} = {v} is outside "
                f"{{{MIN_STATE}..{MAX_STATE}}}; QA state space forbids 0"
            )

    checks_run.append("T1_recurrence_a1_adjusted")
    if len(typed) >= 2:
        for k in range(2, len(typed)):
            expected = _qa_step(typed[k - 2], typed[k - 1])
            actual = typed[k]
            if actual != expected:
                violations.append(
                    f"T1_recurrence_a1_adjusted: at step k={k}, "
                    f"expected ((seq[{k-1}] + seq[{k-2}] - 1) %% 9) + 1 "
                    f"= (({typed[k-1]} + {typed[k-2]} - 1) %% 9) + 1 "
                    f"= {expected}, got {actual}"
                )

    checks_run.append("S1_no_power_operator")

    return {
        "pass": len(violations) == 0,
        "violations": violations,
        "checks_run": checks_run,
    }


if __name__ == "__main__":
    good = [1, 1, 2, 3, 5, 8, 4, 3]
    bad_zero = [1, 0, 2, 3]
    bad_step = [1, 1, 2, 3, 6]
    bad_float = [1, 1, 2.0, 3]
    for name, s in [("good", good), ("bad_zero", bad_zero),
                    ("bad_step", bad_step), ("bad_float", bad_float)]:
        r = validate(s)
        print(name, r["pass"], r["violations"])
```
{% endraw %}

## Artifact 2: mapping_protocol_ref.json (written after validator)

```json
{
  "cert_family": "qa_fibonacci_mod_cert_v1",
  "schema_ref": "docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md",
  "description": "Well-formedness validator for no-zero Fibonacci-like sequences under the A1-adjusted recurrence on the modular state space {1..9}.",
  "observer_projection": {
    "kind": "identity_discrete",
    "input_layer": "integer sequence supplied by caller",
    "qa_layer": "scalar terms seq[k] in {1..9} evolving under ((seq[k-1]+seq[k-2]-1) mod 9) + 1",
    "output_layer": "boolean pass flag plus list of violation strings; no continuous output",
    "boundary_crossings": 0,
    "notes": "Validator is pure discrete->discrete. Theorem NT's firewall trivially satisfied."
  },
  "state_space": {
    "modulus": 9,
    "alphabet": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "zero_excluded": true,
    "element_type": "int (Fraction with denominator=1 tolerated; float/bool rejected)"
  },
  "dynamics": {
    "recurrence": "seq[k] = ((seq[k-1] + seq[k-2] - 1) mod 9) + 1, for k >= 2",
    "seed_requirement": "length(seq) >= 2; seeds must lie in {1..9}",
    "path_time": "k is an integer index (T1); no continuous time variable"
  },
  "applicable_axioms": {
    "A1_no_zero": {"applies": true, "enforcement": "A1_no_zero_range check rejects terms outside {1..9}."},
    "A2_derived_coords": {"applies": false, "reason": "This cert operates on scalar terms, not (b, e) pairs."},
    "T2_firewall": {"applies": true, "enforcement": "S2_T2_integer_state rejects float/bool inputs."},
    "S1_no_power": {"applies": true, "enforcement": "Validator uses only +, *, %. No ** operator."},
    "S2_no_float_state": {"applies": true, "enforcement": "State is Python int; Fraction only if denominator==1."},
    "T1_path_time": {"applies": true, "enforcement": "Loop index k is integer; recurrence check is T1_recurrence_a1_adjusted."}
  },
  "validator_entrypoint": {
    "module": "validator",
    "function": "validate",
    "signature": "validate(sequence) -> {pass: bool, violations: [str], checks_run: [str]}"
  }
}
```

**Axioms (one sentence):** A1 + T2 + S1 + S2 + T1 enforced; A2 declared NA because cert operates on scalar terms not (b,e) pairs.
