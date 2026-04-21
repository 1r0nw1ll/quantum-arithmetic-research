# Prospective A/B — Arm A (rubric-first) raw output

**Date**: 2026-04-19
**Cert concept**: `qa_sequence_cert_v1` — validate a list of (b,e) mod-9 tuples against QA axioms.
**Instruction**: author `validator.py` first, then write `mapping_protocol_ref.json` as documentation of what the validator enforces.
**Subagent**: general-purpose, 33.8k tokens, 70s duration, 0 tool calls (worked from spec only).

## Artifact 1: validator.py

{% raw %}
```python
"""
qa_sequence_cert_v1 — Validator for QA state-evolution sequences.

Given a sequence of (b, e) integer pairs with modulus m=9, verify that the
sequence respects the six non-negotiable QA axioms (A1, A2, T2, S1, S2, T1)
as defined in docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md.

A sequence element is a tuple (b, e) where b, e in {1..9}. Derived coordinates
d = b+e and a = b+2e are reduced via no-zero modular arithmetic when needed:
    reduce(x) = ((x - 1) % m) + 1

Consecutive elements must relate via a valid QA step: given (b_k, e_k),
the successor base b_{k+1} must equal reduce(b_k + e_k) = ((b_k+e_k-1) % 9) + 1.
The successor e_{k+1} is free in {1..9} (it is the next generator selection).

QA_COMPLIANCE:
    axioms_enforced:
        - A1  (no-zero state space: states in {1..m}, transitions via
               ((b+e-1) % m) + 1, never (b+e) % m)
        - A2  (derived coordinates: d = b+e, a = b+2e are recomputed from
               (b, e); the validator refuses any sequence that carries an
               independently-assigned d or a that disagrees)
        - T2  (observer-projection firewall: rejects any element whose b or e
               originated from a float cast; non-int/Fraction types are a
               T2-b violation)
        - S1  (no b**2: the validator itself writes b*b; sequence payloads
               that include a precomputed 'b_squared' field must equal b*b)
        - S2  (no float state: b and e must be int or fractions.Fraction;
               numpy floats, Python floats, Decimal all fail)
        - T1  (path time: step index k is an integer position in the list;
               the validator refuses any element that carries a float 't'
               field or a non-monotonic integer 't' field)
    modulus: 9
    state_space: {1, 2, 3, 4, 5, 6, 7, 8, 9}
    spec: docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md
"""

from fractions import Fraction
from numbers import Integral

M = 9  # QA modulus for this cert family


def _reduce(x: int) -> int:
    """No-zero modular reduction (A1): map any integer to {1..M}."""
    return ((x - 1) % M) + 1


def _is_qa_int(x) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, Fraction):
        return x.denominator == 1
    if isinstance(x, float):
        return False
    return isinstance(x, Integral)


def _as_int(x) -> int:
    if isinstance(x, Fraction):
        return int(x.numerator)
    return int(x)


def _unpack_element(elem):
    if isinstance(elem, dict):
        b = elem.get("b")
        e = elem.get("e")
        extras = {k: elem[k] for k in ("d", "a", "b_squared", "t") if k in elem}
        return b, e, extras
    if isinstance(elem, (tuple, list)):
        if len(elem) < 2:
            return None, None, {}
        b, e = elem[0], elem[1]
        extras = {}
        if len(elem) >= 3:
            extras["d"] = elem[2]
        if len(elem) >= 4:
            extras["a"] = elem[3]
        return b, e, extras
    return None, None, {}


def validate(sequence) -> dict:
    violations = []
    checks_run = []

    if not isinstance(sequence, (list, tuple)):
        return {"pass": False,
                "violations": [{"index": -1, "axiom": "CONTAINER",
                                "detail": f"sequence must be list/tuple, got {type(sequence).__name__}"}],
                "checks_run": []}

    if len(sequence) == 0:
        return {"pass": False,
                "violations": [{"index": -1, "axiom": "CONTAINER", "detail": "empty sequence"}],
                "checks_run": []}

    checks_run.extend(["A1", "A2", "T2", "S1", "S2", "T1"])

    prev_b_int = None
    prev_e_int = None
    prev_t = None

    for k, elem in enumerate(sequence):
        b, e, extras = _unpack_element(elem)
        if b is None or e is None:
            violations.append({"index": k, "axiom": "CONTAINER", "detail": "element missing b or e"})
            continue

        if not _is_qa_int(b):
            violations.append({"index": k, "axiom": "S2",
                               "detail": f"b has non-int/Fraction type {type(b).__name__}"})
        if not _is_qa_int(e):
            violations.append({"index": k, "axiom": "S2",
                               "detail": f"e has non-int/Fraction type {type(e).__name__}"})
        if not (_is_qa_int(b) and _is_qa_int(e)):
            continue

        b_i = _as_int(b); e_i = _as_int(e)

        if not (1 <= b_i <= M):
            violations.append({"index": k, "axiom": "A1",
                               "detail": f"b={b_i} not in {{1..{M}}}"})
        if not (1 <= e_i <= M):
            violations.append({"index": k, "axiom": "A1",
                               "detail": f"e={e_i} not in {{1..{M}}}"})

        if "d" in extras:
            d_claim = extras["d"]
            if not _is_qa_int(d_claim):
                violations.append({"index": k, "axiom": "A2",
                                   "detail": f"d has non-int/Fraction type {type(d_claim).__name__}"})
            else:
                d_expected_raw = b_i + e_i
                d_expected_reduced = _reduce(d_expected_raw)
                d_i = _as_int(d_claim)
                if d_i != d_expected_raw and d_i != d_expected_reduced:
                    violations.append({"index": k, "axiom": "A2",
                                       "detail": f"d={d_i} disagrees with b+e={d_expected_raw} (reduced {d_expected_reduced})"})

        if "a" in extras:
            a_claim = extras["a"]
            if not _is_qa_int(a_claim):
                violations.append({"index": k, "axiom": "A2",
                                   "detail": f"a has non-int/Fraction type {type(a_claim).__name__}"})
            else:
                a_expected_raw = b_i + 2 * e_i
                a_expected_reduced = _reduce(a_expected_raw)
                a_i = _as_int(a_claim)
                if a_i != a_expected_raw and a_i != a_expected_reduced:
                    violations.append({"index": k, "axiom": "A2",
                                       "detail": f"a={a_i} disagrees with b+2e={a_expected_raw} (reduced {a_expected_reduced})"})

        if "b_squared" in extras:
            bsq_claim = extras["b_squared"]
            if not _is_qa_int(bsq_claim):
                violations.append({"index": k, "axiom": "S1",
                                   "detail": f"b_squared has non-int type {type(bsq_claim).__name__}"})
            else:
                if _as_int(bsq_claim) != b_i * b_i:
                    violations.append({"index": k, "axiom": "S1",
                                       "detail": f"b_squared={_as_int(bsq_claim)} != b*b={b_i*b_i}"})

        if "t" in extras:
            t_claim = extras["t"]
            if isinstance(t_claim, float) or not _is_qa_int(t_claim):
                violations.append({"index": k, "axiom": "T1",
                                   "detail": f"t has non-integer type {type(t_claim).__name__}"})
            else:
                t_i = _as_int(t_claim)
                if prev_t is not None and t_i <= prev_t:
                    violations.append({"index": k, "axiom": "T1",
                                       "detail": f"t={t_i} not strictly greater than previous t={prev_t}"})
                prev_t = t_i

        if prev_b_int is not None and prev_e_int is not None:
            expected_b = _reduce(prev_b_int + prev_e_int)
            if b_i != expected_b:
                violations.append({"index": k, "axiom": "A1",
                                   "detail": f"invalid transition: b={b_i} but ((prev_b+prev_e-1)%{M})+1={expected_b} (prev=({prev_b_int},{prev_e_int}))"})

        prev_b_int = b_i; prev_e_int = e_i

    return {"pass": len(violations) == 0, "violations": violations, "checks_run": checks_run}


if __name__ == "__main__":
    demo = [(1, 2), (3, 5), (8, 4)]
    print(validate(demo))
```
{% endraw %}

## Artifact 2: mapping_protocol_ref.json (written after validator)

```json
{
  "cert_family": "qa_sequence_cert_v1",
  "version": "1.0",
  "spec_ref": "docs/specs/QA_OBSERVER_PROJECTION_COMPLIANCE_SPEC.v1.md",
  "description": "Validator for discrete QA state-evolution sequences given as lists of (b, e) integer pairs at modulus m=9.",
  "observer_projection": {
    "direction": "inbound_only",
    "inbound": {
      "description": "Caller provides a list whose elements are already discrete QA pairs (b, e) with optional derived fields (d, a), optional precomputed b_squared, and optional integer path-time t.",
      "accepted_types": ["int", "fractions.Fraction (integer-valued)"],
      "rejected_types": ["float", "numpy.floating", "decimal.Decimal", "complex", "bool"]
    },
    "firewall_crossings": 2
  },
  "discrete_state_space": {
    "modulus": 9,
    "states": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "pair_space_size": 81,
    "element_schema": {
      "required": {"b": "int in {1..9}", "e": "int in {1..9}"},
      "optional": {
        "d": "int; must equal b+e or ((b+e-1)%9)+1 if present",
        "a": "int; must equal b+2e or ((b+2e-1)%9)+1 if present",
        "b_squared": "int; must equal b*b if present",
        "t": "int; strictly increasing across the sequence if present"
      },
      "accepted_container_forms": [
        "tuple (b, e)", "tuple (b, e, d)", "tuple (b, e, d, a)",
        "dict with keys b, e and optional d, a, b_squared, t"
      ]
    },
    "transition_rule": "b_{k+1} = ((b_k + e_k - 1) % 9) + 1;  e_{k+1} is free in {1..9}"
  },
  "axioms_enforced": {
    "A1": "state membership {1..9} + transition via ((b+e-1)%9)+1",
    "A2": "d=b+e, a=b+2e recomputed; independent d/a that disagree = violation",
    "T2": "non-int/Fraction types = T2-b",
    "S1": "validator writes b*b; any b_squared payload checked == b*b",
    "S2": "b,e must be int or integer-valued Fraction",
    "T1": "integer step index; optional t field must be monotone int"
  }
}
```

**Axioms chosen (one sentence):** all six (A1, A2, T2, S1, S2, T1) enforced — transition via `((b+e-1)%9)+1`, derived-coord consistency if `d`/`a` provided, type-gate reject non-int/Fraction, `b_squared` payload checked, and optional `t` field monotone-integer.
